"""Helpers shared by more than one pipeline stage."""

from __future__ import annotations

import csv
import difflib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w


def _git_state() -> dict[str, Any]:
    """Return {'git_commit', 'git_dirty'} for the current working tree.

    Shells out to ``git`` without adding a dependency. Any subprocess
    failure (git missing, not inside a repo, etc.) yields the sentinel
    ``{'git_commit': 'unknown', 'git_dirty': False}`` and does not raise.
    """
    sentinel: dict[str, Any] = {"git_commit": "unknown", "git_dirty": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return sentinel
    return {"git_commit": commit, "git_dirty": bool(porcelain.strip())}


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys with None values from a dict.

    TOML has no null type, so None values must be stripped before
    serialization.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = _strip_none(v)
        else:
            out[k] = v
    return out


# used by: channel, direct, judge
def write_run_config(
    path: Path,
    params: dict[str, Any],
    run_metadata: dict[str, Any],
    derived: dict[str, Any] | None = None,
) -> None:
    """Write effective config and run metadata as TOML.

    Produces a file with ``[params]`` and ``[run]`` sections. If
    ``derived`` is provided and non-empty, a ``[derived]`` section is
    written between the two.

    ``git_commit``, ``git_dirty``, and ``channel_models_version``
    are auto-populated into the ``[run]`` section. Caller-supplied values
    for those keys, if any, take precedence.

    Keys with None values are omitted (TOML has no null type).
    """
    auto_metadata: dict[str, Any] = {
        **_git_state(),
        "channel_models_version": importlib.metadata.version(
            "channel-models"
        ),
    }
    merged_run: dict[str, Any] = {**auto_metadata, **run_metadata}
    data: dict[str, Any] = {"params": _strip_none(params)}
    if derived:
        data["derived"] = _strip_none(derived)
    data["run"] = _strip_none(merged_run)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


# used by: export
def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error over per-sample confidences.

    Bins ``conf`` into ``n_bins`` equal-width bins on [0, 1], weights each
    bin by its share of samples, and sums the weighted absolute gap
    between the bin's accuracy and its mean confidence. The last bin is
    closed on the right so a confidence of exactly 1.0 falls inside it.

    The paper's captions quote the 10-bin figure; run_config.toml and the
    correction summaries record the 15-bin one.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        out += (n / total) * abs(
            float(correct[in_bin].mean()) - float(conf[in_bin].mean())
        )
    return float(out)


# used by: direct, judge
def ece_top1(
    probs: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width ECE on top-1 probability."""
    return ece(probs.max(axis=-1), correct, n_bins)


# used by: direct, judge
def write_eval_csv(
    path: Path,
    label_names: list[str],
    true_indices: np.ndarray,
    pred_indices: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Write per-sample softmax predictions to ``path``."""
    eps = 1e-12
    prob_cols = [f"prob_{name}" for name in label_names]
    fieldnames = [
        "sample_index",
        "sample_label",
        "predicted_label",
        "true_label_index",
        "correct",
        "top1_prob",
        "entropy",
        *prob_cols,
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (t, p, row_probs) in enumerate(
            zip(true_indices, pred_indices, probs)
        ):
            entropy = float(-(row_probs * np.log(row_probs + eps)).sum())
            row: dict[str, Any] = {
                "sample_index": i,
                "sample_label": label_names[int(t)],
                "predicted_label": label_names[int(p)],
                "true_label_index": int(t),
                "correct": int(int(t) == int(p)),
                "top1_prob": float(row_probs[int(p)]),
                "entropy": entropy,
            }
            for name, prob in zip(label_names, row_probs):
                row[f"prob_{name}"] = float(prob)
            writer.writerow(row)


# used by: judge, interpretability
def build_judge_user_message(
    template: str, ordered_labels: dict[str, str], input_text: str
) -> str:
    """Fill the categories list and the item into the judge's user turn.

    Shared so the Fig-4 prefix-reveal traces put exactly the same prompt
    in front of the judge as the main judge run does.
    """
    categories_str = "\n".join(
        f"- {name}: {desc}" for name, desc in ordered_labels.items()
    )
    return template.format(
        categories_str=categories_str, input_text=input_text
    )


# used by: judge, interpretability
def parse_label(
    generated_text: str,
    permitted_labels: list[str],
) -> tuple[str | None, float | None, str]:
    """Parse ``generated_text`` into one of ``permitted_labels``."""
    # 1) JSON: find the first {...} that loads and has a usable "label".
    for match in re.finditer(r"\{[^{}]*\}", generated_text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        label_val = obj.get("label") or obj.get("category") or obj.get("class")
        if not isinstance(label_val, str):
            continue
        for cand in permitted_labels:
            if label_val.strip().lower() == cand.lower():
                conf = obj.get("confidence")
                conf_f = float(conf) if isinstance(conf, (int, float)) else None
                return cand, conf_f, generated_text

    # 2) Word-boundary regex over each permitted label, longest first
    #    (so a multi-word label wins over a shorter label contained in it).
    for cand in sorted(permitted_labels, key=len, reverse=True):
        pat = re.compile(
            rf"\b{re.escape(cand)}\b", re.IGNORECASE
        )
        if pat.search(generated_text):
            return cand, None, generated_text

    # 3) Fuzzy: collapse to first non-empty line, fuzzy-match against
    #    permitted labels.
    needle = generated_text.strip().splitlines()
    needle_str = needle[0].lower() if needle else ""
    if needle_str:
        match = difflib.get_close_matches(
            needle_str,
            [c.lower() for c in permitted_labels],
            n=1,
            cutoff=0.6,
        )
        if match:
            for cand in permitted_labels:
                if cand.lower() == match[0]:
                    return cand, None, generated_text

    return None, None, generated_text
