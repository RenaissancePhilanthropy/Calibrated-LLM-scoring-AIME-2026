#!/usr/bin/env python3
"""Export the finished run data for the R paper figures.

Outputs:
  export/fig1_data.csv   one row per (model-row, method, variant): acc,
                    ece10 (paper caption convention), ece15, n.
                    Direct rows carry every variant CSV found on disk;
                    figures/paper_figs.R draws only DIRECT_VARIANT.
  figures/figdata/channel_9b.csv  compact per-sample panels for Figures 2-3
  figures/figdata/judge_9b.csv    (sample_index, p_correct, y_correct,
  figures/figdata/direct_9b.csv    entropy_bits, correct); direct = DIRECT_VARIANT.

Metric conventions: judge accuracy = verbalized label, judge ECE
confidence = max class prob; channel P(correct) = final-token cumulative
c_prob_0; channel entropy_bits = log2(2) - |info_gain|; direct/judge
entropy converted nats -> bits.
"""
from __future__ import annotations

import math
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
# common.py lives at the package root, one level up.
sys.path.insert(0, str(PKG))
from common import ece  # noqa: E402
EXPORT = PKG / "export"
FIGDATA = PKG / "figures" / "figdata"
CHANNEL_RESULTS = PKG / "results" / "channel"

# A paper cell scores the whole split. Channel run dirs are checked against
# the exact count; direct and judge record eval_size in run_config.toml, and
# anything well short of the full split is a --limit smoke run.
FULL_N = 540
MIN_FULL_RUN = 100


def metrics_row(
    top1: np.ndarray, pred_correct: np.ndarray, gold_correct: np.ndarray
) -> dict:
    """The four per-cell numbers the figures use: n, accuracy, ece10, ece15.

    ``top1`` is the confidence in the predicted class; ``pred_correct`` and
    ``gold_correct`` are boolean per-sample predictions and gold labels.
    """
    pred, gold = pred_correct.astype(int), gold_correct.astype(int)
    correct = (pred == gold).astype(int)
    return {
        "n": len(gold),
        "accuracy": round(float(correct.mean()), 4),
        "ece10": round(ece(top1, correct, 10), 4),
        "ece15": round(ece(top1, correct, 15), 4),
    }


DIRECT_VARIANT_RE = re.compile(r"eval_per_sample(?:_(?P<v>[a-z0-9_]+))?\.csv$")

# The direct variant the paper's figures show, everywhere: the Fig 2-3 direct
# panel written below, and figures/paper_figs.R's own `DIRECT_VARIANT` for
# Fig 1.
DIRECT_VARIANT = "pq_redacted"

JUDGE_ROWS = {
    ("Qwen/Qwen3.5-9B", "think"): "Qwen3.5-9B (think)",
    ("Qwen/Qwen3.5-9B", "nothink"): "Qwen3.5-9B (no think)",
    ("Qwen/Qwen3.5-4B", "think"): "Qwen3.5-4B (think)",
    ("Qwen/Qwen3.5-4B", "nothink"): "Qwen3.5-4B (no think)",
    ("Qwen/Qwen3.5-2B", "nothink"): "Qwen3.5-2B (no think)",
    ("Qwen/Qwen3.5-0.8B", "nothink"): "Qwen3.5-0.8B (no think)",
    ("meta-llama/Llama-3.2-3B-Instruct", "nothink"): "Llama-3.2-3B",
    ("meta-llama/Llama-3.2-1B-Instruct", "nothink"): "Llama-3.2-1B",
    ("gpt-5.5", "effort_medium"): "gpt-5.5 (think)",
    ("gpt-5.5", "effort_none"): "gpt-5.5 (no think)",
    ("gpt-4o", "nothink"): "gpt-4o",
    ("gpt-4o-mini", "nothink"): "gpt-4o-mini",
}
FAMILY = {"Qwen3.5": "Qwen3.5", "Llama": "Llama-3.2", "gpt": "OpenAI"}
BASE_ROWS = {
    "Qwen/Qwen3.5-9B-Base": "Qwen3.5-9B (no think)",
    "Qwen/Qwen3.5-4B-Base": "Qwen3.5-4B (no think)",
    "Qwen/Qwen3.5-2B-Base": "Qwen3.5-2B (no think)",
    "Qwen/Qwen3.5-0.8B-Base": "Qwen3.5-0.8B (no think)",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
}


def family_of(row_label: str) -> str:
    for k, v in FAMILY.items():
        if row_label.startswith(k):
            return v
    raise KeyError(f"no family prefix in FAMILY matches {row_label!r}")


def fig_row(label: str, method: str, variant: str, row: dict) -> dict:
    """One fig1_data.csv record, in the column order the R scripts read."""
    return {"model": label, "family": family_of(label),
            "method": method, "variant": variant,
            "acc": row["accuracy"], "ece": row["ece10"],
            "ece15": row["ece15"], "n": row["n"]}


def channel_final_runs() -> dict[str, Path]:
    """model -> run dir, for channel runs that scored all 540 samples."""
    out: dict[str, Path] = {}
    for p in sorted(CHANNEL_RESULTS.glob("*/run_config.toml")):
        cfg = tomllib.load(open(p, "rb"))
        if "common_context_format" not in cfg.get("params", {}):
            continue
        if not (p.parent / "eval_per_sample.csv").exists():
            continue
        if len(pd.read_csv(p.parent / "eval_per_sample.csv")) != FULL_N:
            continue
        out[cfg["params"]["model"]] = p.parent  # latest wins (sorted)
    return out


def channel_cell(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    sample_df = pd.read_csv(run_dir / "eval_per_sample.csv").sort_values("sample_index")
    token_df = pd.read_csv(run_dir / "eval_per_token.csv")
    last = token_df.groupby("sample_index").tail(1).set_index("sample_index").sort_index()
    p = last["c_prob_0"].reindex(sample_df["sample_index"]).to_numpy(float)
    gold = sample_df["sample_label"].str.upper().eq("CORRECT").to_numpy()
    row = metrics_row(top1=np.maximum(p, 1 - p),
                      pred_correct=p >= 0.5, gold_correct=gold)
    panel = pd.DataFrame({
        "sample_index": sample_df["sample_index"].to_numpy(),
        "p_correct": p,
        "y_correct": gold.astype(int),
        "entropy_bits": math.log2(2) - sample_df["info_gain"].abs().to_numpy(float),
        "correct": sample_df["accuracy"].to_numpy(float),
    })
    return row, panel


def per_sample_cell(csv_path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(csv_path).sort_values("sample_index")
    gold = df["sample_label"].eq("correct").to_numpy()
    p = df["prob_correct"].to_numpy(float)
    # Both methods score the written label: for judge that is the VERBALIZED
    # verdict, for direct the argmax label the runner recorded.
    pred = df["predicted_label"].eq("correct").to_numpy()
    row = metrics_row(top1=np.maximum(p, 1 - p),
                      pred_correct=pred, gold_correct=gold)
    panel = pd.DataFrame({
        "sample_index": df["sample_index"].to_numpy(),
        "p_correct": p,
        "y_correct": gold.astype(int),
        "entropy_bits": df["entropy"].to_numpy(float) / math.log(2),
        "correct": df["correct"].to_numpy(float),
    })
    return row, panel


def main() -> None:
    fig_rows = []
    panels: dict[str, pd.DataFrame] = {}

    # channel runs
    for model, run_dir in channel_final_runs().items():
        row, panel = channel_cell(run_dir)
        if model in BASE_ROWS:
            fig_rows.append(fig_row(BASE_ROWS[model], "channel", "", row))
        if model == "Qwen/Qwen3.5-9B-Base":
            panels["channel_9b"] = panel

    # direct runs, all variant CSVs present
    droot = PKG / "results" / "direct"
    for cfg_path in sorted(droot.glob("*/run_config.toml")):
        cfg = tomllib.load(open(cfg_path, "rb"))
        if cfg["run"].get("eval_size", 0) < MIN_FULL_RUN:
            continue
        model = cfg["params"]["model"]
        for csv_path in sorted(cfg_path.parent.glob("eval_per_sample*.csv")):
            m = DIRECT_VARIANT_RE.search(csv_path.name)
            variant = m.group("v") or "raw"
            row, panel = per_sample_cell(csv_path)
            if model in BASE_ROWS:
                fig_rows.append(
                    fig_row(BASE_ROWS[model], "direct", variant, row))
            if model == "Qwen/Qwen3.5-9B-Base" and variant == DIRECT_VARIANT:
                panels["direct_9b"] = panel

    # judge runs
    jroot = PKG / "results" / "judge"
    for cfg_path in sorted(jroot.glob("*/run_config.toml")):
        cfg = tomllib.load(open(cfg_path, "rb"))
        r = cfg["run"]
        if r.get("eval_size", 0) < MIN_FULL_RUN:
            continue
        model = r.get("model") or cfg["params"].get("model")
        effort = r.get("reasoning_effort_effective", "")
        if effort:
            variant = f"effort_{effort}"
        elif str(model).startswith("gpt-4o"):
            variant = "nothink"
        else:
            variant = "think" if r.get("thinking_mode") == "on" else "nothink"
        key = (model, variant)
        if key not in JUDGE_ROWS:
            continue
        row, panel = per_sample_cell(cfg_path.parent / "eval_per_sample.csv")
        fig_rows.append(fig_row(JUDGE_ROWS[key], "judge", "", row))
        if key == ("Qwen/Qwen3.5-9B", "nothink"):
            panels["judge_9b"] = panel

    out = pd.DataFrame(fig_rows).drop_duplicates(
        subset=["model", "method", "variant"], keep="last")
    out.to_csv(EXPORT / "fig1_data.csv", index=False)
    print(f"wrote fig1_data.csv ({len(out)} rows)")

    figdir = FIGDATA
    figdir.mkdir(exist_ok=True)
    for name, panel in panels.items():
        panel.to_csv(figdir / f"{name}.csv", index=False)
        print(f"wrote figdata/{name}.csv ({len(panel)} rows)")
    missing = {"channel_9b", "judge_9b", "direct_9b"} - set(panels)
    if missing:
        print(f"WARNING: missing panels: {missing}")


if __name__ == "__main__":
    main()
