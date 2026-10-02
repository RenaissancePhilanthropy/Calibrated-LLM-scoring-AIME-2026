#!/usr/bin/env python3
"""Per-question contextual calibration — this produces the paper's Direct cell.

WHAT IT CORRECTS. A base model asked to continue "... the student's answer
is" already leans towards one label before it has read anything the student
wrote: the wording of the prompt alone is enough to skew it. Contextual
calibration (Zhao et al. 2021) measures that lean by scoring a content-free
input, then divides it back out of the real predictions.

WHY PER QUESTION. SciEntsBank asks each question of many students, and the
question and its reference answer are part of the prompt, so the lean
differs from one question to the next. For each question this script
renders the item with the student answer replaced by "[REDACTED]", reads
off the label distribution the model gives that content-free item, and
divides every sample of that question by it (then renormalises). Zhao et
al.'s own procedure estimates one content-free bias for the whole run.
We tried this approach as well, but the per-question variant produced
superior results for an acceptable amount of computational work.

INPUTS AND OUTPUTS. Runs over every full direct run under results/direct/,
reading each run's eval_per_sample.csv (the uncorrected probabilities) and
the exact prompt recorded in its run_config.toml. Writes, beside each:

    eval_per_sample_pq_redacted.csv    the corrected per-sample predictions
    eval_per_question_null.csv         the calibration they were divided by

"pq" is per-question, "redacted" is the null string used. The first file holds
the Direct method results used in all four figures; the second records the
content-free distribution per question, so that p_cal proportional to
p_raw / p_bar_q can be rechecked from the CSVs alone. Needs the GPU: it runs a
forward pass per (question, null), 135 x 4 for this split, and caches nothing
between runs.

DO NOT PRUNE THE OTHER CONTENT-FREE STRINGS. Four null strings are scored — "",
"N/A", "[MASK]", "[REDACTED]" — although only "[REDACTED]" is used. They go
through the model in one batched call, and dropping the other three would
repack those batches. fp16 log-probs depend on batch composition, so
pq_redacted's own numbers would move. The new numbers would of course be no
less correct, but they might be slightly different from what's published in the
paper, so for reproducibility's sake we keep all four.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
DIRECT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(DIRECT_DIR))
from scientsbank import load_eval, load_split, render_item  # noqa: E402
from common import ece_top1, write_eval_csv  # noqa: E402

# All four nulls are scored (batch-composition fidelity, see the module
# docstring); only the "redacted" column is turned into an output CSV.
NULLS = {"empty": "", "na": "N/A", "mask": "[MASK]", "redacted": "[REDACTED]"}
OUTPUT_NULL = "redacted"


def load_rows() -> tuple[list[dict], list[int]]:
    """Raw rows plus gold labels, from the same pinned split as load_eval."""
    _, labels, label_names = load_eval()
    assert label_names == ["correct", "incorrect"], label_names
    return [dict(r) for r in load_split()], labels


def score_cf(model: str, cfg: dict, cf_texts: list[str]) -> np.ndarray:
    from channel_models import NLLCalculator

    nll_kwargs: dict = {"model": model,
                        "model_cache_directory": str(PKG / "models")}
    if cfg["params"].get("key_name"):
        nll_kwargs["key_name"] = str(cfg["params"]["key_name"])
    calc = NLLCalculator(**nll_kwargs)
    d = cfg["derived"]
    label_contents = list(d["label_contents"])
    results = calc.compute_nll(
        contexts=cf_texts, contents=label_contents,
        batch_size=int(cfg["params"]["batch_size"]), use_tqdm=True,
        template=d.get("common_prefix", "") + d["template"])
    n, k = len(cf_texts), len(label_contents)
    log_lik = np.zeros((n, k))
    for kk, result in enumerate(results):
        df = result["df"]
        for i in range(n):
            log_lik[i, kk] = -float(df[f"t_nll_{i}"].sum())
    del calc
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return log_lik


def softmax_rows(a):
    s = a - a.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path,
                   default=PKG / "results" / "direct")
    args = p.parse_args()

    rows, labels = load_rows()
    true_idx = np.asarray(labels, dtype=np.int64)
    print(f"dataset: n={len(rows)}")

    qkeys, qindex = [], {}
    sample_q = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        key = (r["question"], r["reference_answer"])
        if key not in qindex:
            qindex[key] = len(qkeys)
            qkeys.append(key)
        sample_q[i] = qindex[key]
    null_names = list(NULLS)
    # Keep the full four-null context list: it fixes the batch composition of
    # the forward passes below, which the fp16 logits depend on.
    cf_texts = [render_item(q, ref, NULLS[nn]) for (q, ref) in qkeys
                for nn in null_names]
    print(f"unique questions: {len(qkeys)}; CF contexts: {len(cf_texts)}")

    out_ni = null_names.index(OUTPUT_NULL)
    out_name = f"pq_{OUTPUT_NULL}"

    for cfg_path in sorted(args.results_root.glob("*/run_config.toml")):
        run_dir = cfg_path.parent
        cfg = tomllib.load(open(cfg_path, "rb"))
        # Skip smoke runs (`--limit`): the paper's cells score all 540 samples.
        if cfg["run"].get("eval_size", 0) < 100:
            continue
        model = cfg["params"]["model"]
        label_names = list(cfg["derived"]["label_names"])
        df = pd.read_csv(run_dir / "eval_per_sample.csv")
        gold_seq = [label_names[i] for i in true_idx]
        if list(df["sample_label"]) != gold_seq:
            raise AssertionError(f"{run_dir}: gold-label order mismatch")
        p_raw = df[[f"prob_{n}" for n in label_names]].to_numpy(np.float64)

        print(f"\n=== {model} ({run_dir.name}) ===")
        log_lik = score_cf(model, cfg, cf_texts)
        p_cf = softmax_rows(log_lik).reshape(len(qkeys), len(null_names), -1)

        # Single-null variant: p_bar_q is just that null's per-question
        # counterfactual posterior (the mean over a one-element selection).
        p_bar_q = p_cf[:, [out_ni], :].mean(axis=1)          # (Q, 2)
        denom = p_bar_q[sample_q]                            # per question
        p_cal = p_raw / denom
        p_cal /= p_cal.sum(axis=1, keepdims=True)
        pred = p_cal.argmax(axis=1)
        correct = (pred == true_idx).astype(np.int64)
        acc = float(correct.mean())
        ece = float(ece_top1(p_cal, correct))
        write_eval_csv(run_dir / f"eval_per_sample_{out_name}.csv",
                       label_names, true_idx, pred, p_cal)
        # The calibration itself, so the correction can be checked against
        # eval_per_sample.csv without a GPU.
        pd.DataFrame({
            "question_index": np.arange(len(qkeys)),
            "question": [q for q, _ in qkeys],
            "reference_answer": [ref for _, ref in qkeys],
            "n_samples": np.bincount(sample_q, minlength=len(qkeys)),
            **{f"prob_{n}": p_bar_q[:, k] for k, n in enumerate(label_names)},
        }).to_csv(run_dir / "eval_per_question_null.csv", index=False)
        print(f"  {out_name:16s} acc={acc:.4f} ece15={ece:.4f}"
              f"  (nulls for {len(qkeys)} questions written)")


if __name__ == "__main__":
    main()
