"""The direct method: score the label tokens as a continuation of the item.

For each (sample x, label y), scores log P(y_tokens + terminator | x) using
NLLCalculator with:
    contexts = N sample texts
    contents = K label strings + terminator

Softmax across labels per sample produces calibrated per-class
probabilities. Including a terminator in each label content makes the K
candidate sequences mutually exclusive complete events (rather than nested
prefix probabilities), eliminating the spurious length bias of naive
direct scoring.

Output schema (sample_index, sample_label, predicted_label,
true_label_index, correct, top1_prob, entropy, prob_<class>...) is the
one common.write_eval_csv produces, shared with the judge method and read
by export/export_figure_data.py. (The channel method logs a different
per-sample schema, written by the vendored library.)
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax  # type: ignore[reportUnknownVariableType]

from channel_models import NLLCalculator

# scientsbank.py / common.py live at the package root, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ece_top1, write_eval_csv, write_run_config  # noqa: E402
from scientsbank import load_eval  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parent / "config.toml"
# The package ships one dataset; scientsbank.py registers no other loader.
DATASET = "scientsbank_2way_final"


def render_common_prefix(
    ordered_labels: dict[str, str], prefix_skeleton: str
) -> str:
    """Fill the categories list into the fixed head of the prompt.

    The head is the part that is identical for every (sample, label)
    pair. ``prompt_skeleton_suffix`` supplies the rest, keeping the
    literal {context} and {content} placeholders NLLCalculator
    substitutes per pair; ``main`` concatenates the two. That
    concatenation is the exact prompt text that produced the paper's
    numbers, so the whitespace at the join is load-bearing.
    """
    categories_str = "\n".join(
        f"- {name}: {desc}" for name, desc in ordered_labels.items()
    )
    return prefix_skeleton.format(categories_str=categories_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"TOML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap eval split at N samples for smoke testing.",
    )
    p.add_argument("--model", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "rb") as f:
        params: dict[str, Any] = tomllib.load(f)
    if args.model:
        params["model"] = args.model

    if DATASET not in params["datasets"]:
        raise RuntimeError(
            f"No [datasets.{DATASET}] section in {args.config}"
        )
    ds_cfg: dict[str, Any] = params["datasets"][DATASET]

    torch.manual_seed(int(params["seed"]))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Loading {DATASET}...")
    texts, labels, label_names = load_eval()
    N_full = len(texts)
    if args.limit is not None and N_full > args.limit:
        rng = np.random.default_rng(int(params["seed"]))
        idx = rng.choice(N_full, size=args.limit, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    N = len(texts)
    K = len(label_names)
    print(f"  N={N}/{N_full} K={K}")

    # Config-supplied descriptions, in the loader's label order. These are
    # part of the scored prompt, so a missing one is an error rather than
    # something to paper over with a verbalised label name.
    config_labels: dict[str, str] = ds_cfg["labels"]
    missing = [n for n in label_names if n not in config_labels]
    if missing:
        raise RuntimeError(
            f"[datasets.{DATASET}.labels] has no description for {missing}."
        )
    ordered_labels: dict[str, str] = {n: config_labels[n] for n in label_names}

    package_root = Path(__file__).resolve().parents[1]
    cache_dir = str(package_root / "models")

    print(f"Loading model {params['model']}...")
    nll_kwargs: dict[str, Any] = {
        "model": params["model"],
        "model_cache_directory": cache_dir,
    }
    # Forward the gated-model env-var name to NLLCalculator. Non-gated repos
    # ignore the resulting `token=` kwarg, so this is safe to always pass.
    if params.get("key_name"):
        nll_kwargs["key_name"] = str(params["key_name"])
    calculator = NLLCalculator(**nll_kwargs)

    # Appended to each label content so the K candidates are complete
    # events rather than nested prefixes. Recorded in run_config.toml as
    # terminator_resolved: it is part of the scored text, so a run is not
    # interpretable without it.
    terminator = str(params["terminator"])
    print(f"  terminator: {terminator!r}")

    label_contents = [name + terminator for name in label_names]

    # Token-count sanity print (cheap; helps spot tokenizer weirdness).
    tok = calculator.tokenizer
    token_counts = [
        len(tok.encode(lc, add_special_tokens=False)) for lc in label_contents
    ]
    print(
        f"  label-content token counts: "
        f"min={min(token_counts)} max={max(token_counts)} "
        f"mean={sum(token_counts) / len(token_counts):.1f}"
    )

    template = str(ds_cfg["prompt_skeleton_suffix"])
    common_prefix = render_common_prefix(
        ordered_labels, str(ds_cfg["prompt_skeleton_prefix"])
    )
    # Token-count sanity print for the fixed head of the prompt (cheap;
    # helps spot a mis-rendered categories list).
    prefix_tok_count = len(tok.encode(common_prefix, add_special_tokens=True))
    print(f"  common_prefix tokens: {prefix_tok_count}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = package_root / "results" / "direct" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring (flipped direction: contexts=samples, contents=labels)...")
    batch_size = int(params["batch_size"])
    # FLIPPED DIRECTION. Result has K dicts (one per label), each with a df
    # containing per-token NLLs in columns t_nll_0 ... t_nll_{N-1}. The
    # auto-computed c_prob_* columns softmax over CONTEXTS (samples), which
    # is the wrong axis for classification — we ignore them and softmax
    # the raw NLL sums over LABELS ourselves below.
    # The fixed head of the prompt is concatenated in front of the template,
    # so every (sample, label) forward pass re-reads it.
    results: list[dict[str, Any]] = calculator.compute_nll(
        contexts=texts,
        contents=label_contents,
        template=common_prefix + template,
        batch_size=batch_size,
        use_tqdm=True,
    )

    # Pivot from K result dicts (each scored under N contexts) to an
    # (N, K) log-prob matrix. log P(label_k | sample_i) = -sum(t_nll_i).
    log_prob = np.zeros((N, K), dtype=np.float64)
    for k, result in enumerate(results):
        df = result["df"]
        for i in range(N):
            log_prob[i, k] = -float(df[f"t_nll_{i}"].sum())

    # Softmax over labels per sample → per-class probabilities.
    probs = softmax(log_prob, axis=1).astype(np.float64)

    pred_indices = probs.argmax(axis=1)
    true_indices = np.asarray(labels, dtype=np.int64)
    correct = (pred_indices == true_indices).astype(np.int64)
    accuracy = float(correct.mean())
    test_ece = float(ece_top1(probs, correct))

    csv_path = results_dir / "eval_per_sample.csv"
    write_eval_csv(csv_path, label_names, true_indices, pred_indices, probs)
    print(f"  accuracy={accuracy:.4f}  ece={test_ece:.4f}  wrote {csv_path}")

    write_run_config(
        results_dir / "run_config.toml",
        params,
        {
            "timestamp": timestamp,
            "dataset": DATASET,
            "limit": args.limit,
            "eval_size": N,
            "num_classes": K,
            "test_accuracy": round(accuracy, 4),
            "test_ece": round(test_ece, 4),
            "device": str(calculator.device),
            "terminator_resolved": terminator,
        },
        derived={
            "label_names": label_names,            # loader's gold-label names
            "label_contents": label_contents,
            "label_descriptions": ordered_labels,
            "template": template,
            "common_prefix": common_prefix,
            "common_prefix_token_count": prefix_tok_count,
        },
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
