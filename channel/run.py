"""SciEntsBank classification with the channel-models library."""

import argparse
import os
import random
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root

from datasets import ClassLabel  # type: ignore[import-untyped]

from channel_models import NLLCalculator
from channel_models.csv_logging import AsyncCSVLogger

from common import write_run_config
from scientsbank import HF_SPLIT, load_split
from evaluate_per_sample_context import evaluate_with_per_sample_context

DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment parameters from a TOML file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    """Load the config file and apply CLI overrides.

    CLI args with non-None values override their config equivalents.
    ``config`` itself is skipped: it names the file, not a parameter.
    """
    params = load_config(args.config)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            params[key] = value
    return params


def format_content(content_format: str, **fields: str) -> str:
    """Apply content_format.format(**fields) with a readable error on typos."""
    try:
        return content_format.format(**fields)
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(
            f"content_format uses unknown placeholder {{{missing}}}; "
            f"format={content_format!r}, fields={sorted(fields)}"
        ) from e


def subsample(
    dataset: list[dict[str, str]], limit: int | None, seed: int = 42
) -> list[dict[str, str]]:
    """Shuffle and truncate the dataset, for --limit smoke runs.

    Shuffling first means a small sample still spans both classes, rather
    than taking the first N in dataset order.
    """
    if limit is None:
        return dataset
    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]

# TWO_WAY_MAPPING collapses nkazi/SciEntsBank's integer label (its ClassLabel
# index) into the two class names used by the config: class 0 is CORRECT, every
# other class is INCORRECT. It is keyed by the dataset's label order, recorded
# here; load_scientsbank_dataset checks the dataset still matches it, so a
# future reordering can't silently mislabel.
EXPECTED_HF_LABEL_NAMES = [
    "correct",
    "contradictory",
    "partially_correct_incomplete",
    "irrelevant",
    "non_domain",
]

TWO_WAY_MAPPING = {0: "CORRECT", 1: "INCORRECT", 2: "INCORRECT",
                   3: "INCORRECT", 4: "INCORRECT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SciEntsBank with the channel method",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to TOML config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate a random seeded subsample of N questions (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (overrides config)",
    )
    return parser.parse_args()


def load_scientsbank_dataset(
    content_format: str,
    common_context_format: str,
) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    """Load SciEntsBank and format it for the channel evaluate loop.

    Args:
        content_format: Per-sample template for the scored part of the
            prompt ({content}) — the student answer.
        common_context_format: Per-sample template for the unscored
            conditioning text ({common_context}) — the question and the
            reference answer.

    Returns:
        (formatted_dataset, hf_label_names, first_row_fields), where
        hf_label_names is the dataset's native label space and
        first_row_fields holds the raw fields of the first row.
    """
    ds = load_split()

    label_feature = ds.features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise TypeError(
            f"Expected the 'label' column to be a ClassLabel, got "
            f"{type(label_feature).__name__}. TWO_WAY_MAPPING is keyed by the "
            "ClassLabel index, so without one the collapse is unverifiable."
        )
    hf_label_names = list(label_feature.names)
    if hf_label_names != EXPECTED_HF_LABEL_NAMES:
        raise ValueError(
            "SciEntsBank label order does not match what TWO_WAY_MAPPING "
            f"assumes. Expected {EXPECTED_HF_LABEL_NAMES}, but the "
            f"dataset reports {hf_label_names}. Update the mapping and "
            "EXPECTED_HF_LABEL_NAMES before trusting the results."
        )

    first_row_fields: dict[str, str] | None = None
    formatted: list[dict[str, str]] = []
    for row in ds:
        if first_row_fields is None:
            first_row_fields = {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "student_answer": row["student_answer"],
            }
        entry: dict[str, str] = {
            "formatted_text": format_content(
                content_format,
                question=row["question"],
                reference_answer=row["reference_answer"],
                student_answer=row["student_answer"],
            ),
            "label": TWO_WAY_MAPPING[row["label"]],
            "common_context": format_content(
                common_context_format,
                question=row["question"],
                reference_answer=row["reference_answer"],
                student_answer=row["student_answer"],
            ),
        }
        formatted.append(entry)
    assert first_row_fields is not None, "SciEntsBank split was empty"
    return formatted, hf_label_names, first_row_fields


def main() -> None:
    """Run SciEntsBank evaluation."""
    args = parse_args()
    params = build_params(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    contexts = params["contexts"]

    # Check the config before touching the dataset, so a misconfigured run
    # fails fast (both inputs are config-only).
    expected = set(TWO_WAY_MAPPING.values())
    if expected != set(contexts.keys()):
        missing = sorted(expected - set(contexts.keys()))
        extra = sorted(set(contexts.keys()) - expected)
        raise ValueError(
            "Config [contexts] keys do not match TWO_WAY_MAPPING. "
            f"Missing from config: {missing}. Extra in config: {extra}."
        )

    dataset, hf_label_names, first_row_fields = load_scientsbank_dataset(
        content_format=params["content_format"],
        common_context_format=params["common_context_format"],
    )
    dataset = subsample(dataset, params.get("limit"))
    print(
        f"Loaded {len(dataset)} SciEntsBank questions "
        f"({HF_SPLIT} split, {len(contexts)} classes)"
    )

    print("Loading model...")
    calculator = NLLCalculator(
        model=params["model"],
        model_cache_directory=str(project_root / "models"),
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = script_dir.parent / "results" / "channel" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    common_context_format = params["common_context_format"]
    example_common_context = format_content(
        common_context_format, **first_row_fields
    )
    example_content = format_content(
        params["content_format"], **first_row_fields
    )
    derived = {
        "example_formatted_content_sample_0": example_content,
        "example_common_context_sample_0": example_common_context,
        "example_full_prompt_sample_0": params["template"].format(
            common_context=example_common_context,
            context=next(iter(contexts.values())),
            content=example_content,
        ),
        "hf_label_names": hf_label_names,
        "common_context_format": common_context_format,
        "label_mapping": {str(k): v for k, v in TWO_WAY_MAPPING.items()},
    }

    write_run_config(
        results_dir / "run_config.toml",
        params,
        {
            "timestamp": timestamp,
            "limit": params.get("limit"),
            "dataset_size": len(dataset),
            "num_classes": len(contexts),
            "device": str(calculator.device),
        },
        derived=derived,
    )

    print(f"Evaluating {len(dataset)} questions...")
    scores: list[dict[str, Any]] = evaluate_with_per_sample_context(
        calculator,
        formatted_dataset=dataset,
        contexts=list(contexts.values()),
        statement_mapping=contexts,
        template=params["template"],
        async_logger=AsyncCSVLogger(),
        sample_log_file_name=str(results_dir / "eval_per_sample.csv"),
        token_log_file_name=str(results_dir / "eval_per_token.csv"),
        use_tqdm=True,
        print_summary=True,
    )

    accuracy = sum(s["accuracy"] for s in scores) / len(scores)
    mean_info_gain = sum(s["info_gain"] for s in scores) / len(scores)

    print()
    print(f"Accuracy:        {accuracy:.3f}")
    print(f"Mean info gain:  {mean_info_gain:.4f}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
