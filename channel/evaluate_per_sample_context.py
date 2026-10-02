"""Channel `evaluate` with per-sample common_context.

A copy of NLLCalculator.evaluate from the vendored channel-models
library, modified so each sample can carry its own ``common_context``.
This lets the channel prompt place
per-sample conditioning (question + reference) before the per-class label
statement, scoring only the student answer.

Structural change from the base method: instead of one batched
``compute_nll_iter`` call over all contents with a single shared
``common_context``, loop per sample and call ``compute_nll_iter`` for
that sample's content with that sample's ``common_context`` (falling
back to the shared ``common_context`` arg when a sample has none). The
base method's role/regex token-scoping options are dropped; this package
scores the whole content.
"""
from __future__ import annotations

import csv
import logging
from functools import partial
from pathlib import Path
from typing import Any, cast

import pandas as pd
from tqdm import tqdm

from channel_models.csv_logging import AsyncCSVLogger
from channel_models.nll_calculator import NLLCalculator

logger = logging.getLogger(__name__)


def _write_token_csv(df: pd.DataFrame, path: str) -> None:
    """Round float columns to 4 decimals and write ``df`` to ``path``.

    Creates the CSV on the first call for a given path and appends on
    subsequent calls. Copied from the vendored channel_models.nll_calculator,
    where it is a private helper, so we carry our own copy.
    """
    df_out = df.copy()
    float_cols = df_out.select_dtypes(
        include=["float", "float64", "float32"]
    ).columns
    for col in float_cols:
        df_out[col] = df_out[col].round(4).map("{:.4f}".format)
    file_exists = Path(path).exists()
    df_out.to_csv(
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def evaluate_with_per_sample_context(
    calculator: NLLCalculator,
    contexts: list[str],
    formatted_dataset: list[dict[str, Any]],
    statement_mapping: dict[str, Any],
    template: str,
    async_logger: AsyncCSVLogger | None = None,
    sample_log_file_name: str | None = None,
    token_log_file_name: str | None = None,
    use_tqdm: bool = False,
    print_summary: bool = False,
) -> list[dict[str, Any]]:
    """Like ``NLLCalculator.evaluate``, but each ``formatted_dataset``
    entry carries its own ``"common_context"``."""
    if async_logger is None and (sample_log_file_name or token_log_file_name):
        raise ValueError(
            "sample_log_file_name and token_log_file_name need an async_logger."
        )

    scores: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(
        tqdm(formatted_dataset, disable=not use_tqdm)
    ):
        nll_scores = next(
            iter(
                calculator.compute_nll_iter(
                    contexts=contexts,
                    contents=[sample["formatted_text"]],
                    template=template,
                    common_context=sample["common_context"],
                    batch_size=8,
                    print_results=False,
                    use_tqdm=False,
                )
            )
        )

        most_likely_label: str = list(statement_mapping.items())[
            cast("int", nll_scores["most_likely_label_index"])
        ][0]
        true_label_index = list(statement_mapping.keys()).index(sample["label"])
        logger.info("True label: %s [%d]", sample["label"], true_label_index)
        logger.info(
            "Most likely label: %s [%d]",
            most_likely_label,
            nll_scores["most_likely_label_index"],
        )

        score: dict[str, Any] = {}
        if most_likely_label == sample["label"]:
            score["info_gain"] = nll_scores["info_gain"]
            score["llr"] = abs(nll_scores["llr"])
            score["accuracy"] = 1
        else:
            score["info_gain"] = -nll_scores["info_gain"]
            score["llr"] = -abs(nll_scores["llr"])
            score["accuracy"] = 0

        score["sample_index"] = sample_index
        score["sample_label"] = sample["label"]
        score["true_label_index"] = true_label_index
        score["predicted_label"] = most_likely_label

        scores.append(score)

        if async_logger:
            if sample_log_file_name is not None:
                async_logger.queue_csv_row(
                    sample_log_file_name, score, list(score.keys())
                )

            if token_log_file_name is not None:
                token_attributes = {
                    "sample_index": sample_index,
                    "sample_label": sample["label"],
                }
                attr_df = pd.DataFrame(
                    token_attributes, index=nll_scores["df"].index
                )
                nll_scores["df"] = pd.concat([attr_df, nll_scores["df"]], axis=1)  # type: ignore[reportUnknownMemberType]
                async_logger.queue_write(
                    partial(_write_token_csv, nll_scores["df"], token_log_file_name)
                )

    if print_summary and hasattr(calculator, "_print_evaluation_summary"):
        calculator._print_evaluation_summary(scores, statement_mapping)

    return scores
