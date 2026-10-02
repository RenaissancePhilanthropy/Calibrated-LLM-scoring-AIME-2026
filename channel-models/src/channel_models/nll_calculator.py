import csv
import logging
import math
import os
import re
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from matplotlib import cm
from matplotlib.colors import Colormap
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from scipy.special import log_softmax, softmax  # type: ignore[reportUnknownVariableType]  # scipy-stubs overloads contain Unknown
from sklearn.metrics import classification_report  # type: ignore[import-untyped]
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]
from termcolor import colored
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing_extensions import Self

from channel_models.csv_logging import AsyncCSVLogger

logger = logging.getLogger(__name__)

RESET = "\x1b[0m"
# PyTorch cross-entropy ignore index: tokens with this label are excluded from loss.
IGNORE_INDEX = -100
# In the confusion matrix display, labels longer than this are truncated.
CM_LABEL_MAX_LEN = 13
CM_LABEL_PREFIX_LEN = 10


@dataclass
class _TokenizedItem:
    """A single (content, context) pair after tokenization, before batching.

    Produced by ``_tokenize_pairs_iter``. All arrays are 1D numpy on
    CPU. ``_pad_chunk`` converts them to torch and moves to device in
    one step.

    Attributes:
        content_idx: Index into the ``contents`` list this item came from.
        context_idx: Index into the ``contexts`` list this item came from.
        input_ids: Token IDs for the full rendered text, shape ``(seq_len,)``.
        content_mask: Boolean mask, True for tokens in the scored region,
            shape ``(seq_len,)``.
        masked_token_ids: Token IDs of the scored region only. Used to
            verify the content-identity invariant across contexts.
        masked_token_strs: Token strings of the scored region only. Used to
            populate the ``t_str`` column in the result DataFrame.
    """

    content_idx: int
    context_idx: int
    input_ids: np.ndarray[tuple[int], np.dtype[np.int64]]
    content_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]
    masked_token_ids: np.ndarray[tuple[int], np.dtype[np.int64]]
    masked_token_strs: np.ndarray[tuple[int], np.dtype[np.str_]]


ResultBuilder = Callable[[list[np.ndarray], _TokenizedItem], dict[str, Any]]


@dataclass
class PreviewItem:
    """Preview of what a single (content, context) pair will score.

    Built by ``NLLCalculator.preview()`` and ``NLLCalculator.preview_multilabel()``
    using the same rendering and tokenization codepath as ``compute_nll`` /
    ``compute_nll_multilabel``, but without running the model. Inspect to
    verify that the correct text is being scored.

    Attributes:
        content_idx: Index into the ``contents`` list.
        context_idx: Index into the context list that was fed to the
            tokenizer. For single-label preview, this indexes the caller's
            ``contexts``. For multilabel preview, it indexes the flattened
            2L context list; prefer ``label_name`` and ``polarity``.
        rendered_text: The full rendered text that was sent to the
            tokenizer. This is exactly what the model sees.
        scored_char_spans: Half-open character offsets into
            ``rendered_text`` identifying the scored region.
        label_name: For multilabel preview, the name of the label this
            item corresponds to. ``None`` for single-label preview.
        polarity: For multilabel preview, ``"positive"`` or ``"negative"``
            indicating which claim of the label pair this item scores.
            ``None`` for single-label preview.
    """

    content_idx: int
    context_idx: int
    rendered_text: str
    scored_char_spans: list[tuple[int, int]]
    label_name: str | None = None
    polarity: Literal["positive", "negative"] | None = None

    @property
    def scored_text(self) -> str:
        """The scored portion of ``rendered_text``, concatenated."""
        return "".join(
            self.rendered_text[start:end] for start, end in self.scored_char_spans
        )

    def __repr__(self) -> str:
        if self.label_name is not None:
            header = (
                f"PreviewItem(content={self.content_idx}, "
                f"context={self.context_idx}, "
                f"label={self.label_name}, "
                f"polarity={self.polarity})"
            )
        else:
            header = (
                f"PreviewItem(content={self.content_idx}, context={self.context_idx})"
            )
        scored = self.scored_text
        return (
            f"{header}\n"
            f"--- Full text ---\n"
            f"{self.rendered_text}\n"
            f"--- Scored text ---\n"
            f"{scored}"
        )


def _merge_token_offsets(
    mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Collect character offsets of scored tokens and merge adjacent spans.

    Each token has a ``(start, end)`` character offset in the original
    text. This function keeps only the offsets where ``mask`` is True,
    then merges any that are adjacent or overlapping into contiguous
    spans.
    """
    spans: list[tuple[int, int]] = []
    for is_scored, (start, end) in zip(mask, offsets, strict=True):
        if is_scored:
            if spans and spans[-1][1] >= start:
                spans[-1] = (spans[-1][0], max(spans[-1][1], end))
            else:
                spans.append((start, end))
    return spans


@dataclass
class _Batch:
    """A padded group of sequences ready for a single model forward pass.

    Produced by ``_pad_chunk``. All tensors are 2D and reside on
    ``NLLCalculator._device``.

    Attributes:
        input_ids: Padded token IDs, shape ``(B, max_seq_len)``.
        attention_mask: Long tensor, 1 for real tokens, 0 for padding,
            shape ``(B, max_seq_len)``. Long dtype because HuggingFace
            models expect integer attention masks.
        content_mask: Bool tensor, True for scored tokens, False
            elsewhere (including padding), shape ``(B, max_seq_len)``.
            Bool dtype because it is only used internally for
            ``masked_fill``.
        items: The ``_TokenizedItem`` objects in this batch, in the same
            row order as the tensors. Carries metadata through the
            generator pipeline to ``_assemble_results_iter``.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor
    items: list[_TokenizedItem]


def _matched_group_span(m: re.Match[str]) -> tuple[int, int]:
    """Return (start, end) of the first non-None capture group in a match.

    When a regex has multiple capture groups joined by ``|``, only one
    group will match per hit.  This function finds which one and returns
    its half-open character span.

    Raises ValueError if no capture group matched (should never happen
    for a well-formed regex with at least one group).
    """
    for i in range(1, len(m.groups()) + 1):
        if m.group(i) is not None:
            return m.start(i), m.end(i)
    raise ValueError(
        f"Regex matched {m.group()!r} but no capture group was populated. "
        "Ensure regex_string contains at least one capture group, "
        "e.g. r'<tag>(.*?)</tag>'."
    )


def _ansi_fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def _ansi_bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


def _luminance(r: int, g: int, b: int) -> float:
    # sRGB relative luminance
    return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)


def _is_str_pair(obj: object) -> TypeGuard[tuple[str, str]]:
    """True if ``obj`` is a 2-tuple whose elements are both strings."""
    if not isinstance(obj, tuple):
        return False
    tup = cast("tuple[object, ...]", obj)
    return len(tup) == 2 and all(isinstance(x, str) for x in tup)  # noqa: PLR2004


def _normalize_labels(
    labels: dict[str, tuple[str, str]] | list[tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Coerce the multilabel API's two accepted input forms into one dict.

    The public multilabel methods accept ``labels`` either as a
    ``{name: (positive_claim, negative_claim)}`` dict or as a plain list
    of ``(positive, negative)`` pairs. This helper centralizes validation
    and conversion so the rest of the code only has to handle one shape.

    List inputs are given string names ``"0"``, ``"1"``, ... matching
    list order. Dict inputs are returned as a fresh dict preserving
    insertion order. Empty inputs, non-string keys, and malformed pairs
    (not a 2-tuple of strings) all raise ``ValueError`` with a message
    pointing at the offending entry.
    """
    if not labels:
        raise ValueError("labels must be non-empty")

    def _validate_pair(name: str, pair: object) -> tuple[str, str]:
        if _is_str_pair(pair):
            return pair
        raise ValueError(f"labels[{name}] must be a tuple of two strings, got {pair!r}")

    if isinstance(labels, list):
        return {str(i): _validate_pair(str(i), pair) for i, pair in enumerate(labels)}

    raw = cast("dict[object, object]", labels)
    bad_keys = [k for k in raw if not isinstance(k, str)]
    if bad_keys:
        raise ValueError(f"labels keys must all be strings; got {bad_keys!r}")
    return {name: _validate_pair(repr(name), pair) for name, pair in labels.items()}


def _compile_content_regex(regex_string: str | None) -> re.Pattern[str] | None:
    """Compile a user-supplied content regex, or return None if absent.

    Public methods (``compute_nll_iter``, ``compute_nll_multilabel_iter``,
    ``preview``, ``preview_multilabel``) all accept an optional
    ``regex_string`` that restricts scoring to the tokens inside a
    capture group (e.g. ``r"<assistant>(.*?)</assistant>"``). This
    helper centralizes the compilation flags and the "must have at least
    one capture group" check that every caller would otherwise repeat.

    ``None`` in, ``None`` out. A pattern without any capture group
    raises ``ValueError``.
    """
    if regex_string is None:
        return None
    pattern = re.compile(regex_string, flags=re.DOTALL | re.IGNORECASE)
    if pattern.groups == 0:
        raise ValueError(
            "regex_string must contain at least one capture group, "
            "e.g. r'<tag>(.*?)</tag>' not r'<tag>.*?</tag>'"
        )
    return pattern


_POLARITY_LABELS: tuple[Literal["positive"], Literal["negative"]] = (
    "positive",
    "negative",
)


def _normalize_threshold(
    threshold: float | dict[str, float],
    label_names: list[str],
) -> dict[str, float]:
    """Normalize a scalar or per-label threshold into a per-label dict.

    Scalar inputs are broadcast to every label. Dict inputs must cover the
    full label set with no unknown keys.

    Returns a fresh dict keyed by ``label_names`` in insertion order.
    """
    if isinstance(threshold, dict):
        threshold_keys = set(threshold.keys())
        label_name_set = set(label_names)
        missing = label_name_set - threshold_keys
        extra = threshold_keys - label_name_set
        if missing or extra:
            parts: list[str] = []
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            if extra:
                parts.append(f"unknown keys: {sorted(extra)}")
            raise ValueError(
                "threshold dict keys must match labels exactly; " + "; ".join(parts)
            )
        return {name: float(threshold[name]) for name in label_names}
    return dict.fromkeys(label_names, float(threshold))


def _write_token_csv(df: pd.DataFrame, path: str) -> None:
    """Round float columns to 4 decimals and write ``df`` to ``path``.

    Creates the CSV on the first call for a given path and appends on
    subsequent calls. Non-numeric fields are quoted for consistent CSV
    output across the create and append paths.
    """
    df_out = df.copy()  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
    float_cols = df_out.select_dtypes(  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
        include=["float", "float64", "float32"]
    ).columns
    for col in float_cols:  # type: ignore[reportUnknownVariableType]
        df_out[col] = df_out[col].round(4).map("{:.4f}".format)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    file_exists = Path(path).exists()
    df_out.to_csv(  # type: ignore[reportUnknownMemberType]
        path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def _assemble_multilabel_summary(
    labels_dict: dict[str, dict[str, Any]],
    threshold: dict[str, float],
) -> pd.DataFrame:
    """Roll per-label result dicts up into a single summary DataFrame.

    One row per label (in ``labels_dict`` insertion order), indexed by
    label name. Every value is read off the per-label DataFrame: the
    cumulative values (``llr``, ``prob``, ``info_gain``) come from the
    last row, and the NLL totals are sums across all scored tokens.
    The ``predicted`` column is the boolean ``prob >= threshold[name]``.

    Intended for callers who want a compact per-label overview without
    touching the 2-context DataFrames directly.
    """
    rows: dict[str, dict[str, Any]] = {}
    for name, sub in labels_dict.items():
        df = sub["df"]
        pos_nll_total = float(df["t_nll_0"].to_numpy().sum())
        neg_nll_total = float(df["t_nll_1"].to_numpy().sum())
        llr = float(df["c_llr"].to_numpy()[-1])
        prob = float(df["c_prob_0"].to_numpy()[-1])
        info_gain = float(df["c_info_gain"].to_numpy()[-1])
        rows[name] = {
            "pos_nll_total": pos_nll_total,
            "neg_nll_total": neg_nll_total,
            "llr": llr,
            "prob": prob,
            "info_gain": info_gain,
            "predicted": prob >= threshold[name],
        }
    # pandas-stubs types from_dict's overloads with partial Unknowns.
    return pd.DataFrame.from_dict(rows, orient="index")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]


class NLLCalculator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Environment configuration
    env_location: str | None = Field(
        default=None,
        description=(
            "Path to a .env file to load. If None (the default), a .env file "
            "in the working directory is loaded when present. If an explicit "
            "path is given and the file does not exist, FileNotFoundError is raised."
        ),
    )
    model_cache_directory: str = Field(
        default="models/",
        description="Path to the directory where local models will be cached",
    )

    # Model configuration
    model: str = Field(
        ..., description="Name of the model to use for likelihood calculations."
    )
    device: str = Field(
        default="default", description="Device to use for likelihood calculations."
    )
    key_name: str | None = Field(
        default=None,
        description=(
            "Environment variable name holding the API token for the model. "
            "Only needed for gated models."
        ),
    )
    max_text_length: int = Field(
        default=256,
        description="Maximum number of characters to use for context+content.",
    )

    # Private attributes
    _model: PreTrainedModel = PrivateAttr()
    # TODO: Must we have both _device for the model object and device for the string
    # name?
    _device: torch.device = PrivateAttr()
    tokenizer: PreTrainedTokenizerBase | None = Field(
        default=None
    )  # public so callers can tokenize with the model's own tokenizer
    special_token_padding: int = Field(
        default=0,
        description=(
            "Count of special tokens appended to the beginning of "
            "the string when add_special_tokens=True."
        ),
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Load environment variables from a .env file.
        if self.env_location is not None:
            if not Path(self.env_location).exists():
                raise FileNotFoundError(
                    f"env_location {self.env_location} does not exist"
                )
            load_dotenv(self.env_location)
        else:
            # Auto-load .env from working directory if present.
            load_dotenv(".env")

        # Ask to create model cache directory if it doesn't exist
        self.model_cache_directory = os.getenv("CACHE_DIR", self.model_cache_directory)
        if not self.model_cache_directory.endswith("/"):
            self.model_cache_directory += "/"
            if not Path(self.model_cache_directory).exists():
                logger.info(
                    "Creating model cache directory %s", self.model_cache_directory
                )
                Path(self.model_cache_directory).mkdir(parents=True, exist_ok=True)

        # set device
        if self.device == "default":
            # If multiple GPUs are available, default to using the first one ("cuda:0").
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    logger.info(
                        "Multiple GPUs detected (%d). Using cuda:0 by default.",
                        torch.cuda.device_count(),
                    )
                self._device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS device")
                self._device = torch.device("mps")
            else:
                logger.info("Using CPU device")
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.device)

        # load model from huggingface
        model_kwargs = {
            "cache_dir": self.model_cache_directory,
            "token": os.getenv(self.key_name) if self.key_name else None,
            "torch_dtype": (
                torch.float16 if "cuda" in self._device.type else torch.float32
            ),
            "device_map": "cuda" if "cuda" in self._device.type else None,
        }

        self._model = cast(
            "PreTrainedModel",
            AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs),  # type: ignore[reportUnknownMemberType]  # transformers stubs partially unknown
        )
        self.tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(self.model, **model_kwargs),  # type: ignore[reportUnknownMemberType]  # transformers stubs partially unknown
        )
        # Pad token setup required for batched forward passes
        if self.tokenizer.pad_token is None:  # type: ignore[reportUnknownMemberType]  # transformers stubs partially unknown
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[reportUnknownMemberType]  # transformers stubs partially unknown
        self._model.config.pad_token_id = self.tokenizer.pad_token_id  # type: ignore[reportUnknownMemberType]  # transformers stubs partially unknown
        self.tokenizer.padding_side = "right"

        # Manually move model to device if not using device_map
        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            self._model = self._model.to(self._device)  # type: ignore[reportUnknownMemberType, reportArgumentType]  # transformers stubs partially unknown

        # Get number of special tokens added (BOS tokens)
        self.special_token_padding = self.tokenizer.num_special_tokens_to_add(
            pair=False
        )

        return self

    @torch.inference_mode()
    def _compute_masked_nll(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        synchronize: bool = False,
    ) -> tuple[CausalLMOutputWithPast, torch.Tensor]:
        """Run the model forward pass and prepare labels for NLL extraction.

        Builds ``label_tokens`` from ``token_ids`` by writing
        ``IGNORE_INDEX`` (-100) at every position where ``mask`` is
        False. Both tensors are then passed to the model: ``token_ids``
        as ``input_ids`` and ``label_tokens`` as ``labels``.

        The caller (``_extract_per_token_nll``) uses the returned
        ``label_tokens`` to identify which positions carry scored NLL
        values. Positions marked ``IGNORE_INDEX`` are excluded.

        Args:
            token_ids: Input token IDs, shape ``(B, seq_len)``.
            mask: Content mask (``_Batch.content_mask``). Bool tensor,
                True for scored tokens, shape ``(B, seq_len)``.
            attention_mask: Long tensor, 1 for real tokens, 0 for
                padding, shape ``(B, seq_len)``. Must be provided
                whenever sequences in the batch have been padded to
                different lengths.
            synchronize: If True, call ``torch.cuda.synchronize()``
                after key operations. Useful for profiling to force
                the CPU to wait for GPU completion at each step.

        Returns:
            ``(model_outputs, label_tokens)`` where
            ``model_outputs.logits`` has shape
            ``(B, seq_len, vocab_size)`` and ``label_tokens`` has shape
            ``(B, seq_len)`` with ``IGNORE_INDEX`` at non-scored
            positions.
        """
        expected_ndim = 2
        assert token_ids.ndim == expected_ndim, (
            f"token_ids must be a 2D tensor (B, seq_len), got shape {token_ids.shape}"
        )
        if synchronize:
            torch.cuda.synchronize()
        assert mask.shape == token_ids.shape, (
            f"Mask must be the same shape as the input. Got shape {mask.shape}"
        )

        label_tokens = token_ids.masked_fill(~mask, IGNORE_INDEX)
        if synchronize:
            torch.cuda.synchronize()
        outputs = self._model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            labels=label_tokens,  # -100 ignores pads/unscored tokens in loss
        )
        if synchronize:
            torch.cuda.synchronize()
        return outputs, label_tokens

    @staticmethod
    def _validate_template(
        template: str,
        common_context: str,
    ) -> tuple[str, str]:
        """Validate a template string and split it into prefix and suffix.

        Returns (prefix_template, suffix_template) where the full text is
        rendered as:
        prefix_template.format(...) + content + suffix_template.format(...)

        Raises ValueError if the template is invalid.
        """
        if "{content}" not in template:
            raise ValueError("Template must contain {content} placeholder.")
        if "{context}" not in template:
            raise ValueError("Template must contain {context} placeholder.")
        if common_context and "{common_context}" not in template:
            raise ValueError(
                "common_context is non-empty but template does not contain "
                "{common_context} placeholder."
            )

        allowed = {"common_context", "context", "content"}
        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is not None and field_name not in allowed:
                raise ValueError(
                    f"Unrecognized placeholder '{{{field_name}}}' in template. "
                    f"Allowed placeholders: {{{', '.join(sorted(allowed))}}}"
                )

        prefix_template, suffix_template = template.split("{content}", maxsplit=1)
        return prefix_template, suffix_template

    @staticmethod
    def _render_and_resolve_spans(
        content: str,
        context: str,
        common_context: str,
        prefix_template: str,
        suffix_template: str,
        content_regex: re.Pattern[str] | None,
    ) -> tuple[str, list[tuple[int, int]]]:
        """Render template and compute scored character spans.

        Assembles the full text from prefix + content + suffix, then
        resolves which character ranges should be scored: either the
        entire content region (when no regex) or the capture-group
        matches within the content (when regex is provided).

        Args:
            content: The content string (inserted literally).
            context: The context string for this pair.
            common_context: Shared context string.
            prefix_template: Template portion before ``{content}``.
            suffix_template: Template portion after ``{content}``.
            content_regex: If given, only capture-group matches
                within the content are scored.

        Returns:
            ``(full_text, char_spans)`` where ``char_spans`` are
            half-open character offsets into ``full_text``.
        """
        rendered_prefix = prefix_template.format(
            common_context=common_context, context=context
        )
        rendered_suffix = suffix_template.format(
            common_context=common_context, context=context
        )
        full_text = rendered_prefix + content + rendered_suffix
        content_start = len(rendered_prefix)
        content_end = content_start + len(content)

        if content_regex is None:
            char_spans: list[tuple[int, int]] = [(content_start, content_end)]
        else:
            char_spans = []
            for m in content_regex.finditer(content):
                s, e = _matched_group_span(m)
                char_spans.append((content_start + s, content_start + e))

        return full_text, char_spans

    def compute_nll_iter(
        self,
        contexts: list[str],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        batch_size: int = 8,
        print_results: bool = False,
        use_tqdm: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield one NLL result dict per content string, streaming.

        For each content, the model scores it under every context. NLL
        values are compared across contexts to produce log-likelihood
        ratios, entropy, and information gain. Results are yielded as
        soon as all C context scores for a content are available, so
        the caller can process them without waiting for the full run
        or holding all results in memory.

        Args:
            contexts: C context strings. Each is inserted into the
                template via ``{context}``. Every content is scored
                under all C contexts.
            contents: N content strings to score. One result dict is
                yielded per content.
            template: Format string controlling how context and content
                are assembled into the text fed to the model. Must
                contain ``{context}`` and ``{content}``; may also
                contain ``{common_context}``. The portion before
                ``{content}`` becomes the prefix (not scored); the
                portion after becomes the suffix (not scored).
            common_context: String substituted into the
                ``{common_context}`` placeholder. Pass empty string
                (the default) if the template has no such placeholder.
            regex_string: If given, a regex applied to each content
                string. Only text matched by its capture group(s) is
                scored; the rest of the content is treated as unscored
                context. Must contain at least one capture group.
            batch_size: Maximum number of sequences per forward pass.
            print_results: If True, print a per-token table after each
                content is assembled.
            use_tqdm: If True, wrap forward passes in a tqdm progress
                bar.

        Yields:
            One result dict per content, in the same order as
            ``contents``. Each dict contains:

            - ``df`` -- DataFrame with one row per scored token and
              columns ``t_ids``, ``t_str``, ``t_nll_0`` ...
              ``t_nll_{C-1}``, ``t_llr``, ``c_llr``, and (when C > 1)
              ``t_entropy``, ``c_entropy``, ``c_best_index``,
              ``c_info_gain``, ``c_prob_0`` ... ``c_prob_{C-1}``.
            - ``llr`` -- Final cumulative log-likelihood ratio (float,
              or NaN when C != 2).
            - ``entropy`` -- Final cumulative entropy in bits (float,
              or NaN when C == 1).
            - ``info_gain`` -- ``log2(C) - entropy`` (float, or NaN
              when C == 1).
            - ``most_likely_label_index`` -- Index of the context with
              the highest cumulative probability at the last token.

        Raises:
            ValueError: If ``contexts`` is empty, ``batch_size`` < 1,
                or ``regex_string`` has no capture group.
        """
        if not contexts:
            raise ValueError("contexts must be a non-empty list")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        prefix_template, suffix_template = self._validate_template(
            template, common_context
        )

        content_regex = _compile_content_regex(regex_string)

        n_items = len(contents) * len(contexts)
        n_batches = math.ceil(n_items / batch_size)

        items = self._tokenize_pairs_iter(
            contexts=contexts,
            contents=contents,
            common_context=common_context,
            prefix_template=prefix_template,
            suffix_template=suffix_template,
            content_regex=content_regex,
        )
        batches = self._build_batches_iter(items, batch_size=batch_size)
        nll_batches = self._run_forward_iter(
            batches, use_tqdm=use_tqdm, n_batches=n_batches
        )
        yield from self._assemble_results_iter(
            nll_batches,
            n_contexts=len(contexts),
            result_builder=self._build_result_dict,
            print_results=print_results,
        )

    @torch.inference_mode()
    def compute_nll(
        self,
        contexts: list[str],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        batch_size: int = 8,
        print_results: bool = False,
        use_tqdm: bool = False,
    ) -> list[dict[str, Any]]:
        """Compute NLL scores for all contents and return as a list.

        Eager counterpart to ``compute_nll_iter``. All parameters and
        result dict structure are identical; see that method's docstring
        for details.

        If streaming of results is not needed, this method may be faster
        than `compute_nll_iter`, since it optimises the order in which
        to iterate through `contents`.

        Returns:
            List of N result dicts, one per content, in the same order
            as ``contents``. Returns an empty list when ``contents``
            is empty.
        """
        if not contents:
            return []

        sorted_indices = sorted(range(len(contents)), key=lambda i: len(contents[i]))
        sorted_contents = [contents[i] for i in sorted_indices]

        sorted_results = list(
            self.compute_nll_iter(
                contexts=contexts,
                contents=sorted_contents,
                template=template,
                common_context=common_context,
                regex_string=regex_string,
                batch_size=batch_size,
                print_results=print_results,
                use_tqdm=use_tqdm,
            )
        )

        # Unsort back to input order
        results: list[dict[str, Any] | None] = [None] * len(contents)
        for sorted_pos, orig_pos in enumerate(sorted_indices):
            results[orig_pos] = sorted_results[sorted_pos]
        return cast("list[dict[str, Any]]", results)

    def compute_nll_multilabel_iter(
        self,
        labels: dict[str, tuple[str, str]] | list[tuple[str, str]],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        threshold: float | dict[str, float] = 0.5,
        batch_size: int = 8,
        print_results: bool = False,
        use_tqdm: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield one multilabel result dict per content string, streaming.

        For each content, every label is scored by feeding its positive
        and negative claims through the model via ``template`` and
        comparing the two cumulative log-likelihoods; softmax of the
        pair gives the probability that the positive claim (and
        therefore the label) applies to the content. Labels are scored
        independently, so the resulting probabilities are Bernoulli
        estimates of each label's applicability and do not form a joint
        distribution across labels.

        Call ``preview_multilabel`` first to sanity-check the template
        and scoring spans. For a non-streaming variant, see
        ``compute_nll_multilabel``.

        Args:
            labels: Label set as either
                ``{name: (positive_claim, negative_claim)}`` (insertion
                order preserved) or ``[(positive_claim, negative_claim),
                ...]`` (names default to ``"0", "1", ...``). Example:
                ``{"polite": ("The assistant is polite.",
                "The assistant is rude.")}``.
            contents: N content strings to score. One result dict is
                yielded per content, in input order.
            template: Same as ``compute_nll_iter``; shared across all
                labels.
            common_context: Same as ``compute_nll_iter``.
            regex_string: Same as ``compute_nll_iter``.
            threshold: Cutoff at or above which a label is marked
                ``predicted``. A scalar is broadcast to every label; a
                ``dict[str, float]`` specifies per-label cutoffs and
                must cover exactly the label set. Only affects the
                ``predicted`` column of ``summary`` and the
                ``predicted_labels`` list; probabilities, NLLs, and
                other numeric fields are threshold-independent.
            batch_size: Same as ``compute_nll_iter``.
            print_results: If True, print the per-token table once per
                label per content, with the label name as a header.
                Noisy for many labels.
            use_tqdm: Same as ``compute_nll_iter``.

        Yields:
            One result dict per content, in ``contents`` order. Each
            dict contains:

            - ``labels`` -- ``{label_name: per_label_result}`` where
              each ``per_label_result`` has the shape of a 2-context
              ``compute_nll_iter`` result: a ``df`` DataFrame with per-
              token NLLs for the positive and negative claims plus
              cumulative ``c_llr``, ``c_prob_0``, ``c_prob_1``, and
              ``c_info_gain`` columns, alongside the scalars ``llr``,
              ``entropy``, ``info_gain``, and
              ``most_likely_label_index``.
            - ``summary`` -- DataFrame indexed by label name with
              columns ``pos_nll_total``, ``neg_nll_total``, ``llr``,
              ``prob``, ``info_gain``, and ``predicted``.
            - ``predicted_labels`` -- list of label names whose
              probability is at or above their threshold, in label
              definition order.

        Raises:
            ValueError: If ``labels`` is empty or contains malformed
                entries; ``threshold`` is a dict whose keys don't match
                the label set; ``batch_size`` < 1; or ``regex_string``
                has no capture group.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        label_dict = _normalize_labels(labels)
        label_names = list(label_dict.keys())
        threshold_dict = _normalize_threshold(threshold, label_names)
        label_specs: list[tuple[str, int, int]] = []
        contexts: list[str] = []
        for name, (pos, neg) in label_dict.items():
            pos_idx = len(contexts)
            contexts.append(pos)
            neg_idx = len(contexts)
            contexts.append(neg)
            label_specs.append((name, pos_idx, neg_idx))

        prefix_template, suffix_template = self._validate_template(
            template, common_context
        )

        content_regex = _compile_content_regex(regex_string)

        n_items = len(contents) * len(contexts)
        n_batches = math.ceil(n_items / batch_size)

        items = self._tokenize_pairs_iter(
            contexts=contexts,
            contents=contents,
            common_context=common_context,
            prefix_template=prefix_template,
            suffix_template=suffix_template,
            content_regex=content_regex,
        )
        batches = self._build_batches_iter(items, batch_size=batch_size)
        nll_batches = self._run_forward_iter(
            batches, use_tqdm=use_tqdm, n_batches=n_batches
        )

        builder = partial(
            self._build_multilabel_result_dict,
            label_specs=label_specs,
            threshold=threshold_dict,
        )

        for result in self._assemble_results_iter(
            nll_batches,
            n_contexts=len(contexts),
            result_builder=builder,
            print_results=False,
        ):
            if print_results:
                for name, sub in result["labels"].items():
                    logger.info("Label: %s", name)
                    self._print_results(results_df=sub["df"])
            yield result

    @torch.inference_mode()
    def compute_nll_multilabel(
        self,
        labels: dict[str, tuple[str, str]] | list[tuple[str, str]],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        threshold: float | dict[str, float] = 0.5,
        batch_size: int = 8,
        print_results: bool = False,
        use_tqdm: bool = False,
    ) -> list[dict[str, Any]]:
        """Compute multilabel NLL scores for all contents as a list.

        Eager counterpart to ``compute_nll_multilabel_iter``. Sorts contents
        by length internally for batching efficiency, then unsorts the
        results back to input order. See ``compute_nll_multilabel_iter`` for
        parameter and return details.
        """
        if not contents:
            return []

        sorted_indices = sorted(range(len(contents)), key=lambda i: len(contents[i]))
        sorted_contents = [contents[i] for i in sorted_indices]

        sorted_results = list(
            self.compute_nll_multilabel_iter(
                labels=labels,
                contents=sorted_contents,
                template=template,
                common_context=common_context,
                regex_string=regex_string,
                threshold=threshold,
                batch_size=batch_size,
                print_results=print_results,
                use_tqdm=use_tqdm,
            )
        )

        results: list[dict[str, Any] | None] = [None] * len(contents)
        for sorted_pos, orig_pos in enumerate(sorted_indices):
            results[orig_pos] = sorted_results[sorted_pos]
        return cast("list[dict[str, Any]]", results)

    def preview(
        self,
        contexts: list[str],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        max_samples: int = 3,
    ) -> list[PreviewItem]:
        """Preview which tokens would be scored, without running the model.

        Uses the same rendering and tokenization codepath as
        ``compute_nll``, so the preview is guaranteed to match what the
        model would actually receive and score.

        Args:
            contexts: Context strings (same as ``compute_nll``).
            contents: Content strings (same as ``compute_nll``).
            template: Template string (same as ``compute_nll``).
            common_context: Shared context (same as ``compute_nll``).
            regex_string: Optional regex to restrict scoring
                (same as ``compute_nll``).
            max_samples: Maximum total number of (content, context)
                pairs to return. Default 3.

        Returns:
            List of ``PreviewItem`` objects, one per (content, context)
            pair, up to ``max_samples``.
        """
        if not contexts:
            raise ValueError("contexts must be a non-empty list")

        prefix_template, suffix_template = self._validate_template(
            template, common_context
        )

        content_regex = _compile_content_regex(regex_string)

        assert self.tokenizer is not None

        items: list[PreviewItem] = []
        for content_idx, content in enumerate(contents):
            for context_idx, context in enumerate(contexts):
                if len(items) >= max_samples:
                    return items

                full_text, char_spans = self._render_and_resolve_spans(
                    content=content,
                    context=context,
                    common_context=common_context,
                    prefix_template=prefix_template,
                    suffix_template=suffix_template,
                    content_regex=content_regex,
                )

                tokenized_text, mask, _, _ = self.encode_with_masking(
                    input_text=full_text,
                    char_spans=char_spans,
                )

                # Map the token-level mask back to character spans in
                # full_text, using the encoding's per-token offsets.
                # Adjacent scored tokens are merged into one span.
                mask_1d = mask.squeeze(0)
                encoding = tokenized_text.encodings[0]  # type: ignore[reportOptionalSubscript]
                all_offsets = cast(
                    "list[tuple[int, int]]",
                    encoding.offsets,  # type: ignore[reportUnknownMemberType]
                )
                scored_char_spans = _merge_token_offsets(mask_1d, all_offsets)

                items.append(
                    PreviewItem(
                        content_idx=content_idx,
                        context_idx=context_idx,
                        rendered_text=full_text,
                        scored_char_spans=scored_char_spans,
                    )
                )

        return items

    def preview_multilabel(
        self,
        labels: dict[str, tuple[str, str]] | list[tuple[str, str]],
        contents: list[str],
        template: str = "{context}\n{content}",
        common_context: str = "",
        regex_string: str | None = None,
        max_samples: int = 3,
    ) -> list[PreviewItem]:
        """Preview which tokens will be scored for each (content, label, polarity)
        triple, without running the model.

        Same rendering/tokenization path as ``compute_nll_multilabel``, so the
        preview is guaranteed to match what the model would actually receive.

        Args:
            labels: Same as ``compute_nll_multilabel``.
            contents: Content strings.
            template: Same as ``compute_nll_multilabel``.
            common_context: Same as ``compute_nll_multilabel``.
            regex_string: Same as ``compute_nll_multilabel``.
            max_samples: Maximum total number of items to return.

        Returns:
            List of ``PreviewItem`` with ``label_name`` and ``polarity`` set.
            Order: for each content, all labels in input order; within each
            label, positive then negative.
        """
        label_dict = _normalize_labels(labels)

        prefix_template, suffix_template = self._validate_template(
            template, common_context
        )

        content_regex = _compile_content_regex(regex_string)

        assert self.tokenizer is not None

        items: list[PreviewItem] = []
        for content_idx, content in enumerate(contents):
            flat_context_idx = 0
            for name, (pos, neg) in label_dict.items():
                for polarity, context in zip(_POLARITY_LABELS, (pos, neg), strict=True):
                    if len(items) >= max_samples:
                        return items

                    full_text, char_spans = self._render_and_resolve_spans(
                        content=content,
                        context=context,
                        common_context=common_context,
                        prefix_template=prefix_template,
                        suffix_template=suffix_template,
                        content_regex=content_regex,
                    )

                    tokenized_text, mask, _, _ = self.encode_with_masking(
                        input_text=full_text,
                        char_spans=char_spans,
                    )

                    mask_1d = mask.squeeze(0)
                    encoding = tokenized_text.encodings[0]  # type: ignore[reportOptionalSubscript]
                    all_offsets = cast(
                        "list[tuple[int, int]]",
                        encoding.offsets,  # type: ignore[reportUnknownMemberType]
                    )
                    scored_char_spans = _merge_token_offsets(mask_1d, all_offsets)

                    items.append(
                        PreviewItem(
                            content_idx=content_idx,
                            context_idx=flat_context_idx,
                            rendered_text=full_text,
                            scored_char_spans=scored_char_spans,
                            label_name=name,
                            polarity=polarity,
                        )
                    )
                    flat_context_idx += 1

        return items

    def _print_results(self, results_df: pd.DataFrame) -> None:
        """
        This method collects results in a nice format for returning and printing
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.precision", 4)

        # Use sophisticated color printing with true viridis colors
        self._print_df_colored(
            results_df, cmap="viridis", precision=4, include_index=False
        )

    def _print_df_colored(
        self,
        df: pd.DataFrame,
        cmap: str | Colormap = "viridis",
        precision: int = 4,
        include_index: bool = False,
        na_text: str = "",
    ) -> None:
        """
        Print DataFrame to a truecolor terminal with per-column sequential
        color mapping. Numeric columns are colored based on their own
        min..max. Non-numeric are uncolored.
        """
        if isinstance(cmap, str):
            if hasattr(mpl, "colormaps"):
                # For matplotlib >= 3.7
                cmap = mpl.colormaps[cmap]
            else:
                # For older matplotlib versions
                cmap = cm.get_cmap(cmap)

        cols = list(df.columns)
        idx_strs = [str(i) for i in df.index]

        # pre-format strings and compute widths (without ANSI)
        formatted: dict[str, list[str]] = {}
        col_widths: dict[str, int] = {}
        skip_cols = {"t_mask", "t_str", "t_ids"}
        for c in cols:
            if is_numeric_dtype(df[c]):
                # Special handling for integer columns - keep as integers
                if c in ["t_mask", "t_ids", "c_best_index"] or df[c].dtype in [
                    "int64",
                    "int32",
                    "int16",
                    "int8",
                ]:
                    s = [
                        na_text if pd.isna(v) else f"{int(v)}" for v in df[c].to_numpy()
                    ]
                else:
                    s = [
                        na_text if pd.isna(v) else f"{float(v):.{precision}f}"
                        for v in df[c].to_numpy()
                    ]
            else:
                s = [
                    na_text if pd.isna(v) else str(v)
                    for v in df[c].astype(object).to_numpy()
                ]
            formatted[c] = s
            col_widths[c] = max(len(c), max((len(x) for x in s), default=0))

        index_header = "index"
        index_width = 0
        if include_index:
            index_width = max(
                len(index_header), max((len(x) for x in idx_strs), default=0)
            )

        # compute per-column min/max for numeric columns (skip t_mask, t_str, t_ids)
        skip_cols = {"t_mask", "t_str", "t_ids"}
        col_minmax: dict[str, tuple[float, float] | None] = {}

        for c in cols:
            if is_numeric_dtype(df[c]) and c not in skip_cols:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy()
                v = v[np.isfinite(v)]
                if v.size > 0:
                    col_minmax[c] = (float(v.min()), float(v.max()))
                else:
                    col_minmax[c] = (np.nan, np.nan)
            else:
                col_minmax[c] = None

        # header
        parts: list[str] = []
        if include_index:
            parts.append(f"{index_header:>{index_width}}")
        for c in cols:
            header_text = f"{c:>{col_widths[c]}}"
            parts.append(colored(header_text, "yellow", attrs=["bold"]))
        logger.info("\n%s", " ".join(parts))

        # rows
        for i, row in df.iterrows():
            row_out: list[str] = []
            if include_index:
                row_out.append(f"{i!s:>{index_width}}")

            for c in cols:
                cell = formatted[c][cast("int", df.index.get_loc(i))]
                padded = f"{cell:>{col_widths[c]}}"

                # Skip coloring for t_mask, t_str, t_ids
                if c in skip_cols:
                    row_out.append(padded)
                    continue

                mm = col_minmax[c]
                if mm is None or not is_numeric_dtype(df[c]) or cell == na_text:
                    # non-numeric or NA → no color
                    row_out.append(padded)
                    continue

                vmin, vmax = mm
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    # degenerate range → no color
                    row_out.append(padded)
                    continue

                # normalize
                raw: float = row[c]  # type: ignore[reportUnknownMemberType]  # pandas iterrows yields Unknown row values
                if bool(pd.isna(raw)):
                    row_out.append(padded)
                    continue
                n = (float(raw) - vmin) / (vmax - vmin)
                n = min(max(n, 0.0), 1.0)

                r, g, b, _ = (np.array(cmap(n)) * 255).astype(int)

                # Use foreground (text) color only
                row_out.append(_ansi_fg(r, g, b) + padded + RESET)

            logger.info("%s", " ".join(row_out))

    @torch.inference_mode()
    def _extract_per_token_nll(
        self,
        logits: torch.Tensor,
        label_tokens: torch.Tensor,
    ) -> list[np.ndarray]:
        """Extract per-token NLL values from model logits for each batch item.

        Applies the standard causal-LM shift: logits at position t
        predict token t+1, so ``logits[:, :-1]`` is compared against
        ``label_tokens[:, 1:]``. Positions where ``label_tokens`` is
        ``IGNORE_INDEX`` are excluded from the output.

        Args:
            logits: Raw model logits, shape ``(B, L, V)``. Must be
                aligned with ``label_tokens`` (same B and L).
            label_tokens: Target token IDs, shape ``(B, L)``, with
                ``IGNORE_INDEX`` at positions that should not be
                scored. Typically produced by ``_compute_masked_nll``.

        Returns:
            List of B numpy arrays. Each is a 1D float array of NLL
            values for the scored tokens of that batch item. Array
            lengths vary per item (only non-ignored positions are
            included).
        """
        batch, seq_len, vocab = logits.shape
        assert label_tokens.shape == (batch, seq_len), "label_tokens must be (B, L)"
        assert label_tokens.dtype == torch.long, "CE targets must be Long"

        target = label_tokens[:, 1:]  # [B, L-1]
        mask = target != IGNORE_INDEX  # [B, L-1]
        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        batch, seq_len_m1, vocab = shift_logits.shape

        per_token_nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, vocab),
            target.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).reshape(batch, seq_len_m1)

        results: list[np.ndarray] = []
        for i in range(batch):
            item_nll = per_token_nll[i].masked_select(mask[i])
            results.append(item_nll.cpu().float().numpy())
        return results

    def encode_with_masking(
        self,
        input_text: str,
        char_spans: list[tuple[int, int]],
        return_detokenized_text: bool = False,
    ) -> tuple[BatchEncoding, np.ndarray, dict[int, tuple[int, int]], str | None]:
        """Tokenize text and build a boolean mask over the given character spans.

        The ``BatchEncoding`` contains torch CPU tensors (from the
        HuggingFace tokenizer). The mask is a numpy bool array.

        Args:
            input_text: The full text to tokenize.
            char_spans: Half-open character spans ``[(start, end), ...]``
                to score. Each span selects ``input_text[start:end]``.
            return_detokenized_text: If True, also return the decoded
                string of the masked tokens (useful for debugging).

        Returns:
            A 4-tuple of:

            - **tokenized_text** -- ``BatchEncoding`` for the full
              ``input_text`` (with ``add_special_tokens=True``).
            - **mask** -- Bool numpy array, shape ``(1, seq_len)``,
              True for tokens that fall within any of the
              ``char_spans``.
            - **token_boundaries** -- Dict mapping each span index to
              ``(start_token, end_token_inclusive)``. Set to
              ``(-1, -1)`` when a span could not be aligned to token
              boundaries.
            - **detokenized_text** -- Decoded string of the masked
              tokens, or None when ``return_detokenized_text`` is
              False.
        """
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            text_encoded = self.tokenizer(
                input_text,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            # Single sequence expected
            token_boundaries: dict[int, tuple[int, int]] = {}
            for k, (c0, c1) in enumerate(char_spans):
                # Map char start/end-1 to token indices
                # char_to_token returns None if the char sits exactly on a
                # boundary; nudge inward.
                # Map the character index c0 (start of the span) to the
                # corresponding token index in the encoded text.
                # The first argument to char_to_token is the sequence index
                # (for batched inputs).
                # Here, since we're tokenizing a single string, the sequence
                # index is always 0.
                start = text_encoded.char_to_token(
                    0,
                    c0,  # index in batch  # character index
                )
                if start is None and c0 < len(input_text):  # type: ignore[reportUnnecessaryComparison]  # transformers stubs wrongly declare char_to_token as int, but it can return None
                    start = text_encoded.char_to_token(0, c0 + 1)
                end = text_encoded.char_to_token(0, max(0, c1 - 1))
                if end is None and c1 - 2 >= 0:  # type: ignore[reportUnnecessaryComparison]  # transformers stubs wrongly declare char_to_token as int, but it can return None
                    end = text_encoded.char_to_token(0, c1 - 2)

                if start is None or end is None:  # type: ignore[reportUnnecessaryComparison]  # transformers stubs wrongly declare char_to_token as int, but it can return None
                    token_boundaries[k] = (
                        -1,
                        -1,
                    )  # could not align (rare on pathological boundaries)
                else:
                    token_boundaries[k] = (start, end)
            # Build boolean mask from token boundaries
            mask = np.zeros_like(text_encoded["input_ids"], dtype=bool)
            for start, end in token_boundaries.values():
                mask[:, start : end + 1] = True
            if return_detokenized_text:
                detokenized_text = cast(
                    "str",
                    self.tokenizer.decode(  # type: ignore[reportUnknownMemberType]  # transformers tokenizer stubs partially unknown
                        text_encoded.input_ids[mask],  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # transformers BatchEncoding stubs
                        skip_special_tokens=True,
                    ),
                )
                return text_encoded, mask, token_boundaries, detokenized_text
            return text_encoded, mask, token_boundaries, None
        raise NotImplementedError(
            "Tokenization for non-fast tokenizers is not supported."
        )

    def _tokenize_pairs_iter(
        self,
        contexts: list[str],
        contents: list[str],
        common_context: str,
        prefix_template: str,
        suffix_template: str,
        content_regex: re.Pattern[str] | None,
    ) -> Iterator[_TokenizedItem]:
        """Render templates and tokenize all (content, context) pairs.

        For each pair, renders the full text as
        ``prefix_template.format(...) + content + suffix_template.format(...)``
        and tokenizes it via ``encode_with_masking``.

        Before yielding items for a content, validates that the scored
        region tokenizes identically across all contexts. This is
        required for NLL comparison to be meaningful: the same token
        sequence must appear under each context so per-token NLLs are
        aligned.

        Args:
            contexts: Context strings substituted into the templates.
            contents: Content strings to score.
            common_context: String substituted into
                ``{common_context}`` in the templates.
            prefix_template: The portion of the template before
                ``{content}``, as returned by ``_validate_template``.
                Contains ``{context}`` and optionally
                ``{common_context}``.
            suffix_template: The portion of the template after
                ``{content}``, as returned by ``_validate_template``.
            content_regex: If given, only tokens within capture-group
                matches are scored. Otherwise the entire content is
                scored.

        Yields:
            ``_TokenizedItem`` instances in content-major order: for
            each content, all C context items consecutively.

        Raises:
            ValueError: If the scored tokens for a content differ
                across contexts. This typically happens when contexts
                end with different whitespace or punctuation, causing
                the tokenizer to split the content boundary differently.
        """
        assert self.tokenizer is not None

        for content_idx, content in enumerate(contents):
            content_items: list[_TokenizedItem] = []
            for context_idx, context in enumerate(contexts):
                full_text, char_spans = self._render_and_resolve_spans(
                    content=content,
                    context=context,
                    common_context=common_context,
                    prefix_template=prefix_template,
                    suffix_template=suffix_template,
                    content_regex=content_regex,
                )

                tokenized_text, mask, _, _ = self.encode_with_masking(
                    input_text=full_text,
                    char_spans=char_spans,
                )

                input_ids_1d = cast(
                    "np.ndarray[tuple[int], np.dtype[np.int64]]",
                    tokenized_text.input_ids.squeeze(0).numpy(),  # type: ignore[reportUnknownMemberType]
                )
                mask_1d = mask.squeeze(0)

                masked_token_ids = cast(
                    "np.ndarray[tuple[int], np.dtype[np.int64]]",
                    input_ids_1d[mask_1d],
                )
                masked_token_strs = cast(
                    "np.ndarray[tuple[int], np.dtype[np.str_]]",
                    np.array(
                        tokenized_text.encodings[0].tokens  # type: ignore[reportOptionalSubscript, reportUnknownMemberType, reportUnknownArgumentType]
                    )[mask_1d],
                )

                content_items.append(
                    _TokenizedItem(
                        content_idx=content_idx,
                        context_idx=context_idx,
                        input_ids=input_ids_1d,
                        content_mask=mask_1d,
                        masked_token_ids=masked_token_ids,
                        masked_token_strs=masked_token_strs,
                    )
                )

            # Validate content-identity invariant
            ref_ids = content_items[0].masked_token_ids
            for item in content_items[1:]:
                if not np.array_equal(ref_ids, item.masked_token_ids):
                    raise ValueError(
                        f"Content {content_idx} tokenizes differently with "
                        f"context {item.context_idx}! "
                        f"Context 0 tokens: {content_items[0].masked_token_strs} "
                        f"Context {item.context_idx} tokens: "
                        f"{item.masked_token_strs} "
                        f"This makes NLL comparison invalid. "
                        f"Check that contexts end with the same "
                        f"whitespace/punctuation."
                    )

            yield from content_items

    def _pad_chunk(
        self,
        chunk: list[_TokenizedItem],
    ) -> _Batch:
        """Pad a list of tokenized items to uniform length and move to device.

        Args:
            chunk: List of ``_TokenizedItem`` to pad and batch together.

        Returns:
            A ``_Batch`` with padded tensors on ``self._device`` and the
            ``items`` field set to ``chunk``.
        """
        assert self.tokenizer is not None
        pad_id = self.tokenizer.pad_token_id  # type: ignore[reportUnknownMemberType]

        batch_len = len(chunk)
        max_len = max(it.input_ids.shape[0] for it in chunk)

        ids = np.full((batch_len, max_len), pad_id, dtype=np.int64)
        attn = np.zeros((batch_len, max_len), dtype=np.int64)
        mask = np.zeros((batch_len, max_len), dtype=np.bool_)

        for i, it in enumerate(chunk):
            seq_len = it.input_ids.shape[0]
            ids[i, :seq_len] = it.input_ids
            attn[i, :seq_len] = 1
            mask[i, :seq_len] = it.content_mask

        return _Batch(
            input_ids=torch.from_numpy(ids).to(self._device),  # type: ignore[reportUnknownMemberType]
            attention_mask=torch.from_numpy(attn).to(self._device),  # type: ignore[reportUnknownMemberType]
            content_mask=torch.from_numpy(mask).to(self._device),  # type: ignore[reportUnknownMemberType]
            items=chunk,
        )

    def _build_batches_iter(
        self,
        items: Iterator[_TokenizedItem],
        batch_size: int,
    ) -> Iterator[_Batch]:
        """Accumulate tokenized items into padded batches, yielding each
        batch as soon as it reaches ``batch_size`` items.

        No sorting -- items are chunked in the order they arrive.

        Args:
            items: Iterator of tokenized items in content-major order.
            batch_size: Maximum number of sequences per batch.

        Yields:
            Padded ``_Batch`` objects with tensors on ``self._device``.
        """
        chunk: list[_TokenizedItem] = []
        for item in items:
            chunk.append(item)
            if len(chunk) == batch_size:
                yield self._pad_chunk(chunk)
                chunk = []
        if chunk:
            yield self._pad_chunk(chunk)

    def _run_forward_iter(
        self,
        batches: Iterator[_Batch],
        use_tqdm: bool = False,
        n_batches: int | None = None,
    ) -> Iterator[tuple[list[np.ndarray], list[_TokenizedItem]]]:
        """Run model forward passes, yielding per-token NLL results per batch.

        Args:
            batches: Iterator of padded batches.
            use_tqdm: If True, show a progress bar.
            n_batches: Total number of batches (for tqdm ``total``).
                Optional; if None, tqdm shows a count-only bar.

        Yields:
            ``(nll_arrays, items)`` tuples. ``nll_arrays`` is a list of
            B numpy arrays (one per batch item). ``items`` is the
            corresponding list of ``_TokenizedItem`` from the batch.
        """
        batch_iter: Any = (
            tqdm(batches, desc="Forward passes", unit="batch", total=n_batches)
            if use_tqdm
            else batches
        )
        for batch in batch_iter:
            outputs, label_tokens = self._compute_masked_nll(
                token_ids=batch.input_ids,
                mask=batch.content_mask,
                attention_mask=batch.attention_mask,
            )
            assert outputs.logits is not None
            yield (
                self._extract_per_token_nll(outputs.logits, label_tokens),
                batch.items,
            )

    def _build_result_dict(
        self,
        context_nlls: list[np.ndarray],
        ref_item: _TokenizedItem,
    ) -> dict[str, Any]:
        """Build a single content's result dict from per-context NLL arrays.

        Constructs the DataFrame with per-token NLL columns, computes LLR
        (when there are exactly 2 contexts), entropy, cumulative
        probabilities, and information gain.

        Args:
            context_nlls: One NLL array per context, length C.
            ref_item: A ``_TokenizedItem`` from the first context, used
                for ``masked_token_ids`` and ``masked_token_strs``.

        Returns:
            A result dict with keys: ``df``, ``llr``, ``entropy``,
            ``info_gain``, ``most_likely_label_index``.
        """
        n_contexts = len(context_nlls)

        df_dict: dict[str, Any] = {
            "t_ids": ref_item.masked_token_ids,
            "t_str": ref_item.masked_token_strs,
        }
        nll_data: list[np.ndarray] = []
        for ctx_idx in range(n_contexts):
            colname = f"t_nll_{ctx_idx}"
            df_dict[colname] = context_nlls[ctx_idx]
            nll_data.append(context_nlls[ctx_idx])

        results_df = pd.DataFrame(df_dict)
        logp = -np.column_stack(nll_data)

        results_dict: dict[str, Any] = {}

        if n_contexts > 1:
            probs = softmax(logp.astype(np.float64), axis=1)
            assert np.all(probs >= 0), "Probs are not between 0 and 1"
            assert np.all(probs <= 1), "Probs are not between 0 and 1"
            assert np.all(np.allclose(np.sum(probs, axis=1), 1, atol=1e-6)), (
                "Probs do not sum to 1"
            )
            t_entropy = -np.sum(probs * np.log2(probs), axis=1)

            logp_cumulative = np.cumsum(logp.astype(np.float64), axis=0)
            probs_cumulative = softmax(logp_cumulative, axis=1)
            logp_cumulative_normalized = log_softmax(logp_cumulative, axis=1)

            assert np.all(probs_cumulative >= 0), (
                "Cumulative probs are not between 0 and 1"
            )
            assert np.all(probs_cumulative <= 1), (
                "Cumulative probs are not between 0 and 1"
            )
            assert np.all(
                np.allclose(np.sum(probs_cumulative, axis=1), 1, atol=1e-6)
            ), "Cumulative probs do not sum to 1"

            c_entropy = -np.sum(
                probs_cumulative * logp_cumulative_normalized, axis=1
            ) / np.log(2)  # convert nats to bits

            c_prob_dict = {
                f"c_prob_{i}": probs_cumulative[:, i] for i in range(n_contexts)
            }
            c_best_index = np.argmax(probs_cumulative, axis=1).astype(int)
            c_info_gain = np.log2(n_contexts) - c_entropy

            multi_context_cols = pd.DataFrame(
                {
                    "t_entropy": t_entropy,
                    "c_entropy": c_entropy,
                    "c_best_index": c_best_index,
                    "c_info_gain": c_info_gain,
                    **c_prob_dict,
                }
            )
            results_df = pd.concat([results_df, multi_context_cols], axis=1)  # type: ignore[reportUnknownMemberType]

        if n_contexts == 2:  # noqa: PLR2004
            t_llr = results_df["t_nll_1"].to_numpy() - results_df["t_nll_0"].to_numpy()
            c_llr = np.cumsum(t_llr)
            llr_cols = pd.DataFrame({"t_llr": t_llr, "c_llr": c_llr})
            results_df = pd.concat([results_df, llr_cols], axis=1)  # type: ignore[reportUnknownMemberType]
        else:
            llr_cols = pd.DataFrame(
                {"t_llr": float("nan"), "c_llr": float("nan")},
                index=results_df.index,
            )
            results_df = pd.concat([results_df, llr_cols], axis=1)  # type: ignore[reportUnknownMemberType]

        if n_contexts == 1:
            results_dict["llr"] = float("nan")
            results_dict["entropy"] = float("nan")
            results_dict["info_gain"] = float("nan")
            results_dict["most_likely_label_index"] = 0
        else:
            results_dict["llr"] = float(results_df["c_llr"].to_numpy()[-1])
            results_dict["entropy"] = float(results_df["c_entropy"].to_numpy()[-1])
            results_dict["info_gain"] = float(results_df["c_info_gain"].to_numpy()[-1])
            results_dict["most_likely_label_index"] = int(
                results_df["c_best_index"].to_numpy()[-1]
            )

        results_dict["df"] = results_df
        return results_dict

    def _build_multilabel_result_dict(
        self,
        context_nlls: list[np.ndarray],
        ref_item: _TokenizedItem,
        label_specs: list[tuple[str, int, int]],
        threshold: dict[str, float],
    ) -> dict[str, Any]:
        """Build a per-content multilabel result dict from flattened NLLs.

        Args:
            context_nlls: One NLL array per flattened context, length 2L.
            ref_item: A ``_TokenizedItem`` from the first context of the
                content, used for token ids and strings.
            label_specs: ``(label_name, pos_context_idx, neg_context_idx)``
                triples identifying each label's pair within ``context_nlls``.
            threshold: Per-label thresholds. Keys must match the names in
                ``label_specs`` exactly. A label is marked predicted when
                its probability is at or above its threshold.

        Returns:
            Dict with keys ``labels`` (name -> 2-context result dict),
            ``summary`` (DataFrame indexed by label name), and
            ``predicted_labels`` (list of names where predicted is True).
        """
        labels_dict: dict[str, dict[str, Any]] = {}
        for name, pos_idx, neg_idx in label_specs:
            labels_dict[name] = self._build_result_dict(
                context_nlls=[context_nlls[pos_idx], context_nlls[neg_idx]],
                ref_item=ref_item,
            )

        summary = _assemble_multilabel_summary(labels_dict, threshold=threshold)
        # pandas-stubs types Index.tolist() with partial Unknowns.
        predicted_labels: list[str] = summary.index[summary["predicted"]].tolist()  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return {
            "labels": labels_dict,
            "summary": summary,
            "predicted_labels": predicted_labels,
        }

    def _assemble_results_iter(
        self,
        nll_batches: Iterator[tuple[list[np.ndarray], list[_TokenizedItem]]],
        n_contexts: int,
        result_builder: ResultBuilder,
        print_results: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Assemble result dicts incrementally as NLL batches arrive.

        Accumulates NLL arrays and metadata until all C contexts for a
        content are available, then yields that content's result dict.

        Relies on items arriving in strict content-major order (all C
        contexts for content 0, then all C for content 1, etc.).

        Args:
            nll_batches: Iterator of ``(nll_arrays, items)`` tuples from
                ``_run_forward_iter``.
            n_contexts: Number of contexts (C).
            result_builder: Callback that converts the per-context NLL arrays
                and a reference ``_TokenizedItem`` into one per-content result
                dict. Pass ``self._build_result_dict`` for the standard
                single-label layout.
            print_results: If True, print per-token table for each content.
                Expects ``result["df"]`` in the yielded dict and so is only
                valid for single-label ``result_builder``; multilabel callers
                should pass ``False`` and print per-label themselves.

        Yields:
            One result dict per content, in content-major order.
        """
        pending_nlls: list[np.ndarray] = []
        pending_items: list[_TokenizedItem] = []

        for batch_nlls, batch_items in nll_batches:
            pending_nlls.extend(batch_nlls)
            pending_items.extend(batch_items)
            while len(pending_nlls) >= n_contexts:
                context_nlls = pending_nlls[:n_contexts]
                context_items = pending_items[:n_contexts]
                pending_nlls = pending_nlls[n_contexts:]
                pending_items = pending_items[n_contexts:]
                result = result_builder(context_nlls, context_items[0])
                if print_results:
                    self._print_results(results_df=result["df"])
                yield result

    def evaluate(
        self,
        contexts: list[str],
        formatted_dataset: list[dict[str, Any]],
        statement_mapping: dict[str, Any],
        template: str,
        common_context: str = "",
        target_roles: list[str] | None = None,
        regex_string: str | None = None,
        async_logger: AsyncCSVLogger | None = None,
        sample_log_file_name: str | None = None,
        token_log_file_name: str | None = None,
        additional_row_log_attributes: dict[str, Any] | None = None,
        use_tqdm: bool = False,
        print_summary: bool = False,
        print_likelihood_table: bool = False,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Classify each sample in a dataset and score the results.

        For each sample, runs ``compute_nll_iter`` to determine which
        context is most likely, compares against the true label, and
        produces a score dict with accuracy, info gain, and LLR.

        NLL parameters (``contexts``, ``template``, ``common_context``,
        ``regex_string``, ``batch_size``) are forwarded to
        ``compute_nll_iter``; see that method's docstring for details.

        Args:
            contexts: Context strings, one per class. Order must match
                ``statement_mapping``.
            formatted_dataset: One dict per sample, each with keys
                ``"formatted_text"`` (the content string) and
                ``"label"`` (a key in ``statement_mapping``).
            statement_mapping: Ordered mapping from label strings to
                context descriptions. The iteration order determines
                the correspondence between label names and context
                indices.
            template: Forwarded to ``compute_nll_iter``.
            common_context: Forwarded to ``compute_nll_iter``.
            target_roles: If given, score only tokens within these
                XML-tag roles (e.g. ``["assistant"]`` scores text
                inside ``<assistant>...</assistant>``). Converted to a
                regex internally. Mutually exclusive with
                ``regex_string``.
            regex_string: Forwarded to ``compute_nll_iter``. Mutually
                exclusive with ``target_roles``.
            async_logger: If provided, enables CSV logging of
                per-sample and per-token results.
            sample_log_file_name: CSV path for per-sample scores.
                Requires ``async_logger``.
            token_log_file_name: CSV path for per-token NLL values.
                Requires ``async_logger``.
            additional_row_log_attributes: Extra key-value pairs
                appended to every logged row. Requires
                ``async_logger``.
            use_tqdm: Forwarded to ``compute_nll_iter``.
            print_summary: If True, print a classification report and
                confusion matrix after the run.
            print_likelihood_table: If True, print the per-token NLL
                table for each sample.
            batch_size: Forwarded to ``compute_nll_iter``.

        Returns:
            List of score dicts, one per sample. Each contains:

            - ``accuracy`` -- 1 if predicted label matches true label,
              0 otherwise.
            - ``info_gain`` -- Positive if correct, negated if wrong.
            - ``llr`` -- Absolute LLR, positive if correct, negated if
              wrong.
            - ``sample_index``, ``sample_label``,
              ``true_label_index``, ``predicted_label``,
              ``string_length``.

        Raises:
            ValueError: If both ``target_roles`` and ``regex_string``
                are provided, or if logging file names are given
                without ``async_logger``.
        """
        if target_roles is None:
            target_roles = []
        if additional_row_log_attributes is None:
            additional_row_log_attributes = {}

        if target_roles and regex_string is not None:
            raise ValueError("target_roles and regex_string are mutually exclusive.")

        if async_logger is None and (
            sample_log_file_name or token_log_file_name or additional_row_log_attributes
        ):
            raise ValueError(
                "If sample_log_file_name, token_log_file_name, or"
                " additional_row_log_attributes is provided, async_logger"
                " must also be provided."
            )

        # Convert target_roles to regex_string
        effective_regex: str | None = regex_string
        if target_roles:
            effective_regex = "|".join(
                f"<{role}>(.*?)</{role}>" for role in target_roles
            )

        contents = [conv["formatted_text"] for conv in formatted_dataset]

        scores: list[dict[str, Any]] = []

        for sample_index, (sample, nll_scores) in enumerate(
            zip(
                formatted_dataset,
                self.compute_nll_iter(
                    contexts=contexts,
                    contents=contents,
                    template=template,
                    common_context=common_context,
                    regex_string=effective_regex,
                    batch_size=batch_size,
                    print_results=print_likelihood_table,
                    use_tqdm=use_tqdm,
                ),
                strict=True,
            )
        ):
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

            score["string_length"] = sum([len(i) for i in contexts])
            score["sample_index"] = sample_index
            score["sample_label"] = sample["label"]
            score["true_label_index"] = true_label_index
            score["predicted_label"] = most_likely_label

            scores.append(score)

            if async_logger:
                if sample_log_file_name is not None:
                    row_data = {
                        **score,
                        **additional_row_log_attributes,
                    }
                    headers = list(row_data.keys())
                    async_logger.queue_csv_row(sample_log_file_name, row_data, headers)

                if token_log_file_name is not None:
                    token_attributes = {
                        "sample_index": sample_index,
                        "sample_label": sample["label"],
                        **additional_row_log_attributes,
                    }
                    attr_df = pd.DataFrame(
                        token_attributes, index=nll_scores["df"].index
                    )
                    nll_scores["df"] = pd.concat([attr_df, nll_scores["df"]], axis=1)  # type: ignore[reportUnknownMemberType]
                    async_logger.queue_write(
                        partial(_write_token_csv, nll_scores["df"], token_log_file_name)
                    )

        if print_summary:
            self._print_evaluation_summary(scores, statement_mapping)

        return scores

    def evaluate_multilabel(
        self,
        labels: dict[str, tuple[str, str]] | list[tuple[str, str]],
        formatted_dataset: list[dict[str, Any]],
        template: str,
        threshold: float | dict[str, float] = 0.5,
        common_context: str = "",
        target_roles: list[str] | None = None,
        regex_string: str | None = None,
        async_logger: AsyncCSVLogger | None = None,
        sample_log_file_name: str | None = None,
        token_log_file_name: str | None = None,
        additional_row_log_attributes: dict[str, Any] | None = None,
        use_tqdm: bool = False,
        print_summary: bool = False,
        print_likelihood_table: bool = False,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Run ``compute_nll_multilabel_iter`` against a labeled dataset
        and score each sample.

        For every sample, each label's probability is compared to its
        threshold to yield a binary prediction, which is then compared
        to the sample's ground truth to produce per-label correctness
        plus per-sample aggregate metrics.

        When ``async_logger`` is provided, this method writes CSV files
        as a side effect (see "CSV output files" below); the caller is
        responsible for shutting the logger down.

        CSV output files:
            The sample-log file (``sample_log_file_name``) has one row
            per sample in ``formatted_dataset`` order, with sample-level
            scalar columns (``sample_index``, ``hamming_accuracy``,
            ``subset_accuracy``, ``string_length``, and semicolon-joined
            ``true_labels`` / ``predicted_labels``) plus, per label,
            ``prob_<name>``, ``correct_<name>``, ``info_gain_<name>``,
            ``pos_nll_total_<name>``, ``neg_nll_total_<name>``,
            ``llr_<name>``, and ``predicted_<name>``.

            The token-log file (``token_log_file_name``) has one row per
            scored token per label per sample. Leading columns are
            ``sample_index`` and ``label_name``, followed by the columns
            of the per-label 2-context ``df``. Rows for a given sample
            are grouped by label in label-definition order.

            ``additional_row_log_attributes`` are merged into every
            sample row and added as columns on every token row.

        Args:
            labels: Same as ``compute_nll_multilabel``.
            formatted_dataset: List of sample dicts. Each sample must
                have ``"formatted_text": str`` (the content to score)
                and ``"labels": list[str]`` (names of the labels that
                apply; pass ``[]`` for a sample with no applicable
                labels). Unknown label names raise; duplicates within
                a sample's list are silently deduplicated.
            template: Forwarded to ``compute_nll_multilabel_iter``.
            threshold: Scalar or per-label dict. See
                ``compute_nll_multilabel_iter`` for details.
            common_context: Forwarded to ``compute_nll_multilabel_iter``.
            target_roles: Same as ``evaluate``; mutually exclusive with
                ``regex_string``.
            regex_string: Forwarded to ``compute_nll_multilabel_iter``;
                mutually exclusive with ``target_roles``.
            async_logger: Required whenever any logging argument
                (``sample_log_file_name``, ``token_log_file_name``,
                ``additional_row_log_attributes``) is set.
            sample_log_file_name: CSV path for the per-sample table.
                See "CSV output files" above for the row format.
            token_log_file_name: CSV path for the per-token table.
                See "CSV output files" above for the row format.
            additional_row_log_attributes: Constant key/value pairs
                added to every logged row. See "CSV output files"
                above.
            use_tqdm: Forwarded to ``compute_nll_multilabel_iter``.
            print_summary: If True, log
                ``sklearn.metrics.classification_report`` over the
                multilabel indicator matrices, plus mean hamming and
                subset accuracy across samples.
            print_likelihood_table: If True, print the per-token table
                once per (sample, label). Noisy for many labels.
            batch_size: Forwarded to ``compute_nll_multilabel_iter``.

        Returns:
            List of score dicts, one per sample, in input order. Each
            dict contains:

            - ``sample_index`` -- ``int``, position in
              ``formatted_dataset``.
            - ``true_labels`` -- ``list[str]``, ground-truth names
              (deduplicated), in label-definition order.
            - ``predicted_labels`` -- ``list[str]``, names whose
              probability meets the threshold, also in
              label-definition order.
            - ``per_label`` -- ``{label_name: {"prob", "info_gain",
              "correct"}}``. ``correct`` is True when the label's
              binary prediction matches the ground truth.
              ``info_gain`` follows the single-label ``evaluate()``
              sign convention: the raw magnitude when ``correct`` is
              True, negated when False.
            - ``hamming_accuracy`` -- ``float`` in [0, 1], fraction
              of labels correctly classified for this sample.
            - ``subset_accuracy`` -- ``int``, 1 if every label is
              correct (set-equal prediction), else 0.
            - ``string_length`` -- ``int``, total character length of
              all ``2L`` context strings (constant across the run).

        Raises:
            ValueError: If a sample references an unknown label name;
                if both ``target_roles`` and ``regex_string`` are
                given; or if any logging argument is set without
                ``async_logger``. Errors from
                ``compute_nll_multilabel_iter`` (empty ``labels``,
                ``threshold`` dict mismatch, ``batch_size`` < 1, bad
                ``regex_string``) also propagate.
        """
        if target_roles is None:
            target_roles = []
        if additional_row_log_attributes is None:
            additional_row_log_attributes = {}

        if target_roles and regex_string is not None:
            raise ValueError("target_roles and regex_string are mutually exclusive.")

        if async_logger is None and (
            sample_log_file_name or token_log_file_name or additional_row_log_attributes
        ):
            raise ValueError(
                "If sample_log_file_name, token_log_file_name, or"
                " additional_row_log_attributes is provided, async_logger"
                " must also be provided."
            )

        label_dict = _normalize_labels(labels)
        label_names = list(label_dict.keys())
        label_name_set = set(label_names)

        # Validate samples up-front so we fail before any forward passes.
        for i, sample in enumerate(formatted_dataset):
            sample_labels = sample["labels"]
            unknown = [n for n in sample_labels if n not in label_name_set]
            if unknown:
                raise ValueError(
                    f"Sample {i} references unknown label(s): {unknown}. "
                    f"Known labels: {label_names}"
                )

        effective_regex: str | None = regex_string
        if target_roles:
            effective_regex = "|".join(
                f"<{role}>(.*?)</{role}>" for role in target_roles
            )

        contents = [sample["formatted_text"] for sample in formatted_dataset]
        string_length = sum(len(pos) + len(neg) for pos, neg in label_dict.values())

        scores: list[dict[str, Any]] = []

        for sample_index, (sample, nll_result) in enumerate(
            zip(
                formatted_dataset,
                self.compute_nll_multilabel_iter(
                    labels=label_dict,
                    contents=contents,
                    template=template,
                    common_context=common_context,
                    regex_string=effective_regex,
                    threshold=threshold,
                    batch_size=batch_size,
                    print_results=print_likelihood_table,
                    use_tqdm=use_tqdm,
                ),
                strict=True,
            )
        ):
            true_set = set(sample["labels"])
            true_labels_ordered = [n for n in label_names if n in true_set]
            predicted_labels_ordered: list[str] = list(nll_result["predicted_labels"])
            pred_set = set(predicted_labels_ordered)

            summary_df: pd.DataFrame = nll_result["summary"]

            per_label: dict[str, dict[str, Any]] = {}
            n_correct = 0
            for name in label_names:
                sub = nll_result["labels"][name]
                prob = float(summary_df.loc[name, "prob"])  # type: ignore[reportUnknownMemberType]  # pandas-stubs loc is partially unknown
                is_predicted = name in pred_set
                is_true = name in true_set
                correct = is_predicted == is_true
                raw_info_gain = float(sub["info_gain"])
                # TODO: rename this to signed_info_gain (and expose raw
                # info_gain separately) in a follow-up PR. The current
                # sign-on-error convention matches single-label evaluate()
                # but is misleading as a per-label CSV column.
                signed_info_gain = raw_info_gain if correct else -raw_info_gain
                per_label[name] = {
                    "prob": prob,
                    "info_gain": signed_info_gain,
                    "correct": correct,
                }
                if correct:
                    n_correct += 1

            score = {
                "sample_index": sample_index,
                "true_labels": true_labels_ordered,
                "predicted_labels": predicted_labels_ordered,
                "per_label": per_label,
                "hamming_accuracy": n_correct / len(label_names),
                "subset_accuracy": 1 if pred_set == true_set else 0,
                "string_length": string_length,
            }
            scores.append(score)

            if async_logger is not None and sample_log_file_name is not None:
                row_data: dict[str, Any] = {
                    "sample_index": sample_index,
                    "hamming_accuracy": score["hamming_accuracy"],
                    "subset_accuracy": score["subset_accuracy"],
                    "string_length": score["string_length"],
                    "true_labels": ";".join(true_labels_ordered),
                    "predicted_labels": ";".join(predicted_labels_ordered),
                }
                for name in label_names:
                    # Materialize the summary row to a plain dict so the
                    # per-cell pyright narrowing all happens on the cast
                    # below, not at each column read.
                    summary_row: dict[str, Any] = summary_df.loc[name].to_dict()  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                    row_data[f"prob_{name}"] = per_label[name]["prob"]
                    row_data[f"correct_{name}"] = per_label[name]["correct"]
                    row_data[f"info_gain_{name}"] = per_label[name]["info_gain"]
                    row_data[f"pos_nll_total_{name}"] = float(
                        summary_row["pos_nll_total"]
                    )
                    row_data[f"neg_nll_total_{name}"] = float(
                        summary_row["neg_nll_total"]
                    )
                    row_data[f"llr_{name}"] = float(summary_row["llr"])
                    row_data[f"predicted_{name}"] = bool(summary_row["predicted"])
                row_data.update(additional_row_log_attributes)
                headers = list(row_data.keys())
                async_logger.queue_csv_row(sample_log_file_name, row_data, headers)

            if async_logger is not None and token_log_file_name is not None:
                token_frames: list[pd.DataFrame] = []
                for name in label_names:
                    label_df = nll_result["labels"][name]["df"].copy()
                    label_df.insert(0, "label_name", name)
                    label_df.insert(0, "sample_index", sample_index)
                    for key, value in additional_row_log_attributes.items():
                        label_df[key] = value
                    token_frames.append(label_df)
                combined = pd.concat(token_frames, ignore_index=True)  # type: ignore[reportUnknownMemberType]
                async_logger.queue_write(
                    partial(_write_token_csv, combined, token_log_file_name)
                )

        if print_summary:
            self._print_multilabel_evaluation_summary(scores, label_names)

        return scores

    def _print_multilabel_evaluation_summary(
        self,
        scores: list[dict[str, Any]],
        label_names: list[str],
    ) -> None:
        """Print a multilabel evaluation summary.

        Prints sklearn's classification_report on the multilabel indicator
        matrices, plus mean hamming and subset accuracy.
        """
        if not scores:
            logger.info("No samples to summarize.")
            return

        n_samples = len(scores)
        n_labels = len(label_names)
        name_to_idx = {name: i for i, name in enumerate(label_names)}

        y_true = np.zeros((n_samples, n_labels), dtype=np.int64)
        y_pred = np.zeros((n_samples, n_labels), dtype=np.int64)
        for row, score in enumerate(scores):
            for name in score["true_labels"]:
                y_true[row, name_to_idx[name]] = 1
            for name in score["predicted_labels"]:
                y_pred[row, name_to_idx[name]] = 1

        logger.info("\n%s", "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("%s\n", "=" * 80)
        logger.info("--- Classification Report ---")

        report = classification_report(  # type: ignore[reportUnknownVariableType]
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,  # type: ignore[arg-type]
        )
        logger.info("%s", report)  # type: ignore[reportUnknownArgumentType]

        mean_hamming = float(np.mean([s["hamming_accuracy"] for s in scores]))
        mean_subset = float(np.mean([s["subset_accuracy"] for s in scores]))
        logger.info("Hamming accuracy (mean over samples): %.4f", mean_hamming)
        logger.info("Subset accuracy (mean over samples): %.4f", mean_subset)
        logger.info("\n%s\n", "=" * 80)

    def _print_evaluation_summary(
        self,
        scores: list[dict[str, Any]],
        statement_mapping: dict[str, Any],
        cmap: str | Colormap = "viridis",
    ) -> None:
        """
        Prints an evaluation summary, including micro/macro stats
        and a color-coded confusion matrix heatmap.

        Note: This function was largely generated from Gemini 2.5 Pro.

        Args:
            scores: The list of score dictionaries from the evaluate run.
            cmap: The matplotlib colormap to use for the heatmap.
        """

        all_labels = list(statement_mapping.keys())

        y_true = [s["sample_label"] for s in scores]
        y_pred = [s["predicted_label"] for s in scores]

        logger.info("\n%s", "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("%s\n", "=" * 80)
        logger.info("--- Classification Report ---")

        report = classification_report(  # type: ignore[reportUnknownVariableType]  # sklearn has no stubs; return type is unknown
            y_true,
            y_pred,
            labels=all_labels,
            target_names=all_labels,
            zero_division=0,  # type: ignore[arg-type]  # sklearn stubs type zero_division as str only
        )
        logger.info("%s", report)  # type: ignore[reportUnknownArgumentType]  # report type unknown due to missing sklearn stubs

        logger.info("\n--- Confusion Matrix ---")
        cm_data = confusion_matrix(y_true, y_pred, labels=all_labels)

        short_names = [
            name[:CM_LABEL_PREFIX_LEN] + "..." if len(name) > CM_LABEL_MAX_LEN else name
            for name in all_labels
        ]
        cm_df = pd.DataFrame(
            cm_data,
            index=[f"True: {name}" for name in short_names],
            columns=[f"Pred: {name}" for name in short_names],
        )

        if isinstance(cmap, str):
            if hasattr(mpl, "colormaps"):
                # For matplotlib >= 3.7
                cmap_obj = mpl.colormaps[cmap]
            else:
                # For older matplotlib versions
                cmap_obj = cm.get_cmap(cmap)
        else:
            cmap_obj = cmap

        vmin, vmax = 0, cm_data.max()
        if vmax == 0:
            vmax = 1

        header_names = list(cm_df.columns)
        index_names = list(cm_df.index)
        cell_width = max(
            *(len(str(c)) for c in header_names),
            *(len(str(v)) for v in cm_data.flatten()),
        )

        index_width = max(len(str(i)) for i in index_names)

        header_parts = [" " * (index_width + 1)]
        header_parts.extend(f"{name:>{cell_width}}" for name in header_names)
        logger.info("%s", " ".join(header_parts))

        for idx_name, row in cm_df.iterrows():
            row_parts = [f"{idx_name:>{index_width}}"]

            for cell_val in row.to_numpy():
                n = (cell_val - vmin) / (vmax - vmin)
                n = min(max(n, 0.0), 1.0)

                r, g, b, _ = (np.array(cmap_obj(n)) * 255).astype(int)  # type: ignore[reportUnknownArgumentType]  # matplotlib colormap __call__ return type unknown

                bg = _ansi_bg(r, g, b)

                lum = _luminance(r, g, b)
                fg = _ansi_fg(0, 0, 0) if lum > 0.5 else _ansi_fg(255, 255, 255)  # noqa: PLR2004

                padded_str = f"{cell_val:>{cell_width}}"
                row_parts.append(f"{bg}{fg}{padded_str}{RESET}")

            logger.info("%s", " ".join(row_parts))

        logger.info("\n%s\n", "=" * 80)
