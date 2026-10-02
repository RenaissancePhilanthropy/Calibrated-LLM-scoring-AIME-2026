"""Unified LLM-as-Judge runner — multi-provider, flat prompt, verbalized confidence.

Talks to any OpenAI-compatible chat-completions endpoint through a single
OpenAI-SDK-based runner; this package configures two providers, OpenAI
and a local vLLM server (see judge/config.toml).

Provider, model, and per-call options resolve in this precedence
(low → high):
    global config < [providers.X] < [[sweep.models]] entry < CLI flag

Execution mode: threaded — plain sync ChatCompletions in a
ThreadPoolExecutor (see scripts/04_local_judges.sh,
scripts/05_openai_judges.sh).

Output schema and CSV/run_config layout come from ``common.write_eval_csv``
and ``common.write_run_config``, shared with the direct method, so
export/export_figure_data.py reads both the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package root
from common import (  # noqa: E402
    build_judge_user_message, ece_top1, parse_label, write_eval_csv,
    write_run_config,
)
from scientsbank import load_eval  # noqa: E402

DEFAULT_CONFIG = HERE / "config.toml"
# The package ships one dataset; scientsbank.py registers no other loader.
DATASET = "scientsbank_2way_final"
# Label descriptions come from direct/config.toml so the judge and the
# scoring methods describe the classes with the same words.
DESCS_CFG = HERE.parent / "direct" / "config.toml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "config", type=Path, nargs="?", default=DEFAULT_CONFIG,
        help=f"TOML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument("--model", required=True,
                   help="Repo identifier matching a [[sweep.models]] entry's "
                        "`repo` field — e.g. 'gpt-4o-mini', "
                        "'Qwen/Qwen3.5-9B', 'meta-llama/Llama-3.2-3B-Instruct'")
    p.add_argument("--provider", type=str, default=None,
                   help="Force a specific provider for --model. Default: "
                        "auto-detect from sweep.models.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap eval split at N samples.")
    p.add_argument("--max-workers", type=int, default=None,
                   help="Override the thread-pool size.")
    p.add_argument("--max-completion-tokens", type=int, default=None,
                   help="Override max_completion_tokens cap.")
    p.add_argument(
        "--no-thinking", action="store_true",
        help="Force enable_thinking=False (provider-specific param injected "
             "via providers.X.thinking_kwarg_path).",
    )
    p.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high"),
        default=None,
        help="Override the model entry's reasoning_effort (OpenAI gpt-5.x "
             "thinking control).",
    )
    p.add_argument(
        "--run-tag-suffix", type=str, default=None,
        help="Suffix appended to the auto-generated run-tag.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` from path ``'a.b.c'``, creating dicts."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _first_not_none(*vals: Any) -> Any:
    """First argument that is not None, else None. Unlike ``a or b`` this
    treats falsy-but-valid values (e.g. ``temperature=0.0``) as present."""
    for v in vals:
        if v is not None:
            return v
    return None


def served_model_matches(requested: str, served: str) -> bool:
    """True if the API-reported served model is an acceptable identity for the
    requested model.

    Exact match, or an OpenAI dated snapshot of the requested model
    (``requested + '-YYYY-MM-DD'``, e.g. ``gpt-4o`` -> ``gpt-4o-2024-08-06``).
    Anything else (e.g. a request for ``Qwen/Qwen3.5-0.8B`` served by
    ``Qwen/Qwen3.5-9B`` after the provider silently forwards a deprecated
    model) is a mismatch.
    """
    if served == requested:
        return True
    return re.fullmatch(re.escape(requested) + r"-\d{4}-\d{2}-\d{2}", served) is not None


def _load_api_key(env_var: str) -> str:
    if k := os.environ.get(env_var):
        return k
    load_dotenv(Path.home() / ".env")
    if k := os.environ.get(env_var):
        return k
    raise RuntimeError(f"{env_var} not in env or ~/.env")


def load_descriptions(dataset_name: str) -> dict[str, str]:
    """Read the per-class label descriptions from direct/config.toml."""
    cfg = tomllib.load(open(DESCS_CFG, "rb"))
    if dataset_name not in cfg.get("datasets", {}):
        raise RuntimeError(f"No [datasets.{dataset_name}] in {DESCS_CFG}")
    return dict(cfg["datasets"][dataset_name].get("labels", {}))


def find_model_entry(
    models_list: list[dict[str, Any]], repo: str, provider_hint: str | None = None,
) -> dict[str, Any]:
    """Find the [[sweep.models]] entry for a given repo + optional provider."""
    matches = [m for m in models_list if m.get("repo") == repo]
    if provider_hint is not None:
        matches = [m for m in matches if m.get("provider") == provider_hint]
    if not matches:
        raise RuntimeError(
            f"No [[sweep.models]] entry found for repo={repo!r}"
            + (f" (provider={provider_hint!r})" if provider_hint else "")
        )
    if len(matches) > 1:
        provs = sorted({m.get("provider") for m in matches})
        raise RuntimeError(
            f"Repo {repo!r} mapped to multiple providers {provs}. "
            "Pass --provider to disambiguate."
        )
    return matches[0]


def build_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_msg: str,
    *,
    max_completion_tokens: int,
    response_format_json: bool,
    enable_thinking: bool,
    provider_cfg: dict[str, Any],
    reasoning_effort_override: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a single chat-completions request body in OpenAI schema, with
    provider-specific injection of the "thinking" control kwarg.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if response_format_json:
        body["response_format"] = {"type": "json_object"}

    # Reasoning models (gpt-5.x with reasoning_effort) reject temperature, so
    # it goes only on non-reasoning requests: those send temperature = 0.0,
    # the config's global default. Seed is broadly supported and helps any
    # residual tie-breaking randomness.
    if reasoning_effort_override is None and temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed

    # Provider-specific thinking control.
    thinking_path = provider_cfg.get("thinking_kwarg_path")
    if thinking_path:
        val = (
            provider_cfg.get("thinking_on_value") if enable_thinking
            else provider_cfg.get("thinking_off_value")
        )
        # Skip injection if the off-value is null/empty — some providers
        # want the param simply omitted in the off state.
        if val is not None and val != "":
            _set_nested(body, thinking_path, val)

    # Per-model reasoning-effort override (e.g. gpt-5.5 always wants
    # reasoning_effort="none" regardless of global enable_thinking).
    if reasoning_effort_override is not None:
        body["reasoning_effort"] = reasoning_effort_override

    return {
        "custom_id": custom_id, "body": body,
        "method": "POST", "url": "/v1/chat/completions",
    }


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
def _run_threaded(
    client: OpenAI, requests: list[dict[str, Any]], max_workers: int,
) -> list[dict[str, Any]]:
    """Sync ChatCompletions in a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out: dict[str, dict[str, Any]] = {}

    def _call(req: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        body = dict(req["body"])
        try:
            resp = client.chat.completions.create(**body)
            return req["custom_id"], {
                "custom_id": req["custom_id"],
                "response": {"body": resp.model_dump()},
            }
        except Exception as e:
            return req["custom_id"], {
                "custom_id": req["custom_id"],
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }

    completed = 0
    total = len(requests)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_call, r) for r in requests]
        for fut in as_completed(futures):
            cid, rec = fut.result()
            out[cid] = rec
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"  threaded: {completed}/{total} done", flush=True)
    return [out[r["custom_id"]] for r in requests]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    with open(args.config, "rb") as f:
        params: dict[str, Any] = tomllib.load(f)
    # ---- find model entry + provider ----
    sweep_models = params.get("sweep", {}).get("models") or []
    model_entry = find_model_entry(sweep_models, args.model, args.provider)
    provider_name = model_entry["provider"]
    if provider_name not in params.get("providers", {}):
        raise RuntimeError(
            f"Provider {provider_name!r} not defined in [providers.{provider_name}]"
        )
    provider_cfg = params["providers"][provider_name]

    # ---- resolve effective params (global < provider < model < CLI) ----
    max_workers = (
        args.max_workers
        or model_entry.get("max_workers")
        or provider_cfg.get("max_workers")
        or 16
    )
    max_completion_tokens = int(
        args.max_completion_tokens
        or model_entry.get("max_completion_tokens")
        or provider_cfg.get("max_completion_tokens")
        or params.get("max_completion_tokens", 1024)
    )
    enable_thinking = (
        (not args.no_thinking)
        and bool(model_entry.get("enable_thinking", params.get("enable_thinking", True)))
    )
    response_format_json = bool(model_entry.get(
        "response_format_json", provider_cfg.get("response_format_json", True)
    ))
    reasoning_effort_override = (
        args.reasoning_effort or model_entry.get("reasoning_effort")
    )  # may be None
    # Temperature resolves None-aware so a valid 0.0 isn't overridden by a
    # later default (global < provider < model). Default 0.0 = greedy.
    temperature = _first_not_none(
        model_entry.get("temperature"),
        provider_cfg.get("temperature"),
        params.get("temperature"),
    )
    if temperature is None:
        temperature = 0.0

    # ---- dataset ----
    if DATASET not in params["datasets"]:
        raise RuntimeError(
            f"No [datasets.{DATASET}] section in {args.config}"
        )
    ds_cfg = params["datasets"][DATASET]
    # Sent to the API as `seed`; also seeds the subsample RNG below.
    seed = int(params["seed"])

    print(f"Loading {DATASET}...")
    texts, labels, label_names = load_eval()
    N_full = len(texts)
    if args.limit is not None and N_full > args.limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_full, size=args.limit, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]
    N = len(texts)
    K = len(label_names)
    print(f"  N={N}/{N_full} K={K}")

    # Verbalized-confidence -> probability conversion (below) puts the
    # judge's confidence c on its chosen class and spreads the remaining
    # 1-c uniformly over the others. That uniform spread is only
    # defensible when there is exactly one other class, so this runner is
    # restricted to binary tasks. Fail fast (before any API calls) rather
    # than silently producing an arbitrary K>2 probability vector.
    if K != 2:
        raise ValueError(
            f"This judge runner is binary-only (K=2), but dataset "
            f"{DATASET!r} has K={K}. The verbalized-confidence "
            f"probability conversion would spread 1-c uniformly over "
            f"{K - 1} classes, which is not a justified posterior. Use a "
            f"binary dataset or extend the conversion before running."
        )

    # ---- descriptions ----
    desc_map = load_descriptions(DATASET)
    missing = [n for n in label_names if n not in desc_map]
    if missing:
        raise RuntimeError(
            f"{DESCS_CFG} has no description for {missing}. The descriptions "
            "are part of the judge prompt."
        )
    ordered_labels: dict[str, str] = {n: desc_map[n] for n in label_names}

    # ---- requests ----
    user_template = ds_cfg["user_template"]
    system_prompt = ds_cfg["system_prompt"]
    requests = []
    for i, t in enumerate(texts):
        user_msg = build_judge_user_message(
            template=user_template, ordered_labels=ordered_labels, input_text=t
        )
        requests.append(
            build_request(
                custom_id=f"{DATASET}-{i}",
                model=args.model,
                system_prompt=system_prompt,
                user_msg=user_msg,
                max_completion_tokens=max_completion_tokens,
                response_format_json=response_format_json,
                enable_thinking=enable_thinking,
                provider_cfg=provider_cfg,
                reasoning_effort_override=reasoning_effort_override,
                temperature=temperature,
                seed=seed,
            )
        )

    # ---- results dir ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"{provider_name}_{model_entry['short']}"
    if args.run_tag_suffix:
        run_tag = f"{run_tag}_{args.run_tag_suffix}"
    dir_name = f"{run_tag}_{timestamp}"
    results_dir = HERE.parent / "results" / "judge" / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "input.jsonl"
    with open(jsonl_path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    # ---- client ----
    client = OpenAI(
        api_key=_load_api_key(provider_cfg["api_key_env_var"]),
        base_url=provider_cfg.get("api_base_url"),
    )

    # ---- dispatch ----
    print(
        f"Running provider={provider_name}, model={args.model}, "
        f"n={N}, max_workers={max_workers}, enable_thinking={enable_thinking}..."
    )
    out_records = _run_threaded(client, requests, max_workers=max_workers)

    output_path = results_dir / "output.jsonl"
    with open(output_path, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    # ---- parse ----
    by_id = {rec["custom_id"]: rec for rec in out_records}
    parsed_labels: list[str | None] = [None] * N
    parsed_confs: list[float | None] = [None] * N
    raw_outputs: list[str] = [""] * N
    in_tok = out_tok = 0
    api_failures = 0
    served_models: Counter[str] = Counter()
    for i in range(N):
        rec = by_id.get(f"{DATASET}-{i}")
        if rec is None or rec.get("error"):
            api_failures += 1
            continue
        body = rec["response"]["body"]
        served = body.get("model")
        if served:
            served_models[served] += 1
        usage = body.get("usage", {})
        in_tok += int(usage.get("prompt_tokens", 0) or 0)
        out_tok += int(usage.get("completion_tokens", 0) or 0)
        text = body["choices"][0]["message"]["content"] or ""
        raw_outputs[i] = text
        lbl, conf, _raw = parse_label(text, label_names)
        parsed_labels[i] = lbl
        parsed_confs[i] = conf

    # ---- served-model guard ----
    # Fail loudly if the provider served a model other than the one requested
    # (some providers silently forward a deprecated model to a replacement
    # and bill for it). Abort BEFORE writing eval_per_sample.csv / run_config.toml
    # so a substituted run can never be mistaken for a valid result; the raw
    # output.jsonl (carrying the served `model` field) stays on disk as evidence.
    distinct_served = sorted(served_models)
    served_summary = (
        ", ".join(f"{m}×{served_models[m]}" for m in distinct_served)
        or "(none reported)"
    )
    mismatched = [m for m in distinct_served if not served_model_matches(args.model, m)]
    if not served_models:
        print("  WARNING: no response carried a `model` field; cannot verify "
              "served-model identity.")
    elif mismatched:
        sys.stderr.write(
            "\nERROR: served model does not match the requested model.\n"
            f"  requested: {args.model}\n"
            f"  served:    {served_summary}\n"
            "  The provider forwarded the request to a different model. "
            "Refusing to write eval_per_sample.csv / run_config.toml.\n"
            f"  Raw responses (with served `model`) kept at: {output_path}\n"
        )
        raise SystemExit(2)
    else:
        print(f"  served-model verified: {served_summary}")

    # ---- prob matrix: verbalized confidence (Tian 2023) ----
    # K is guaranteed 2 by the guard above, so the (K-1) below is 1: the
    # chosen class gets c and the single other class gets 1-c.
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    probs = np.full((N, K), 1.0 / K, dtype=np.float64)
    pred_indices = np.zeros(N, dtype=np.int64)
    parse_failures = 0
    confidence_count = 0
    for i, lbl in enumerate(parsed_labels):
        if lbl is None:
            parse_failures += 1
            pred_indices[i] = int(probs[i].argmax())
            continue
        k = name_to_idx[lbl]
        pred_indices[i] = k
        conf = parsed_confs[i]
        if conf is None:
            probs[i] = 0.0
            probs[i, k] = 1.0
        else:
            c = float(min(max(conf, 0.0), 1.0))
            probs[i] = 1.0 - c
            probs[i, k] = c
            confidence_count += 1
    confidence_rate = confidence_count / N

    true_indices = np.asarray(labels, dtype=np.int64)
    correct = (pred_indices == true_indices).astype(np.int64)
    accuracy = float(correct.mean())
    test_ece = float(ece_top1(probs, correct))
    parse_failure_rate = parse_failures / N
    api_failure_rate = api_failures / N

    # ---- cost ----
    cost_usd = (
        in_tok / 1_000_000 * float(model_entry.get("input_per_million", 0))
        + out_tok / 1_000_000 * float(model_entry.get("output_per_million", 0))
    )

    # ---- write ----
    csv_path = results_dir / "eval_per_sample.csv"
    write_eval_csv(csv_path, label_names, true_indices, pred_indices, probs)
    with open(results_dir / "raw_outputs.txt", "w") as f:
        for i, raw in enumerate(raw_outputs):
            f.write(f"=== {i} ===\n{raw}\n")

    print(
        f"  accuracy={accuracy:.4f}  ece={test_ece:.4f}  "
        f"parse_fail={parse_failure_rate:.4f}  api_fail={api_failure_rate:.4f}  "
        f"conf_rate={confidence_rate:.4f}\n"
        f"  tokens: in={in_tok:,} out={out_tok:,}  cost≈${cost_usd:.4f}"
    )

    write_run_config(
        results_dir / "run_config.toml",
        params,
        {
            "timestamp": timestamp,
            "dataset": DATASET, "model": args.model,
            "provider": provider_name,
            "limit": args.limit, "eval_size": N, "num_classes": K,
            "test_accuracy": round(accuracy, 4),
            "test_ece": round(test_ece, 4),
            "parse_failure_rate": round(parse_failure_rate, 4),
            "api_failure_rate": round(api_failure_rate, 4),
            "confidence_rate": round(confidence_rate, 4),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "estimated_cost_usd": round(cost_usd, 4),
            "max_workers": max_workers,
            "max_completion_tokens_effective": max_completion_tokens,
            "enable_thinking": enable_thinking,
            "thinking_mode": "on" if enable_thinking else "off",
            "reasoning_effort_effective": reasoning_effort_override or "",
            "temperature": temperature,
            "api_seed": seed,
            "served_model": distinct_served[0] if len(distinct_served) == 1 else "",
            "served_models": served_summary,
        },
        derived={
            "label_names": label_names,
            "label_descriptions": ordered_labels,
            # Recorded relative to the repo root so shared results carry no local paths.
            "descriptions_source": str(DESCS_CFG.relative_to(HERE.parent)),
            "user_template": user_template,
            "system_prompt": system_prompt,
            "provider_config": provider_cfg,
        },
    )
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
