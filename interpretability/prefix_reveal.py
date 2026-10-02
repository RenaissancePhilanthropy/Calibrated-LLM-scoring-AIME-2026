#!/usr/bin/env python3
"""Prefix-reveal interpretability data for the paper's Fig-4 examples.

For each selected sample, reveal the student answer token-by-token with a
"[REDACTED]" tail marking the hidden remainder, and record each method's
P(correct) after every reveal:

  channel  read from the 9B channel run's eval_per_token.csv
           (cumulative c_prob_0 + per-token t_llr) — the paper's own view.
  direct   p_raw(label | question, prefix+"[REDACTED]") via the same
           direct prompt/scorer, corrected per-question by the t=0 case:
           p_cal(t) ∝ p_raw(t) / p_raw(0). At t=0 (nothing revealed,
           student = "[REDACTED]") this is exactly the pq_redacted null, so
           the corrected trajectory starts at 0.5 by construction.
  judge    the judge prompt with the same prefix+"[REDACTED]" answer,
           Qwen3.5-9B instruct no-thinking at temperature 0 via a local vLLM
           OpenAI endpoint; verbalized label+confidence mapped to P(correct).

Prefixes follow the CHANNEL run's own token boundaries (t_str column), so all
three trajectories share an x-axis of the same revealed tokens.

Outputs figdata/interp_<idx>_{channel,direct,judge}.csv.

Usage (direct needs the GPU; judge needs a running vLLM server):
  python interpretability/prefix_reveal.py --samples 491 229 --do-channel --do-direct
  python interpretability/prefix_reveal.py --samples 491 229 --do-judge \
      --judge-base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
FIGDATA = PKG / "figures" / "figdata"

sys.path.insert(0, str(PKG))
from common import build_judge_user_message, parse_label  # noqa: E402
from scientsbank import load_split, render_item  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen3.5-9B"
DIRECT_MODEL = "Qwen/Qwen3.5-9B-Base"


def find_run(root: Path, model: str, channel_run: bool) -> Path:
    """Latest run dir under ``root`` for ``model`` (smoke runs skipped).

    ``channel_run`` requires the config to carry ``common_context_format``,
    which only the channel runner writes.
    """
    found: Path | None = None
    for p in sorted(root.glob("*/run_config.toml")):
        cfg = tomllib.load(open(p, "rb"))
        if cfg["params"].get("model") != model:
            continue
        if channel_run and "common_context_format" not in cfg.get("params", {}):
            continue
        # Skip smoke runs (`--limit`): the paper's cells score all 540 samples.
        if cfg.get("run", {}).get("eval_size", 540) < 100:
            continue
        found = p.parent
    if found is None:
        raise SystemExit(
            f"no run for model {model} under {root}; run scripts 01 and 02 first"
        )
    return found


def channel_tokens(run_dir: Path, idx: int) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "eval_per_token.csv")
    sub = df[df.sample_index == idx].reset_index(drop=True)
    # "Ġ" is the byte-BPE marker for a leading space in the token strings.
    toks = [t.replace("Ġ", " ") for t in sub["t_str"]]
    return pd.DataFrame({
        "step": np.arange(1, len(sub) + 1),
        "token": toks,
        "p_correct": sub["c_prob_0"].to_numpy(float),
        "bar": sub["t_llr"].to_numpy(float),   # per-token LLR (nats)
    })


def load_examples(samples: list[int]) -> dict[int, dict]:
    ds = load_split()
    return {i: dict(ds[int(i)]) for i in samples}


def prefixes_from_channel(chan: pd.DataFrame, student_answer: str) -> list[str]:
    """Cumulative decoded prefixes along the channel run's token boundaries.

    The channel scores the answer inside its own rendering; its t_str
    sequence includes the closing punctuation/quote tokens of the template.
    We accumulate tokens and clip to the raw student answer so the last
    prefix equals the full answer.
    """
    acc, out = "", []
    for tok in chan["token"]:
        acc += tok
        clipped = acc.strip()
        if len(clipped) > len(student_answer):
            clipped = student_answer
        out.append(clipped)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def run_direct(examples: dict[int, dict], prefix_map: dict[int, list[str]]) -> None:
    from channel_models import NLLCalculator

    run_dir = find_run(PKG / "results" / "direct",
                       DIRECT_MODEL, channel_run=False)
    cfg = tomllib.load(open(run_dir / "run_config.toml", "rb"))
    d = cfg["derived"]
    label_contents = list(d["label_contents"])
    calc = NLLCalculator(model=DIRECT_MODEL,
                         model_cache_directory=str(PKG / "models"))

    for idx, ex in examples.items():
        prefs = [""] + prefix_map[idx]           # t=0 (null) first
        contexts = [
            render_item(ex["question"], ex["reference_answer"],
                         (p + "[REDACTED]") if p else "[REDACTED]")
            for p in prefs
        ]
        results = calc.compute_nll(
            contexts=contexts, contents=label_contents, batch_size=8,
            use_tqdm=True,
            template=d.get("common_prefix", "") + d["template"])
        ll = np.zeros((len(contexts), 2))
        for k, result in enumerate(results):
            df = result["df"]
            for i in range(len(contexts)):
                ll[i, k] = -float(df[f"t_nll_{i}"].sum())
        p_raw = np.exp(ll - ll.max(1, keepdims=True))
        p_raw /= p_raw.sum(1, keepdims=True)
        p_cal = p_raw / p_raw[0]                 # per-question null = t=0 row
        p_cal /= p_cal.sum(1, keepdims=True)
        pc = p_cal[:, 0]
        out = pd.DataFrame({
            "step": np.arange(len(prefs)),
            "token": ["<null>"] + list(
                pd.read_csv(FIGDATA / f"interp_{idx}_channel.csv")["token"]),
            "p_correct": pc,
            "p_raw_correct": p_raw[:, 0],
            "bar": np.concatenate([[0.0], np.diff(logit(pc))]),  # per-reveal dlogit
        })
        out.to_csv(FIGDATA / f"interp_{idx}_direct.csv", index=False)
        print(f"direct #{idx}: start={pc[0]:.3f} end={pc[-1]:.3f} "
              f"(raw end={p_raw[-1,0]:.3f})")


def run_judge(examples: dict[int, dict], prefix_map: dict[int, list[str]],
              base_url: str) -> None:
    from openai import OpenAI

    jcfg = tomllib.load(open(PKG / "judge" / "config.toml", "rb"))
    dcfg = tomllib.load(open(PKG / "direct" / "config.toml", "rb"))
    block = jcfg["datasets"]["scientsbank_2way_final"]
    labels_cfg = dcfg["datasets"]["scientsbank_2way_final"]["labels"]
    system_prompt = block["system_prompt"]
    user_template = block["user_template"]

    client = OpenAI(base_url=base_url, api_key="local-dummy")
    for idx, ex in examples.items():
        rows = []
        prefs = [""] + prefix_map[idx]
        for t, p in enumerate(prefs):
            student = (p + "[REDACTED]") if p else "[REDACTED]"
            input_text = render_item(ex["question"], ex["reference_answer"], student)
            user = build_judge_user_message(
                user_template, labels_cfg, input_text)
            messages = ([{"role": "system", "content": system_prompt}]
                        if system_prompt else []) + [
                       {"role": "user", "content": user}]
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, messages=messages, temperature=0.0,
                max_completion_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content or ""
            lbl, conf, _ = parse_label(text, ["correct", "incorrect"])
            if lbl is None:
                pc = float("nan")
            else:
                c = 0.5 if conf is None else float(min(max(conf, 0.0), 1.0))
                pc = c if lbl == "correct" else 1.0 - c
            rows.append({"step": t,
                         "token": "<null>" if t == 0 else None,
                         "p_correct": pc})
        chan_tokens = list(pd.read_csv(FIGDATA / f"interp_{idx}_channel.csv")["token"])
        for t, r in enumerate(rows):
            if t > 0:
                r["token"] = chan_tokens[t - 1]
        out = pd.DataFrame(rows)
        pc = out["p_correct"].to_numpy(float)
        out["bar"] = np.concatenate([[0.0], np.diff(logit(pc))])
        out.to_csv(FIGDATA / f"interp_{idx}_judge.csv", index=False)
        print(f"judge #{idx}: start={pc[0]:.3f} end={pc[-1]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", nargs="+", type=int, default=[491, 229])
    ap.add_argument("--do-channel", action="store_true")
    ap.add_argument("--do-direct", action="store_true")
    ap.add_argument("--do-judge", action="store_true")
    ap.add_argument("--judge-base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()

    FIGDATA.mkdir(exist_ok=True)
    examples = load_examples(args.samples)

    chan_run = find_run(PKG / "results" / "channel",
                        DIRECT_MODEL, channel_run=True)
    prefix_map: dict[int, list[str]] = {}
    for idx, ex in examples.items():
        chan = channel_tokens(chan_run, idx)
        if args.do_channel:
            chan.to_csv(FIGDATA / f"interp_{idx}_channel.csv", index=False)
            print(f"channel #{idx}: {len(chan)} tokens, "
                  f"end P(correct)={chan['p_correct'].iloc[-1]:.3f}")
        prefix_map[idx] = prefixes_from_channel(chan, ex["student_answer"])

    if args.do_direct:
        run_direct(examples, prefix_map)
    if args.do_judge:
        run_judge(examples, prefix_map, args.judge_base_url)


if __name__ == "__main__":
    main()
