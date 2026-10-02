#!/bin/bash
# Reproduce one cell of Fig 1 -- the Qwen3.5-0.8B-Base channel and direct cells --
# and compare it against the paper's numbers, printing PASS/NEAR/FAIL. Cheap way to
# check an environment before committing to the six-model sweep. Run from the
# package root; a few minutes, plus the model download.
# Usage: [PY=/path/to/python] bash scripts/verify_baseline.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

run () {  # stage-name command...
  local name=$1; shift
  echo "=== $name ==="
  "$@" || { echo "FATAL: $name failed" >&2; exit 1; }
}

run channel     "$PY" channel/run.py channel/config.toml --model Qwen/Qwen3.5-0.8B-Base
run direct      "$PY" direct/run.py --model Qwen/Qwen3.5-0.8B-Base
run corrections bash scripts/03_direct_corrections.sh
echo "(the missing-9B-panel warning from export is expected in this 0.8B-only tree)"
run export      bash scripts/07_export.sh

"$PY" - <<'PYEOF'
import csv
import sys

# The paper's Qwen3.5-0.8B-Base cells.
EXPECTED = {
    ("channel", ""): {"acc": 0.5796, "ece": 0.1316},
    ("direct", "pq_redacted"): {"acc": 0.6574, "ece": 0.0348},
}

# PASS means exact. These are the NEAR bands, deliberately wider than the ~0.5pp
# the README leads you to expect, so that ordinary GPU and driver variation reads
# as NEAR rather than as a scary FAIL. pq_redacted gets the wider band: its
# corrected posteriors sit near the decision boundary by construction, which is
# what gives it good ECE, so a tiny log-prob wobble flips several samples.
NEAR = {"channel": 0.02, "direct": 0.04}

try:
    with open("export/fig1_data.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("model", "").startswith("Qwen3.5-0.8B")]
except FileNotFoundError:
    sys.exit("export/fig1_data.csv not found")

worst = "PASS"


def record(label, expected, got, verdict):
    global worst
    if verdict == "FAIL" or (verdict == "NEAR" and worst == "PASS"):
        worst = verdict
    print(f"{label:<26}{expected:<12}{got:<12}{verdict}")


print(f"{'metric':<26}{'expected':<12}{'got':<12}verdict")
for (method, variant), expected in EXPECTED.items():
    label = method if not variant else f"{method}/{variant}"
    match = [r for r in rows
             if r.get("method") == method and (r.get("variant") or "") == variant]
    if not match:
        record(label, "(row missing)", "", "FAIL")
        continue
    row = match[0]
    if row.get("n", "").strip() != "540":
        record(f"{label}/n", "540", row.get("n", ""), "FAIL")
    for metric, want in expected.items():
        try:
            got = float(row.get(metric, ""))
        except (TypeError, ValueError):
            record(f"{label}/{metric}", want, "(bad value)", "FAIL")
            continue
        delta = abs(got - want)
        record(f"{label}/{metric}", want, got,
               "PASS" if delta < 1e-9 else "NEAR" if delta <= NEAR[method] else "FAIL")

print({
    "PASS": "Overall: PASS",
    "NEAR": "Overall: NEAR -- plausible GPU/dtype wobble. The paper's numbers come "
            "from an A100; judgment call.",
    "FAIL": "Overall: FAIL",
}[worst])
sys.exit({"PASS": 0, "NEAR": 2, "FAIL": 1}[worst])
PYEOF
