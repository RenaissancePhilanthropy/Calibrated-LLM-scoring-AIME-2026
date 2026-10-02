#!/bin/bash
# Prefix-reveal interpretability data for the Fig-4 examples (samples 491, 229). Run from the package root.
# Needs the 9B runs from 01/02 on disk, then a 9B vLLM server (as in script 04) for the judge trace.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p logs
source scripts/lib.sh

$PY interpretability/prefix_reveal.py --samples 491 229 --do-channel --do-direct

if start_server Qwen/Qwen3.5-9B qwen3 20480 c202236235762e1c871ad0ccb60c8ee5ba337b9a; then
  trap stop_server EXIT
  $PY interpretability/prefix_reveal.py --samples 491 229 --do-judge \
      --judge-base-url http://localhost:8000/v1
  stop_server
  trap - EXIT
else
  stop_server
  echo "FATAL: vLLM server for Qwen/Qwen3.5-9B failed to start; Fig 4's judge CSVs were not produced." >&2
  exit 1
fi
