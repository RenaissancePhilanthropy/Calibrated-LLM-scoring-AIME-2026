#!/bin/bash
# Run the 4 OpenAI-API judge cells. Requires OPENAI_API_KEY. Run from the package root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p logs

run_cell () { local tag=$1; shift
  $PY judge/run.py "$@" 2>&1 | tee "logs/judge_$tag.log"; }
run_cell gpt4o_mini    --model gpt-4o-mini --max-workers 32
run_cell gpt4o         --model gpt-4o --max-workers 32
run_cell gpt55_nothink --model gpt-5.5 --max-workers 32
run_cell gpt55_think   --model gpt-5.5 --max-workers 32 \
    --reasoning-effort medium --max-completion-tokens 16384 --run-tag-suffix effort_medium
