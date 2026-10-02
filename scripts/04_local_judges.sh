#!/bin/bash
# Serve + run the 8 local-vLLM judge cells, one model server at a time. Run from the package root.
# No -e: a failed start_server must not abort the script, so a dead server is reported and skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p logs
source scripts/lib.sh

# Client-side key for the local server; judge/config.toml reads it as
# local_vllm's api_key_env_var. start_server strips it from the server's
# own environment.
export VLLM_API_KEY="${VLLM_API_KEY:-local-dummy}"

run_cell () { local tag=$1; shift
  $PY judge/run.py --provider local_vllm "$@" \
      2>&1 | tee "logs/judge_$tag.log"; }

start_server Qwen/Qwen3.5-9B qwen3 20480 c202236235762e1c871ad0ccb60c8ee5ba337b9a && {
  run_cell qwen9b_think   --model Qwen/Qwen3.5-9B --max-workers 16 --max-completion-tokens 16384
  run_cell qwen9b_nothink --model Qwen/Qwen3.5-9B --max-workers 16 --no-thinking --run-tag-suffix nothink
}; stop_server
start_server Qwen/Qwen3.5-4B qwen3 20480 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a && {
  run_cell qwen4b_think   --model Qwen/Qwen3.5-4B --max-workers 24 --max-completion-tokens 16384
  run_cell qwen4b_nothink --model Qwen/Qwen3.5-4B --max-workers 24 --no-thinking --run-tag-suffix nothink
}; stop_server
start_server Qwen/Qwen3.5-2B qwen3 20480 15852e8c16360a2fea060d615a32b45270f8a8fc && run_cell qwen2b_nothink --model Qwen/Qwen3.5-2B --max-workers 24 --no-thinking --run-tag-suffix nothink; stop_server
start_server Qwen/Qwen3.5-0.8B qwen3 20480 2fc06364715b967f1860aea9cf38778875588b17 && run_cell qwen08b_nothink --model Qwen/Qwen3.5-0.8B --max-workers 24 --no-thinking --run-tag-suffix nothink; stop_server
start_server meta-llama/Llama-3.2-3B-Instruct "" 8192 0cb88a4f764b7a12671c53f0838cd831a0843b95 && run_cell llama3b --model meta-llama/Llama-3.2-3B-Instruct --max-workers 24 --no-thinking; stop_server
start_server meta-llama/Llama-3.2-1B-Instruct "" 8192 9213176726f574b556790deb65791e0c5aa438b6 && run_cell llama1b --model meta-llama/Llama-3.2-1B-Instruct --max-workers 24 --no-thinking; stop_server
