# Shared by the pipeline scripts. Sourced, not run:  source scripts/lib.sh
# (from the package root, after the caller has set PY).

# The six base models behind every channel and direct cell in Figures 1-4.
# Stated once so the two sweeps can never cover different sets.
BASE_MODELS=(Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-2B-Base Qwen/Qwen3.5-4B-Base \
             Qwen/Qwen3.5-9B-Base meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B)

# run_sweep <method> <command...> — run <command> --model M for every base
# model, tee-ing each to logs/<method>_<model>.log. Keeps going after a
# failure so one bad model does not cost the whole sweep, then reports.
run_sweep () {
  local method=$1; shift
  local fail=0 model slug
  for model in "${BASE_MODELS[@]}"; do
    slug=$(echo "$model" | tr '/.' '__')
    echo "=== $method $model ==="
    if "$@" --model "$model" 2>&1 | tee "logs/${method}_${slug}.log"; then
      echo "OK: $model"
    else
      echo "FAILED (see log): $model"
      fail=1
    fi
  done
  return $fail
}

# vLLM serving, used by the judge cells (04) and the Fig-4 judge trace (06).
# Model revisions are pinned to the snapshots the paper's numbers came from,
# so callers pass the revision explicitly.
#
# FlashInfer's top-k/top-p sampler is off by default: it JIT-compiles a kernel
# and so needs `ninja` on PATH, and without it the server dies during startup
# memory profiling. The judge decodes greedily (temperature 0, no top_p/top_k),
# so that sampler never runs on a real request and switching it off does not
# change any output. Install ninja and set VLLM_USE_FLASHINFER_SAMPLER=1 to
# restore it.
SERVER_PID=""

start_server () {  # repo parser maxlen rev
  local repo=$1 parser=$2 maxlen=$3 rev=$4 extra=()
  [ -n "$parser" ] && extra+=(--reasoning-parser "$parser")
  env -u VLLM_API_KEY HF_HUB_CACHE=models \
      VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
      vllm serve "$repo" --dtype bfloat16 \
      --port 8000 --served-model-name "$repo" --max-model-len "$maxlen" --revision "$rev" "${extra[@]}" \
      > "logs/vllm_$(echo "$repo" | tr '/.' '__').log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 600); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server for $repo died"; return 1; }
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null \
      | grep -q 200 && return 0
    sleep 3
  done
  return 1
}

stop_server () {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 10
}
