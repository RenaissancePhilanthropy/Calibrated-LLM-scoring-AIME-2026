#!/bin/bash
# Run the channel method across all six base models. Run from the package root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p logs
source scripts/lib.sh

run_sweep channel $PY channel/run.py channel/config.toml
