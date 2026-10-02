#!/bin/bash
# Export per-cell metrics and per-sample panels into export/ and figures/figdata/. Run from the package root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

$PY export/export_figure_data.py
