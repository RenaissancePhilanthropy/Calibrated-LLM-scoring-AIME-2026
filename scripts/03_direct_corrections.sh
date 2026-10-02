#!/bin/bash
# Per-question content-free calibration of the direct runs, writing the
# figures' pq_redacted CSV beside each. Needs the GPU. Run from the package
# root. Read per_question_cf.py's docstring before editing it: the
# counterfactual forward passes must not be pruned.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

$PY direct/per_question_cf.py
