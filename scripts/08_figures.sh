#!/bin/bash
# Render the four paper figures into figures/generated/. Run from the package root.
set -euo pipefail
cd "$(dirname "$0")/.."

Rscript figures/paper_figs.R && Rscript figures/interp_figs.R
echo "Compare figures/generated/ against figures/reference/"
