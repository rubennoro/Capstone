#!/usr/bin/env bash
# Run the analysis tests with the base python, and again with $ZARR_PYTHON (a venv with
# zarr==2.18.3 numcodecs==0.13.1 imagecodecs) to cover dataset_report's zarr path.
#   train/analysis/tests/run.sh
#   ZARR_PYTHON=~/envs/analysis/bin/python train/analysis/tests/run.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
echo "== $(command -v python3)"
python3 -m unittest discover -s train/analysis/tests "$@"
if [[ -n "${ZARR_PYTHON:-}" ]]; then
  echo "== $ZARR_PYTHON (zarr path)"
  "$ZARR_PYTHON" -m unittest discover -s train/analysis/tests "$@"
fi
