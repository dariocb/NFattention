#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/02_zi_fixed_bivariate/outputs}"
"$PYTHON" rebuttal/02_zi_fixed_bivariate/main.py --output-dir "$OUTPUT_DIR" "${@}"
