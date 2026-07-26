#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/03_zi_elu_gap/outputs}"
"$PYTHON" rebuttal/03_zi_elu_gap/main.py --output-dir "$OUTPUT_DIR" "${@}"
