#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/04_zi_capacity_qk/outputs}"
"$PYTHON" rebuttal/04_zi_capacity_qk/main.py --output-dir "$OUTPUT_DIR" "${@}"
