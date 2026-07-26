#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/05_zi_efficiency/outputs}"
"$PYTHON" rebuttal/05_zi_efficiency/main.py --output-dir "$OUTPUT_DIR" "${@}"
