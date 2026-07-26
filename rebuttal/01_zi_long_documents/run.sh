#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/01_zi_long_documents/outputs}"
"$PYTHON" rebuttal/01_zi_long_documents/main.py --output-dir "$OUTPUT_DIR" "${@}"
