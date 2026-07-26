#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PYTHON="${PYTHON:-python}"; OUTPUT_DIR="${OUTPUT_DIR:-rebuttal/06_zi_kl_20ng/outputs}"
"$PYTHON" rebuttal/06_zi_kl_20ng/main.py --output-dir "$OUTPUT_DIR" "${@}"
