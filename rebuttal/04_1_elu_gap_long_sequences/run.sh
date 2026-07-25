#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-cuda}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_ID}}"
LENGTHS="${SEQ_LENGTHS:-128,256,512,1024,2000,4096}"

"${PYTHON_BIN}" -c "import torch, scipy, normflows"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" --smoke --device "${DEVICE}" --output-dir "${RESULT_ROOT}/smoke"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" --device "${DEVICE}" --lengths "${LENGTHS}" --output-dir "${RESULT_ROOT}"
