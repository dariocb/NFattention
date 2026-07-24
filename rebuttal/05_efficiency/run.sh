#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-cuda}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_ID}}"
# Log-spaced by default: this separates fixed GPU/flow overhead at short
# contexts from the asymptotic long-context scaling regime.  Override with
# SEQ_LENGTHS when hardware capacity or a focused follow-up requires it.
LENGTH_LIST="${SEQ_LENGTHS:-128,256,512,1024,2048,4096,8192,16384,32768}"

"${PYTHON_BIN}" -c "import torch, sklearn, normflows"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
    --smoke --device "${DEVICE}" --output-dir "${RESULT_ROOT}/smoke"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --device "${DEVICE}" --lengths "${LENGTH_LIST}" \
  --output-dir "${RESULT_ROOT}"
