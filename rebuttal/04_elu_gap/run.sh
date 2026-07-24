#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-cuda}"
SEED_LIST="${SEEDS:-0,1,2,3,4}"
DATA_ROOT="${DATA_DIR:-${REPO_ROOT}/rebuttal/data}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_ID}}"

"${PYTHON_BIN}" -c "import torch, sklearn, scipy, datasets, normflows"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
    --smoke --device "${DEVICE}" --seeds 0 \
    --output-dir "${RESULT_ROOT}/smoke"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --device "${DEVICE}" --seeds "${SEED_LIST}" \
  --data-dir "${DATA_ROOT}" --output-dir "${RESULT_ROOT}"

