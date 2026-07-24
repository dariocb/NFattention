#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-cuda}"
PRIMARY_SEED_LIST="${SEEDS:-0,1,2,3,4}"
ABLATION_SEED_LIST="${ABLATION_SEEDS:-0,1,2}"
DATA_ROOT="${DATA_DIR:-${REPO_ROOT}/rebuttal/data/listops}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_ID}}"
STEP_COUNT="${STEPS:-5000}"
MICROBATCH_SIZE="${MICROBATCH:-4}"
ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"

"${PYTHON_BIN}" -c "import torch, sklearn, normflows"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
    --smoke --device "${DEVICE}" --primary-seeds 0 --ablation-seeds 0 \
    --output-dir "${RESULT_ROOT}/smoke"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --device "${DEVICE}" \
  --primary-seeds "${PRIMARY_SEED_LIST}" \
  --ablation-seeds "${ABLATION_SEED_LIST}" \
  --steps "${STEP_COUNT}" \
  --microbatch "${MICROBATCH_SIZE}" \
  --gradient-accumulation "${ACCUMULATION}" \
  --data-dir "${DATA_ROOT}" \
  --output-dir "${RESULT_ROOT}"

