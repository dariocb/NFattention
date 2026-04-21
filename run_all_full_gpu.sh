#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${1:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/full_gpu_${TS}"
STATUS_FILE="${ROOT_DIR}/logs/full_gpu_stage_status.tsv"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
USE_CURRENT_ENV="${USE_CURRENT_ENV:-0}"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/logs"
touch "${STATUS_FILE}"

cd "${ROOT_DIR}"

print_stage_done() {
  local stage="$1"
  local total="$2"
  local label="$3"
  echo
  echo "######################################################################"
  echo "######################################################################"
  echo "###                                                                ###"
  echo "###              EXPERIMENT ${stage}/${total} FINISHED                    ###"
  echo "###              ${label}"
  echo "###                                                                ###"
  echo "######################################################################"
  echo "######################################################################"
  echo
}

mark_stage_status() {
  local stage_key="$1"
  local status="$2"
  local log_path="$3"
  local now
  now="$(date +%Y-%m-%dT%H:%M:%S%z)"
  printf "%s\t%s\t%s\t%s\n" "${stage_key}" "${status}" "${now}" "${log_path}" >> "${STATUS_FILE}"
}

stage_already_success() {
  local stage_key="$1"
  grep -q "^${stage_key}"$'\t'"SUCCESS"$'\t' "${STATUS_FILE}"
}

if [[ "${USE_CURRENT_ENV}" == "1" ]]; then
  echo "[INFO] Using currently active environment (USE_CURRENT_ENV=1)"
else
  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[ERROR] venv not found at ${VENV_PATH}"
    echo "Set VENV_PATH to a valid env or use USE_CURRENT_ENV=1 after manual activation."
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "[INFO] Logs will be written to: ${LOG_DIR}"
echo "[INFO] Stage status file: ${STATUS_FILE}"
echo "[INFO] Python executable: $(command -v python)"
python -V | tee "${LOG_DIR}/python_version.txt"
echo "[INFO] Checking GPU visibility..."
nvidia-smi | tee "${LOG_DIR}/nvidia_smi.txt"

python - <<'PY' | tee "${LOG_DIR}/torch_cuda_check.txt"
import torch
print("cuda_available=", torch.cuda.is_available())
print("torch_cuda=", torch.version.cuda)
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name=", torch.cuda.get_device_name(0))
PY

if ! python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[ERROR] torch.cuda.is_available() is False. Fix driver/CUDA/PyTorch before running full GPU benchmarks."
  exit 1
fi

if stage_already_success "text_full"; then
  echo "[INFO] Skipping text_full (already marked SUCCESS in ${STATUS_FILE})"
else
  echo "[INFO] Running full text benchmark..."
  if (
    cd "${ROOT_DIR}/experiments/text_classification/text_seq_classification_benchmark"
    python run_benchmark.py \
      --datasets rt_polarity sst2 sst5 trec ag_news dbpedia_14 yelp_review_full quora \
      --models transformer mikan performer rka gmm_rks mgk kpca_scaled metala ours_latest \
      --seeds 0 1 2 3 4 \
      --data_dir ../data \
      --gpu 0
  ) 2>&1 | tee "${LOG_DIR}/text_full.log"; then
    mark_stage_status "text_full" "SUCCESS" "${LOG_DIR}/text_full.log"
    print_stage_done "1" "4" "TEXT CLASSIFICATION BENCHMARK"
  else
    mark_stage_status "text_full" "FAILED" "${LOG_DIR}/text_full.log"
    echo "[ERROR] text_full failed. See: ${LOG_DIR}/text_full.log"
    exit 1
  fi
fi

if stage_already_success "lra_full"; then
  echo "[INFO] Skipping lra_full (already marked SUCCESS in ${STATUS_FILE})"
else
  echo "[INFO] Running full LRA benchmark..."
  if (
    cd "${ROOT_DIR}/experiments/lra_benchmark"
    python run_full_benchmark.py \
      --device cuda \
      --models transformer mikan performer rka gmm_rks mgk kpca_scaled metala ours_latest \
      --seq_lens 256 512 1024 2048 4096 \
      --batch_size 4 \
      --d_model 128 \
      --n_heads 4 \
      --n_layers 2 \
      --M 64
  ) 2>&1 | tee "${LOG_DIR}/lra_full.log"; then
    mark_stage_status "lra_full" "SUCCESS" "${LOG_DIR}/lra_full.log"
    print_stage_done "2" "4" "LRA BENCHMARK"
  else
    mark_stage_status "lra_full" "FAILED" "${LOG_DIR}/lra_full.log"
    echo "[ERROR] lra_full failed. See: ${LOG_DIR}/lra_full.log"
    exit 1
  fi
fi

if stage_already_success "mnist_full"; then
  echo "[INFO] Skipping mnist_full (already marked SUCCESS in ${STATUS_FILE})"
else
  echo "[INFO] Running full MNIST ablation benchmark..."
  if (
    cd "${ROOT_DIR}"
    PYTHONPATH=. python experiments/mnist_ablation/mnist_classification_ablation.py \
      --use_full_data \
      --N_EPOCHS 50 \
      --output_dir mnist_results_gpu
  ) 2>&1 | tee "${LOG_DIR}/mnist_full.log"; then
    mark_stage_status "mnist_full" "SUCCESS" "${LOG_DIR}/mnist_full.log"
    print_stage_done "3" "4" "MNIST FULL ABLATION"
  else
    mark_stage_status "mnist_full" "FAILED" "${LOG_DIR}/mnist_full.log"
    echo "[ERROR] mnist_full failed. See: ${LOG_DIR}/mnist_full.log"
    exit 1
  fi
fi

if stage_already_success "mnist_focused_full"; then
  echo "[INFO] Skipping mnist_focused_full (already marked SUCCESS in ${STATUS_FILE})"
else
  echo "[INFO] Running focused MNIST ablation benchmark..."
  if (
    cd "${ROOT_DIR}"
    PYTHONPATH=. python experiments/mnist_ablation/mnist_focused_ablation.py \
      --use_full_data \
      --epochs 20 \
      --seed 42 \
      --output_dir mnist_focused_results_gpu
  ) 2>&1 | tee "${LOG_DIR}/mnist_focused_full.log"; then
    mark_stage_status "mnist_focused_full" "SUCCESS" "${LOG_DIR}/mnist_focused_full.log"
    print_stage_done "4" "4" "MNIST FOCUSED ABLATION"
  else
    mark_stage_status "mnist_focused_full" "FAILED" "${LOG_DIR}/mnist_focused_full.log"
    echo "[ERROR] mnist_focused_full failed. See: ${LOG_DIR}/mnist_focused_full.log"
    exit 1
  fi
fi

echo "[INFO] All full GPU benchmarks completed."
echo "[INFO] Consolidated logs: ${LOG_DIR}"
echo "[INFO] Stage status file: ${STATUS_FILE}"
