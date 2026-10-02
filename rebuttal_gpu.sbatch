#!/bin/bash
#SBATCH --job-name=fska_rebuttal
#SBATCH --output=slurm_logs/rebuttal_%A_%a.out
#SBATCH --error=slurm_logs/rebuttal_%A_%a.err
#SBATCH -p gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${MODE:?Submit with MODE=smoke,01,02,03,04,05,06p,or 06a}"

module purge
module load python/3.11.14

REPO=/home/dacabeza/lustre/dacabeza/NFattention
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

cd "${REPO}"

if [[ ! -f "${REPO}/rebuttal/README.md" ]]; then
    echo "ERROR: ${REPO} is not the NFattention repository root" >&2
    exit 2
fi

if [[ ! -f "${REPO}/.venv/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at ${REPO}/.venv" >&2
    exit 2
fi

source "${REPO}/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

DATA_ROOT="${REPO}/rebuttal/data"
RESULT_ROOT="${REPO}/rebuttal/slurm_outputs"
TASK_OUTPUT="${RESULT_ROOT}/${MODE}/job_${ARRAY_JOB_ID}/task_${TASK_ID}"

mkdir -p "${TASK_OUTPUT}"

echo "Host: $(hostname)"
echo "Repository: ${REPO}"
echo "Mode: ${MODE}"
echo "Task: ${TASK_ID}"
echo "Output: ${TASK_OUTPUT}"

which python
python -V

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Visible CUDA devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('Selected GPU:', torch.cuda.get_device_name(0))
"

nvidia-smi

case "${MODE}" in
    smoke)
        python rebuttal/01_sst5_diagnostics/main.py \
            --smoke \
            --device cuda:0 \
            --seeds 0 \
            --output-dir "${TASK_OUTPUT}/01"

        python rebuttal/02_fixed_density/main.py \
            --smoke \
            --device cuda:0 \
            --seeds 0 \
            --output-dir "${TASK_OUTPUT}/02"

        python rebuttal/03_kl_sensitivity/main.py \
            --smoke \
            --device cuda:0 \
            --seeds 0 \
            --output-dir "${TASK_OUTPUT}/03"

        python rebuttal/04_elu_gap/main.py \
            --smoke \
            --device cuda:0 \
            --seeds 0 \
            --output-dir "${TASK_OUTPUT}/04"

        python rebuttal/05_efficiency/main.py \
            --smoke \
            --device cuda:0 \
            --output-dir "${TASK_OUTPUT}/05"

        python rebuttal/06_listops/main.py \
            --smoke \
            --device cuda:0 \
            --primary-seeds 0 \
            --ablation-seeds 0 \
            --output-dir "${TASK_OUTPUT}/06"

        echo "Smoke validation completed successfully."

        echo "Staging SST-5 and ListOps datasets..."

        python -c "
from pathlib import Path
from rebuttal._shared.datasets import load_sst5, load_listops

root = Path('${DATA_ROOT}')

sst5 = load_sst5(root)
print('SST-5:', sst5.manifest())

listops = load_listops(root / 'listops', download=True)
print('ListOps:', listops.manifest())
"

        echo "Dataset staging completed."
        ;;

    01|02|03|04)
        case "${MODE}" in
            01) EXPERIMENT="01_sst5_diagnostics" ;;
            02) EXPERIMENT="02_fixed_density" ;;
            03) EXPERIMENT="03_kl_sensitivity" ;;
            04) EXPERIMENT="04_elu_gap" ;;
        esac

        export PYTHON=python
        export CUDA_DEVICE=cuda:0
        export SEEDS="${TASK_ID}"
        export DATA_DIR="${DATA_ROOT}"
        export OUTPUT_DIR="${TASK_OUTPUT}"
        export SKIP_SMOKE=1

        bash "rebuttal/${EXPERIMENT}/run.sh"
        ;;

    05)
        export PYTHON=python
        export CUDA_DEVICE=cuda:0
        export OUTPUT_DIR="${TASK_OUTPUT}"
        export SEQ_LENGTHS=256,512,1024,2000,4096
        export SKIP_SMOKE=1

        bash rebuttal/05_efficiency/run.sh
        ;;

    06p)
        python rebuttal/06_listops/main.py \
            --device cuda:0 \
            --primary-seeds "${TASK_ID}" \
            --ablation-seeds "" \
            --steps 5000 \
            --microbatch 4 \
            --gradient-accumulation 8 \
            --no-download \
            --data-dir "${DATA_ROOT}/listops" \
            --output-dir "${TASK_OUTPUT}"
        ;;

    06a)
        python rebuttal/06_listops/main.py \
            --device cuda:0 \
            --primary-seeds "" \
            --ablation-seeds "${TASK_ID}" \
            --steps 5000 \
            --microbatch 4 \
            --gradient-accumulation 8 \
            --no-download \
            --data-dir "${DATA_ROOT}/listops" \
            --output-dir "${TASK_OUTPUT}"
        ;;

    *)
        echo "ERROR: unknown MODE=${MODE}" >&2
        exit 2
        ;;
esac

echo "Job completed successfully."