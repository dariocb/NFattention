#!/bin/bash
# Generic Slurm worker for all standalone rebuttal experiments.
#
# Do not submit this file manually unless MODE is provided. The companion
# rebuttal/run_all_slurm.sh script validates the environment and submits the
# complete dependency graph.

# Step 1: Request one GPU and the common host resources.
# Experiment 06 overrides the time limit to two days at submission time.
#SBATCH --job-name=fska_rebuttal
#SBATCH --output=slurm_logs/rebuttal_%A_%a.out
#SBATCH --error=slurm_logs/rebuttal_%A_%a.err
#SBATCH -p gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

# Step 2: Require an explicit work mode from the submission script.
: "${MODE:?Submit with MODE=smoke,01,02,03,04,04_1,05,06p,06a,or 06cal}"

# Step 3: Load the same Python module used to create the virtual environment.
module purge
module load python/3.11.14

# Step 4: Resolve the repository from the directory in which sbatch was run.
# run_all_slurm.sh always submits from the repository root.
REPO="${SLURM_SUBMIT_DIR}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

if [[ ! -f "${REPO}/rebuttal/README.md" ]]; then
    echo "ERROR: ${REPO} is not the NFattention repository root." >&2
    exit 2
fi

cd "${REPO}"

# Step 5: Activate the repository-local environment prepared by the launcher.
if [[ ! -f "${REPO}/.venv/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at ${REPO}/.venv." >&2
    exit 2
fi
source "${REPO}/.venv/bin/activate"
hash -r

# Step 6: Configure noninteractive execution and CPU thread usage.
# Slurm controls CUDA_VISIBLE_DEVICES; do not overwrite it here.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

# Step 7: Give every array task an isolated output directory.
DATA_ROOT="${REPO}/rebuttal/data"
RESULT_ROOT="${REPO}/rebuttal/slurm_outputs"
TASK_OUTPUT="${RESULT_ROOT}/${MODE}/job_${ARRAY_JOB_ID}/task_${TASK_ID}"
mkdir -p "${TASK_OUTPUT}"

# Step 8: Record the execution environment in the Slurm log.
echo "Host: $(hostname)"
echo "Repository: ${REPO}"
echo "Mode: ${MODE}"
echo "Task: ${TASK_ID}"
echo "Output: ${TASK_OUTPUT}"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV}"
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

# Step 9: Dispatch the requested experiment group.
case "${MODE}" in
    smoke)
        # Run all synthetic correctness checks sequentially on one small GPU.
        python rebuttal/01_sst5_diagnostics/main.py \
            --smoke --device cuda:0 --seeds 0 \
            --output-dir "${TASK_OUTPUT}/01"

        python rebuttal/02_fixed_density/main.py \
            --smoke --device cuda:0 --seeds 0 \
            --output-dir "${TASK_OUTPUT}/02"

        python rebuttal/03_kl_sensitivity/main.py \
            --smoke --device cuda:0 --seeds 0 \
            --output-dir "${TASK_OUTPUT}/03"

        python rebuttal/04_elu_gap/main.py \
            --smoke --device cuda:0 --seeds 0 \
            --output-dir "${TASK_OUTPUT}/04"

        python rebuttal/04_1_elu_gap_long_sequences/main.py \
            --smoke --device cuda:0 \
            --output-dir "${TASK_OUTPUT}/04_1"

        python rebuttal/05_efficiency/main.py \
            --smoke --device cuda:0 \
            --output-dir "${TASK_OUTPUT}/05"

        python rebuttal/06_listops/main.py \
            --smoke --device cuda:0 \
            --primary-seeds 0 --ablation-seeds 0 \
            --output-dir "${TASK_OUTPUT}/06"

        # Stage shared datasets once. Dependent array jobs use these read-only
        # files and therefore do not race while downloading or extracting.
        python -c "
from pathlib import Path
from rebuttal._shared.datasets import load_listops, load_sst5

root = Path('${DATA_ROOT}')
print('SST-5:', load_sst5(root).manifest())
print('ListOps:', load_listops(root / 'listops', download=True).manifest())
"
        ;;

    01|02|03|04)
        # Each task runs every paired arm for one SST-5 seed. This preserves
        # paired comparisons while parallelizing the five seeds.
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

    04_1)
        # Supplemental long-sequence numerical diagnostic. It does not alter
        # the SST-5 or standard ListOps quality protocols.
        export PYTHON=python
        export CUDA_DEVICE=cuda:0
        export OUTPUT_DIR="${TASK_OUTPUT}"
        export SKIP_SMOKE=1
        bash rebuttal/04_1_elu_gap_long_sequences/run.sh
        ;;

    05)
        # Keep the complete timing comparison on one GPU so device variation
        # cannot confound latency and throughput comparisons.
        export PYTHON=python
        export CUDA_DEVICE=cuda:0
        export OUTPUT_DIR="${TASK_OUTPUT}"
        # Use the log-spaced default declared in 05_efficiency/run.sh.
        export SKIP_SMOKE=1

        bash rebuttal/05_efficiency/run.sh
        ;;

    06p)
        # One primary variant/seed per GPU task (5 variants x 5 seeds).
        LISTOPS_VARIANT_INDEX=$(( TASK_ID / 5 ))
        LISTOPS_SEED=$(( TASK_ID % 5 ))
        PRIMARY_VARIANTS=(transformer performer rka fska_main fska_learned_qk)
        python rebuttal/06_listops/main.py \
            --device cuda:0 \
            --primary-seeds "${LISTOPS_SEED}" \
            --ablation-seeds "" \
            --primary-variants "${PRIMARY_VARIANTS[${LISTOPS_VARIANT_INDEX}]}" \
            --steps 5000 \
            --microbatch "${LISTOPS_MICROBATCH:-8}" \
            --gradient-accumulation "${LISTOPS_GRADIENT_ACCUMULATION:-4}" \
            --no-download \
            --data-dir "${DATA_ROOT}/listops" \
            --output-dir "${TASK_OUTPUT}"
        ;;

    06a)
        # One ablation variant/seed per GPU task (4 variants x 3 seeds).
        LISTOPS_VARIANT_INDEX=$(( TASK_ID / 3 ))
        LISTOPS_SEED=$(( TASK_ID % 3 ))
        ABLATION_VARIANTS=(fska_fixed_bivariate fska_raw_rff fska_m32 fska_m128)
        python rebuttal/06_listops/main.py \
            --device cuda:0 \
            --primary-seeds "" \
            --ablation-seeds "${LISTOPS_SEED}" \
            --ablation-variants "${ABLATION_VARIANTS[${LISTOPS_VARIANT_INDEX}]}" \
            --steps 5000 \
            --microbatch "${LISTOPS_MICROBATCH:-8}" \
            --gradient-accumulation "${LISTOPS_GRADIENT_ACCUMULATION:-4}" \
            --no-download \
            --data-dir "${DATA_ROOT}/listops" \
            --output-dir "${TASK_OUTPUT}"
        ;;

    06cal)
        # One true 2,000-token model step for memory calibration. Transformer
        # is the default because its dense attention is the memory worst case;
        # set LISTOPS_CAL_VARIANT=fska_main only when sizing an FSKA-only arm.
        python rebuttal/06_listops/main.py \
            --device cuda:0 --primary-seeds 0 --ablation-seeds "" \
            --primary-variants "${LISTOPS_CAL_VARIANT:-transformer}" --steps 1 \
            --microbatch "${LISTOPS_MICROBATCH:?Set LISTOPS_MICROBATCH}" \
            --gradient-accumulation 1 --eval-every 1 --no-download \
            --data-dir "${DATA_ROOT}/listops" --output-dir "${TASK_OUTPUT}"
        ;;

    *)
        echo "ERROR: unknown MODE=${MODE}." >&2
        exit 2
        ;;
esac

# Step 10: Reaching this line means every requested cell completed.
echo "Mode ${MODE}, task ${TASK_ID} completed successfully."
