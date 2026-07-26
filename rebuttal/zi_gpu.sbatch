#!/bin/bash
# Slurm worker for the PDF-protocol reconstruction suite.
# Submit through rebuttal/run_all_zi_slurm.sh, which sets MODE and the array.
#SBATCH --job-name=fska_zi
#SBATCH --output=slurm_logs/zi_%A_%a.out
#SBATCH --error=slurm_logs/zi_%A_%a.err
#SBATCH -p gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${MODE:?MODE must be zi_smoke,zi01,zi02,zi03,zi04,zi05,or zi06}"
module purge
module load python/3.11.14
REPO="${SLURM_SUBMIT_DIR}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
cd "$REPO"
source "$REPO/.venv/bin/activate"
hash -r
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
DATA_ROOT="${REPO}/rebuttal/data"
OUT="${REPO}/rebuttal/slurm_outputs/${MODE}/job_${ARRAY_JOB_ID}/task_${TASK_ID}"
mkdir -p "$OUT"
echo "Host: $(hostname)"; echo "Mode: $MODE; task: $TASK_ID; output: $OUT"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi

common=(--dataset-id "${ZI_DATASET_ID:-SetFit/20_newsgroups}" --data-dir "$DATA_ROOT" --device cuda:0 --seeds "$TASK_ID")
case "$MODE" in
  zi_smoke)
    python rebuttal/01_zi_long_documents/main.py --smoke --device cuda:0 --output-dir "$OUT/01"
    python rebuttal/02_zi_fixed_bivariate/main.py --smoke --device cuda:0 --output-dir "$OUT/02"
    python -c "from rebuttal._shared.datasets import load_hf_text_classification; print(load_hf_text_classification('SetFit/20_newsgroups', __import__('pathlib').Path('$DATA_ROOT')).manifest())"
    ;;
  zi01) python rebuttal/01_zi_long_documents/main.py --output-dir "$OUT" "${common[@]}" ;;
  zi02) python rebuttal/02_zi_fixed_bivariate/main.py --output-dir "$OUT" "${common[@]}" ;;
  zi03)
    # Raw RFF instability is the outcome being measured.  Leave failures.json
    # intact but make the array dependency succeed only when every failure is
    # the designated raw-RFF arm.
    set +e
    python rebuttal/03_zi_elu_gap/main.py --output-dir "$OUT" "${common[@]}"
    status=$?
    set -e
    python - "$OUT/failures.json" "$status" <<'PY'
import json, sys
path, status = sys.argv[1], int(sys.argv[2])
failures = json.load(open(path, encoding='utf-8'))
if status and failures and all(x.get('variant') == 'signed_raw_rff' for x in failures):
    print('Accepted expected signed-RFF failures:', len(failures))
    raise SystemExit(0)
raise SystemExit(status)
PY
    ;;
  zi04)
    # 18 variants x 5 seeds: one model/seed per GPU task.
    variant=$(( TASK_ID / 5 )); seed=$(( TASK_ID % 5 ))
    python rebuttal/04_zi_capacity_qk/main.py --output-dir "$OUT" "${common[@]}" --seeds "$seed" --variant-index "$variant"
    ;;
  zi05) python rebuttal/05_zi_efficiency/main.py --output-dir "$OUT" --device cuda:0 ;;
  zi06)
    # 5 lambda values x 5 seeds: one coefficient/seed per GPU task.
    variant=$(( TASK_ID / 5 )); seed=$(( TASK_ID % 5 ))
    python rebuttal/06_zi_kl_20ng/main.py --output-dir "$OUT" "${common[@]}" --seeds "$seed" --variant-index "$variant"
    ;;
esac
echo "${MODE} task ${TASK_ID} completed."
