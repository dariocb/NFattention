#!/bin/bash
# Submit the PDF-protocol reconstruction experiments to Slurm.
# Usage, from any directory:
#   bash /home/dacabeza/lustre/dacabeza/NFattention/rebuttal/run_all_zi_slurm.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER="$SCRIPT_DIR/zi_gpu.sbatch"
cd "$REPO"
module purge; module load python/3.11.14
if [[ ! -f .venv/bin/activate ]]; then echo "Missing $REPO/.venv" >&2; exit 2; fi
source .venv/bin/activate; hash -r
python -c "import datasets, normflows, torch; print('Python:', __import__('sys').executable); print('PyTorch:', torch.__version__)"
mkdir -p slurm_logs rebuttal/slurm_outputs rebuttal/slurm_submissions
chmod +x "$WORKER"

# First validate GPU execution and stage the explicit default 20NG dataset.
SMOKE="$(sbatch --parsable --job-name=fska_zi_smoke --array=0-0 --export=ALL,MODE=zi_smoke "$WORKER")"
DEP="afterok:${SMOKE}"
submit_seed_grid() { sbatch --parsable --dependency="$DEP" --array=0-4 --export=ALL,MODE="$1",ZI_DATASET_ID="${ZI_DATASET_ID:-SetFit/20_newsgroups}" --job-name="$2" "$WORKER"; }
J01="$(submit_seed_grid zi01 fska_zi_01)"
J02="$(submit_seed_grid zi02 fska_zi_02)"
J03="$(submit_seed_grid zi03 fska_zi_03)"
J04="$(sbatch --parsable --dependency="$DEP" --array=0-89 --export=ALL,MODE=zi04,ZI_DATASET_ID="${ZI_DATASET_ID:-SetFit/20_newsgroups}" --job-name=fska_zi_04 "$WORKER")"
J05="$(sbatch --parsable --dependency="$DEP" --array=0-0 --export=ALL,MODE=zi05 --job-name=fska_zi_05 "$WORKER")"
J06="$(sbatch --parsable --dependency="$DEP" --array=0-24 --export=ALL,MODE=zi06,ZI_DATASET_ID="${ZI_DATASET_ID:-SetFit/20_newsgroups}" --job-name=fska_zi_06 "$WORKER")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="rebuttal/slurm_submissions/${STAMP}_zi.env"
printf 'SMOKE=%s\nJ01=%s\nJ02=%s\nJ03=%s\nJ04=%s\nJ05=%s\nJ06=%s\n' "$SMOKE" "$J01" "$J02" "$J03" "$J04" "$J05" "$J06" > "$MANIFEST"
echo "Submitted ZI suite. Manifest: $REPO/$MANIFEST"
echo "Smoke $SMOKE; 01 $J01; 02 $J02; 03 $J03; 04 $J04; 05 $J05; 06 $J06"
echo "Monitor: squeue -u \"$USER\""
