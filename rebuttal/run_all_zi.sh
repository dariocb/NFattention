#!/usr/bin/env bash
# Launch the colleague-PDF reconstruction suite without touching the original
# rebuttal experiments.  Every module writes to its own output directory.
#
# Required environment (normally only PYTHON/CUDA_DEVICE need changing):
#   PYTHON=/path/to/python CUDA_DEVICE=cuda:0 SEEDS=0,1,2,3,4 \
#     bash rebuttal/run_all_zi.sh
#
# The PDF does not disclose dataset repository IDs/splits for BBC, SCOTUS, or
# arXiv.  The suite therefore runs its documented 20NG default, then optionally
# runs Table 1 on each whitespace-separated explicit ID in ZI_LONG_DATASETS.
# Example:
#   ZI_LONG_DATASETS='org/bbc org/scotus org/arxiv' bash rebuttal/run_all_zi.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-cuda}"
SEEDS="${SEEDS:-0,1,2,3,4}"
DATA_DIR="${DATA_DIR:-rebuttal/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-rebuttal/zi_outputs}"
DATASET_ID="${ZI_DATASET_ID:-SetFit/20_newsgroups}"
COMMON=(--dataset-id "$DATASET_ID" --data-dir "$DATA_DIR" --device "$DEVICE" --seeds "$SEEDS")

run_module() {
  local module="$1"
  local output="$2"
  shift 2
  echo "==> ${module} -> ${output}"
  "$PYTHON" "rebuttal/${module}/main.py" --output-dir "$output" "${COMMON[@]}" "$@"
}

# Table 1 reconstruction on the default explicit dataset (20NG by default).
run_module 01_zi_long_documents "$OUTPUT_ROOT/01_20ng"

# Optional Table-1 datasets.  Pass each exact ID and, if necessary, replace
# split/column arguments in an individual launcher invocation afterwards.
for dataset in ${ZI_LONG_DATASETS:-}; do
  safe_name="${dataset//\//_}"
  echo "==> 01_zi_long_documents for explicit dataset ${dataset}"
  "$PYTHON" rebuttal/01_zi_long_documents/main.py --output-dir "$OUTPUT_ROOT/01_${safe_name}" \
    --dataset-id "$dataset" --data-dir "$DATA_DIR" --device "$DEVICE" --seeds "$SEEDS"
done

# Table 2, Table 3, Table 4, and Table 7 reconstructions on the same 20NG
# provenance used above.  This ensures their comparisons are internally valid.
run_module 02_zi_fixed_bivariate "$OUTPUT_ROOT/02_fixed_bivariate"

# Signed raw-RFF is expected to become non-finite under the current strict
# implementation.  Preserve the failure artefact but continue the suite.
if ! run_module 03_zi_elu_gap "$OUTPUT_ROOT/03_elu_gap"; then
  echo "03_zi_elu_gap returned non-zero; inspect failures.json. Continuing because raw-RFF divergence is an experimental outcome."
fi

run_module 04_zi_capacity_qk "$OUTPUT_ROOT/04_capacity_qk"

# Original PDF efficiency grid: only GPU attention timing, no dataset loading.
echo "==> 05_zi_efficiency -> $OUTPUT_ROOT/05_efficiency"
"$PYTHON" rebuttal/05_zi_efficiency/main.py --output-dir "$OUTPUT_ROOT/05_efficiency" --device "$DEVICE"

run_module 06_zi_kl_20ng "$OUTPUT_ROOT/06_kl_20ng"

echo "ZI reconstruction suite submitted/completed. Outputs: $OUTPUT_ROOT"
