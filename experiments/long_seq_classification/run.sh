#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# 长文本分类实验 (max_seq_len = 1024)
#   数据集: hyperpartisan (RFGPA), imdb (Stanford)
#   模型:   ours = hybrid {NoPrior, NGSM, RBF}
#           baseline (文章全部) = transformer mikan performer rka gmm_rks kpca_scaled metala
#
# 用法:
#   bash run.sh            # 默认 GPU0, seed 42
#   GPU=1 SEED=0 bash run.sh
# ────────────────────────────────────────────────────────────────────
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/../../experiments/text_classification/text_seq_classification_benchmark"
DATA="$HERE/../../experiments/text_classification/data"
PY="${PY:-/home/kyzhang/miniconda3/envs/py310/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"

MODELS=(
  ours_hybrid_noprior
  ours_hybrid_ngsm
  ours_hybrid_rbf
  transformer
  mikan
  performer
  rka
  gmm_rks
  kpca_scaled
  metala
)

# 1) 准备数据（幂等）
"$PY" "$HERE/prepare_data.py"

# 2) 运行 benchmark：10 个模型 × 2 个数据集
"$PY" "$BENCH/run_benchmark.py" \
  --datasets hyperpartisan imdb \
  --models "${MODELS[@]}" \
  --seeds "$SEED" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --gpu "$GPU" \
  --data_dir "$DATA" \
  --output_dir "$HERE/results"

echo "结果在: $HERE/results/run_<timestamp>/"
