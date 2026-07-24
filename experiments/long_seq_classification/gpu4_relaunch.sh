#!/usr/bin/env bash
# GPU4 重启：imdb 上先跑 4 个快 baseline（50 epoch，含 early-stop），
# 最后单独跑 metala（capped 10 epoch）。hyperpartisan 的结果已在旧 run 里，不重跑。
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/../../experiments/text_classification/text_seq_classification_benchmark"
DATA="$HERE/../../experiments/text_classification/data"
PY="${PY:-/home/kyzhang/miniconda3/envs/py310/bin/python}"
OUT="$HERE/results/gpu4"

export CUDA_VISIBLE_DEVICES=4

# 1) 4 个快 baseline（imdb, 默认 50 epoch）
"$PY" "$BENCH/run_benchmark.py" \
  --datasets imdb \
  --models performer rka gmm_rks kpca_scaled \
  --seeds 42 --max_seq_len 1024 --gpu 0 --no_progress \
  --data_dir "$DATA" --output_dir "$OUT"

# 2) metala 最后跑，capped 10 epoch（imdb）
"$PY" "$BENCH/run_benchmark.py" \
  --config "$HERE/config_metala_cap10.yaml" \
  --datasets imdb \
  --models metala \
  --seeds 42 --max_seq_len 1024 --gpu 0 --no_progress \
  --data_dir "$DATA" --output_dir "$OUT"

echo "GPU4 relaunch done."
