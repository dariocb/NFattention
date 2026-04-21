#!/usr/bin/env bash
set -euo pipefail

# 仅 kpca_scaled + metala，七数据集全跑；保存每轮 checkpoint 与复现清单（见 run_* 目录）。
#   repro_manifest.json, requirements-frozen.txt, requirements-cloud.txt, config.yaml
#   checkpoints/{dataset}__{model}__seed0.pt（7×2×1 组）
#
#   cd experiments/text_classification/text_seq_classification_benchmark
#   pip install -r requirements-cloud.txt
#   bash run_baselines_cloud.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# 避免漏装导致 No module named 'yaml'（包名 PyYAML，import 名为 yaml）
python -m pip install -r requirements-cloud.txt

python "run_benchmark.py" \
  --datasets rt_polarity sst2 sst5 trec ag_news dbpedia_14 yelp_review_full \
  --models kpca_scaled metala \
  --seeds 0 \
  --save_checkpoints \
  --output_dir "benchmark_kpca_metala_repro"
