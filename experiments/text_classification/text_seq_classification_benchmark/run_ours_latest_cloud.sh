#!/usr/bin/env bash
set -euo pipefail

# Run full benchmark for the head-conditioned shared-kernel Ours setting.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

python "run_benchmark.py" \
  --datasets rt_polarity sst2 sst5 trec ag_news dbpedia_14 yelp_review_full \
  --models ours_latest_head_kernel \
  --seeds 0 1 2 \
  --output_dir "benchmark_results_ours_latest_head_kernel"
