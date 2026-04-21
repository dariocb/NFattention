# Hugging Face Transformers Benchmark

Standalone benchmark runner for pretrained Hugging Face Transformer models on the NFattention text classification datasets.

## What it runs

- Datasets (default): `trec`, `ag_news`, `dbpedia_14`, `yelp_review_full`
- Model presets (default):
  - `distilbert` -> `distilbert-base-uncased`
  - `bert` -> `bert-base-uncased`
  - `roberta` -> `roberta-base`

You can also pass any Hugging Face model id directly via `--models`.

## Setup

From repo root:

```bash
source .venv/bin/activate
pip install -r experiments/hf_transformers_benchmark/requirements.txt
```

## Quick GPU smoke test

```bash
cd experiments/hf_transformers_benchmark
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --quick --gpu 0
```

## Full benchmark example

```bash
cd experiments/hf_transformers_benchmark
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
  --datasets trec ag_news dbpedia_14 yelp_review_full \
  --models distilbert bert roberta \
  --seeds 0 1 2 \
  --gpu 0 \
  --epochs 5 \
  --train_batch_size 16 \
  --eval_batch_size 32
```

Outputs are written under `benchmark_results_hf/run_<timestamp>/` in this folder.
