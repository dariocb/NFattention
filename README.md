# NFattention

Research code for kernel- and flow-based attention variants, text-sequence classification benchmarks, long-sequence memory profiling, and MNIST ablations.

## Repository layout

| Path | Purpose |
|------|---------|
| `models/` | Shared building blocks (`baseTransformer`, `MHA`, `frska`, etc.) used by MNIST experiments. |
| `experiments/text_classification/text_seq_classification_benchmark/` | Main text classification benchmark (`train.py`, `run_benchmark.py`, models, configs). |
| `experiments/lra_benchmark/` | Long-sequence memory / throughput benchmark and figure generation. |
| `experiments/mnist_ablation/` | MNIST classification and focused ablation scripts. |
| `baseline/` | Third-party reference code (e.g. Performer, RFA); not required to run the experiments above. |

**Bundled dataset directory removed.** The large tree `experiments/text_classification/data/` is **not** shipped in this repository (it was removed to keep the clone small). You must recreate it locally; see [Data setup](#data-setup).

## Requirements

Install dependencies from the repository root:

```bash
cd /path/to/NFattention
pip install -r requirements.txt
```

If `torch` / `torchvision` fail to install from this file alone, install a CUDA-enabled build from [pytorch.org](https://pytorch.org/) first, then run `pip install -r requirements.txt` again for the rest.

Optional: the text benchmark folder also ships a minimal cloud list at `experiments/text_classification/text_seq_classification_benchmark/requirements-cloud.txt` (core numeric stack only).

## Data setup

Recreate this directory (default expected by the text benchmark):

```bash
mkdir -p experiments/text_classification/data
```

### Local text datasets (RT Polarity, SST-2, SST-5)

The loader (`data/dataset_loader.py`) expects:

1. **RT Polarity** — under  
   `experiments/text_classification/data/rt-polaritydata/rt-polaritydata/`  
   with files `rt-polarity.pos` and `rt-polarity.neg`.  
   Obtain the original “sentence polarity dataset v2.0” distribution and unpack so the paths above match.

2. **Stanford Sentiment Treebank (SST-2 / SST-5)** — under  
   `experiments/text_classification/data/stanfordSentimentTreebank/`  
   with the usual release files (`datasetSentences.txt`, `datasetSplit.txt`, `dictionary.txt`, `sentiment_labels.txt`, etc.).  
   Download from the official SST release and place the folder as shown.

### Hugging Face–style datasets (TREC, AG News, DBpedia-14, Yelp Full)

The code **first** looks for offline JSON exports:

`experiments/text_classification/data/hf_datasets/<name>/data.json`

If missing, it downloads from the Hub at runtime (requires network and the `datasets` package).

To pre-download and snapshot JSON locally (recommended for offline or server runs):

```bash
cd experiments/text_classification/text_seq_classification_benchmark
python download_datasets.py
```

That writes under `../data/hf_datasets/` (i.e. `experiments/text_classification/data/hf_datasets/`).

### MNIST

`experiments/mnist_ablation/mnist_classification_ablation.py` uses `torchvision.datasets.MNIST` with `root='./data'`. MNIST is downloaded automatically into `./data` relative to your **current working directory** when you run the script (typically the repo root).

---

## Experiments and how to run them

Unless noted, commands assume the **repository root** as the working directory and use relative paths.

### 1. Text sequence classification benchmark

**Location:** `experiments/text_classification/text_seq_classification_benchmark/`

**Datasets:** `rt_polarity`, `sst2`, `sst5`, `trec`, `ag_news`, `dbpedia_14`, `yelp_review_full`.

**Models (examples):** `transformer`, `mikan`, `performer`, `rka`, `gmm_rks`, `kpca_scaled`, `metala`, `ours_latest`, `ours_fixed_qk`, `ours_trainable_qk`.

Single run (default `--data_dir` is `../data` → `experiments/text_classification/data`):

```bash
cd experiments/text_classification/text_seq_classification_benchmark

python train.py --dataset sst2 --model ours_latest --seed 0 --gpu 0

# RT Polarity uses 10-fold CV when requested:
python train.py --dataset rt_polarity --model transformer --seed 0 --use_cv --gpu 0
```

Full benchmark grid (override models; default in `run_benchmark.py` is only `ours_latest`):

```bash
cd experiments/text_classification/text_seq_classification_benchmark

python run_benchmark.py \
  --datasets rt_polarity sst2 sst5 trec ag_news dbpedia_14 yelp_review_full \
  --models transformer mikan performer rka gmm_rks ours_latest \
  --seeds 0 1 2 3 4 \
  --data_dir ../data \
  --gpu 0
```

Quick smoke test (small subset, single seed):

```bash
python run_benchmark.py --quick --gpu 0
```

Optional: reproducible cloud run for `kpca_scaled` + `metala` with checkpoints (uses `requirements-cloud.txt` inside that folder):

```bash
cd experiments/text_classification/text_seq_classification_benchmark
bash run_baselines_cloud.sh
```

Outputs land under `benchmark_results/run_<timestamp>/` (CSV, LaTeX table, logs, `config.yaml`, etc.).

### 2. LRA-style memory / throughput benchmark

**Location:** `experiments/lra_benchmark/`

Measures peak memory and throughput across sequence lengths; generates figures from the saved JSON.

```bash
cd experiments/lra_benchmark

# Small CPU sanity check
python run_full_benchmark.py --device cpu --quick_test

# GPU full run (adjust batch size if OOM)
python run_full_benchmark.py --device cuda --batch_size 4
```

Regenerate figures from an existing result file:

```bash
python visualize.py --results_file results/memory_benchmark_<timestamp>.json --output_dir figures
```

### 3. MNIST ablations

**Location:** `experiments/mnist_ablation/`

Imports `models` from the **repository root**, so set `PYTHONPATH` (or always run from root as below).

**Full comparison suite** (`mnist_classification_ablation.py`):

```bash
cd /path/to/NFattention
PYTHONPATH=. python experiments/mnist_ablation/mnist_classification_ablation.py \
  --use_full_data --N_EPOCHS 50 --output_dir mnist_results
```

**Focused ablation list** (`mnist_focused_ablation.py`):

```bash
PYTHONPATH=. python experiments/mnist_ablation/mnist_focused_ablation.py \
  --seed 42 --use_full_data --epochs 20 --output_dir mnist_focused_results
```

Run a single named experiment from the focused list:

```bash
PYTHONPATH=. python experiments/mnist_ablation/mnist_focused_ablation.py \
  --only_experiment "Transformer" --seed 42
```

---

## Note on removed `data/`

The directory **`experiments/text_classification/data/`** previously held RT Polarity, SST, and exported Hugging Face JSON (~hundreds of MB). It has been **removed from this tree** to reduce size. Restore it by following [Data setup](#data-setup); for HF datasets you can either run `download_datasets.py` or let `train.py` / `run_benchmark.py` fetch from the Hub on first use.

For questions about file formats, see `experiments/text_classification/text_seq_classification_benchmark/data/dataset_loader.py`.
