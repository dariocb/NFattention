# WikiText Perplexity Benchmark

This benchmark mirrors the text classification setup and evaluates next-token perplexity on WikiText.

## What it does

- Loads WikiText through Hugging Face `datasets`
- Tokenizes the corpus with a configurable tokenizer
- Packs text into fixed-length causal LM blocks
- Trains the benchmark models with a next-token objective
- Reports loss and perplexity on train, validation, and test splits

## Entry points

- `train.py` for a single run
- `run_benchmark.py` for multi-model, multi-seed sweeps

## Example

```bash
python -m experiments.language_modelling.wikitext_perplexity_benchmark.train \
  --model transformer \
  --seed 0
```

