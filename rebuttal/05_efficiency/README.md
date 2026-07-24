# Efficiency comparison

Measures synchronized inference and forward/backward latency, tokens per
second, peak CUDA memory, and parameter counts for Transformer, Performer,
linear RKA, and FSKA. FSKA additionally reports sampling, feature-map,
contraction, and output-projection time.

```bash
bash rebuttal/05_efficiency/run.sh
```

Use `SEQ_LENGTHS`, `CUDA_DEVICE`, `OUTPUT_DIR`, `PYTHON`, or `SKIP_SMOKE` to
override defaults. OOMs are recorded and cause a nonzero exit.

