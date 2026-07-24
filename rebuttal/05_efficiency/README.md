# Efficiency comparison

Measures synchronized inference and forward/backward latency, tokens per
second, peak CUDA memory, and parameter counts for Transformer, Performer,
linear RKA, and FSKA. FSKA additionally reports sampling, feature-map,
contraction, and output-projection time.

```bash
bash rebuttal/05_efficiency/run.sh
```

Use `SEQ_LENGTHS`, `CUDA_DEVICE`, `OUTPUT_DIR`, `PYTHON`, or `SKIP_SMOKE` to
override defaults. The default sequence grid is `128,256,512,1024,2048,4096,
8192,16384,32768`; this exposes both fixed overhead and long-context scaling.
OOMs are recorded and cause a nonzero exit.

The output table includes trainable and total parameter counts. They must be
reported with latency because the FSKA and baseline adapters are not
parameter-matched in this harness.
