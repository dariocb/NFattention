# Official-protocol ListOps

Downloads the official LRA release when needed and reruns Transformer,
Performer, linear RKA, and FSKA under one PyTorch pipeline. Focused FSKA arms
cover a fixed bivariate density, raw features, learned Q/K, and the
32/64/128-pair capacity curve.

```bash
bash rebuttal/06_listops/run.sh
```

Defaults are five primary seeds, three ablation seeds, 5,000 optimizer steps,
microbatch 4, and gradient accumulation 8. Override them with `SEEDS`,
`ABLATION_SEEDS`, `STEPS`, `MICROBATCH`, or `GRADIENT_ACCUMULATION`.

The output contains validation-selected predictions, length-bin accuracy,
length/truncation statistics, wall-clock, throughput, peak CUDA memory,
checkpoints, tables, and capacity plots.

