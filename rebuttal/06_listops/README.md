# Official-protocol ListOps

Downloads the official LRA release when needed and reruns Transformer,
Performer, linear RKA, and FSKA under one PyTorch pipeline. Focused FSKA arms
cover a fixed bivariate density, raw features, learned Q/K, and the
32/64/128-pair capacity curve.

```bash
bash rebuttal/06_listops/run.sh
```

Defaults are five primary seeds, three ablation seeds, 5,000 optimizer steps,
microbatch 8, and gradient accumulation 4 (effective batch 32). Before a
final A40 run, submit `MODE=06cal` with microbatches 8, 16, and 32 using the
default Transformer calibration (the dense-attention memory worst case); use
the largest value that completes with headroom, adjusting accumulation to
preserve the effective batch. Each 50-step validation now writes a progress JSON and
an updated best-checkpoint file. Override defaults with `SEEDS`,
`ABLATION_SEEDS`, `STEPS`, `MICROBATCH`, or `GRADIENT_ACCUMULATION`.

The output contains validation-selected predictions, length-bin accuracy,
length/truncation statistics, wall-clock, throughput, peak CUDA memory,
checkpoints, tables, and capacity plots.

ListOps now defaults to FP32 (`--no-mixed-precision`) because the earlier AMP
run produced non-finite RKA/FSKA training losses. RKA and FSKA write a
fail-fast `nonfinite_traces/<variant>__seed<seed>.json` record at the first
non-finite module boundary. Set `--mixed-precision` only for a separate AMP
comparison, never as the primary stability result.
