# SST-5 diagnostics

Runs Transformer, Performer, RKA, and FSKA on the standard SST-5 splits for
five seeds. The primary checkpoint is selected by validation loss; the same
training history is also evaluated under validation-accuracy and
validation-macro-F1 selection.

Outputs include raw predictions, classwise metrics, confusion matrices,
selection-policy results, `summary.csv`, `table.tex`, and PNG/PDF plots.

```bash
bash rebuttal/01_sst5_diagnostics/run.sh
```

Set `DATA_DIR`, `OUTPUT_DIR`, `SEEDS`, `CUDA_DEVICE`, `PYTHON`, or
`SKIP_SMOKE=1` to override launcher defaults.

