# SST-5 diagnostics

Runs Transformer, Performer, RKA, and FSKA on the standard SST-5 splits for
five seeds. The primary checkpoint is selected by validation loss; the same
training history is also evaluated under validation-accuracy and
validation-macro-F1 selection.

Outputs include raw predictions, classwise metrics, confusion matrices,
selection-policy results, `summary.csv`, `table.tex`, and PNG/PDF plots.

## Locked final protocol

The default full run is the paper-text architecture, fixed before reading test
results: maximum length 256; hidden size 128; 4 heads; 2 encoder layers; FFN
size 256; dropout 0.1; 64 spectral pairs (128 feature width); batch size 32;
Adam learning rate 1e-3; weight decay 1e-4; at most 100 epochs; and
validation-loss early stopping with patience 20.  Every variant uses the same
encoder/training budget, and every seed writes its complete history under
`histories/`.  This is a locked paper-aligned protocol, not a test-selected
claim of optimal tuning.

```bash
bash rebuttal/01_sst5_diagnostics/run.sh
```

Set `DATA_DIR`, `OUTPUT_DIR`, `SEEDS`, `CUDA_DEVICE`, `PYTHON`, or
`SKIP_SMOKE=1` to override launcher defaults.
