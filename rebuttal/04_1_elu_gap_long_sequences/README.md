# ELU+1 gap at long sequence lengths

This supplemental diagnostic extends experiment 04 from short held-out SST-5
examples to controlled sequence lengths of 128 through 4,096. It does **not**
claim a new downstream ListOps result and does not replace the reviewer-requested
SST-5 kernel diagnostic.

For each length it draws nested spectral samples from the frozen paper
10-component bivariate prior (64 candidate pairs; 2,048 raw-reference pairs), creates normalized
synthetic hidden states, and compares raw-RFF and ELU+1 maps against that raw
reference. It reports sampled kernel error/correlation, linear-context error,
and signed-normalizer pathologies. No dense sequence-square attention or kernel
matrix is allocated.

```bash
bash rebuttal/04_1_elu_gap_long_sequences/run.sh
```

Use `SEQ_LENGTHS=128,256,...` to override the length grid. Outputs include
`config.json`, `runs.jsonl`, `summary.csv`, `table.tex`, `length_records.json`,
and PNG/PDF figures.
