# Rebuttal experiments

This directory is intentionally standalone. It does not modify or rely on the
existing text, MNIST, or synthetic-LRA runners.

Install a CUDA-enabled PyTorch build for the target GPU, then install the
remaining dependencies:

```bash
python -m pip install -r rebuttal/requirements.txt
python -m pytest -q rebuttal/tests
```

Run experiments individually:

```bash
bash rebuttal/01_sst5_diagnostics/run.sh
bash rebuttal/02_fixed_density/run.sh
bash rebuttal/03_kl_sensitivity/run.sh
bash rebuttal/04_elu_gap/run.sh
bash rebuttal/05_efficiency/run.sh
bash rebuttal/06_listops/run.sh
```

Each launcher performs an inexpensive synthetic smoke run before the real grid
and exits nonzero if any required cell fails. Results are written below the
corresponding experiment's `outputs/` directory unless `OUTPUT_DIR` is set.

Common environment variables:

- `PYTHON`: Python executable.
- `CUDA_DEVICE`: PyTorch device string, such as `cuda` or `cuda:0`.
- `SEEDS`: comma-separated primary seeds.
- `DATA_DIR`: dataset cache/root.
- `OUTPUT_DIR`: exact output directory.
- `RUN_ID`: output subdirectory name.
- `SKIP_SMOKE=1`: skip the launcher smoke run.

ListOps also accepts `ABLATION_SEEDS`, `STEPS`, `MICROBATCH`, and
`GRADIENT_ACCUMULATION`. Efficiency accepts `SEQ_LENGTHS`.

