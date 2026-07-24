# ELU+1 theory/practice gap

Trains raw-RFF and ELU+1 FSKA variants, then evaluates the learned ELU model
with identical spectral draws under both feature maps. A 2,048-pair raw kernel
is used as the bounded diagnostic reference.

Outputs include downstream metrics, per-example distortion records,
denominator pathology rates, context errors, and PNG/PDF figures.

```bash
bash rebuttal/04_elu_gap/run.sh
```

