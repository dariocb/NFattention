# Fixed versus learned density

This experiment separates density learning from density-family and parameter
count effects. It compares a fixed single Gaussian, fixed 10-component
bivariate prior, learned two-component GMM, frozen RealNVP, and learned
RealNVP while holding the encoder, feature map, Q/K mode, seeds, and training
budget fixed.

`frozen_flow` has exactly the same RealNVP architecture and initialization as
`learned_flow`, but all flow parameters are frozen and its KL weight is zero.
The paired frozen-flow versus learned-flow comparison is therefore the strict
parameter-matched test of adapting the flow density itself.

```bash
bash rebuttal/02_fixed_density/run.sh
```

Initial/final prior and density checksums are included in `runs.jsonl`. The
fixed arms and frozen-flow arm must remain unchanged; any failed seed produces
a nonzero launcher exit.
