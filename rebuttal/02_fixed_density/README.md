# Fixed bivariate density

Compares a frozen, untrained bivariate spectral density against the learned
RealNVP density while holding the attention architecture and seeds fixed.
Initial and final prior/density checksums are included in `runs.jsonl`.

```bash
bash rebuttal/02_fixed_density/run.sh
```

The fixed arm must have identical initial and final checksums. Any failed seed
causes a nonzero launcher exit.

