# Fixed bivariate density

Compares four paired arms while holding the attention architecture and seeds
fixed: a conventional fixed single joint Gaussian, a frozen fixed
10-component bivariate spectral-mixture prior, a learned two-component
equal-weight Gaussian mixture without a flow, and the learned RealNVP flow.
The frozen-mixture, learned-two-component, and flow arms share the same frozen
10-component KL target; the single-Gaussian arm is an additional conventional
stationary baseline. Initial and final
prior/density checksums are included in `runs.jsonl`.

```bash
bash rebuttal/02_fixed_density/run.sh
```

The fixed arm must have identical initial and final checksums. Any failed seed
causes a nonzero launcher exit.
