# KL sensitivity

Runs the predeclared coefficient grid
`0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1` on SST-5. Histories contain cross-entropy,
weighted KL, the KL/cross-entropy ratio, validation accuracy, and macro-F1.
The paper default remains `0.001`; test results are not used to select it.

```bash
bash rebuttal/03_kl_sensitivity/run.sh
```
