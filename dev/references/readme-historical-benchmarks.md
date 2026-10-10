# Historical README benchmark attribution

This record preserves the benchmark section from PR #168 head
`bf9400063ee4f92e54091be50dc3a4e6dbea7c6e`. The archived text below is not a
validated benchmark of the current source.

## Provenance mismatch

The README labeled this section RTX 4090, but both cited reports identify
Tesla P100-SXM2-16GB hardware. The GLM report also uses N=100,000 and P=50,
not the table's 678K-by-42 workload, and its reference is NumPy/FISTA rather
than the displayed sklearn comparison. Neither cited report supplies the
listed 196.9x/97.9x figures or the displayed package-version environment.
These two citations therefore do not substantiate that table's attribution.
The figures are preserved as historical transcription, not relabeled as P100
measurements or used as current performance/correctness evidence.

- [Original README at the recorded source](https://github.com/TheHiddenObserver/statgpu/blob/bf9400063ee4f92e54091be50dc3a4e6dbea7c6e/README.md)
- [2026-06-23 GLM solver report](../../results/glm_solver_benchmark_2026-06-23.md)
- [2026-06-27 unsupervised report](../../results/unsupervised_bench_2026-06-27.md)

A fresh performance claim requires numerical correctness, aligned objectives,
source identity, actual hardware/software, synchronized timing, transfer scope
and the relevant workload. No benchmark or physical-GPU rerun was performed in
this documentation correction. Missing provenance must not be filled with
inferred hardware or commit information.

## Archived README text

## Benchmark Results (RTX 4090)

Full reports: `results/unsupervised_bench_2026-06-27.md`, `results/glm_solver_benchmark_2026-06-23.md`

Test environment: RTX 4090 (24GB), CuPy 14.1.0, PyTorch 2.8.0+cu128,
scikit-learn 1.8.0, statsmodels 0.14.6, lifelines 0.30.3.
These are environment-specific benchmark results, not installation requirements or
universal speed guarantees.

### Selected Benchmark Results

| Module | Dataset | n | p | Best Speedup | Precision |
|---|---|---:|---:|---:|---|
| Poisson GLM | freMTPL2 | 678K | 42 | 196.9x vs sklearn | coef_corr=1.000000 |
| Gamma GLM | synthetic | 678K | 42 | 97.9x vs sklearn | coef_corr=0.9995 |
| CoxPH | synthetic | 1.9K | 500 | 1.2x vs CPU | coef_corr=1.000 |
| adjust_pvalues (BH) | synthetic | — | 1M | 0.55x | 100% agreement |
| PenalizedPoisson (L1) | freMTPL2 | 678K | 42 | — | OK |
| PenalizedCoxPH (L2) | synthetic | 1.9K | 500 | — | C-index match |

### Precision Summary

| Module | Metric | Result |
|---|---|---|
| Poisson GLM | coefficient correlation vs sklearn | 1.000000 |
| Gamma GLM | coefficient correlation vs sklearn | 0.9995 |
| CoxPH | coefficient correlation vs lifelines | 1.000 |
| adjust_pvalues (BH) | rejection agreement vs statsmodels | 100% |
| Penalized models | self-consistency | validated across supported penalties |

