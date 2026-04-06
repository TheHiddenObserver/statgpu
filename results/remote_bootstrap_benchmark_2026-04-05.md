# Remote Bootstrap Benchmark (2026-04-05)

## Environment

- Host: hz-4.matpool.com:27613
- Remote workspace: /tmp/statgpu_inference_full_1775387750
- Python: /root/miniconda3/envs/myconda/bin/python (3.9.16)
- CUDA available: true
- Rscript available: false

## Benchmark Configuration

- Model path: Lasso with inference_method=bootstrap
- Data: n_samples=3000, n_features=24
- Parameters: alpha=0.08, max_iter=1500, tol=1e-4
- Bootstrap count: n_bootstrap=100
- Warmup runs: 1
- Measured repeats: 5
- Seed: 20260405

Timing includes: base fit + residual-bootstrap refits.

## Results (mean over 5 runs)

| Implementation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| statgpu bootstrap (CPU) | 89.961 | 4.179 | 82.678 | 93.909 |
| statgpu bootstrap (GPU) | 86.346 | 1.596 | 84.746 | 88.892 |
| sklearn residual bootstrap (CPU) | 160.997 | 2.036 | 157.832 | 163.554 |
| statsmodels residual bootstrap (CPU) | 3631.271 | 64.078 | 3579.813 | 3752.621 |

## Derived Ratios

- statgpu CPU vs GPU: 1.042x (GPU slightly faster)
- sklearn vs statgpu CPU: 1.790x slower
- statsmodels vs statgpu CPU: 40.365x slower

## Notes

- This benchmark is method-level comparable for residual-bootstrap loop runtime.
- statsmodels path uses OLS.fit_regularized as a Lasso-like baseline for timing comparison.
- Since Rscript is unavailable on the remote machine, no R bootstrap baseline was run.
