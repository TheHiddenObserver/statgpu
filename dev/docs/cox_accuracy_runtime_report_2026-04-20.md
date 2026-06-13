# CoxPH Precision and Runtime Report (2026-04-20)

## Scope

- Model: `CoxPH`
- Tie methods: `efron`, `breslow`
- Data scales:
  - `n=2000, p=20`
  - `n=5000, p=20`
  - `n=10000, p=50`
- Compared implementations:
  - `statgpu` (`cpu`, `cuda`, `torch`)
  - `statsmodels`
  - `lifelines`
  - `scikit-survival`
  - `R survival::coxph`
  - `PySurvival` (best effort)
- Reference for precision: `statsmodels` coefficients / log-likelihood

Source results:
- `dev/docs/cox_scaling_compare.json`
- `dev/docs/cox_efron_full_compare.json`

---

## Executive Summary

- `statgpu` now matches `statsmodels` at high precision for:
  - **Efron**: CPU/CUDA/Torch (coef diff typically `1e-16 ~ 1e-14`)
  - **Breslow**: CPU/CUDA/Torch (coef diff typically `1e-15 ~ 1e-14`)
- The previous `Breslow + CPU` mismatch is fixed.
- `Efron + CUDA` was heavily optimized by adding a no-tie fast path (`Efron == Breslow` when all event groups are singleton), yielding large runtime reduction at larger scales.
- `PySurvival` is still unavailable in current remote env (`No module named 'pysurvival'`).

---

## Runtime Comparison (fit_sec)

### Efron

| n, p | statgpu_cpu | statgpu_cuda | statgpu_torch | statsmodels | lifelines | scikit-survival | R survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2000, 20 | 0.6003 | 1.3594 | 2.4766 | 0.7617 | 0.2094 | 0.2707 | 0.787 |
| 5000, 20 | 1.2859 | 0.0266 | 0.1172 | 1.5396 | 0.4363 | 0.5223 | 0.818 |
| 10000, 50 | 5.4548 | 0.1283 | 0.3050 | 4.5882 | 1.3896 | 1.5167 | 1.304 |

### Breslow

| n, p | statgpu_cpu | statgpu_cuda | statgpu_torch | statsmodels | lifelines | scikit-survival | R survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2000, 20 | 0.0612 | 0.0159 | 0.0173 | 0.3724 | 0.2088 | 0.2562 | 0.808 |
| 5000, 20 | 0.3023 | 0.0174 | 0.0302 | 0.7945 | 0.4445 | 0.5121 | 0.817 |
| 10000, 50 | 3.3616 | 0.1121 | 0.1273 | 2.6768 | 1.3628 | 1.4385 | 1.301 |

Notes:
- Cross-framework runtime is environment-dependent (library versions, BLAS, GPU clocks, warmup, etc.).
- `R survival` times include only `coxph` fit in the invoked script.

---

## Precision vs statsmodels

### Efron (statgpu)

- `statgpu_cpu`: coef max abs diff `~5.6e-16` to `1.15e-14`
- `statgpu_cuda`: coef max abs diff `~3.3e-16` to `2.84e-14`
- `statgpu_torch`: coef max abs diff `~4.4e-16` to `2.84e-14`
- Log-likelihood differences are near floating-point noise (`~1e-11` to `1e-10`)

### Breslow (statgpu)

- `statgpu_cpu`: coef max abs diff `~1.78e-15` to `5.55e-15`
- `statgpu_cuda`: coef max abs diff `~3.33e-16` to `2.82e-14`
- `statgpu_torch`: coef max abs diff `~4.44e-16` to `2.82e-14`
- Log-likelihood differences are near floating-point noise (`~1e-12` to `1e-11`)

### Other frameworks

- `lifelines`: very close, typically `~1e-8` to `1e-6` coef gap
- `scikit-survival`: very close, typically `~1e-9` to `1e-14` coef gap
- `R survival`: coef output in current pipeline is rounded in JSON, observed gap `~5e-5` (not algorithmic mismatch)

---

## What Changed in This Iteration

- Fixed `Efron + Cython` non-convergence and sign-consistency issues.
- Fixed `Breslow + CPU` Newton direction issue via adaptive direction choice in line search.
- Added Efron singleton fast path for GPU (`Efron == Breslow` when no ties), significantly reducing CUDA Efron runtime on continuous-time synthetic data.

---

## Current Limitations

- `PySurvival` not installed in remote env; no direct runtime/precision numbers yet.
- Efron CUDA kernel in heavy-tie scenarios can still be optimized further (current big gain is from singleton fast path).

---

## Recommended Next Steps

- Add a dedicated **heavy-tie stress benchmark** (discretized times) to evaluate true Efron-kernel throughput.
- Add regression tests for:
  - CPU Breslow convergence and coefficient parity vs statsmodels
  - Efron Cython/CUDA parity checks
- If PySurvival comparison is required, prepare a compatible env (toolchain and package versions) and rerun same benchmark script.

