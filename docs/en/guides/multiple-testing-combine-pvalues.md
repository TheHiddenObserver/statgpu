# Multiple-testing: P-value Adjustment & Combination (BH/BY/Holm/Bonferroni/Hochberg + Fisher/Cauchy/Stouffer)

> Language: English  
> Last updated: 2026-04-26  
> This page: Guide  
> Switch: [Chinese](../../guides/multiple-testing-combine-pvalues.md)

Language switch: [Chinese](../../guides/multiple-testing-combine-pvalues.md)

## API Summary

### P-value Adjustment (Multiple Hypothesis Testing)

Use `statgpu.adjust_pvalues` for FDR/FWER correction:

```python
reject, pvals_adj = statgpu.adjust_pvalues(
    pvalues,
    method="bh",           # "bh" | "by" | "holm" | "bonferroni" | "hochberg"
    alpha=0.05,              # FWER/FDR level
    axis=None,               # None = flatten all values
    backend="auto",         # "auto" | "numpy" | "cupy" | "torch"
)
```

Equivalent: `statgpu.multipletests(...)` has identical parameters to `statgpu.adjust_pvalues(...)`.

### Global P-value Combination

Use `statgpu.combine_pvalues` to combine many p-values into one global p-value:

```python
statistic, pvalue = statgpu.combine_pvalues(
    pvalues,
    method="fisher",        # "fisher" | "cauchy" | "stouffer"
    weights=None,            # used by cauchy/stouffer only
    axis=None,               # None = flatten all values
    backend="auto",         # "auto" | "numpy" | "cupy" | "torch"
)
```

## Method Semantics

### P-value Adjustment Methods

1. **Bonferroni**: `p_i * m`, FWER control, most conservative.
2. **Holm**: Step-down FWER, more powerful than Bonferroni.
3. **BH** (Benjamini-Hochberg): Step-up FDR, alias `fdr_bh`.
4. **BY** (Benjamini-Yekutieli): FDR control, robust to arbitrary dependence.
5. **Hochberg**: Step-up FWER, alias `fdr_hochberg`/`step_up`/`stepup`.

All adjusted p-values are clipped to [0, 1].

### Global P-value Combination Methods

1. **Fisher**
- Statistic: `-2 * sum(log(p_i))`
- Reference distribution: chi-square with `2m` degrees of freedom.
- Weight input is not used.

2. **Cauchy** (ACAT alias)
- Statistic: weighted tangent transform over p-values.
- P-value: Cauchy tail transform.
- Requires non-negative weights when `weights` is provided.
- `method="acat"` is an alias of `method="cauchy"`.

3. **Stouffer** (Weighted Z-test)
- Statistic: `sum(w_i * Z_i) / sqrt(sum(w_i^2))`, where `Z_i = norm.ppf(1 - p_i)`.
- P-value: `norm.sf(statistic)`.
- Supports weights, consistent with the cauchy weight interface.
- Aliases: `ztest`/`weighted_z`.

## Shape Rules

- `axis=None`: flatten all values, return scalar statistic and scalar p-value.
- `axis=k`: combine along axis `k`, return arrays with that axis reduced.
- `weights` length must match the combine axis length.

## Examples

### P-value Adjustment

#### 1) Vector adjustment

```python
import numpy as np
from statgpu import adjust_pvalues

p = np.array([0.003, 0.02, 0.50, 0.10, 0.001])
reject, pvals_adj = adjust_pvalues(p, method='bh', alpha=0.05)
```

#### 2) Hochberg step-up FWER

```python
import numpy as np
from statgpu import adjust_pvalues

p = np.array([0.003, 0.02, 0.50, 0.10, 0.001])
reject, pvals_adj = adjust_pvalues(p, method='hochberg', alpha=0.05)
# More powerful than Holm, controls FWER
```

#### 3) Row-wise adjustment (`axis=1`)

```python
import numpy as np
from statgpu import adjust_pvalues

p_matrix = np.random.default_rng(0).uniform(0, 1, size=(100, 16))
reject_row, adj_row = adjust_pvalues(p_matrix, method='bh', axis=1)
```

### Global P-value Combination

#### 1) One global p-value from a vector

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.01, 0.07, 0.03, 0.40])
stat, p_global = combine_pvalues(p, method="fisher", backend="numpy")
```

#### 2) Row-wise combination (`axis=1`)

```python
import numpy as np
from statgpu import combine_pvalues

p_matrix = np.random.default_rng(0).uniform(1e-8, 1 - 1e-8, size=(100, 16))
stat_row, p_row = combine_pvalues(p_matrix, method="fisher", axis=1)
```

#### 3) Weighted Cauchy / ACAT

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_c, p_c = combine_pvalues(p, method="cauchy", weights=w)
stat_a, p_a = combine_pvalues(p, method="acat", weights=w)  # alias
```

#### 4) Weighted Stouffer Z-test

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_s, p_s = combine_pvalues(p, method='stouffer', weights=w)
stat_z, p_z = combine_pvalues(p, method='ztest', weights=w)  # alias
```

#### 5) GPU path with CuPy

```python
import cupy as cp
from statgpu import combine_pvalues

p_cp = cp.random.uniform(1e-8, 1 - 1e-8, size=(4000, 64), dtype=cp.float64)
stat_cp, p_cp_out = combine_pvalues(p_cp, method="fisher", axis=1, backend="cupy")
```

#### 6) GPU path with Torch

```python
import torch
from statgpu import combine_pvalues, adjust_pvalues

p_torch = torch.rand(5000, 64, dtype=torch.float64, device='cuda')
reject, adj = adjust_pvalues(p_torch, method='bh', backend='torch')
stat, p_global = combine_pvalues(p_torch, method='stouffer', axis=1, backend='torch')
```

## Large-Scale Performance (p=50k-1M, Tesla P100)

Benchmark script: `dev/benchmarks/_bench_inference_timing_large.py`

### adjust_pvalues BH (sort + cummin, O(n log n))

| p | NumPy | CuPy | Torch | CuPy vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 33.3 ms | 1.75 ms | 114 ms | 19.0x |
| 100,000 | 69.7 ms | 3.49 ms | 232 ms | 20.0x |
| 500,000 | 374 ms | 28.5 ms | 1.01 s | 13.1x |
| 1,000,000 | 799 ms | 76.7 ms | 1.96 s | **10.4x** |

CuPy excels at sort-heavy operations (adjust family).

### combine_pvalues Stouffer (norm.ppf + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 2.01 ms | 1.47 ms | 0.51 ms | 3.9x |
| 100,000 | 3.88 ms | 2.93 ms | 0.65 ms | 6.0x |
| 500,000 | 17.8 ms | 17.6 ms | 1.67 ms | 10.7x |
| 1,000,000 | 36.8 ms | 34.0 ms | 3.09 ms | **11.9x** |

Torch excels at compute-bound operations (norm.ppf).

### combine_pvalues Fisher (sum+log, O(n) bandwidth-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 5.43 ms | 6.93 ms | 2.04 ms | **2.7x** |

Bandwidth-bound operations (sum+log) see modest GPU speedup (~2-3x).

### combine_pvalues Cauchy (tan + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 49.5 ms | 42.6 ms | 11.2 ms | **4.4x** |

### GPU Speedup Summary

- **p < 10,000**: GPU kernel launch overhead (>300 us) dominates; CPU may be faster
- **p > 50,000**: GPU advantage becomes clear
- **Sort-heavy** (adjust BH): CuPy best at 10x+
- **Compute-bound** (Stouffer norm.ppf): Torch best at 12x
- **Bandwidth-bound** (Fisher sum+log): modest GPU speedup at 2-3x
- **Mixed** (Cauchy tan+sum): moderate GPU speedup at 4-5x

## Benchmark Interpretation Notes (old, p=4000x64)

Remote supplement artifact:
- JSON: `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- Summary: `results/remote_fisher_cauchy_benchmark_2026-04-05.md`

Workload used in that artifact:
- `n_groups=4000`, `group_size=64`, `axis=1`, `warmup=1`, `repeats=5`

Key runtime means:

| Method | statgpu NumPy (ms) | statgpu CuPy (ms) | SciPy (ms) |
|---|---:|---:|---:|
| Fisher | 3.979 | 0.816 | 350.721 |
| Cauchy | 4.052 | 0.874 | N/A |
| ACAT alias | 3.971 | N/A | N/A |

How to read these numbers:
- Fisher: SciPy is much slower than statgpu NumPy on this workload (`~88.15x`).
- NumPy to CuPy speedup in statgpu is about `~4.88x` (Fisher) and `~4.63x` (Cauchy).
- Cauchy and ACAT alias are numerically identical in this benchmark (`max abs p-value diff = 0.0`).
- NumPy/CuPy output differences are at floating-point noise level (about `1e-15` in p-values).

## Reproducibility

Primary local benchmark script:
- `dev/benchmarks/benchmark_inference_backends.py`

Example run:

```bash
python dev/benchmarks/benchmark_inference_backends.py --output-tag local_check
```

Generated output:
- `results/inference_backend_benchmark_<date>_local_check.json`