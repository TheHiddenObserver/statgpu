# Multiple-testing: Global P-value Combination (Fisher/Cauchy/ACAT)

> Language: English  
> Last updated: 2026-04-06  
> This page: Guide  
> Switch: [中文](../../guides/multiple-testing-combine-pvalues.md)

Language switch: [中文](../../guides/multiple-testing-combine-pvalues.md)

## API Summary

Use `statgpu.combine_pvalues` to combine many p-values into one global p-value:

```python
statistic, pvalue = statgpu.combine_pvalues(
    pvalues,
    method="fisher",      # "fisher" | "cauchy" | "acat"
    weights=None,          # used by cauchy/acat only
    axis=None,             # None = flatten all values
    backend="auto",       # "auto" | "numpy" | "cupy"
)
```

Related utilities:
- `statgpu.adjust_pvalues(...)` / `statgpu.multipletests(...)`: control FDR/FWER across many hypotheses.
- `statgpu.combine_pvalues(...)`: produce one global signal from multiple p-values.

## Method Semantics

1. Fisher
- Statistic: `-2 * sum(log(p_i))`
- Reference distribution: chi-square with `2m` degrees of freedom.
- Weight input is not used.

2. Cauchy (ACAT alias)
- Statistic: weighted tangent transform over p-values.
- P-value: Cauchy tail transform.
- Requires non-negative weights when `weights` is provided.
- `method="acat"` is an alias of `method="cauchy"`.

## Shape Rules

- `axis=None`: flatten all values, return scalar statistic and scalar p-value.
- `axis=k`: combine along axis `k`, return arrays with that axis reduced.
- `weights` length must match the combine axis length.

## Examples

### 1) One global p-value from a vector

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.01, 0.07, 0.03, 0.40])
stat, p_global = combine_pvalues(p, method="fisher", backend="numpy")
```

### 2) Row-wise combination (`axis=1`)

```python
import numpy as np
from statgpu import combine_pvalues

p_matrix = np.random.default_rng(0).uniform(1e-8, 1 - 1e-8, size=(100, 16))
stat_row, p_row = combine_pvalues(p_matrix, method="fisher", axis=1)
```

### 3) Weighted Cauchy / ACAT

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_c, p_c = combine_pvalues(p, method="cauchy", weights=w)
stat_a, p_a = combine_pvalues(p, method="acat", weights=w)  # alias
```

### 4) GPU path with CuPy

```python
import cupy as cp
from statgpu import combine_pvalues

p_cp = cp.random.uniform(1e-8, 1 - 1e-8, size=(4000, 64), dtype=cp.float64)
stat_cp, p_cp_out = combine_pvalues(p_cp, method="fisher", axis=1, backend="cupy")
```

## Benchmark Interpretation Notes

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