# ANOVA

> Language: English  
> Last updated: 2026-05-28  
> This page: Model documentation  
> Switch: [Chinese](../../models/anova.md)

Language switch: [Chinese](../../models/anova.md)

## Overview

`f_oneway` performs one-way Analysis of Variance (ANOVA), testing whether group means are equal. It is a GPU-accelerated drop-in replacement for `scipy.stats.f_oneway`, supporting numpy, cupy, and torch backends.

## Path

`statgpu.anova.f_oneway`, `statgpu.anova.AnovaResult`

## Objective Function

Grand mean:
\[
\bar{y} = \frac{\sum_i n_i \bar{y}_i}{\sum_i n_i}
\]

Between-group sum of squares:
\[
SSB = \sum_{i=1}^k n_i (\bar{y}_i - \bar{y})^2
\]

Within-group sum of squares:
\[
SSW = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_i)^2
\]

F-statistic:
\[
F = \frac{SSB / (k-1)}{SSW / (N-k)}
\]

where $k$ is the number of groups, $n_i$ is the size of group $i$, and $N = \sum_i n_i$ is the total number of observations.

## Estimating Equation

Direct computation, no iterative solver needed. The F-statistic is computed in a single pass over the data using backend-native reduction operations.

## Covariance/Inference

P-value is obtained from the F-distribution survival function $1 - F_{k-1,\,N-k}(F)$. Effect size is reported as:
\[
\eta^2 = \frac{SSB}{SSB + SSW}
\]

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | (required) | Two or more 1-D arrays, one per group |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |

## CPU+GPU Examples

```python
from statgpu.anova import f_oneway
import numpy as np

# CPU
g1 = np.random.randn(100)
g2 = np.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="numpy")
print(f"F={result.statistic:.4f}, p={result.pvalue:.4e}, eta2={result.eta_squared:.4f}")

# GPU (cupy)
import cupy as cp
g1_gpu = cp.asarray(g1)
g2_gpu = cp.asarray(g2)
result_gpu = f_oneway(g1_gpu, g2_gpu, backend="cupy")

# GPU (torch)
import torch
g1_t = torch.from_numpy(g1).cuda()
g2_t = torch.from_numpy(g2).cuda()
result_torch = f_oneway(g1_t, g2_t, backend="torch")
```

## strict/approx difference

No strict/approx modes. Single computation path with backend selection.

## Outputs

`AnovaResult` dataclass with fields:

| Field | Type | Description |
|---|---|---|
| `statistic` | float | F-statistic value |
| `pvalue` | float | P-value from F-distribution |
| `df_between` | int | Between-group degrees of freedom ($k - 1$) |
| `df_within` | int | Within-group degrees of freedom ($N - k$) |
| `eta_squared` | float | Effect size $\eta^2$ |

## FAQ

- **How many groups are supported?** Two or more.
- **What if all observations are identical?** Returns `NaN` for `statistic`, `pvalue`, and `eta_squared`.
- **What if groups are perfectly separated?** Returns `inf` for `statistic`, `0.0` for `pvalue`, `1.0` for `eta_squared`.
- **Is this a drop-in replacement for scipy?** Yes. The function signature and output fields are compatible with `scipy.stats.f_oneway`, with the addition of `eta_squared`, `df_between`, and `df_within`.

## External Validation

Validated against `scipy.stats.f_oneway` with relative error < 1e-15 across a wide range of group sizes and effect magnitudes.

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
