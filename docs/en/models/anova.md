# ANOVA

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/models/anova.md)

## Overview

The ANOVA module provides:

- `f_oneway`
- `f_twoway`
- `f_welch`
- `tukey_hsd`
- `bonferroni`
- `cohens_f`
- `partial_eta_squared`

Group reductions support NumPy, CuPy, and Torch backends. Distribution functions that
are unavailable on a selected GPU backend may use scalar CPU evaluation after the
backend-native sufficient statistics have been computed.

## One-Way ANOVA

For groups with sizes $n_i$ and means $\bar y_i$, the grand mean is

$$
\bar y = \frac{\sum_i n_i\bar y_i}{\sum_i n_i}.
$$

The F statistic is

$$
F = \frac{SSB/(k-1)}{SSW/(N-k)},
$$

where

$$
SSB = \sum_i n_i(\bar y_i-\bar y)^2,
\qquad
SSW = \sum_i\sum_j(y_{ij}-\bar y_i)^2.
$$

`AnovaResult` reports `statistic`, `pvalue`, `df_between`, `df_within`, and
`eta_squared`.

## Two-Way, Welch, and Post-Hoc Tests

- `f_twoway` supports balanced two-way designs with either a full interaction model or
  an additive model. Unbalanced designs raise until Type I/II/III sum-of-squares
  semantics are explicitly supported.
- `f_welch` handles unequal group variances and preserves the fractional
  Welch–Satterthwaite denominator degrees of freedom.
- `tukey_hsd` uses the studentized-range distribution.
- `bonferroni` performs Bonferroni-adjusted pairwise Welch tests.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | required | Two or more one-dimensional samples |
| `backend` | `"auto"` | `"auto"`, `"numpy"`, `"cupy"`, or `"torch"` |

Function-specific parameters are documented in the public API docstrings.

## Examples

```python
import numpy as np
from statgpu.anova import f_oneway

g1 = np.random.randn(100)
g2 = np.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="numpy")
print(result.statistic, result.pvalue, result.eta_squared)
```

```python
import cupy as cp
from statgpu.anova import f_oneway

g1 = cp.random.randn(100)
g2 = cp.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="cupy")
```

```python
import torch
from statgpu.anova import f_oneway

g1 = torch.randn(100, device="cuda", dtype=torch.float64)
g2 = torch.randn(100, device="cuda", dtype=torch.float64) + 0.5
result = f_oneway(g1, g2, backend="torch")
```

## Execution Boundary

Means, variances, sums of squares, and group reductions remain on the selected backend.
Only scalar studentized-range, t, normal, or F distribution evaluations may cross to
CPU when CuPy or Torch does not provide the required function. Complete group arrays
are not transferred solely for p-value evaluation.

## Validation

This page does not maintain a global GPU completion flag. Validation evidence is scoped
to the exact function, backend, hardware, and commit recorded by maintained tests or
hardware-specific artifacts.

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
