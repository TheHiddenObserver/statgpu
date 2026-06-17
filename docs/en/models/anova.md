# ANOVA

> Language: English  
> Last updated: 2026-06-17  
> This page: Model documentation  
> Switch: [Chinese](../../models/anova.md)

Language switch: [Chinese](../../models/anova.md)

## Overview

`f_oneway` performs one-way Analysis of Variance (ANOVA), testing whether group means are equal. It is a GPU-accelerated drop-in replacement for `scipy.stats.f_oneway`, supporting numpy, cupy, and torch backends.

## Path

`statgpu.anova.f_oneway`, `statgpu.anova.AnovaResult`
`statgpu.anova.f_twoway`, `statgpu.anova.TwoWayAnovaResult`
`statgpu.anova.f_welch`
`statgpu.anova.tukey_hsd`, `statgpu.anova.TukeyResult`
`statgpu.anova.bonferroni`, `statgpu.anova.PosthocResult`
`statgpu.anova.cohens_f`
`statgpu.anova.partial_eta_squared`

## Objective Function

Grand mean:
$$
\bar{y} = \frac{\sum_i n_i \bar{y}_i}{\sum_i n_i}
$$

Between-group sum of squares:
$$
SSB = \sum_{i=1}^k n_i (\bar{y}_i - \bar{y})^2
$$

Within-group sum of squares:
$$
SSW = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_i)^2
$$

F-statistic:
$$
F = \frac{SSB / (k-1)}{SSW / (N-k)}
$$

where $k$ is the number of groups, $n_i$ is the size of group $i$, and $N = \sum_i n_i$ is the total number of observations.

## Estimating Equation

Direct computation, no iterative solver needed. The F-statistic is computed in a single pass over the data using backend-native reduction operations.

## Covariance/Inference

P-value is obtained from the F-distribution survival function $1 - F_{k-1,\,N-k}(F)$. Effect size is reported as:
$$
\eta^2 = \frac{SSB}{SSB + SSW}
$$

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

## Three-backend support

All ANOVA functions (`f_oneway`, `f_twoway`, `f_welch`, `tukey_hsd`, `bonferroni`, `cohens_f`, `partial_eta_squared`) support three compute backends via the `backend` parameter:

| Backend | Description |
|---|---|
| `"numpy"` | CPU computation using NumPy |
| `"cupy"` | GPU computation using CuPy (NVIDIA CUDA) |
| `"torch"` | GPU computation using PyTorch (NVIDIA CUDA) |
| `"auto"` | Automatically selects the best available backend |

---

## f_twoway

Two-way ANOVA with optional interaction term.

### Path

`statgpu.anova.f_twoway`, `statgpu.anova.TwoWayAnovaResult`

### Overview

`f_twoway` performs a two-factor analysis of variance, testing the effects of factor A, factor B, and their interaction. It accepts data as a nested list of cell observations and supports both full (with interaction) and additive (without interaction) models.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `data` | (required) | Nested list/array of shape `(a, b)` where each element is an array of cell observations |
| `interaction` | `True` | If `True`, include the interaction term (full model); if `False`, fit additive model |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |
| `dtype` | `None` | Float dtype for computation; `None` uses `float64` |

### Outputs

`TwoWayAnovaResult` dataclass with fields:

| Field | Type | Description |
|---|---|---|
| `factor_a_statistic` | float | F-statistic for factor A |
| `factor_a_pvalue` | float | P-value for factor A |
| `factor_a_df` | int | Degrees of freedom for factor A ($a - 1$) |
| `factor_a_eta_squared` | float | Eta-squared for factor A |
| `factor_b_statistic` | float | F-statistic for factor B |
| `factor_b_pvalue` | float | P-value for factor B |
| `factor_b_df` | int | Degrees of freedom for factor B ($b - 1$) |
| `factor_b_eta_squared` | float | Eta-squared for factor B |
| `interaction_statistic` | float or None | F-statistic for interaction (`None` if `interaction=False`) |
| `interaction_pvalue` | float or None | P-value for interaction (`None` if `interaction=False`) |
| `interaction_df` | int or None | Degrees of freedom for interaction (`None` if `interaction=False`) |
| `interaction_eta_squared` | float or None | Eta-squared for interaction (`None` if `interaction=False`) |
| `df_within` | int | Residual degrees of freedom |
| `ss_within` | float | Residual sum of squares |

### Example

```python
from statgpu.anova import f_twoway
import numpy as np

# 2x3 balanced design, 5 observations per cell
data = [[np.random.randn(5) for _ in range(3)] for _ in range(2)]
result = f_twoway(data, interaction=True)
print(f"Factor A: F={result.factor_a_statistic:.4f}, p={result.factor_a_pvalue:.4e}")
print(f"Factor B: F={result.factor_b_statistic:.4f}, p={result.factor_b_pvalue:.4e}")
print(f"Interaction: F={result.interaction_statistic:.4f}, p={result.interaction_pvalue:.4e}")

# Additive model (no interaction)
result_add = f_twoway(data, interaction=False)
```

---

## f_welch

Welch's one-way ANOVA for groups with unequal variances.

### Path

`statgpu.anova.f_welch`

### Overview

`f_welch` performs Welch's ANOVA, which does not assume equal variances across groups. It is a GPU-accelerated alternative to `scipy.stats.alexandergovern` and R's `oneway.test`. Uses the Welch-Satterthwaite equation for degrees of freedom.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | (required) | Two or more 1-D arrays, one per group |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |
| `dtype` | `None` | Float dtype for computation; `None` uses `float64` |

### Outputs

Returns `AnovaResult` (same as `f_oneway`):

| Field | Type | Description |
|---|---|---|
| `statistic` | float | Welch F-statistic |
| `pvalue` | float | P-value from F-distribution |
| `df_between` | int | Between-group degrees of freedom ($k - 1$) |
| `df_within` | int | Approximate within-group df (Welch-Satterthwaite) |
| `eta_squared` | float | `NaN` (not meaningful for Welch's test) |

### Example

```python
from statgpu.anova import f_welch
import numpy as np

# Groups with very different variances
g1 = np.random.randn(100)
g2 = np.random.randn(100) * 5 + 2
g3 = np.random.randn(50) * 0.5 - 1

result = f_welch(g1, g2, g3, backend="numpy")
print(f"Welch F={result.statistic:.4f}, p={result.pvalue:.4e}")
```

---

## tukey_hsd

Tukey's Honestly Significant Difference post-hoc test.

### Path

`statgpu.anova.tukey_hsd`, `statgpu.anova.TukeyResult`

### Overview

`tukey_hsd` performs all pairwise comparisons between group means using the studentized range distribution. It controls the family-wise error rate and provides simultaneous confidence intervals. Use after a significant ANOVA result to identify which specific group means differ.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | (required) | Two or more 1-D arrays, one per group |
| `alpha` | `0.05` | Family-wise significance level |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |
| `dtype` | `None` | Float dtype for computation; `None` uses `float64` |

### Outputs

`TukeyResult` dataclass with fields:

| Field | Type | Description |
|---|---|---|
| `comparisons` | list of `PairwiseComparison` | All pairwise comparisons |
| `alpha` | float | Significance level used |
| `n_groups` | int | Number of groups |
| `df_within` | int | Within-group degrees of freedom |
| `mse` | float | Mean square error (within-group variance) |

Each `PairwiseComparison` has:

| Field | Type | Description |
|---|---|---|
| `group_i` | int | Index of first group |
| `group_j` | int | Index of second group |
| `mean_diff` | float | Difference in means ($\bar{x}_i - \bar{x}_j$) |
| `pvalue` | float | P-value from studentized range distribution |
| `ci_lower` | float | Lower bound of simultaneous confidence interval |
| `ci_upper` | float | Upper bound of simultaneous confidence interval |
| `reject` | bool | `True` if `pvalue < alpha` |

### Example

```python
from statgpu.anova import f_oneway, tukey_hsd
import numpy as np

g1 = np.random.randn(30)
g2 = np.random.randn(30) + 1.0
g3 = np.random.randn(30) + 0.5

# Check overall significance first
f_result = f_oneway(g1, g2, g3)
if f_result.pvalue < 0.05:
    # Pairwise comparisons
    t_result = tukey_hsd(g1, g2, g3, alpha=0.05)
    for c in t_result.comparisons:
        print(f"Group {c.group_i} vs {c.group_j}: diff={c.mean_diff:.4f}, "
              f"p={c.pvalue:.4e}, reject={c.reject}")
```

---

## bonferroni

Bonferroni-corrected pairwise t-tests.

### Path

`statgpu.anova.bonferroni`, `statgpu.anova.PosthocResult`

### Overview

`bonferroni` performs Welch's t-test for each pair of groups with Bonferroni correction for multiple comparisons. Unlike Tukey HSD, it does not assume equal variances and uses a simpler correction. The per-comparison significance level is $\alpha / m$ where $m = k(k-1)/2$.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | (required) | Two or more 1-D arrays, one per group |
| `alpha` | `0.05` | Family-wise significance level |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |
| `dtype` | `None` | Float dtype for computation; `None` uses `float64` |

### Outputs

`PosthocResult` dataclass with fields:

| Field | Type | Description |
|---|---|---|
| `comparisons` | list of `PairwiseComparison` | All pairwise comparisons (same fields as Tukey) |
| `alpha` | float | Family-wise significance level |
| `n_comparisons` | int | Number of pairwise comparisons ($k(k-1)/2$) |

### Example

```python
from statgpu.anova import bonferroni
import numpy as np

g1 = np.random.randn(30)
g2 = np.random.randn(30) + 1.0
g3 = np.random.randn(30) + 0.5

result = bonferroni(g1, g2, g3, alpha=0.05)
print(f"Number of comparisons: {result.n_comparisons}")
for c in result.comparisons:
    print(f"Group {c.group_i} vs {c.group_j}: diff={c.mean_diff:.4f}, "
          f"p={c.pvalue:.4e}, reject={c.reject}")
```

---

## cohens_f

Cohen's f effect size measure.

### Path

`statgpu.anova.cohens_f`

### Overview

`cohens_f` computes Cohen's f effect size from group data. It is derived from eta-squared via $f = \sqrt{\eta^2 / (1 - \eta^2)}$. Benchmarks: small = 0.10, medium = 0.25, large = 0.40 (Cohen 1988).

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | (required) | Two or more 1-D arrays, one per group |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |
| `dtype` | `None` | Float dtype for computation; `None` uses `float64` |

### Outputs

Returns `float`: Cohen's f value.

### Example

```python
from statgpu.anova import cohens_f
import numpy as np

g1 = np.random.randn(50)
g2 = np.random.randn(50) + 0.5

f_val = cohens_f(g1, g2, backend="numpy")
print(f"Cohen's f = {f_val:.4f}")
# Interpret: < 0.10 small, < 0.25 medium, < 0.40 large
```

---

## partial_eta_squared

Partial eta-squared effect size from sum of squares.

### Path

`statgpu.anova.partial_eta_squared`

### Overview

`partial_eta_squared` computes partial eta-squared from pre-computed sum of squares: $\eta_p^2 = SS_{\text{effect}} / (SS_{\text{effect}} + SS_{\text{error}})$. This is equivalent to eta-squared in one-way ANOVA but differs in multi-factor designs where $SS_{\text{error}}$ is the residual SS. Useful with `TwoWayAnovaResult` fields.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ss_effect` | (required) | Sum of squares for the effect of interest |
| `ss_error` | (required) | Sum of squares for the error term |
| `backend` | `"auto"` | Not used (kept for API consistency) |

### Outputs

Returns `float`: Partial eta-squared value in $[0, 1]$, or `NaN` if both SS are zero.

### Example

```python
from statgpu.anova import f_twoway, partial_eta_squared
import numpy as np

data = [[np.random.randn(10) for _ in range(3)] for _ in range(2)]
result = f_twoway(data)

# Partial eta-squared for factor A
eta_a = partial_eta_squared(
    result.factor_a_statistic * result.factor_a_df * (result.ss_within / result.df_within),
    result.ss_within
)
print(f"Partial eta-squared for factor A: {eta_a:.4f}")
```

---

## FAQ

- **How many groups are supported?** Two or more.
- **What if all observations are identical?** Returns `NaN` for `statistic`, `pvalue`, and `eta_squared`.
- **What if groups are perfectly separated?** Returns `inf` for `statistic`, `0.0` for `pvalue`, `1.0` for `eta_squared`.
- **Is this a drop-in replacement for scipy?** Yes. The function signature and output fields are compatible with `scipy.stats.f_oneway`, with the addition of `eta_squared`, `df_between`, and `df_within`.

## External Validation

Validated against `scipy.stats.f_oneway` with relative error < 1e-15 across a wide range of group sizes and effect magnitudes.

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
