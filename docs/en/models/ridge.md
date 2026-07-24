# Ridge

> Language: English  
> Last updated: 2026-07-12  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/ridge.md)

Language switch: [Chinese](../../cn/models/ridge.md)

## Overview

`Ridge` provides L2-regularized linear regression with the same inference surface as `LinearRegression` (including robust covariance options). It is used when multicollinearity or shrinkage is required while keeping interpretable coefficient inference in aligned settings.

## Path

`statgpu.linear_model.Ridge`

## Objective Function

For unweighted observations, statgpu minimizes the average-loss objective

$$
\min_{b,\beta}
\frac{1}{2n}\sum_{i=1}^n
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

With `sample_weight=w`, the data-fit term is normalized by the total weight:

$$
\min_{b,\beta}
\frac{1}{2\sum_i w_i}\sum_{i=1}^n
w_i\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

The intercept is not penalized. Multiplying every sample weight by the same positive constant therefore leaves the fitted model unchanged.

## Estimating Equation

After centering the data using the corresponding ordinary or weighted means, the first-order condition is

$$
\left(X_c^\top W X_c + \alpha\,s_w I\right)\hat\beta
= X_c^\top W y_c,
$$

where $W=I$ and $s_w=n$ without sample weights, while $W=\operatorname{diag}(w)$ and $s_w=\sum_iw_i$ for weighted fitting.

`Ridge` defaults to `solver="exact"`. The same objective scale is used by the exact and FISTA paths, by `PenalizedLinearRegression(loss="squared_error", penalty="l2")`, and by `RidgeCV`.

scikit-learn uses an unnormalized residual sum of squares. For coefficient comparisons, use

- unweighted: `sklearn_alpha = n_samples * statgpu_alpha`;
- weighted: `sklearn_alpha = sample_weight.sum() * statgpu_alpha`.

Comparing the two libraries with the same numerical `alpha` compares different objectives.

## Covariance/Inference

- `cov_type="nonrobust"`: classical ridge covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: sandwich-style robust covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- `compute_inference=True` returns `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.
- Weighted inference uses the weighted design `[sqrt(w), sqrt(w) * X]`, so the intercept column, residuals, bread, and meat follow the same weighting convention as estimation.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 regularization strength on the average-loss scale |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | Number of parallel jobs |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows a Newey-West-style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after each fit |
| `solver` | `"exact"` | Exact L2 solution by default; `fista` uses the same objective |

## CPU+GPU Examples

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc3", compute_inference=True)
m_cpu.fit(X, y, sample_weight=w)

# CuPy CUDA
m_gpu = Ridge(
    alpha=1.0,
    device="cuda",
    cov_type="hc3",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y, sample_weight=w)
```

## strict/approx difference

No separate public approximate mode is exposed. CPU tests cover the exact/FISTA, weighted/unweighted, formula, inference, and RidgeCV contracts. Physical CuPy/Torch CUDA numerical and performance validation remains part of the remote validation gate.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `rsquared`, `rsquared_adj`, `fvalue`, `aic`, `bic`
- Methods: `fit`, `predict`, `score`, `summary`

## FAQ

- How should `alpha` be chosen? Use `RidgeCV` or a task-specific log grid on statgpu's average-loss scale.
- Why does the same `alpha` differ from sklearn? The residual term has a different normalization; apply the mapping above.
- Does rescaling all sample weights change the model? No. The weighted loss is divided by `sum(sample_weight)`.
- When should I set `hac_maxlags`? When using `cov_type="hac"` with time dependence; otherwise leave the default.

## External Validation

- Internal consistency is tested against the average-loss closed form and the generic penalized-linear estimator.
- sklearn comparisons use the explicit unweighted or weighted alpha mapping.
- Weighted exact/FISTA, formula-row alignment, inference, and RidgeCV weight-rescaling invariance are covered in `dev/tests/test_ridge_weighted_consistency.py`.

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
