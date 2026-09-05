# Elastic Net

> Language: English
> Last updated: 2026-09-05
> Switch: [简体中文](../../cn/models/elastic-net.md)

## What problem does it solve?

`ElasticNet` combines the L1 penalty of [Lasso](lasso.md) with the L2 penalty of [Ridge](ridge.md). It is designed for situations where you want a sparse model **and** your predictors are correlated enough that pure Lasso selection is unstable.

The motivation is a practical tension:

- Ridge handles correlated predictors well, but normally keeps every coefficient nonzero.
- Lasso can remove features, but among nearly interchangeable predictors it may keep one and discard another arbitrarily.
- Elastic Net adds both penalties so it can create zeros while still encouraging correlated predictors to share signal more smoothly.

## A motivating example

Suppose four useful predictors come in two highly correlated pairs. Within each pair, both variables really carry signal.

Pure Lasso may split each pair unevenly:

```text
              x1     x2     x3     x4
Lasso        1.62   0.71  -0.68  -1.24
Elastic Net  1.19   1.14  -0.95  -0.98
```

Both fits can predict well, but Elastic Net better reflects the idea that the paired predictors are nearly interchangeable measurements of shared latent signals.

## Intuition

Elastic Net asks the model to satisfy two regularization preferences at once:

1. **L1 part:** weak coefficients may be pushed all the way to zero;
2. **L2 part:** large or unstable coefficients are smoothly shrunk, which helps correlated predictors behave more like a group.

Two parameters control the trade-off:

- `alpha` controls **how much total regularization** is applied;
- `l1_ratio` controls **what kind of regularization** it is.

A useful continuum is:

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

In statgpu's objective scaling, `l1_ratio=0` gives the Ridge penalty at the same public `alpha`, while `l1_ratio=1` gives the Lasso penalty.

## When to use it

Elastic Net is a strong choice when:

- you want feature selection but many predictors are correlated;
- predictors naturally come in groups that may carry similar information;
- pure Lasso changes its selected feature among nearly duplicate variables;
- you have many candidate predictors and want a sparse but more stable model;
- you are willing to tune both regularization strength and mixture using validation.

Prefer another method when:

- you do **not** need exact zeros and mainly want stable prediction — Ridge is simpler;
- you strongly expect one sparse representative from each feature group and want the most aggressive sparsity — Lasso may be enough;
- the response is not appropriate for Gaussian linear regression — use the corresponding penalized GLM or other model family;
- feature selection is not scientifically meaningful because predictors are arbitrary encodings or strongly confounded.

## Model and objective

With intercept $b$, Elastic Net minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2,
$$

where $\lambda$ is `l1_ratio`.

Here:

- $\alpha\ge 0$ sets the overall penalty strength;
- $0\le\lambda\le 1$ mixes L1 and L2;
- the intercept is not penalized.

Increasing `alpha` shrinks the model more strongly. Increasing `l1_ratio` makes exact zeros more likely; decreasing it makes the fit behave more like Ridge.

## Minimal runnable example

This example creates two pairs of almost-duplicate useful predictors and compares pure Lasso with Elastic Net.

```python
import numpy as np
from statgpu.linear_model import ElasticNet, Lasso

rng = np.random.default_rng(2)
n = 500

z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
X = np.column_stack([
    z1 + 0.05 * rng.normal(size=n),
    z1 + 0.05 * rng.normal(size=n),
    z2 + 0.05 * rng.normal(size=n),
    z2 + 0.05 * rng.normal(size=n),
    rng.normal(size=n),
    rng.normal(size=n),
])

true_coef = np.array([1.2, 1.2, -1.0, -1.0, 0.0, 0.0])
y = 0.5 + X @ true_coef + rng.normal(scale=0.8, size=n)

lasso = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

elastic = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("Lasso:      ", np.round(lasso.coef_, 2))
print("Elastic Net:", np.round(elastic.coef_, 2))
```

With the fixed seed, Lasso should split the first correlated pair quite unevenly (roughly `1.62` and `0.71`) and the second around `-0.68` and `-1.24`. Elastic Net should distribute the signal more evenly, around `1.19`, `1.14`, `-0.95`, and `-0.98`, while leaving the two noise features at or near zero.

The point is not that equal coefficients are always correct. It is that the L2 component reduces the arbitrary winner-takes-most behavior pure Lasso can show among highly correlated predictors.

## How to read the result

- `coef_[j] == 0` means the combined penalty removed feature `j` at the selected hyperparameters.
- Nonzero coefficients are still shrunk; they are not unpenalized OLS estimates.
- `intercept_` is fitted separately and is not penalized.
- `predict(X_new)` returns continuous predictions.
- `score(X, y)` returns $R^2$.
- The active set depends on **both** `alpha` and `l1_ratio`; changing either can change which variables survive.

If two correlated variables stay nonzero together, that is a common Elastic Net behavior, not evidence that both are independently causal.

## Key parameters and how to choose them

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Overall regularization strength. Larger values shrink more strongly and can remove more features. Tune with validation. |
| `l1_ratio` | `0.5` | L1/L2 mixture. Values near 1 behave more like Lasso; values near 0 behave more like Ridge. Tune jointly with `alpha` when possible. |
| `fit_intercept` | `True` | Usually keep it unless theory fixes the intercept or the design already contains one. |
| `device` | `"auto"` | CPU is simplest for small problems; GPU becomes useful when the optimization workload is large enough. |
| `solver` | `"fista"` | Stable default for the declared non-smooth objective. Change primarily for numerical/performance reasons. |
| `stopping` | `"coef_delta"` | Use `"kkt"` when optimality-based convergence diagnostics are more important than coefficient movement. |
| `compute_inference` | `False` | Keep off for ordinary prediction/selection. Enable only when you need a supported post-fit inference procedure. |

For predictive modeling, `ElasticNetCV` is usually preferable to choosing `alpha` and `l1_ratio` by hand.

### Standardize your features

Both L1 and L2 penalties act on coefficient magnitude. Different feature units therefore change the effective penalty.

Standardize continuous predictors before fitting unless the raw feature scale is deliberately part of the modeling convention.

## Compare with Ridge and Lasso

| Property | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| Smooth coefficient shrinkage | yes | yes | yes |
| Exact zero coefficients | usually no | yes | yes |
| Stable with correlated predictors | strong | can be unstable | stronger than pure Lasso |
| Main tuning choices | `alpha` | `alpha` | `alpha` + `l1_ratio` |
| Best mental model | stabilize | select | select + stabilize |

A practical rule of thumb:

- start with **Ridge** when you care about prediction and correlated predictors but not deletion;
- start with **Lasso** when sparsity is the main goal and correlations are modest;
- start with **Elastic Net** when you want sparsity and know correlated feature groups are important.

## CPU and GPU example

```python
from statgpu.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    compute_inference=False,
).fit(X, y)
```

The public wrapper supports NumPy CPU, CuPy CUDA, and Torch CUDA paths where available. Backend speed depends on sample size, feature dimension, dtype, data residency, and transfer cost; benchmark the workload you actually care about.

A one-fit warm start can be supplied with `fit(initial_coef=...)`.

## Advanced: solver and optimization details

Elastic Net is non-smooth whenever `l1_ratio > 0`, so proximal methods are the normal numerical path.

| `solver` value | CPU | CuPy / Torch | Notes |
|---|:---:|:---:|---|
| `fista` (default) | yes | yes | Recommended proximal path |
| `auto` | FISTA | FISTA | Current squared-error + Elastic Net dispatch |
| `fista_bb` | yes | yes | Adaptive spectral steps |
| `admm` | yes | yes | Alternative split solver; uniform sample weights only |
| `coordinate_descent` | yes | no | CPU-only compatibility path |

`newton`, `lbfgs`, `irls`, and `exact` are rejected for the non-smooth Elastic Net estimator surface. `cpu_solver` does not override `solver` for a single estimator fit. Full numerical mechanics are documented in the [solver guide](../guides/solver-algorithms.md).

The first-order KKT condition for the coefficient vector is

$$
\frac{1}{n}X^\top(X\hat\beta-y)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1
=0.
$$

`stopping="kkt"` checks this kind of optimality condition; it does not define a different statistical approximation.

## Advanced: inference

`ElasticNet` is estimation-only by default. With `compute_inference=True`, statgpu runs a post-fit inference method without changing the penalized coefficients.

| `inference_method` | Intended role | Important limitation |
|---|---|---|
| `debiased` (default inference method) | bias-corrected coefficient inference using the shared penalized-linear engine | assumptions for de-biasing matter; inference is conditional on selected regularization parameters |
| `cpu_ols` | lightweight post-selection OLS-style path | heuristic after selection; not a general selective-inference guarantee |
| `bootstrap` | resampling-based alternative | higher computational cost and conditional on the implemented bootstrap assumptions |

When inference succeeds, `summary()` and reporting fields such as standard errors, z-style statistics, p-values, and confidence intervals become available according to the selected method.

For `ElasticNetCV`, `compute_inference=True` applies only to the final full-data refit after `alpha` and `l1_ratio` have been selected. Fold models remain estimation-only.

## Common pitfalls

- **Do not tune only `alpha` while treating `l1_ratio` as irrelevant.** The mixture parameter changes the kind of model you are fitting.
- **Do not assume correlated nonzero features have separate causal effects.** Elastic Net stabilizes prediction/selection; it does not identify causal structure.
- **Do not forget standardization.** Both parts of the penalty depend on coefficient scale.
- **Do not choose hyperparameters from training $R^2$.** Use held-out or cross-validation performance.
- **Do not expect the same active set under tiny data perturbations when signals are weak.** Elastic Net improves correlated-feature stability but does not eliminate sampling uncertainty.
- **Do not attach ordinary unpenalized inference to a data-selected active set without acknowledging selection.** Use the supported post-fit methods and their stated assumptions.

## API and validation

Import path:

```python
from statgpu.linear_model import ElasticNet
```

The public wrapper also exposes advanced controls including `max_iter`, `tol`, `cpu_solver`, `lipschitz_L`, `gpu_memory_cleanup`, `cov_type`, and `hac_maxlags`. It does not expose separate constructor parameters named `backend`, `warm_start`, or `random_state`; backend selection is controlled by `device`, and a one-fit warm start uses `fit(initial_coef=...)`.

Maintained validation checks the declared Elastic Net objective, solver/KKT behavior, CPU and supported GPU paths, post-fit inference, and the final-refit inference contract for `ElasticNetCV`. Physical CUDA validation remains part of exact release acceptance where applicable.

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
