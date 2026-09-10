# Elastic Net

> Language: English  
> Last updated: 2026-09-10  
> Switch: [简体中文](../../cn/models/elastic-net.md)

## What problem does it solve?

`ElasticNet` combines L1 sparsity with L2 stabilization. It is useful when you want some coefficients to become exactly zero but important predictors are correlated enough that pure Lasso can be unstable.

A practical continuum is:

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

This is an objective-level statement. `ElasticNet(l1_ratio=0)` has a pure-L2 objective, but the `ElasticNet` wrapper keeps its own solver/default/inference contract; use `Ridge` when you want the dedicated Ridge estimator surface.

## Model and objective

With unpenalized intercept $b$ and $\lambda=$ `l1_ratio`, statgpu minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2.
$$

`alpha` controls total regularization and `l1_ratio` controls how much of it is L1. Standardize continuous predictors before regularization unless raw scale is intentionally part of the model.

## Minimal example

```python
import numpy as np
from statgpu.linear_model import ElasticNet

rng = np.random.default_rng(2)
X = rng.normal(size=(500, 12))
beta = np.zeros(12)
beta[[1, 2, 7, 8]] = [1.2, 1.0, -0.9, -0.8]
y = 0.5 + X @ beta + rng.normal(scale=0.8, size=500)

model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print(model.coef_)
print(model.score(X, y))
```

When `l1_ratio>0`, exact zeros are possible. Nonzero coefficients are still penalized estimates, not ordinary OLS effects.

## Key parameters

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Overall regularization strength. |
| `l1_ratio` | `0.5` | L1 share: values near 1 are more Lasso-like; values near 0 more Ridge-like. |
| `device` | `"auto"` | CPU/CuPy/Torch execution choice. Explicit unavailable GPU devices fail visibly. |
| `solver` | `"fista"` | Authoritative direct-fit numerical solver. |
| `compute_inference` | `False` | Enable only when a supported post-fit inference procedure is needed. |
| `nodewise_alpha` | `None` | Separate tuning for the node-wise precision construction used by `debiased` inference only. |

For predictive tuning of both `alpha` and `l1_ratio`, prefer `ElasticNetCV` to training-fit criteria.

## Compare with Ridge and Lasso

| Property | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| Smooth shrinkage | yes | yes | yes |
| Exact zeros | usually no | yes | yes when `l1_ratio>0` |
| Correlated predictors | stable | can select one arbitrarily | more group-friendly |
| Main tuning | `alpha` | `alpha` | `alpha` + `l1_ratio` |

## CPU, GPU, Formula, weights, warm starts

```python
model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    compute_inference=False,
).fit(X, y)
```

The public estimator supports NumPy CPU, CuPy CUDA, and Torch CUDA where available. `fit()` accepts `sample_weight=` and forwards the shared `formula=` / `data=` interface. A one-fit warm start can be supplied with `initial_coef=`.

## Advanced: optimization

For the centered coefficient problem, the KKT relation is

$$
\frac{1}{n}X_c^\top(X_c\hat\beta-y_c)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1=0.
$$

For direct `ElasticNet.fit`, `solver` is authoritative on every backend. The historical `cpu_solver` is compatibility state, not a second direct-fit selector.

## Advanced: inference

`ElasticNet` is estimation-only by default. With `compute_inference=True`, post-fit inference does not change the penalized `coef_` / `intercept_` prediction fit.

| `inference_method` | Intended role | Main limitation |
|---|---|---|
| `debiased` | one-step bias-corrected coefficient inference using the shared node-wise precision engine | software reuse does not imply every Lasso theorem automatically applies to every `l1_ratio`; validity depends on the actual initial estimator and assumptions |
| `post_selection_ols` | OLS/WLS active-set diagnostic on the fit-resolved NumPy/CuPy/Torch backend | not a general selective-inference guarantee |
| `bootstrap` | residual-resampling alternative | more expensive and assumption-dependent |

Legacy `cpu_ols` / `gpu_ols` are deprecated aliases for the hardware-neutral `post_selection_ols`; device selection remains separate.

### Node-wise tuning in `debiased` inference

`nodewise_alpha` is separate from the main Elastic Net `alpha`. It changes only the node-wise Lasso regressions used to approximate the design precision matrix; it must not change the penalized prediction fit.

An explicit finite positive value is authoritative. With `nodewise_alpha=None` and $p\ge2$, statgpu standardizes the canonical centered/weighted working design and resolves

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

where $n_{\mathrm{nw}}=n$ without analytic weights and a Kish-style effective sample size is used for non-uniform analytic weights. The exact constant and effective-n convention are statgpu defaults rather than a unique theorem requirement.

The rule is response-scale independent. It intentionally replaces the historical internal response-residual-scaled rule. The standardized node-wise solution must pass an independent KKT publication gate before the precision estimate is transformed back to the working-feature scale. For `p=1`, statgpu uses analytic univariate precision and does not consume `nodewise_alpha`.

NumPy, CuPy, and Torch implement the same maintained statistical definition; explicit CUDA/Torch numerical inference does not silently fall back to CPU. Successful multi-feature debiased inference publishes `nodewise_alpha_` and detailed provenance in `_inference_result.metadata`.

For `ElasticNetCV`, `nodewise_alpha` is final-full-data-refit inference configuration only. It does not enter the `alpha`/`l1_ratio` grid, fold scoring, or tuning selection. Current CV inference remains conditional on selected tuning values. See [the node-wise tuning migration guide](../guides/nodewise-alpha-migration.md).

## Common pitfalls

- Tune `alpha` and `l1_ratio` jointly when predictive performance matters.
- Do not assume `l1_ratio=0` makes the wrapper identical to `Ridge`.
- Do not infer causality from a stable active set.
- Do not ignore feature scaling.
- Do not use training $R^2$ or compatibility AIC/BIC/F fields as penalty-aware tuning criteria.
- Do not confuse main `alpha` with inference-only `nodewise_alpha`.
- Treat post-selection OLS as a diagnostic, not automatic selective inference.

## Complete API reference

The runtime public constructor is the static wrapper constructor plus the `nodewise_alpha=None` extension installed by the maintained node-wise inference contract:

```python
ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    device="auto",
    n_jobs=None,
    solver="fista",
    cpu_solver="fista",
    lipschitz_L=None,
    gpu_memory_cleanup=False,
    compute_inference=False,
    inference_method="debiased",
    cov_type="nonrobust",
    hac_maxlags=None,
    nodewise_alpha=None,
)
```

The marked table remains the static-wrapper AST inventory used by this Draft's source-only docs checker. The runtime-installed public extension is listed immediately after it.

<!-- API-CONSTRUCTOR-START:ElasticNet -->
| Parameter | Default | Reference meaning |
|---|---:|---|
| `alpha` | `1.0` | Overall regularization strength. |
| `l1_ratio` | `0.5` | L1 penalty share. |
| `fit_intercept` | `True` | Fit an unpenalized intercept. |
| `max_iter` | `1000` | Maximum solver iterations. |
| `tol` | `1e-4` | Numerical convergence tolerance. |
| `stopping` | `"coef_delta"` | `coef_delta` or `kkt`. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | `None` | Parallelism hint where supported. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver. |
| `cpu_solver` | `"fista"` | Legacy/shared compatibility control; not authoritative for direct fit. |
| `lipschitz_L` | `None` | Optional precomputed Lipschitz constant. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU cache cleanup after fit. |
| `compute_inference` | `False` | Run the selected post-fit inference path. |
| `inference_method` | `"debiased"` | `debiased`, canonical `post_selection_ols`, or `bootstrap`; legacy `cpu_ols` / `gpu_ols` aliases are deprecated. |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable. |
| `hac_maxlags` | `None` | HAC lag count where supported. |
<!-- API-CONSTRUCTOR-END:ElasticNet -->

**Runtime-installed public extension:** `nodewise_alpha=None` — `None` uses the standardized design-side automatic rule; a finite positive real scalar explicitly sets the node-wise penalty. Successful multi-feature debiased inference publishes its resolved value in `nodewise_alpha_`.

### `fit` and important outputs

`fit(X=None, y=None, sample_weight=None, initial_coef=None, **kwargs)` returns `self`; shared forwarded keywords include `formula` and `data`.

| Attribute | Meaning |
|---|---|
| `coef_`, `intercept_` | penalized prediction fit |
| `n_iter_` | numerical iteration count |
| `nodewise_alpha_` | resolved node-wise tuning after successful multi-feature debiased inference; otherwise `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | inference/reporting arrays when inference succeeds |
| `_inference_result` | structured result with node-wise and backend provenance |

`predict`, `score`, `summary`, `get_params`, and `set_params` follow the shared estimator contract. Changing `nodewise_alpha` through `set_params` invalidates stale inference state while preserving the requested constructor value for clone/introspection.

## Validation

Maintained coverage checks the Elastic Net objective, solver/KKT behavior, direct-fit invariance to node-wise tuning, `ElasticNetCV` final-refit isolation, weighted inference, formula/public API behavior, NumPy/CuPy/Torch precision parity, and physical CUDA acceptance of the shared node-wise implementation.

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *JRSS B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
