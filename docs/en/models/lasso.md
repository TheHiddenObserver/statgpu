# Lasso

> Language: English
> Last updated: 2026-09-05
> Switch: [简体中文](../../cn/models/lasso.md)

## What problem does it solve?

`Lasso` is linear regression with an L1 penalty. Its distinctive feature is that regularization can drive some coefficients **exactly to zero**, so the fitted model performs prediction and feature selection at the same time.

Suppose you have 100 candidate predictors but believe only a small subset carries useful signal. OLS gives every predictor a coefficient. Ridge shrinks all of them but normally keeps them nonzero. Lasso can return a much smaller active set.

Typical questions include:

- Which predictors can be removed while keeping a useful linear model?
- Can I reduce overfitting when many candidate features are mostly noise?
- Can I obtain a sparse model that is easier to inspect or deploy?

## A motivating example

Imagine a dataset with 12 measured variables, but the response was actually generated from only three of them.

An unregularized model can assign small noisy coefficients to many variables. Lasso instead tends to produce a pattern like:

```text
feature       0    1    2    3    4    5    6    7    8    9   10   11
true coef     0   2.0   0    0    0  -1.5   0    0    0   0.8   0    0
Lasso coef    0  ~2.0   0    0    0  ~-1.4  0    0    0  ~0.7   0    0
```

That is the central attraction of Lasso: it turns a continuous regression problem into a sparse representation without a separate hard feature-selection step.

## Intuition

Lasso makes coefficients expensive in proportion to their absolute magnitude:

$$
\lVert\beta\rVert_1=\sum_j|\beta_j|.
$$

This creates a **soft-thresholding** effect. A coefficient with weak evidence can be pulled all the way to zero instead of merely being made smaller.

A useful mental comparison is:

```text
OLS   : keep whatever improves fit
Ridge : shrink everything
Lasso : shrink, and remove weak coefficients entirely
```

The price of that sparsity is that feature selection can become unstable when several predictors carry nearly the same information. Pure Lasso may keep one member of a correlated group and discard another almost arbitrarily. [Elastic Net](elastic-net.md) is often preferable in that situation.

## When to use it

Lasso is especially useful when:

- you expect the true signal to be sparse;
- the number of candidate predictors is large relative to the amount of data;
- interpretability benefits from a small active feature set;
- storage, deployment, or downstream modeling benefits from dropping features;
- you are willing to tune the amount of regularization using validation.

Choose another method when:

- most predictors are expected to have small but real effects — [Ridge](ridge.md) may be better;
- predictors form strongly correlated groups and you would rather keep/shrink the group together — start with [Elastic Net](elastic-net.md);
- the response is not well modeled by Gaussian linear regression — use the corresponding penalized GLM, survival, or other model family;
- your primary goal is formal post-selection inference rather than prediction/selection — selection-aware inference requires additional assumptions and care.

## Model and objective

With intercept $b$, Lasso minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lVert\beta\rVert_1.
$$

Here:

- $n$ is the number of observations;
- $x_i$ is the feature vector for observation $i$;
- $\beta$ is the coefficient vector;
- $\alpha\ge 0$ controls regularization strength;
- the intercept is not penalized.

Larger `alpha` makes zero coefficients more likely. Very small `alpha` approaches an unregularized linear fit; very large `alpha` can remove almost all signal.

### Why can L1 produce exact zeros?

The absolute-value penalty has a sharp corner at zero. In the optimization step, small coefficient updates are soft-thresholded:

$$
\mathcal S_\lambda(z)
=
\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

If $|z|\le\lambda$, the result is exactly zero. This is the computational mechanism behind Lasso sparsity.

## Minimal runnable example

The following data has 12 features but only features `1`, `5`, and `9` truly affect the response.

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 12))

true_coef = np.zeros(12)
true_coef[[1, 5, 9]] = [2.0, -1.5, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("coefficients:", np.round(model.coef_, 3))
print("selected features:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("R²:", round(model.score(X, y), 3))
```

With the fixed seed and this `alpha`, the clearly nonzero coefficients should be at indices `1`, `5`, and `9`, with values roughly `1.97`, `-1.41`, and `0.73`. The exact values are smaller in magnitude than the generating coefficients because shrinkage is part of the estimator.

## How to read the result

- `coef_[j] == 0` means Lasso removed that feature from the fitted linear predictor at the selected `alpha`.
- A nonzero coefficient is still **shrunken**; do not read it as the unpenalized OLS effect estimate.
- `intercept_` is fitted separately and is not part of the L1 penalty.
- `predict(X_new)` returns predicted continuous outcomes.
- `score(X, y)` returns $R^2$.
- `n_iter_` reports optimization iterations for the selected numerical path.

Selection is data-dependent. A feature being zero in one sample does not prove its true effect is exactly zero, especially when predictors are correlated or the sample is small.

## Key parameters and how to choose them

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Main statistical choice. Larger values create more shrinkage and more zeros. Prefer `LassoCV` or another validation procedure for predictive selection. |
| `fit_intercept` | `True` | Usually keep it unless theory fixes the intercept or the design already contains one. |
| `device` | `"auto"` | CPU is usually simplest for small problems; GPU is useful when the optimization workload is large enough to amortize transfer/setup cost. |
| `solver` | `"fista"` | Stable default for a single estimator fit. Change it mainly for numerical/performance reasons, not to change the statistical model. |
| `stopping` | `"coef_delta"` | `"kkt"` is useful when you want convergence judged against optimality conditions rather than coefficient movement alone. |
| `compute_inference` | `True` | Turn it off for pure prediction/selection. If inference matters, choose `inference_method` deliberately and read the limitations below. |

### Standardize before regularizing

L1 penalizes coefficient magnitude directly. Features measured on very different scales therefore receive effectively different penalties.

For most Lasso workflows, standardize continuous predictors before fitting. The synthetic example above already puts every feature on approximately the same scale.

## Compare with nearby methods

| Method | Keeps correlated groups? | Exact zeros? | Typical reason to choose it |
|---|---|:---:|---|
| OLS / `LinearRegression` | no regularization | no | unpenalized estimation when the design is stable |
| [Ridge](ridge.md) | tends to share signal | no | stabilize correlated predictors without feature deletion |
| **Lasso** | may select one member | yes | sparse prediction / automatic feature selection |
| [Elastic Net](elastic-net.md) | more group-friendly than Lasso | yes | sparse model with correlated predictors |
| Adaptive Lasso | data-dependent L1 weights | yes | reduce uniform-penalty bias under a stronger sparse-model assumption |

If your main uncertainty is “Ridge or Lasso?”, ask whether exact feature removal is actually valuable. If not, Ridge is often the safer low-variance choice.

## CPU and GPU example

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

Explicit `device="cuda"` and `device="torch"` use their corresponding GPU backends when supported; an unavailable explicit device should fail rather than silently change the statistical execution path.

## Advanced: solver support

| `solver` value | CPU | CuPy / Torch | Meaning |
|---|:---:|:---:|---|
| `fista` (default) | yes | yes | Stable proximal-gradient path for the L1 objective |
| `auto` | FISTA | FISTA | Current squared-error + L1 automatic destination |
| `fista_bb` | yes | yes | FISTA with Barzilai-Borwein step adaptation |
| `admm` | yes | yes | Alternative split solver; uniform sample weights only |
| `coordinate_descent` | yes | no | CPU-only compatibility path for a single squared-error fit |

`newton`, `lbfgs`, `irls`, and `exact` are rejected for the non-smooth L1 objective. `cpu_solver` is used by Lasso CV/path helpers and does not override `solver` on a single `Lasso.fit`. General algorithm mechanics are in the [solver guide](../guides/solver-algorithms.md).

## Advanced: inference after Lasso

Inference after data-driven selection is substantially harder than inference after a prespecified OLS model. statgpu exposes several practical paths, but they do **not** all make the same statistical claim.

| `inference_method` | Intended use | Important limitation |
|---|---|---|
| `cpu_ols_inference` | lightweight CPU post-selection diagnostic | heuristic OLS-style intervals; not valid selective-inference intervals |
| `gpu_ols_inference` | same style while reducing GPU→CPU transfer | same post-selection validity limitation |
| `debiased` | de-biased/de-sparsified coefficient inference | current `_conf_int` is marginal per coefficient; assumptions for high-dimensional de-biasing still matter |
| `bootstrap` | residual-bootstrap alternative | materially more expensive and still conditional on the implemented resampling/model assumptions |

With `compute_inference=True`, the reporting surface can include `_bse`, `_tvalues` or z-style statistics depending on the selected method, `_pvalues`, and `_conf_int`.

For `inference_method="debiased"`, optional simultaneous intervals are available with:

```python
model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=7,
).fit(X, y)

marginal_ci = model._conf_int
simultaneous_ci = model._conf_int_simultaneous
```

A marginal interval answers a different question from a simultaneous family-wise interval. Do not describe one as the other.

## Common pitfalls

- **Do not interpret “selected” as “proved causal” or even “certainly nonzero in the population.”** Lasso selection is sample- and tuning-dependent.
- **Do not ignore feature scaling.** An L1 penalty is not scale invariant.
- **Do not expect stable choices among nearly duplicate predictors.** Pure Lasso can arbitrarily prefer one correlated feature; Elastic Net is often more appropriate.
- **Do not choose `alpha` by maximizing training $R^2$.** Use held-out validation or cross-validation.
- **Do not attach ordinary OLS p-values after selection and treat them as if the model had been prespecified.** Use an inference method whose assumptions match your question.
- **Do not confuse numerical convergence with statistical correctness.** A tiny KKT residual only says the declared optimization problem was solved accurately.

## API and validation

Import path:

```python
from statgpu.linear_model import Lasso
```

Advanced constructor controls include `max_iter`, `tol`, `cpu_solver`, `gpu_memory_cleanup`, `inference_method`, and simultaneous-inference settings. They are deliberately placed after the modeling workflow on this learner page.

Maintained validation covers solver convergence, CPU/GPU consistency, KKT stopping, de-biased inference, bootstrap/inference paths, and physical-GPU behavior where required. Relevant maintained entry points include `dev/tests/test_lasso_debiased_inference.py`, `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`, and `dev/comparisons/compare_lasso_kkt_stopping.py`.

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
