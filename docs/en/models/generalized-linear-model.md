# GeneralizedLinearModel and penalized GLMs

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/models/generalized-linear-model.md)

## What problem do GLMs solve?

A generalized linear model (GLM) extends linear regression to outcomes whose range and variance do not fit a Gaussian continuous model. It keeps an interpretable linear predictor, but connects that predictor to the expected response through a link function.

In statgpu, `GeneralizedLinearModel` is the common entry point. Typed estimators such as `PoissonRegression` make the response family explicit, while penalized variants add shrinkage or feature selection.

## When to use a GLM

Choose the family from the kind of outcome and its mean-variance behavior:

| Outcome | Family / default link | Good starting estimator |
|---|---|---|
| continuous, roughly symmetric | Gaussian / identity | `LinearRegression` or `GeneralizedLinearModel(family="gaussian")` |
| 0/1 response | Binomial / logit | `LogisticRegression` |
| non-negative counts with variance near the mean | Poisson / log | `PoissonRegression` |
| overdispersed counts | Negative binomial / log | `NegativeBinomialRegression` |
| positive, right-skewed continuous response | Gamma / log | `GammaRegression` |
| positive response with variance growing roughly as $\mu^3$ | Inverse Gaussian / log | `InverseGaussianRegression` |
| zeros plus positive continuous values or power variance | Tweedie / log | `TweedieRegression` |

Do not select a family from the outcome name alone. Check the observed range, the variance pattern, excess zeros, dependence, and whether the link gives a scientifically meaningful relationship.

## Intuition

Every GLM has three parts:

1. a response distribution, called the **family**;
2. a linear predictor $\eta_i=\beta_0+x_i^\top\beta$;
3. a **link function** $g$ connecting the mean to that predictor:

$$
g(\mu_i)=\eta_i,
\qquad
\mu_i=\mathbb E(Y_i\mid x_i).
$$

For a log link, $\mu_i=\exp(\eta_i)$, so predictions stay positive. For a logit link, the inverse link maps any linear predictor into a probability between zero and one.

## Model, objective, and assumptions

Ordinary GLMs estimate coefficients by minimizing the family's average negative log-likelihood:

$$
\hat\beta
=
\arg\min_\beta
\frac{1}{n}\sum_{i=1}^{n}
\ell(y_i,x_i^\top\beta).
$$

A penalized GLM adds a regularization term:

$$
\hat\beta
=
\arg\min_\beta
\left[
\frac{1}{n}\sum_{i=1}^{n}
\ell(y_i,x_i^\top\beta)
+\alpha P(\beta)
\right],
$$

where the intercept is not penalized.

Interpretation and inference require an appropriate family, link, linear predictor, independent or correctly modeled dependence, and no severe rank deficiency. A GLM does not automatically fix omitted variables, excess zeros, or clustered observations.

## Minimal runnable example

The following Poisson example generates count data and fits an unpenalized model with robust inference.

```python
import numpy as np
from statgpu.linear_model import PoissonRegression

rng = np.random.default_rng(1)
X = rng.normal(size=(800, 2))
true_coef = np.array([0.45, -0.25])
mean_count = np.exp(0.30 + X @ true_coef)
y = rng.poisson(mean_count)

model = PoissonRegression(
    solver="newton",
    device="cpu",
    compute_inference=True,
    cov_type="hc1",
).fit(X, y)

print("intercept:", model.intercept_)
print("log-rate coefficients:", model.coef_)
print("rate ratios:", np.exp(model.coef_))
print("p-values:", model._pvalues)
print("expected counts:", model.predict(X[:3]))
```

`solver="newton"` is explicit here because the IRLS path's `C` parameter adds L2 regularization. The fitted coefficients should be near `[0.45, -0.25]`.

## How to read the result

For a log-link model:

$$
\log(\mu_i)=\beta_0+x_i^\top\beta.
$$

- `coef_[j]` is a change on the log-mean scale.
- `exp(coef_[j])` is a multiplicative mean ratio. For example, `0.45` corresponds to about `exp(0.45)=1.57` times the expected count for a one-unit increase, holding other features fixed.
- `predict(X_new)` returns the expected response on its natural scale.
- `_bse`, `_zvalues`, `_pvalues`, and `_conf_int` describe uncertainty when inference is enabled.

For a logit-link model, `exp(coef_[j])` is an odds ratio, not a probability change.

## Key parameters and how to choose them

| Parameter | Guidance |
|---|---|
| `family` | Choose from the response support and variance pattern; supported values are `gaussian`, `binomial`, `poisson`, `gamma`, `inverse_gaussian`, `negative_binomial`, and `tweedie` |
| `fit_intercept` | Usually keep `True` unless the design already contains an intercept or theory fixes it at zero |
| `solver` | Start with `"auto"`; use an explicit solver when you need reproducible dispatch or an unpenalized path |
| `device` | Use `cpu`, `cuda`, `torch`, or `auto` according to the estimator and installed backend |
| `compute_inference` | Enable for ordinary-model standard errors and tests; it is off by default for many GLM wrappers |
| `cov_type` | Ordinary GLMs support `nonrobust`, `hc0`, and `hc1` |
| `alpha` | In penalized models, larger values produce stronger shrinkage; choose it by cross-validation |
| `l1_ratio` | For Elastic Net, values near 1 favor sparsity and values near 0 favor L2-like shrinkage |

## Ordinary or penalized?

| Goal | Recommended API |
|---|---|
| estimate and interpret a small, prespecified model | typed ordinary wrapper such as `PoissonRegression` |
| shrink correlated coefficients | typed penalized wrapper with `penalty="l2"` |
| select features | `penalty="l1"`, `"elasticnet"`, SCAD, or MCP |
| tune regularization | `PenalizedGLM_CV` |
| use a formula and DataFrame | install `statgpu[formula]` and pass `formula=...` with `data=...` |

Available penalized typed wrappers cover Gaussian, logistic, Poisson, Gamma, inverse Gaussian, negative binomial, and Tweedie models.

## CPU, GPU, solver, and inference support

Explicit `device="cuda"` uses CuPy and `device="torch"` uses Torch CUDA on supported paths; an explicit GPU request does not silently fall back to CPU. Formula parsing is CPU-side preprocessing, so explicit arrays are preferable for large GPU workloads.

`solver="auto"` is family-, penalty-, and device-aware. Smooth objectives can use IRLS, Newton, or L-BFGS; non-smooth penalties use proximal solvers such as FISTA. Invalid solver/penalty combinations raise an error.

Ordinary typed GLMs expose post-fit inference through `compute_inference=True`. Penalized inference depends on the penalty and model; consult the [solver and penalty matrix](../guides/solver-penalty-matrix.md) instead of assuming ordinary-model p-values remain valid after selection.

`PenalizedGLM_CV` uses strict cross-validation by default. `cv_strategy="two_stage"` is an explicit approximate screening option and emits `ApproximateCVWarning` unless acknowledged.

## Common pitfalls

- Poisson variance that is much larger than the mean suggests overdispersion; compare negative binomial or robust inference.
- A log-link coefficient is not an additive change in the response.
- Zero-heavy counts may need a zero-inflated or hurdle model, which is not implied by choosing Poisson.
- Regularized coefficients are biased by design; ordinary standard errors cannot simply be attached after feature selection.
- Parameter scales differ between libraries. Do not copy an `alpha` or `C` value from another framework without checking its objective definition.
- Always inspect convergence state, predictive validation, and residual or calibration diagnostics.

## API and validation

Main imports:

```python
from statgpu.linear_model import (
    GeneralizedLinearModel,
    LogisticRegression,
    PoissonRegression,
    GammaRegression,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)
```

Common fitted outputs are `coef_`, `intercept_`, `n_iter_`, `predict`, and family-specific `predict_proba` or `score` methods. Inference outputs use `_bse`, `_zvalues`, `_pvalues`, and `_conf_int`.

External-framework and backend checks cover coefficient agreement, objective gaps, KKT residuals, inference, and CPU/CuPy/Torch behavior. See the validation references linked from the [implemented methods guide](../guides/implemented-methods.md).

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.
