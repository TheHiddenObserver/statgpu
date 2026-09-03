# GLM families and complete API

> Language: English
> Last updated: 2026-09-04
> Switch: [简体中文](../../cn/models/glm-family-reference.md)

This advanced reference complements the [beginner GLM guide](generalized-linear-model.md). In every row below,

$$
g(\mu_i)=\beta_0+x_i^\top\beta,
\qquad
\operatorname{Var}(Y_i\mid x_i)=\phi V(\mu_i).
$$

The displayed $V(\mu)$ omits the dispersion multiplier $\phi$.

## Family map

| Family | Response accepted by the implementation | Default link | $V(\mu)$ | Typed estimator |
|---|---|---|---|---|
| Gaussian | finite real values | identity | $1$ | `LinearRegression` or generic GLM |
| Binomial | finite values in $[0,1]$ | logit | $\mu(1-\mu)$ | generic GLM; standalone `LogisticRegression` is array-only |
| Poisson | finite non-negative values | log | $\mu$ | `PoissonRegression` |
| Negative binomial | finite non-negative values | log | $\mu+\alpha\mu^2$ | `NegativeBinomialRegression` |
| Gamma | finite strictly positive values | log | $\mu^2$ | `GammaRegression` |
| Inverse Gaussian | finite strictly positive values | log | $\mu^3$ | `InverseGaussianRegression` |
| Tweedie | finite non-negative values | log | $\mu^p$ | `TweedieRegression` |

The validation contract allows non-integer non-negative responses in count-like losses. Scientific use still requires a family appropriate to the data-generating process.

## Gaussian

The identity link gives $\mu=\eta$. Use [`LinearRegression`](linear-regression.md) for ordinary least squares and its full HC/HAC covariance set. The generic `GeneralizedLinearModel(family="gaussian")` instead uses the common iterative GLM interface.

## Binomial

The logit link is

$$
g(\mu)=\log\frac{\mu}{1-\mu},
\qquad
\mu=\frac{1}{1+\exp(-\eta)}.
$$

Use `LogisticRegression` when you want the dedicated classification API (`predict_proba`, thresholds, ROC, and classification metrics). It currently accepts arrays only. Use `GeneralizedLinearModel(family="binomial")` when Formula input is required; `predict` then returns the fitted mean probability.

## Poisson

The log link keeps $\mu>0$. A coefficient exponentiates to a mean ratio. Compare negative binomial when conditional variance materially exceeds the mean, and do not treat robust covariance as a replacement for a misspecified conditional distribution.

## Negative binomial

`NegativeBinomialRegression(alpha=1.0, ...)` exposes the positive overdispersion parameter $\alpha$. As $\alpha\to0$, the variance approaches the Poisson variance. Here `alpha` belongs to the response distribution; it is not a regularization parameter.

## Gamma

`GammaRegression(link="log", ...)` is the stable default. `link="inverse_power"` is also implemented:

$$
g(\mu)=\mu^{-1}.
$$

The inverse link requires extra care because the linear predictor must remain positive after numerical safeguarding.

## Inverse Gaussian

`InverseGaussianRegression` uses a log link and targets positive, strongly right-skewed outcomes whose variance grows approximately with the cube of the mean.

## Tweedie

`TweedieRegression(power=1.5, ...)` requires $1<p<2$ in the implemented loss. This compound Poisson–Gamma range can represent an exact mass at zero plus positive continuous values. `power` controls the variance, not regularization strength.

## Covariance by family, link, and covariance type

The three choices play different roles:

- the **family** supplies the response domain and variance function $V(\mu)$;
- the **link** maps the linear predictor $\eta$ to $\mu=g^{-1}(\eta)$ and therefore changes the derivatives used by the solver and covariance;
- `cov_type` chooses model-based or empirical-score uncertainty **after the same coefficients have been fitted**.

Changing `cov_type` does not refit the coefficients. Changing the family or link changes the objective and usually changes the fitted coefficients.

### Estimating equations

Let $\tilde x_i$ include the intercept when one is fitted, $k$ be the number of fitted parameters, and

$$
\eta_i=\tilde x_i^\top\hat\beta,
\qquad
\mu_i=g^{-1}(\eta_i).
$$

For an unweighted fit, statgpu forms the per-observation loss-gradient contribution

$$
s_i(\hat\beta)
=
\frac{\partial \ell_i}{\partial\eta_i}\tilde x_i,
\qquad
\bar J
=
\frac{1}{n}\sum_{i=1}^n s_i s_i^\top.
$$

The family and link enter the information matrix through both $V(\mu_i)$ and $d\mu_i/d\eta_i$. In standard GLM notation, its expected working weight is proportional to

$$
w_i
=
\frac{\left(d\mu_i/d\eta_i\right)^2}{V(\mu_i)},
\qquad
\bar H_F
=
\frac{1}{n}\tilde X^\top W\tilde X.
$$

The implementation uses the loss object's expected Fisher information for `nonrobust` when available and otherwise falls back to its Hessian. Robust covariance uses the observed Hessian $\bar H_O$. If an IRLS fit has `C>0`, the diagonal Ridge curvature $D_P$ is added to the bread; set `C=0` or select an unpenalized smooth solver when ordinary unpenalized likelihood inference is intended.

### Implemented covariance formulas

The following formulas show the unweighted convention used by the implementation:

$$
\widehat{\operatorname{Cov}}_{\text{nonrobust}}(\hat\beta)
=
\frac{\hat\phi}{n}
\left(\bar H_F+D_P\right)^{-1},
$$

$$
\widehat{\operatorname{Cov}}_{\text{HC0}}(\hat\beta)
=
\frac{1}{n}
\left(\bar H_O+D_P\right)^{-1}
\bar J
\left(\bar H_O+D_P\right)^{-1},
$$

$$
\widehat{\operatorname{Cov}}_{\text{HC1}}(\hat\beta)
=
\frac{n}{n-k}
\widehat{\operatorname{Cov}}_{\text{HC0}}(\hat\beta).
$$

| `cov_type` | Meaning | Use when |
|---|---|---|
| `nonrobust` | Model-based expected-information covariance scaled by $\hat\phi$ | The family, link, conditional variance, and independence assumptions are credible |
| `hc0` | Empirical score-outer-product sandwich | Conditional variance may be misspecified and the sample is not small |
| `hc1` | HC0 multiplied by $n/(n-k)$ | The same robust target with a degrees-of-freedom correction |

With analytic `sample_weight`, statgpu replaces $n$ in the average-scale normalization by $n_{\mathrm{eff}}=\sum_i w_i$, uses squared analytic weights in $\bar J$, and retains the raw observation count $n$ in the HC1 correction. Globally rescaling all weights therefore leaves fitted diagnostics and the HC1 correction invariant.

`hc0` and `hc1` protect against conditional variance misspecification, but they do not model clusters, repeated measurements, or serial dependence. `hc2`, `hc3`, and `hac` currently raise `NotImplementedError` for this ordinary GLM inference path. For a Gaussian identity-link model requiring those covariance types, use [`LinearRegression` and its inference API](linear-regression-inference.md).

### Supported family/link combinations

| Family | Link accepted here | $V(\mu)$ | `nonrobust` dispersion $\hat\phi$ | Robust covariance |
|---|---|---|---|---|
| Gaussian | identity | $1$ | $\mathrm{RSS}/(n-k)$ | `hc0`, `hc1` |
| Binomial | logit | $\mu(1-\mu)$ | `1.0` | `hc0`, `hc1` |
| Poisson | log | $\mu$ | `1.0` | `hc0`, `hc1` |
| Negative binomial | log | $\mu+\alpha\mu^2$ | `1.0` | `hc0`, `hc1` |
| Gamma | log; `inverse_power` in `GammaRegression` | $\mu^2$ | Pearson $\chi^2/(n-k)$ | `hc0`, `hc1` |
| Inverse Gaussian | log | $\mu^3$ | Pearson $\chi^2/(n-k)$ | `hc0`, `hc1` |
| Tweedie | log | $\mu^p$ | Pearson $\chi^2/(n-k)$ | `hc0`, `hc1` |

This table documents the combinations exposed by the current public estimators; it is not an arbitrary family/link registry. In particular, only Gamma exposes an alternative link on its typed ordinary wrapper. `NegativeBinomialRegression.alpha` and `TweedieRegression.power` also enter $V(\mu)$, so changing them changes both the fitted loss and its covariance.

## Complete API reference

### Common constructor

```python
GeneralizedLinearModel(
    family="gaussian",
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    C=1.0,
    device="auto",
    n_jobs=None,
    solver="auto",
    gpu_memory_cleanup=False,
    compute_inference=False,
    cov_type="nonrobust",
)
```

| Parameter | Contract |
|---|---|
| `family` | `gaussian`, `binomial`, `poisson`, `gamma`, `inverse_gaussian`, `negative_binomial`, or `tweedie` |
| `fit_intercept` | Adds an intercept unless Formula syntax explicitly removes it |
| `max_iter`, `tol` | Iteration budget and convergence tolerance |
| `C` | IRLS inverse ridge strength; $\lambda=1/(2C)$ when $C>0$, while `C=0` removes that ridge term |
| `device` | `auto`, `cpu`, `cuda` (CuPy), or `torch` (Torch CUDA) |
| `n_jobs` | Optional estimator-level parallelism hint |
| `solver` | `auto`, `irls`, `fista`, `newton`, or `lbfgs` |
| `gpu_memory_cleanup` | Releases cached GPU blocks after public work when enabled |
| `compute_inference` | Computes post-fit M-estimation inference when supported |
| `cov_type` | Ordinary GLM inference supports `nonrobust`, `hc0`, and `hc1` |

`solver="auto"` currently selects IRLS for this ordinary GLM surface. If an unpenalized likelihood fit is intended, set `C=0` on IRLS or choose the supported unpenalized smooth solver explicitly.

For algorithm mechanics and stopping behavior, use the independent [solver algorithms guide](../guides/solver-algorithms.md). For the exact loss × penalty dispatch and rejected combinations in penalized GLMs, use the [solver/penalty compatibility matrix](../guides/solver-penalty-matrix.md).

### Family-specific constructor inputs

| Estimator | Additional input |
|---|---|
| `GammaRegression` | `link="log"` or `"inverse_power"` |
| `NegativeBinomialRegression` | `alpha=1.0`, a finite positive dispersion |
| `TweedieRegression` | `power=1.5` with $1<p<2$ |
| other typed ordinary GLMs | no family-specific constructor input beyond the common controls |

### `fit`

```python
fit(X=None, y=None, sample_weight=None, formula=None, data=None)
```

Pass either `X` and `y`, or `formula` and a pandas `data` frame. `sample_weight` must be one-dimensional, finite, non-negative, and have a positive sum. Patsy missing-value removal happens before retained weights are aligned and validated.

### Outputs and methods

| Name | Meaning |
|---|---|
| `coef_`, `intercept_` | Coefficients on the linear-predictor scale |
| `n_iter_` | Solver iteration count where the selected solver reports it |
| `llf` / `loglikelihood` | Fitted pseudo-loglikelihood; additive constants can differ from R or statsmodels |
| `aic`, `bic` | Information criteria based on that pseudo-loglikelihood |
| `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | Current compatibility inference arrays |
| `_inference_result` | Structured result with covariance, distribution, solver, and backend metadata |
| `predict(X_new)` | Conditional mean on the response scale; Formula fits accept aligned DataFrames |
| `summary()` | Returns a formatted string with model metadata, coefficient inference, log-likelihood, AIC, and BIC; call `print(model.summary())` to display it and see the [runnable output example](generalized-linear-model.md#minimal-runnable-example) |

The dedicated `LogisticRegression` has additional classification methods and its own [model page](logistic-regression.md). Penalized GLMs have more controls—`penalty`, `alpha`, `l1_ratio`, `penalty_kwargs`, `cpu_solver`, `lipschitz_L`, inference mode, HAC lag, stopping rule, and LLA controls—documented in the [solver/penalty matrix](../guides/solver-penalty-matrix.md) and [inference API](../guides/inference-api.md).

## Documentation boundary

- Beginner model pages explain the problem, assumptions, one runnable path, interpretation, and common mistakes.
- Advanced references inventory public constructor/fit parameters, fitted attributes, formulas, backend differences, and failure conditions.
- Private fields used only to coordinate solvers are not a stable user contract. Underscore-prefixed inference arrays are documented because users already consume them, but structured inference results are safer for reusable reporting.

This split keeps first use readable without hiding information from advanced users.

## Source map

- Ordinary GLM orchestration: `statgpu/linear_model/_glm_base.py`
- Families and links: `statgpu/glm_core/_family.py`
- Typed wrappers: `statgpu/linear_model/wrappers/`
- Response validation: `statgpu/glm_core/_base.py` and `_validation.py`
- M-estimation inference: `statgpu/inference/_sandwich.py`

See [Formula interface](../guides/formula-interface.md) and [CPU/GPU acceleration internals](../guides/acceleration-internals.md) for cross-model behavior.
