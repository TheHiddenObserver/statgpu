# GLM families and complete API

> Language: English
> Last updated: 2026-09-03
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
| `summary()` | Returns a formatted string; call `print(model.summary())` to display it |

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
