# Penalized GLM Inference

> Language: English  
> Last updated: 2026-10-05  
> This page: statistical targets and supported coefficient-inference behavior  
> Switch: [Chinese](../../cn/guides/penalized-glm-inference.md)

## What this interface means

`PenalizedGeneralizedLinearModel` and typed penalized GLM wrappers expose coefficient inference through `compute_inference`, `inference_method`, and `cov_type`.

For the generic interface, the usual starting point is:

```python
inference_method="auto"
```

A successful inference-enabled fit distinguishes the method requested by the caller from the method resolved for the fitted model. Relevant fitted attributes include:

- `inference_requested_method_`;
- `inference_resolved_method_`;
- `inference_method_`;
- `inference_target_`;
- `penalty_conditioning_`;
- `penalty_selection_adjusted_`.

Where populated, these fields help identify what parameter the reported uncertainty refers to, especially after penalization or CV selection. The current `post_selection_ols` path leaves these public provenance fields as `None`. Its executed method and selected-set refit details are instead available through `model._inference_result.method` and `model._inference_result.metadata`; `params` holds the refit estimates. Do not interpret the missing public fields as disabled inference. See the [post-selection example](inference-modes.md#post_selection_ols).

## Support overview

| Loss / penalty | Supported inference | `auto` behavior |
|---|---|---|
| squared error + L2/no penalty | Gaussian classical/robust covariance | classical or Gaussian robust covariance according to `cov_type` |
| squared error + L1/ElasticNet | de-biased inference; `post_selection_ols` when explicitly selected | `debiased` |
| smooth non-Gaussian GLM + L2/no penalty | fixed-penalty M-estimation | `m_estimation` |
| squared error + L1/ElasticNet/SCAD/MCP | residual bootstrap when explicitly requested; not available for L2 | not selected automatically |
| squared error + SCAD/MCP | active-set oracle refit after a CPU parent fit when explicitly requested; see child-device caveat | not selected automatically |
| non-Gaussian SCAD/MCP | `oracle` accepts scalar GLM families but currently may change the refit model; see the limitation below | not selected automatically |
| non-Gaussian L1/ElasticNet | coefficient inference not implemented | unsupported request raises |
| group penalties | estimation only | inference request raises |
| `PenalizedCoxPHModel` / Cox branch of `PenalizedGLM_CV` | estimation only | inference request raises |

For sparse Gaussian methods such as `debiased` and `post_selection_ols`, see [Inference Modes](inference-modes.md) for interpretation and method choice.

## Fixed-penalty M-estimation

For a supported smooth non-Gaussian L2/no-penalty fit, statgpu treats the fitted coefficient as the solution of an estimating equation at the chosen penalty strength.

For positive L2 penalty the target is reported as:

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

For an unpenalized fit (`alpha=0` or the corresponding no-penalty configuration), the target is the ordinary unpenalized population coefficient.

With score contribution $\psi_i$, average Hessian $H$, L2 curvature $P''$, and average score outer product $J$, the HC0 covariance is

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

For HC1, multiply this covariance by $n/(n-k)$ when $n>k$, where $n$ is the observation count and $k$ counts fitted parameters, including the intercept when present. This is a finite-sample multiplier, not a correction for penalty selection. The current implementation omits that multiplier when $n\le k$; do not interpret such a result as a valid HC1 degrees-of-freedom correction. With analytic weights, this multiplier still uses the row count rather than the sum of weights.

`cov_type="nonrobust"` uses model-based penalized-information covariance.

The supported covariance choices for this non-Gaussian fixed-penalty path are:

- `nonrobust`;
- `hc0`;
- `hc1`.

HC2, HC3, and HAC are not available for this path and raise when requested.

## Analytic weights

Where the selected smooth GLM solver supports analytic `sample_weight`, the fit uses the same normalized weighted objective throughout optimization:

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

The corresponding M-estimation calculation uses the same relative-weight interpretation. For covariance calculations the weights can be represented on an equivalent mean-one scale,

$$
\widetilde w_i
=
\frac{n w_i}{\sum_j w_j},
\qquad
\sum_i \widetilde w_i=n.
$$

Therefore multiplying every positive analytic weight by the same constant leaves the statistical objective and inferential target unchanged, up to numerical solver tolerance.

These are **analytic/relative-importance weights**, not frequency weights. Multiplying all weights by a common factor does not claim that the sample size has increased through replication.

Losses that do not define the requested weighted fit reject genuine non-uniform weights rather than silently dropping them.

## Solver selection and weights

An explicit supported solver request remains authoritative. For smooth non-Gaussian L2/no-penalty rows, Newton and L-BFGS use the weighted objective above where analytic weights are supported.

With `solver="auto"`, statgpu follows the model's normal solver dispatch. The public request remains `auto`; inference describes the model that actually completed the supported fit rather than selecting an unrelated solver solely for inference.

Solver compatibility itself is documented in the [Solver × Penalty Matrix](solver-penalty-matrix.md).

## Backend and device behavior

Supported non-Gaussian M-estimation follows the backend/device used by the successful fit for its numerical covariance/statistic/p-value/CI work.

An explicit `device="cuda"` or `device="torch"` request is not silently replaced by NumPy inference. Small reporting arrays may be converted to NumPy after numerical inference is complete; that reporting boundary does not change where the numerical procedure ran.

Some inference methods have narrower backend support than the parent estimator. For example, the SCAD/MCP oracle interface rejects an already-executed GPU fit. Its child refit currently defaults to `device="auto"`, however, so a CPU parent does not guarantee a CPU child on a GPU-equipped system. See the oracle limitation below; an explicit separate refit lets you control the device as well as the statistical model.

## Residual bootstrap scope

`inference_method="bootstrap"` is a Gaussian penalized-model residual bootstrap, not a universal GLM bootstrap.

For each draw, statgpu:

1. computes fitted values and residuals from the fitted Gaussian model;
2. resamples residuals with replacement;
3. forms a bootstrap response;
4. refits the same penalized model with the same tuning configuration; and
5. summarizes the bootstrap coefficient distribution.

Set `bootstrap_random_state` for reproducible resamples. `n_bootstrap` controls the number of refits and must be at least 2.

The supported residual-bootstrap path requires:

- `sample_weight=None`;
- `cov_type="nonrobust"`.

Weighted residual bootstrap, robust/HC or HAC/block bootstrap, non-Gaussian bootstrap, and Cox bootstrap are not supplied by this interface.

The resulting intervals describe a fixed-design, fixed-tuning residual-bootstrap procedure. They are not general selective-inference intervals and do not automatically account for tuning or variable-selection uncertainty.

The reported intervals are the 2.5th and 97.5th percentiles of the refitted
coefficients, and `bse` is their sample standard deviation (`ddof=1`). The
reported `pvalues` are twice the smaller fraction of bootstrap coefficients
at or below zero and at or above zero, capped at one. They can be exactly
zero and use no plus-one correction. The displayed `z` is the original fitted
coefficient divided by its bootstrap SE; the p-values are not normal-tail
probabilities computed from that statistic. These sign proportions are not
a null-imposed resampling test and have no general finite-sample calibration
for a biased penalized estimator. More draws reduce simulation error but do
not remove shrinkage bias or establish valid hypothesis tests.

## SCAD/MCP oracle inference

`inference_method="oracle"` is explicit because it conditions on the active set selected by the penalized fit. `auto` does not silently choose that interpretation.

The intended procedure refits the selected active set without the original
non-convex penalty. Even a correct unpenalized refit does not automatically
adjust for selecting variables on the same data. Inactive coefficients have
unavailable (`NaN`) uncertainty; if no feature is selected, uncertainty is
unavailable for all parameters, including the intercept.

### Current non-Gaussian oracle limitation

The interface accepts `squared_error`, `logistic`, `poisson`, `gamma`,
`inverse_gaussian`, `negative_binomial`, and `tweedie` with SCAD/MCP. Currently,
the non-Gaussian refit can silently use constructor defaults instead of the
original model configuration: logistic uses `C=1` rather than an unpenalized
refit, negative-binomial dispersion can reset to `alpha=1`, Gamma's link to
`"log"`, and Tweedie's power to `1.5`. Child solver and device also default to
`"auto"`; for example, Poisson then uses the default ridge-penalized IRLS
fit rather than the intended unpenalized Newton refit. A successful result labeled `oracle` therefore does not establish
that the requested statistical target was preserved. Do not use these
non-Gaussian oracle coefficient tables for inference.

For a diagnostic refit, fit the selection model with `compute_inference=False`,
then explicitly refit its selected columns with the desired family, link,
dispersion/power, no penalty, and device. For logistic regression, the
standalone wrapper's `C=0` selects its unregularized fit:

<!-- inference-example: explicit-logistic-diagnostic -->
```python
import numpy as np
from statgpu.linear_model import PenalizedLogisticRegression, LogisticRegression

rng = np.random.default_rng(48)
X = rng.normal(size=(200, 2))
probability = 1 / (1 + np.exp(-(0.6 + X @ np.array([2.0, -1.0]))))
y = rng.binomial(1, probability)
selection = PenalizedLogisticRegression(
    penalty="scad", alpha=0.01, solver="fista", device="cpu",
    compute_inference=False, max_iter=10000, tol=1e-10,
).fit(X, y)
active = np.flatnonzero(np.abs(selection.coef_) > 1e-10)
if active.size == 0:
    raise ValueError("No feature was selected; specify a model with only an intercept")
refit = LogisticRegression(
    C=0, device="cpu", compute_inference=True, max_iter=10000, tol=1e-10,
).fit(X[:, active], y)
print(active.tolist())
print(np.round(np.r_[refit.intercept_, refit.coef_], 6))
```

This selects `[0, 1]` and prints `[0.626286, 2.295259, -1.037398]`.
Prediction from `selection` remains the original penalized model; `refit` is
a separate diagnostic model. Its ordinary intervals still ignore data-driven
selection. For confirmatory inference, use independent data for selection and
testing, or a method that explicitly accounts for selection under applicable
assumptions. Do not treat this manual refit as a selective-inference repair.

## Cross-validation

`PenalizedGLM_CV` separates tuning from coefficient inference:

```text
fold/path/grid fits
    -> select alpha
    -> refit selected model on all observations
    -> run inference once on the final refit
```

A successful inference-enabled CV fit reports selection conditioning such as:

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

The reported standard errors, p-values, and confidence intervals are therefore conditional on the CV-selected penalty. They do not automatically adjust for tuning-selection uncertainty.

For residual bootstrap, resampling starts only after CV has selected the tuning parameter; the candidate-selection process itself is not bootstrapped.

The Cox branch remains estimation-only.

See [Cross-Validation](cross-validation.md) for the general selection/refit contract.

## Example

This complete CPU example generates a finite design, nonnegative integer
counts, and finite positive analytic weights.

<!-- inference-example: weighted-poisson-m-estimation -->
```python
import numpy as np
from statgpu.linear_model import PenalizedPoissonRegression

rng = np.random.default_rng(72)
X = rng.normal(size=(100, 2))
y = rng.poisson(np.exp(0.2 + X @ np.array([0.3, -0.2])))
w = np.linspace(0.5, 1.5, len(y))
model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    solver="lbfgs",
    device="cpu",
)
model.fit(X, y, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
inference = model._inference_result.to_dict()
print(inference["params"])
print(inference["conf_int"])
```

The printed method is `m_estimation`; the positive L2 penalty means its target
is `penalized_estimating_equation`. HC0 does not undo the fitted shrinkage or
provide selection-adjusted intervals.

The generic penalized GLM and typed non-Gaussian wrappers do not currently
provide `summary()`. Use the result's `params`, `bse`, `pvalues`, and `conf_int`,
or `to_dict()` as above. `to_dataframe()` additionally requires pandas.

For sparse non-Gaussian L1/ElasticNet fits, coefficient inference is not currently provided; use `compute_inference=False` rather than assuming that a Gaussian debiasing/bootstrap procedure applies to a different family.

## Related documentation

- [Inference Modes](inference-modes.md) — method selection and interpretation
- [Cross-Validation](cross-validation.md) — tuning and final refit
- [Solver × Penalty Matrix](solver-penalty-matrix.md) — solver compatibility
- [Device and GPU Memory](device-and-memory.md) — device semantics

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
