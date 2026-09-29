# Penalized GLM Inference

> Language: English  
> Last updated: 2026-09-17  
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

These fields help identify what parameter the reported uncertainty refers to, especially after penalization or CV selection.

## Support overview

| Loss / penalty | Supported inference | `auto` behavior |
|---|---|---|
| squared error + L2/no penalty | Gaussian classical/robust covariance | classical or Gaussian robust covariance according to `cov_type` |
| squared error + L1/ElasticNet | de-biased inference; `post_selection_ols` when explicitly selected | `debiased` |
| smooth non-Gaussian GLM + L2/no penalty | fixed-penalty M-estimation | `m_estimation` |
| supported Gaussian penalized models | residual bootstrap when explicitly requested | not selected automatically |
| supported SCAD/MCP scalar GLM models | active-set oracle refit when explicitly requested | not selected automatically |
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

With score contribution $\psi_i$, average Hessian $H$, L2 curvature $P''$, and average score outer product $J$, the HC0/HC1 covariance has the form

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

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

Some inference methods have narrower backend support than the parent estimator. For example, the current SCAD/MCP oracle refit is CPU-only; requesting that method after a GPU fit raises instead of presenting the result as backend-native oracle inference.

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

## SCAD/MCP oracle inference

`inference_method="oracle"` is explicit because it conditions on the active set selected by the penalized fit. `auto` does not silently choose that interpretation.

Where supported, the procedure refits the selected active set without the original non-convex penalty and reports uncertainty for that conditional refit. Check the model/backend support before requesting it.

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

```python
from statgpu.linear_model import PenalizedPoissonRegression

model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    device="cpu",
)
model.fit(X, y, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
print(model.summary())
```

For sparse non-Gaussian L1/ElasticNet fits, coefficient inference is not currently provided; use `compute_inference=False` rather than assuming that a Gaussian debiasing/bootstrap procedure applies to a different family.

## Related documentation

- [Inference Modes](inference-modes.md) — method selection and interpretation
- [Cross-Validation](cross-validation.md) — tuning and final refit
- [Solver × Penalty Matrix](solver-penalty-matrix.md) — solver compatibility
- [Device and GPU Memory](device-and-memory.md) — device semantics

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
