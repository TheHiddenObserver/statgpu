# Penalized GLM inference

> Language: English  
> Status: targeted for 0.2.6; the currently published release remains 0.2.5.  
> Switch: [Chinese](../../cn/guides/penalized-glm-inference.md)

## What this interface means

`PenalizedGeneralizedLinearModel` and the typed penalized GLM wrappers expose coefficient inference through `compute_inference`, `inference_method`, and `cov_type`. The interface distinguishes the **method requested by the caller**, the **method resolved by statgpu**, and the **method actually reported by the result**.

For the generic and typed penalized-GLM surfaces, the recommended default is:

```python
inference_method="auto"
```

A successful inference-enabled fit publishes:

- `inference_requested_method_`;
- `inference_resolved_method_`;
- `inference_method_`;
- `inference_target_`;
- `penalty_conditioning_`;
- `penalty_selection_adjusted_`.

The same provenance is recorded in `_inference_result.metadata` together with the numerical backend/device where applicable.

## Supported methods

| Loss / penalty | Supported inference | `auto` resolution |
|---|---|---|
| squared error + L2/no penalty | existing Gaussian classical/robust covariance | `classical` for `cov_type="nonrobust"`, otherwise the maintained Gaussian sandwich path |
| squared error + L1/ElasticNet | debiased inference; existing `post_selection_ols` contract | `debiased` |
| smooth non-Gaussian GLM + L2/no penalty | fixed-penalty M-estimation | `m_estimation` |
| Gaussian L1/ElasticNet/SCAD/MCP | unweighted residual bootstrap on NumPy/CuPy/Torch when explicitly requested | not selected automatically |
| SCAD/MCP scalar GLM families | active-set oracle refit when explicitly requested | not selected automatically |
| non-Gaussian L1/ElasticNet | not implemented | fails closed |
| group penalties | estimation-only | fails closed |
| `PenalizedCoxPHModel` / Cox branch of `PenalizedGLM_CV` | estimation-only | fails closed |

The historical `cpu_ols` and `gpu_ols` spellings keep the existing one-cycle migration to `post_selection_ols` for sparse Gaussian models. They do not select execution hardware.

For L2/no-penalty models, explicit `inference_method="debiased"` is accepted temporarily as a deprecated compatibility spelling. It warns and resolves to the method that was actually used for that model; L2 inference is not debiased-Lasso inference.

## Fixed-penalty M-estimation

For a supported non-Gaussian smooth L2/no-penalty fit, statgpu treats the fitted coefficient as the solution of a penalized estimating equation. For positive L2 penalty the inferential target is therefore reported as:

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

For an unpenalized (`alpha=0`) fit, the target is the ordinary unpenalized population coefficient.

The numerical engine uses average-loss scaling. With score contribution `psi_i`, average Hessian `H`, and L2 curvature `P''`, HC0/HC1 covariance has the form

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1} J (H+P'')^{-1}/n,
$$

where `J` is the average score outer product under statgpu's analytic-weight convention and `n` is the original observation count. `cov_type="nonrobust"` uses the corresponding model-based penalized-information covariance, also on the original-observation average scale.

Supported covariance choices for this non-Gaussian path are:

- `nonrobust`;
- `hc0`;
- `hc1`.

HC2, HC3, and HAC are not implemented for penalized non-Gaussian M-estimation and fail visibly.

## Analytic weights

Analytic weights are supported by the non-Gaussian L2 M-estimation covariance path. The numerical inference follows the backend/device that actually executed the fit.

The maintained smooth GLM solvers apply non-uniform analytic weights to the **same normalized average-loss objective throughout the solve**: objective value, gradient, Hessian where applicable, line-search trial evaluation, and accepted-point derivatives all use

$$
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

Therefore analytic weights describe **relative observation importance**, not replicated frequency counts. Before M-estimation covariance is computed, statgpu uses the equivalent mean-one weights

$$
\widetilde w_i
=
\frac{n w_i}{\sum_j w_j},
\qquad
\sum_i \widetilde w_i=n.
$$

The Hessian is unchanged by this normalization. For HC0/HC1, `J` uses the same normalized analytic weights; for `nonrobust`, model-based information and dispersion use the same mean-one convention. Consequently, multiplying every analytic weight by any positive constant leaves the fitted parameters, standard errors, test statistics, p-values, and confidence intervals unchanged. Positive constant weights reduce to the same inference problem as omitted weights. This is intentionally different from a frequency-weight interpretation in which multiplying all counts would assert a larger replicated sample.

Floating-point vectors that satisfy the historical effectively-uniform rule retain the established unweighted-equivalent path.

With public `solver="auto"`, weighted smooth non-Gaussian L2/no-penalty fits use the same canonical solver-dispatch table as their unweighted counterparts. Applicable logistic/Poisson rows resolve to backend-native Newton while the public solver request remains `auto`.

Explicit solver requests remain authoritative and are never silently replaced. Losses whose statistical contract does not define weighting (for example Cox) continue to reject genuine non-uniform weights rather than silently dropping them.

## Backend and device provenance

Supported non-Gaussian M-estimation performs covariance/statistic/p-value/CI work on the backend selected by the fit:

- NumPy on CPU;
- CuPy on the concrete CUDA device selected by the fit;
- Torch on the concrete selected Torch device.

An explicit `device="cuda"` or `device="torch"` request never silently substitutes NumPy inference. Small reporting arrays may be copied to NumPy only after numerical inference is complete. Result metadata records `numerical_backend`, `numerical_device`, `reporting_backend`, and the reporting boundary.

## Residual bootstrap scope

`inference_method="bootstrap"` is deliberately **not** a universal GLM bootstrap. It is available for supported Gaussian penalized models and keeps the fitted design and tuning configuration fixed.

For each bootstrap draw, statgpu:

1. computes fitted values and residuals from the fitted Gaussian model;
2. resamples the residuals with replacement;
3. forms a bootstrap response `y_star = y_hat + residual_star`;
4. refits the same penalized model with the same `alpha`, penalty family, ElasticNet mixing/penalty options, intercept convention, solver/stopping controls, and SCAD/MCP LLA controls; and
5. summarizes the resulting coefficient distribution with bootstrap standard errors, sign-based two-sided p-values, and percentile confidence intervals.

Set `bootstrap_random_state` when you need reproducible resamples. `n_bootstrap` controls the number of refits and must be at least 2.

The bootstrap refits follow the backend and concrete device of the successful parent fit. A CPU fit refits on NumPy; a CuPy or Torch CUDA fit keeps the bootstrap responses and numerical refits on the same GPU device. GPU execution changes **where** the refits run, not the statistical procedure. Final reporting arrays use the standard NumPy result boundary.

This method requires `sample_weight=None` and `cov_type="nonrobust"`. Weighted residual bootstrap, robust/HC or HAC/block bootstrap, non-Gaussian bootstrap, and Cox bootstrap are not defined by this interface and fail closed instead of guessing a resampling scheme.

The resulting intervals describe the sampling variation of the penalized estimator under this fixed-design, fixed-tuning residual-bootstrap procedure. They are not general selective-inference confidence intervals and do not correct for variable-selection uncertainty.

## SCAD/MCP oracle boundary

`inference_method="oracle"` is explicit because it conditions on the selected active set. `auto` never silently selects it. The current oracle implementation uses a CPU active-set refit, so an inference-enabled fit that actually executed on CuPy/Torch fails visibly instead of masquerading as backend-native oracle inference.

## Cross-validation

`PenalizedGLM_CV` adds these controls:

```python
PenalizedGLM_CV(
    ...,
    compute_inference=False,
    inference_method="auto",
    cov_type="nonrobust",
    hac_maxlags=None,
)
```

Fold/path/grid fits remain estimation-only. If inference is requested, statgpu first selects `alpha`, then performs inference exactly once on the full-data selected-penalty refit. Successful CV inference reports:

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

Therefore the reported standard errors, p-values, and confidence intervals are **conditional on the CV-selected penalty**. They do not adjust for tuning-selection uncertainty. For residual bootstrap, resampling begins only after CV has selected `alpha`; folds and candidate fits are not bootstrapped.

The Cox branch remains estimation-only.

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
print(model.inference_method_)            # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
print(model._bse)
print(model._pvalues)
```

For sparse non-Gaussian L1/ElasticNet fits, coefficient inference is not currently provided; use `compute_inference=False` rather than expecting a Gaussian debiasing or bootstrap rule to be applied to a different family.

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
