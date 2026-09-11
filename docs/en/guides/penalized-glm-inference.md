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
| Gaussian L1/ElasticNet/SCAD/MCP | unweighted CPU residual bootstrap when explicitly requested | not selected automatically |
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
(H+P'')^{-1} J (H+P'')^{-1}/n_{\mathrm{eff}},
$$

where `J` is the average score outer product under statgpu's analytic-weight convention. `cov_type="nonrobust"` uses the maintained model-based penalized-information covariance.

Supported covariance choices for this non-Gaussian path are:

- `nonrobust`;
- `hc0`;
- `hc1`.

HC2, HC3, and HAC are not implemented for penalized non-Gaussian M-estimation and fail visibly.

## Analytic weights

Analytic weights are supported by the non-Gaussian L2 M-estimation covariance path. The numerical inference follows the backend/device that actually executed the fit.

One solver-policy boundary is important. The current general `solver="auto"` benchmark table can choose Newton for smooth GLMs, while Newton does not accept non-uniform analytic weights. For **inference-enabled, weighted, non-Gaussian L2/no-penalty fits only**, the inference contract therefore makes a fit-local execution choice to the existing backend-native FISTA path. The public request remains `solver="auto"`; this avoids changing the estimation-only benchmark policy in this targeted repair.

Explicit solver requests remain authoritative and are never silently replaced.

## Backend and device provenance

Supported non-Gaussian M-estimation performs covariance/statistic/p-value/CI work on the backend selected by the fit:

- NumPy on CPU;
- CuPy on the concrete CUDA device selected by the fit;
- Torch on the concrete selected Torch device.

An explicit `device="cuda"` or `device="torch"` request never silently substitutes NumPy inference. Small reporting arrays may be copied to NumPy only after numerical inference is complete. Result metadata records `numerical_backend`, `numerical_device`, `reporting_backend`, and the reporting boundary.

## Residual bootstrap scope

`inference_method="bootstrap"` in this repair is deliberately **not** a universal GLM bootstrap. It is an unweighted Gaussian residual bootstrap with:

- fixed design;
- the same fixed `alpha`;
- the same penalty family;
- the same ElasticNet `l1_ratio` / penalty kwargs where applicable;
- the same intercept semantics;
- penalized refitting inside each bootstrap sample;
- no CV/tuning rerun inside bootstrap.

The maintained implementation is CPU-executed in this repair and requires `cov_type="nonrobust"`. Weighted residual bootstrap, robust/HAC bootstrap semantics, and family-aware non-Gaussian bootstrap require separate statistical designs and are rejected rather than guessed.

The resulting intervals are heuristic penalized-estimator bootstrap intervals, not selective-inference coverage guarantees.

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

Therefore the reported standard errors, p-values, and confidence intervals are **conditional on the CV-selected penalty**. They do not adjust for tuning-selection uncertainty.

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

For sparse non-Gaussian L1/ElasticNet fits, set `compute_inference=False`; this repair intentionally does not invent a debiasing method for those rows.

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
