# Lasso inference

> Language: English  
> Last updated: 2026-09-10  
> Model guide: [Lasso](lasso.md)  
> Switch: [简体中文](../../cn/models/lasso-inference.md)

> **Version note:** the current published release is **0.2.5**. The public `nodewise_alpha` control and the new automatic node-wise rule described here are already implemented on current `master` and are targeted for **0.2.6**; published 0.2.5 does not yet expose this public interface or new default rule.

This page is the statistical-inference companion to the learner-first [Lasso guide](lasso.md). The main model page answers “when should I use Lasso?”; this page explains what the post-fit inference objects mean, how statgpu constructs the node-wise approximation, and what the software does **not** claim.

## Why raw Lasso coefficients need a different inferential construction

For the centered Gaussian linear model, Lasso solves

$$
\hat\beta
=\arg\min_\beta
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1
\right\}.
$$

The L1 penalty creates sparsity by shrinking coefficients. Its KKT relation is schematically

$$
\frac{X^\top(y-X\hat\beta)}{n}=\alpha\hat\kappa,
\qquad \hat\kappa_j\in\partial|\hat\beta_j|,
$$

so the score of the unpenalized loss is deliberately not zero. The resulting regularization bias is the primary reason ordinary fixed-model Wald inference cannot simply be attached to raw Lasso coefficients.

Selection uncertainty is a second issue. If the active set selected by Lasso is refitted with OLS on the same data, ordinary OLS intervals generally do not account for that selection step. statgpu therefore presents `post_selection_ols` as a diagnostic rather than a general selective-inference procedure.

## Choose the inference path by the claim you need

| `inference_method` | What statgpu computes | Interpretation | Main limitation |
|---|---|---|---|
| `debiased` | de-biased/de-sparsified coefficient estimator, SE, z, p-values, marginal CI | coefficient-wise high-dimensional inference under de-biasing assumptions | depends on sparsity/design/noise and tuning conditions |
| `post_selection_ols` | OLS/WLS refit on the selected active set using the fit-resolved backend | post-selection engineering/statistical diagnostic | not a general selective-inference confidence procedure |
| `bootstrap` | residual-bootstrap refits | resampling diagnostic | expensive and not a universal correction for model selection |

`post_selection_ols` is hardware-neutral. Deprecated `cpu_ols` and `gpu_ols` names both map to it; they do not select CPU versus GPU. `device="cpu"`, `device="cuda"`, and `device="torch"` control execution, while only genuine `device="auto"` may choose among available backends.

For prediction or feature selection only, set `compute_inference=False`.

## Minimal de-biased example

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(700, 10))
beta = np.zeros(10)
beta[[1, 4, 8]] = [1.5, -1.1, 0.7]
y = 0.4 + X @ beta + rng.normal(scale=1.0, size=X.shape[0])

model = Lasso(
    alpha=0.05,
    nodewise_alpha=None,
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model.nodewise_alpha_)
print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

`coef_` remains the penalized prediction coefficient vector. `_params` is the inference/reporting parameter vector and can differ from `coef_` because it contains the de-biased estimate.

## What de-biasing changes

Let the fitted residual be $r=y-b-X\hat\beta$. statgpu forms the one-step correction

$$
\hat\theta^{\mathrm{db}}
=\hat\beta+\frac{1}{n}MX^\top r,
$$

where $M$ is a data-dependent approximate inverse/precision matrix for the design Gram matrix. Writing $\widehat\Sigma=X^\top X/n$ gives the familiar decomposition

$$
\hat\theta^{\mathrm{db}}-\beta^0
=
\frac{1}{n}MX^\top\varepsilon
+
(I-M\widehat\Sigma)(\hat\beta-\beta^0).
$$

The first term is the leading noise term. The second is the remainder. The method is useful when the node-wise construction makes $M\widehat\Sigma$ sufficiently close to identity and the high-dimensional assumptions make the remainder negligible on the inferential scale.

## Canonical centered/weighted working design

The maintained sparse-Gaussian path first constructs one canonical centered/weighted average-loss working design $X_w$. With analytic `sample_weight`, this uses the same weighted-centering and row-rescaling convention across NumPy, CuPy, and Torch. Global positive rescaling of all weights does not change the intended statistical problem, and all-one weights agree with the unweighted definition.

Node-wise tuning is based only on this design-side problem. It does not use the response residual scale.

## Standardization and node-wise Lasso

Let

$$
d_j^2=\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad D=\operatorname{diag}(d_1,\ldots,d_p),
\qquad Z=X_wD^{-1}.
$$

For each feature $j$, statgpu solves the standardized node-wise Lasso

$$
\hat\gamma_j
=\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{
\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\}.
$$

The theoretical order is of the familiar form

$$
\lambda_j\asymp\sqrt{\frac{\log p}{n}},
$$

up to constants and conventions. statgpu's automatic rule is a concrete library default, not a claim that one theorem uniquely mandates this exact constant.

## Public `nodewise_alpha` contract

The main Lasso `alpha` and the inference-only `nodewise_alpha` are separate controls.

- `alpha` controls the penalized prediction/selection fit.
- `nodewise_alpha` controls only the node-wise precision regressions used by `debiased` inference.
- An explicit finite positive real scalar is authoritative.
- `None` selects the library default.
- `bool`, complex/non-scalar values, NaN/inf, zero, and negative values are rejected.
- `get_params`, `set_params`, and sklearn clone preserve the requested constructor value; changing it invalidates stale inference state.

For $p\ge2$, the automatic value is

$$
\boxed{
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}
}
$$

where

$$
n_{\mathrm{nw}}=n
$$

without analytic weights and, for non-uniform analytic weights,

$$
n_{\mathrm{nw}}
=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}
$$

is the Kish-style effective sample size.

The rule is deliberately **response-scale independent**. The superseded internal implementation used a response-residual-scaled quantity of the form $\hat\sigma_y\sqrt{2\log(p)/n}$. That historical rule is not the current contract and is not exposed as a legacy public mode. See the [node-wise tuning migration guide](../guides/nodewise-alpha-migration.md).

For `p=1`, there is no nuisance node-wise regression. statgpu uses analytic univariate precision and does not consume the requested `nodewise_alpha`; consequently `nodewise_alpha_` remains `None`.

## Paper-style normalizer and precision back-transform

For a standardized node-wise residual

$$
r_j=Z_j-Z_{-j}\hat\gamma_j,
$$

statgpu uses

$$
\hat\tau_j^2
=\frac{\lVert r_j\rVert_2^2}{n}
+\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1.
$$

The standardized precision row has diagonal $1/\hat\tau_j^2$ and off-diagonal entries $-\hat\gamma_j/\hat\tau_j^2$. After all rows are constructed, statgpu returns to the working-feature scale through

$$
M=D^{-1}\Theta_ZD^{-1}.
$$

This is the matrix used by the de-biasing and variance calculations.

## Numerical publication gate

A solver's internal stopping condition is not by itself sufficient to publish inference. The maintained contract uses node-wise FISTA with `coef_delta` stopping, internal tolerance `1e-8`, and up to 3000 iterations, then recomputes an **independent full KKT residual**. Publication requires the KKT residual to be no larger than `1e-5` and also requires finite/nondegenerate scales, normalizers, precision state, and reporting arrays.

These values are internal numerical settings, not additional public tuning parameters. A KKT pass establishes numerical consistency with the declared node-wise optimization problem; it does not prove the statistical sparsity/design assumptions.

## Backend contract

NumPy, CuPy, and Torch solve the same standardized statistical problem. The maintained CuPy/Torch path batches the Gram subproblems for execution efficiency, but batching does not change the definition of $\lambda_{\mathrm{nw}}$, $\hat\gamma_j$, $\hat\tau_j^2$, or $M$.

For explicit CUDA/Torch inference, the numerical node-wise solve, KKT validation, back-transform, and maintained simultaneous inference stay on the concrete selected GPU backend/device. The precision cache may transfer chunks of backend-resident design data to host for **cache identity hashing**, but that is not a CPU numerical fallback. Small reporting arrays are converted to NumPy only after the numerical-inference boundary. `_inference_result.metadata` records `numerical_backend`, `numerical_device`, node-wise tuning provenance, KKT diagnostics, and cache provenance.

## `LassoCV`: final-refit inference only

`LassoCV(nodewise_alpha=...)` treats this parameter as final-full-data-refit inference configuration only. Changing `nodewise_alpha` must not change the main alpha grid, fold MSEs, selected `alpha_`, or penalized final-refit coefficients. The outer `nodewise_alpha_` mirrors the final estimator after successful multi-feature de-biased inference.

Inference after CV is still conditional on the selected main tuning value; the current implementation does not add a separate correction for CV tuning uncertainty.

## Standard errors, z statistics, and marginal intervals

With

$$
V=M\widehat\Sigma M^\top,
$$

statgpu forms coefficient standard errors from the fitted residual scale and $V_{jj}/n$, then uses a standard-normal reference for the maintained de-biased marginal z statistics, p-values, and confidence intervals.

`_conf_int` is **marginal**. Reporting many 95% marginal intervals does not imply 95% simultaneous coverage over the whole target family.

When an intercept is fitted, prediction and inference intentionally have different ownership. Public `coef_` / `intercept_` remain the penalized prediction fit. Debiased reporting uses corrected slopes and the coherent original-coordinate intercept

$$
\hat\theta_0^{\mathrm{db}}
=\bar y_w-\bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

## Simultaneous inference

Marginal inference treats one parameter at a time. Simultaneous inference instead asks whether the intervals for an entire target family can cover their true values **at the same time**. Let $\mathcal J$ denote the coordinates to be covered jointly, with de-biased estimates $\hat\theta_j^{\mathrm{db}}$ and estimated standard errors $\widehat{\mathrm{se}}_j$. The ideal max statistic is

$$
\boxed{
T
=
\max_{j\in\mathcal J}
\left|
\frac{\hat\theta_j^{\mathrm{db}}-\theta_j^0}
{\widehat{\mathrm{se}}_j}
\right|
}.
$$

If the $1-\alpha_{\mathrm{sim}}$ quantile $c_{1-\alpha_{\mathrm{sim}}}$ of $T$ were known, one common critical value would give

$$
CI_j^{\mathrm{sim}}
=
\left[
\hat\theta_j^{\mathrm{db}}-c_{1-\alpha_{\mathrm{sim}}}\widehat{\mathrm{se}}_j,
\;
\hat\theta_j^{\mathrm{db}}+c_{1-\alpha_{\mathrm{sim}}}\widehat{\mathrm{se}}_j
\right],
\qquad j\in\mathcal J,
$$

with the target approximation

$$
\Pr\!\left(
\theta_j^0\in CI_j^{\mathrm{sim}}
\;\text{for all }j\in\mathcal J
\right)
\approx 1-\alpha_{\mathrm{sim}}.
$$

This is the key distinction from reporting many 95% marginal intervals: the simultaneous procedure needs the distribution of the **largest standardized error over the whole target family**, not the quantile of one standard-normal coordinate.

### How the Gaussian multiplier bootstrap approximates the max distribution

The de-biasing expansion above gives the leading stochastic term

$$
\hat\theta^{\mathrm{db}}-\theta^0
\approx
\frac{1}{n}MX^\top\varepsilon.
$$

statgpu replaces the unknown errors by fitted residuals $\hat r_i$ and, for bootstrap draw $b$, samples independent Gaussian multipliers

$$
\xi_i^{(b)}\overset{\mathrm{iid}}{\sim}N(0,1),
\qquad i=1,\ldots,n.
$$

For the slope coordinates, the maintained implementation corresponds to the multiplier perturbation

$$
\boxed{
S^{*(b)}
=
\frac{1}{n}
MX^\top
\bigl(\xi^{(b)}\odot\hat r\bigr)
},
$$

or coordinate-wise,

$$
S_j^{*(b)}
=
\frac{1}{n}
\sum_{i=1}^{n}
\xi_i^{(b)}\hat r_i(Mx_i)_j.
$$

It then standardizes by the already computed marginal standard errors,

$$
Z_j^{*(b)}
=
\frac{S_j^{*(b)}}{\widehat{\mathrm{se}}_j}.
$$

Within a given bootstrap draw, **every coordinate uses the same multiplier vector** $\xi^{(b)}$. That shared perturbation preserves the joint dependence induced by the observations, the design, and the common approximate precision matrix $M$ rather than treating the coordinates as independent bootstrap problems.

The bootstrap max statistic is then

$$
\boxed{
T^{*(b)}
=
\max_{j\in\mathcal J}|Z_j^{*(b)}|
}.
$$

After $B=$ `simultaneous_n_bootstrap` draws, statgpu obtains

$$
T^{*(1)},\ldots,T^{*(B)}
$$

and uses their empirical $1-\alpha_{\mathrm{sim}}$ quantile

$$
\boxed{
\hat c_{1-\alpha_{\mathrm{sim}}}
=
\widehat Q_{1-\alpha_{\mathrm{sim}}}
\left(T^{*(1)},\ldots,T^{*(B)}\right)
}.
$$

The reported simultaneous intervals are therefore

$$
\boxed{
CI_j^{\mathrm{sim}}
=
\hat\theta_j^{\mathrm{db}}
\pm
\hat c_{1-\alpha_{\mathrm{sim}}}\widehat{\mathrm{se}}_j,
\qquad j\in\mathcal J.
}
$$

Thus max-|Z| calibration is not merely a mechanical widening of marginal intervals, nor does it bootstrap each coordinate independently. It directly approximates the distribution of the largest standardized fluctuation over the target family while retaining cross-coordinate dependence through the shared multipliers. The resulting common critical value is typically more conservative than a single-coordinate marginal critical value because it targets joint coverage.

### Including the intercept in the same calibration problem

By default, the target family contains slope coordinates. With `simultaneous_include_intercept=True`, statgpu constructs original-coordinate intercept influence weights $\hat h_i$ from the same centered/weighted fitted problem. For bootstrap draw $b$, the intercept perturbation is

$$
S_0^{*(b)}
=
\sum_{i=1}^{n}
\xi_i^{(b)}\hat r_i\hat h_i,
\qquad
Z_0^{*(b)}
=
\frac{S_0^{*(b)}}{\widehat{\mathrm{se}}_0}.
$$

The max statistic then genuinely becomes

$$
\boxed{
T^{*(b)}
=
\max\left\{
|Z_0^{*(b)}|,
\max_{j=1,\ldots,p}|Z_j^{*(b)}|
\right\}.
}
$$

The intercept therefore participates in the **same max-|Z| calibration**; it is not appended afterward using a feature-only critical value.

Set the corresponding controls as follows:

```python
model = Lasso(
    alpha=0.05,
    nodewise_alpha=None,
    inference_method="debiased",
    compute_inference=True,
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=2000,
    simultaneous_random_state=123,
    simultaneous_include_intercept=True,
).fit(X, y)

print(model._conf_int)
print(model._conf_int_simultaneous)
print(model._simultaneous_critical_value)
```

`_conf_int` remains the marginal interval array; `_conf_int_simultaneous` stores intervals formed with the common max-|Z| critical value; and `_simultaneous_critical_value` is the estimated $\hat c_{1-\alpha_{\mathrm{sim}}}$ above. Maintained centered CuPy/Torch simultaneous inference also records its numerical backend/device provenance.

The joint-coverage statement still relies on the high-dimensional approximation underlying de-biased Lasso and on the multiplier bootstrap adequately approximating the max-statistic distribution. It should not be read as an exact finite-sample $1-\alpha_{\mathrm{sim}}$ guarantee for arbitrary data-generating processes.

## Multiple testing is a separate layer

Simultaneous confidence intervals and multiple-testing adjustment solve related but different problems. If you already have valid marginal p-values and need FDR/FWER adjustment, use estimator-context helpers such as `adjust_pvalues` or the functions in the [Inference API](../guides/inference-api.md). An adjustment procedure cannot repair invalid underlying p-values or remove the assumptions required by de-biased inference.

## What is actually reported?

Important fields include:

- `coef_`, `intercept_`: penalized prediction fit;
- `nodewise_alpha_`: resolved node-wise tuning after successful multi-feature debiased inference;
- `_params`: de-biased reporting parameters;
- `_bse`, `_zvalues`, `_pvalues`, `_conf_int`: marginal inference arrays;
- `_conf_int_simultaneous`: simultaneous intervals when requested;
- `_simultaneous_critical_value`: common max-|Z| critical value;
- `_inference_result`: structured result with method, precision, backend/device, node-wise, KKT, and cache provenance.

## Interpretation checklist

- Decide whether your scientific question is prediction/selection, coefficient-wise de-biased inference, or a post-selection diagnostic.
- Keep main `alpha` separate from `nodewise_alpha`.
- Remember that the automatic node-wise rule is design-side and response-scale independent.
- Treat `post_selection_ols` as a diagnostic after selection.
- Treat CV-final-refit inference as conditional on selected tuning parameters.
- Simultaneous calibration targets the maximum standardized error over the requested family; all coordinates share each multiplier draw so their joint dependence is retained.
- Use simultaneous calibration or multiple-testing correction only when the underlying inferential object is appropriate for the question.
- Numerical backend parity does not weaken or strengthen the statistical assumptions.

## References

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional linear models. *JASA*, 112(518), 757–768.