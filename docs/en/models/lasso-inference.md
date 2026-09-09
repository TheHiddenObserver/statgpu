# Lasso inference

> Language: English  
> Last updated: 2026-09-09  
> Model guide: [Lasso](lasso.md)  
> Switch: [简体中文](../../cn/models/lasso-inference.md)

This page is the statistical-inference companion to the learner-first [Lasso guide](lasso.md). It explains what statgpu's post-fit inference modes compute, how to interpret the reported intervals, and where the current implementation deliberately stops making a stronger claim.

## Why ordinary Lasso coefficients need a different inferential construction

Lasso is designed first as a **regularized estimator**. For the centered Gaussian linear model, it solves an objective of the form

$$
\hat\beta
= \arg\min_\beta
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+ \alpha\lVert\beta\rVert_1
\right\}.
$$

The L1 penalty is exactly what creates sparsity, but it also shrinks fitted coefficients toward zero. At a solution, the KKT relation is schematically

$$
\frac{X^\top(y-X\hat\beta)}{n}
= \alpha\hat\kappa,
\qquad
\hat\kappa_j\in\partial|\hat\beta_j|,
$$

so the score is intentionally not zero as it would be for unpenalized OLS. The resulting shrinkage bias is the central reason why one cannot simply take the raw Lasso coefficient, attach an ordinary OLS-style standard error, and expect a centered Gaussian statistic. In high-dimensional regimes, the regularization scale is typically large enough that this bias is not automatically negligible on the $n^{-1/2}$ inferential scale. Sparse estimators such as Lasso also have non-smooth, parameter-dependent limiting behavior rather than the simple fixed-model OLS distribution used by classical Wald inference (van de Geer et al., 2014; Javanmard & Montanari, 2014).

**Selection uncertainty is a second, distinct issue.** It becomes especially important if one takes the active set chosen by Lasso and then refits OLS on that same data. Ordinary OLS intervals after data-driven variable selection generally do not account for the selection step. That is why statgpu labels `post_selection_ols` as a diagnostic rather than a general selective-inference procedure.

So the main motivation for the `debiased` path is to remove the leading regularization bias and recover an approximately Gaussian coefficient-level inferential object. The warning about selection uncertainty explains why the separate active-set OLS path makes a weaker claim. These are related high-dimensional inference problems, but they are not the same problem.

statgpu therefore exposes several inference modes with different purposes. They should not be treated as interchangeable ways to print the same p-values.

## Choose the inference path by the claim you need

| `inference_method` | What statgpu computes | Appropriate interpretation | Main limitation |
|---|---|---|---|
| `debiased` | De-biased/de-sparsified coefficient estimator, standard errors, z statistics, p-values, and marginal confidence intervals | Coefficient-wise high-dimensional inference under de-biasing assumptions | Validity depends on sparsity/design/noise conditions and on the regularization/inference construction |
| `post_selection_ols` | Unpenalized OLS/WLS refit on the active set selected by the penalized fit, using the fit-resolved backend | Engineering/post-selection diagnostic | Not a general selective-inference confidence procedure |
| `bootstrap` | Residual-bootstrap refits of the penalized model | Resampling-based uncertainty diagnostic | Expensive and not, by itself, a general correction for data-driven model selection |

`post_selection_ols` is the canonical hardware-neutral spelling. The legacy unified aliases `cpu_ols` and `gpu_ols` are deprecated together and remain accepted for one compatibility cycle with `FutureWarning`; both normalize to `post_selection_ols`. `LassoCV` additionally recognizes the older `cpu_ols_inference` / `gpu_ols_inference` spellings at its compatibility boundary and normalizes them to the same statistical method.

The method name does **not** choose the execution device. `device="cpu"`, `device="cuda"`, and `device="torch"` are authoritative; only genuine `device="auto"` may preserve a backend-native CuPy or Torch-CUDA input as part of automatic routing.

For formal coefficient inference after Lasso, `debiased` is the main statgpu path. For prediction or feature selection only, set `compute_inference=False` and avoid paying for an inference procedure you do not need.

## Minimal de-biased inference example

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
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

The penalized coefficients remain available in `coef_`. Inference reporting uses the de-biased parameter vector, so a penalized coefficient and its de-biased inferential estimate need not be numerically identical.

## What de-biasing changes

Let the fitted Lasso coefficient vector be $\hat\beta$, with residual vector

$$
r = y - b - X\hat\beta.
$$

The L1 penalty creates shrinkage bias. statgpu corrects the coefficient vector with

$$
\hat\theta^{\mathrm{db}}
= \hat\beta + \frac{1}{n} M X^\top r,
$$

where $M$ is a data-dependent decorrelation matrix intended to approximate an inverse of the feature Gram/covariance matrix. This is the de-biased/de-sparsified Lasso construction developed in closely related forms by Zhang & Zhang (2014), van de Geer et al. (2014), and Javanmard & Montanari (2014).

The reason the correction helps becomes clearer by writing the linear model as $y=X\beta^0+\varepsilon$ and defining $\widehat\Sigma=X^\top X/n$. Ignoring the intercept notation for the moment,

$$
\hat\theta^{\mathrm{db}}-\beta^0
=
\frac{1}{n}MX^\top\varepsilon
+
\left(I-M\widehat\Sigma\right)
\left(\hat\beta-\beta^0\right).
$$

The first term is a noise-driven approximately Gaussian term. The second is the remaining regularization/nuisance-estimation error. If $M\widehat\Sigma$ is sufficiently close to the identity and the required sparsity/design conditions make the remainder small, the leading shrinkage bias is removed on the inferential scale. This decomposition is the core reason the de-biased estimator can support asymptotically normal coefficient-wise inference even though the original Lasso estimator generally cannot.

The correction does **not** mean that the original sparse Lasso estimate has become unpenalized. `coef_` is still the fitted penalized model used for prediction. The de-biased vector is an inference object used for coefficient-wise uncertainty reporting.

## How statgpu constructs the decorrelation matrix

The node-wise construction follows the approximate precision-matrix idea used by de-sparsified Lasso methods; see van de Geer et al. (2014) and the related low-dimensional projection construction of Zhang & Zhang (2014).

For each feature $j$, statgpu regresses $x_j$ on the remaining columns $X_{-j}$ with a node-wise Lasso:

$$
\hat\gamma_j
=
\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{
\frac{1}{2n}
\left\lVert x_j-X_{-j}\gamma\right\rVert_2^2
+
\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\},
$$

using

$$
\lambda_{\mathrm{nw}}
= \hat\sigma\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

Embed $\hat\gamma_j$ back into $p$ coordinates by defining $\tilde\gamma_j\in\mathbb R^p$ with $(\tilde\gamma_j)_j=0$ and the fitted node-wise coefficients in the remaining positions. Then let

$$
\hat a_j=e_j-\tilde\gamma_j,
\qquad
z_j=X\hat a_j=x_j-X_{-j}\hat\gamma_j,
$$

and use the row-specific normalization

$$
C_j
=\frac{x_j^\top z_j}{n}.
$$

The $j$th decorrelation row is therefore written compactly as

$$
\boxed{
\hat m_j^\top
=
\frac{\hat a_j^\top}{C_j}
=
\frac{(e_j-\tilde\gamma_j)^\top}{x_j^\top z_j/n}
}
$$

and

$$
M
=
\begin{bmatrix}
\hat m_1^\top\\
\vdots\\
\hat m_p^\top
\end{bmatrix}.
$$

This matrix expression is exactly the same implementation rule as placing $1/C_j$ in position $j$ and $-\hat\gamma_{j,k}/C_j$ in the off-diagonal positions, but it makes the statistical object clearer: each row is a normalized residualization direction intended to make $M\widehat\Sigma$ close to the identity. In the common node-wise-Lasso notation of van de Geer et al. (2014), the same construction is written as a row-normalized approximate inverse/precision matrix; the precise symbol used for the normalizer depends on the objective-scaling convention.

### CPU and GPU implementations solve the same statistical problem

The formula above is **not CPU-specific**. The maintained NumPy, CuPy, and Torch debiased paths use the same node-wise penalty scale, the same $\hat\gamma_j$, $C_j$, and $M$ definitions, and the same downstream de-biasing/variance formulas.

The difference is computational:

- the CPU path solves the node-wise Lasso problems one feature at a time;
- the CuPy and Torch paths form the corresponding Gram subproblems in batches and run batched FISTA-style node-wise solves on the concrete GPU device;
- batching changes execution and memory traffic, not the statistical definition of the decorrelation matrix.

Numerical convergence of the node-wise optimization is necessary, but it does not replace the sparsity, design, and noise assumptions behind de-biased inference.

## Standard errors, z statistics, and marginal intervals

With

$$
\widehat\Sigma = \frac{X^\top X}{n},
\qquad
V = M\widehat\Sigma M^\top,
$$

statgpu uses

$$
\widehat{\mathrm{se}}_j
= \sqrt{\frac{\hat\sigma^2 V_{jj}}{n}},
\qquad
Z_j = \frac{\hat\theta_j^{\mathrm{db}}}{\widehat{\mathrm{se}}_j}.
$$

Two-sided p-values use a standard-normal reference distribution. The current marginal confidence interval stored in `_conf_int` is a 95% interval based on the normal critical value.

Important distinctions:

- `_conf_int[j]` is a **marginal** interval for one parameter at a time.
- A collection of 95% marginal intervals does not automatically have 95% simultaneous coverage over the whole coefficient vector.
- `simultaneous_alpha` does not change these marginal intervals; it controls the separate simultaneous procedure described below.

When an intercept is fitted, prediction and inference deliberately have separate parameter ownership. Public `coef_` and `intercept_` remain the penalized prediction fit. The inference slopes are de-biased, and the reported original-coordinate intercept is the coherent centered value

$$
\hat\theta^{\mathrm{db}}_0
= \bar y_w - \bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

Its standard error and influence representation are therefore tied to the same centered de-biased parameterization rather than treating the intercept as another node-wise-Lasso feature coordinate.

## From marginal intervals to simultaneous coverage

A 95% marginal interval controls the error probability for **one coefficient at a time**. If many coordinates are reported together, the probability that at least one interval misses its target can be much larger than 5%. Under independence, for example, $m$ separate 95% intervals would have joint coverage $(0.95)^m$, not 0.95. A Bonferroni correction is a simple way to recover family-wise protection, but it can be conservative because it does not exploit the dependence structure among the de-biased coefficient statistics.

High-dimensional simultaneous-inference methods instead calibrate the distribution of the **maximum** standardized error over the requested target set. Bootstrap-assisted simultaneous inference for de-sparsified Lasso estimators is developed, for example, by Zhang & Cheng (2017) and Dezeure, Bühlmann & Zhang (2017). The max statistic automatically reflects dependence among coordinates rather than treating every coefficient as an isolated test.

### max-|Z| multiplier-bootstrap calibration

statgpu can optionally calibrate a common critical value with a Gaussian multiplier bootstrap:

```python
model = Lasso(
    alpha=0.05,
    inference_method="debiased",
    compute_inference=True,
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=2000,
    simultaneous_random_state=123,
).fit(X, y)

marginal = model._conf_int
simultaneous = model._conf_int_simultaneous
critical = model._simultaneous_critical_value
```

The method draws independent standard-normal multipliers $\xi_i$, forms bootstrap score perturbations from the fitted residuals, and records the maximum absolute standardized perturbation over the requested target family. Schematically,

$$
T^*
= \max_{j\in\mathcal J}
\left|
\frac{n^{-1}\sum_i \xi_i r_i (MX_i)_j}
{\widehat{\mathrm{se}}_j}
\right|.
$$

The empirical $(1-\alpha)$-quantile of $T^*$ becomes the common critical value $c_{1-\alpha}$, giving

$$
\hat\theta_j^{\mathrm{db}}
\pm c_{1-\alpha}\widehat{\mathrm{se}}_j.
$$

Because the critical value is calibrated for the maximum over the whole target family, these intervals are designed for simultaneous/family-wise coverage and are normally wider than the corresponding marginal intervals. As with the marginal de-biased procedure, the theoretical guarantee still depends on the relevant high-dimensional assumptions; the bootstrap does not make those assumptions disappear.

### Simultaneous-inference controls

| Parameter | Default | Meaning |
|---|---:|---|
| `enable_simultaneous_inference` | `False` | Run simultaneous calibration after successful de-biased inference. |
| `simultaneous_method` | `"maxz_bootstrap"` | Current supported simultaneous calibration method. |
| `simultaneous_alpha` | `0.05` | Family-wise error level used to select the common critical value. |
| `simultaneous_n_bootstrap` | `1000` | Number of multiplier-bootstrap draws. |
| `simultaneous_random_state` | `None` | Seed for reproducible multiplier draws. |
| `simultaneous_include_intercept` | `False` | Include the centered de-biased intercept in both the bootstrap max-|Z| target and the reported joint interval family. |

`enable_simultaneous_inference=True` requires `compute_inference=True`, `inference_method="debiased"`, and `simultaneous_method="maxz_bootstrap"`. `simultaneous_alpha` must lie strictly in `(0, 1)` and `simultaneous_n_bootstrap` must be positive; unsupported combinations fail instead of silently returning ordinary marginal intervals.

### Intercept-inclusive simultaneous family

When `simultaneous_include_intercept=True`, the same centered-nodewise original-coordinate intercept influence used by the marginal standard error participates in the bootstrap maximum itself. The intercept is therefore not merely an extra row receiving a feature-only critical value: it is part of the calibrated joint target family.

The default remains `False`, so callers who only need simultaneous coverage over the feature vector do not pay for an enlarged target family.

## Simultaneous intervals are not p-value adjustment

These operations answer related but different questions:

- `_conf_int` gives coefficient-wise marginal intervals.
- `_conf_int_simultaneous` uses a common max-|Z| critical value for an interval family.
- `model.adjust_pvalues(method="bh")` or module-level `adjust_pvalues(...)` applies a multiple-testing correction to an existing vector of p-values.
- `model.combine_pvalues(...)` asks whether a set of p-values contains global evidence; it does not construct simultaneous coefficient intervals.

The generic estimator-bound and module-level multiple-testing APIs are documented in the [Inference API reference](../guides/inference-api.md).

## Backend behavior and reporting boundary

Post-fit method identity and backend identity are separate.

For `post_selection_ols`, inference always reuses the successful penalized fit's recorded backend and concrete device. The active-set OLS/WLS refit, covariance calculation, reference-distribution inference, and confidence-interval numerics run on NumPy, CuPy, or Torch according to that fit-resolved backend. Explicit GPU requests fail closed when the requested backend is unavailable; they do not silently become CPU OLS inference.

For `debiased`, the maintained marginal CuPy/Torch paths likewise keep their numerical coefficient inference on the executed GPU backend. The established reporting layer remains NumPy-oriented: final arrays such as `_bse`, `_pvalues`, `_conf_int`, and structured result metadata are snapshotted to host-side reporting objects only after numerical inference is complete.

For centered `fit_intercept=True` simultaneous inference on CuPy/Torch, the expensive B×n multiplier draws, feature/intercept scores, max-|Z| reduction, quantile calibration, and joint confidence-interval numerics remain on the same concrete GPU device before the final reporting snapshot. The historical `fit_intercept=False` simultaneous path still uses the pre-existing generic reporting-stage helper and is not claimed as GPU-native by the PR #138 contract.

Residual `bootstrap` is different: its current residual-refit implementation is CPU-native. A GPU `device` on the estimator therefore does not imply that residual-bootstrap resampling/refits are GPU-native.

## Sample weights and inferential scope

`Lasso.fit(..., sample_weight=...)` is a supported fitting surface. The maintained NumPy/CuPy/Torch `debiased` paths use the same weighted-centered average-loss working problem, so multiplying all analytic weights by the same positive constant leaves the penalized fit and maintained de-biased inference unchanged.

That implementation contract is not, by itself, a theorem for every scientific interpretation of analytic weights. If weighted coefficient inference is central to the application, validate the weighting convention and target estimand against the assumptions of the intended inferential analysis.

## Residual bootstrap path

`inference_method="bootstrap"` performs repeated residual resampling and penalized refits. Its constructor controls are `n_bootstrap` (default `200`) and `bootstrap_random_state` (default `None`).

The current implementation derives standard errors from bootstrap variability, sign-based two-sided p-values, and percentile confidence intervals. It is computationally much more expensive than one de-biased fit, and it is not advertised as a complete selective-inference procedure.

## OLS-style post-selection path

`inference_method="post_selection_ols"` refits an **unpenalized OLS or WLS model on exactly the active set** chosen by the penalized fit. `coef_` and `intercept_` remain the penalized prediction fit; `_params` and `_inference_result` own the active-set refit used for inferential reporting.

The refit uses the fit-resolved NumPy/CuPy/Torch backend. Rank-deficient active designs use an effective-rank Moore-Penrose/SVD calculation instead of ordinary normal equations. Classical `cov_type="nonrobust"` reporting uses the maintained Student-t convention; robust/HAC covariance choices use the shared Gaussian robust-covariance layer and its normal-reference convention where exposed by the estimator.

Inactive full-space coordinates retain compatibility placeholders (`SE=0`, statistic `0`, `p=1`, and `[0, 0]` intervals). These are not claims that an omitted coefficient is known exactly; use the selected-feature metadata to identify coordinates that received the active-set refit.

Most importantly, ordinary OLS/WLS intervals formed after choosing variables from the same data remain a **post-selection diagnostic**, not a general selective-inference confidence procedure.

## Fitted inference outputs

When the selected inference path succeeds, the reporting surface can include:

| Attribute | Meaning |
|---|---|
| `_params` | Parameter vector used by inference reporting; for `debiased`, feature entries are de-biased estimates; for `post_selection_ols`, active entries belong to the unpenalized refit |
| `_bse` | Standard errors |
| `_tvalues` | Historical/statistic storage used by some paths; de-biased reporting has z semantics |
| `_zvalues` | z-style statistic field when populated through the structured result layer |
| `_pvalues` | Two-sided coefficient p-values |
| `_conf_int` | Marginal confidence intervals |
| `_conf_int_simultaneous` | Simultaneous intervals after successful max-|Z| calibration |
| `_simultaneous_critical_value` | Calibrated common critical value |
| `_inference_result` | Structured inference result and method/backend metadata |

`summary()` uses the inference result available for the fitted model. Underscore-prefixed arrays are established reporting surfaces in the current release, but their meaning remains method-specific.

## Reproducibility and cost

For reproducible resampling, set `bootstrap_random_state` or `simultaneous_random_state` as appropriate. Increasing `simultaneous_n_bootstrap` reduces Monte Carlo noise in the critical-value estimate at the cost of more work and memory traffic on the path's actual numerical backend.

De-biased inference can be substantially more expensive than fitting the original Lasso because it solves many node-wise sparse regressions. Backend acceleration changes the numerical cost profile but not the statistical assumptions.

## What the current implementation does not claim

- Ordinary post-selection OLS/WLS intervals are not general selective-inference intervals.
- Residual bootstrap is not a universal correction for model-selection uncertainty.
- Marginal de-biased intervals do not provide joint coverage without simultaneous calibration.
- Simultaneous max-|Z| intervals do not replace FDR procedures such as Benjamini-Hochberg when FDR is the scientific target.
- Numerical convergence, GPU execution, and a small KKT residual do not prove that the high-dimensional assumptions required for de-biased inference hold for a dataset.
- The historical `fit_intercept=False` simultaneous path is not claimed as GPU-native merely because a GPU was used for the penalized/de-biased fit.

## References

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242. [doi:10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202. [doi:10.1214/14-AOS1221](https://doi.org/10.1214/14-AOS1221)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909. [JMLR](https://jmlr.org/papers/v15/javanmard14a.html)
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional linear models. *Journal of the American Statistical Association*, 112(518), 757–768. [doi:10.1080/01621459.2016.1166114](https://doi.org/10.1080/01621459.2016.1166114)
- Dezeure, R., Bühlmann, P., & Zhang, C.-H. (2017). High-dimensional simultaneous inference with the bootstrap. *TEST*, 26(4), 685–719. [doi:10.1007/s11749-017-0554-2](https://doi.org/10.1007/s11749-017-0554-2)
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.