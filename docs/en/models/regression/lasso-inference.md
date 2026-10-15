# Lasso inference: from de-biasing to simultaneous intervals

> Language: English  
> Last updated: 2026-10-15  
> Model guide: [Lasso](lasso.md)  
> Switch: [Chinese](../../../cn/models/regression/lasso-inference.md)

Lasso is useful for prediction and sparse feature selection, but its penalty
shrinks coefficients and selects variables using the same data. Ordinary
fixed-model confidence intervals cannot simply be attached to those penalized
coefficients. This companion explains the de-biased estimator, the node-wise
regressions that construct its correction, and the shared multiplier bootstrap
used for simultaneous intervals. Start with the [Lasso guide](lasso.md) for
model selection, solver limitations and alternatives; complete controls and
outputs are in the [Lasso API reference](../../reference/linear-model-api.md#lasso).

The public `nodewise_alpha` interface and automatic rule described here are
[unreleased changes targeted for 0.2.6](../../changelog.md); they are not
available in the 0.2.5 release.

## Which inferential claim do you need?

- `debiased` estimates Gaussian linear-model coefficients using a one-step
  correction. Its approximate validity needs sparsity, suitable design/noise
  conditions and sufficiently small estimation and precision-matrix errors.
- `post_selection_ols` refits OLS/WLS on selected columns. Treat it as a
  post-selection diagnostic, not a general selective-inference procedure.
- `bootstrap` resamples residuals and refits the penalized estimator with fixed
  tuning. It currently requires unweighted Gaussian data and nonrobust
  covariance; it does not universally correct selection uncertainty.

The residual-refit `bootstrap` method is different from the **Gaussian multiplier
bootstrap of inference scores** below. The latter accompanies `debiased` and
does not refit Lasso in every multiplier draw. For prediction alone, use
`compute_inference=False`. Inference after `LassoCV` is conditional on the
selected penalty and does not separately correct CV tuning uncertainty.

## Define the working problem first

Let the raw design be $X\in\mathbb R^{n\times p}$ and the response be $y$.
With `fit_intercept=True`, define weighted means and row scales by

$$
W=\sum_i w_i,\qquad
\bar x_w=\frac{1}{W}\sum_i w_i x_i,\qquad
\bar y_w=\frac{1}{W}\sum_i w_i y_i,\qquad
a_i=\sqrt{\frac{n w_i}{W}}.
$$

Weights must be finite and nonnegative with positive total. Use $w_i=1$ in the
unweighted case. The centered/weighted working arrays are

$$
X_{w,i}=a_i(x_i-\bar x_w),\qquad
y_{w,i}=a_i(y_i-\bar y_w),\qquad
\hat r_w=y_w-X_w\hat\beta.
$$

Here $X_{w,i}$ denotes row $i$. With `fit_intercept=False`, omit centering:
$X_{w,i}=a_i x_i$ and $y_{w,i}=a_i y_i$; there is no intercept parameter.
The fitted slopes minimize

$$
\hat\beta=\arg\min_\beta
\left\{\frac{1}{2n}\lVert y_w-X_w\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1\right\}.
$$

Thus $n$ remains the row count in the average-loss formulas. The effective
sample size used later for node-wise tuning does **not** replace every $n$.
Multiplying all weights by a positive constant leaves these working arrays
unchanged. Statistical validity still depends on why the weights are used and
the corresponding error model; scale invariance alone does not establish it.

## Why the one-step correction appears

The Lasso optimality relation is

$$
\frac{X_w^\top\hat r_w}{n}=\alpha\hat\kappa,
\qquad \hat\kappa_j\in\partial|\hat\beta_j|.
$$

The unpenalized score is deliberately nonzero. Let $M$ approximate the inverse
of $\widehat\Sigma_w=X_w^\top X_w/n$. De-biasing takes one score step:

$$
\boxed{\hat\theta^{\rm db}=\hat\beta+
\frac{1}{n}MX_w^\top\hat r_w.}
$$

For a correctly specified linear working model
$y_w=X_w\beta^0+\varepsilon_w$, this gives the exact decomposition

$$
\hat\theta^{\rm db}-\beta^0
=\frac{1}{n}MX_w^\top\varepsilon_w
+(I-M\widehat\Sigma_w)(\hat\beta-\beta^0).
$$

The first term is the leading noise contribution; the second is a remainder.
Approximate inference requires that remainder to be small on the standard-error
scale. A converged solver or a sparse fitted coefficient vector alone does not
establish that condition.

## Construct M with node-wise Lasso

Each feature becomes a temporary response predicted from the other features.
These regressions describe the **design**, not the original response $y$.
First standardize the working columns:

$$
d_j^2=\frac{1}{n}\sum_i X_{w,ij}^2,\qquad
D=\operatorname{diag}(d_1,\ldots,d_p),\qquad Z=X_wD^{-1}.
$$

For each $j$, solve

$$
\hat\gamma_j=\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\rm nw}\lVert\gamma\rVert_1\right\}.
$$

The main `alpha` controls prediction shrinkage; `nodewise_alpha` controls these
precision regressions only. An explicit finite positive scalar sets
$\lambda_{\rm nw}$ on the standardized design. With `nodewise_alpha=None` and
$p\ge2$, the default is

$$
\lambda_{\rm nw}=\sqrt{\frac{2\log(\max(p,2))}{n_{\rm nw}}},
\qquad
n_{\rm nw}=\begin{cases}
n,&\text{unweighted},\\
\dfrac{(\sum_i w_i)^2}{\sum_i w_i^2},&\text{analytic weights}.
\end{cases}
$$

Equal positive weights also give $n_{\rm nw}=n$. The $\sqrt{\log p/n}$ order is
theory-motivated; the constant and weighted effective-sample-size convention are
library defaults, not a uniquely mandated theorem. The rule is independent of
the response scale. For $p=1$, there is no nuisance regression: analytic
univariate precision is used and `nodewise_alpha_` remains `None`.

For $r_j=Z_j-Z_{-j}\hat\gamma_j$, define

$$
\hat\tau_j^2=\frac{\lVert r_j\rVert_2^2}{n}
+\lambda_{\rm nw}\lVert\hat\gamma_j\rVert_1.
$$

Row $j$ of $\Theta_Z$ has diagonal $1/\hat\tau_j^2$ and entries
$-\hat\gamma_{j,k}/\hat\tau_j^2$ in the other columns. Transform back with

$$
\boxed{M=D^{-1}\Theta_ZD^{-1}.}
$$

The row-wise approximation need not be exactly symmetric. Its orientation in
$MX_w^\top\hat r_w$ matters. The software independently checks node-wise KKT
optimality after solving; invalid scales, a degenerate normalizer, non-finite
results or a failed KKT check raise an error rather than yielding placeholder
inference. This numerical check does not verify the statistical assumptions.
See [node-wise tuning](../../guides/nodewise-alpha-migration.md) for parameter
validation and migration details. In `LassoCV`, this setting affects only
inference on the full-data final refit, not the alpha grid or fold scores.

## Marginal uncertainty and the reporting intercept

The Gaussian calculation uses

$$
V=M\widehat\Sigma_wM^\top,\qquad
\widehat{\mathrm{se}}_j^2=\hat\sigma^2V_{jj}/n.
$$

The current residual-scale estimate is
$\hat\sigma^2=\lVert\hat r_w\rVert_2^2/\max(n-\hat s,1)$, where
$\hat s$ counts nonzero penalized slopes. This is an estimated scale, not a
general selection-adjusted degrees-of-freedom result. Marginal z statistics,
p-values and intervals use a standard-normal reference. The resulting
`_conf_int` is marginal per parameter, not a joint coverage statement.

Public `coef_` and `intercept_` remain the penalized prediction fit. With an
intercept, `_params[1:]` contains $\hat\theta^{\rm db}$ and `_params[0]` contains
the matching original-coordinate reporting intercept

$$
\hat\theta_0^{\rm db}=\bar y_w-\bar x_w^\top\hat\theta^{\rm db}.
$$

Its influence vector, expressed against the working residuals, is

$$
q=a-X_wM^\top\bar x_w,\qquad h=q/n,\qquad
\widehat{\mathrm{se}}_0^2=\hat\sigma^2q^\top q/n^2.
$$

The fitted intercept's SE and interval therefore describe this corrected
reporting intercept, not the penalized `intercept_`. Translating all raw
features by a constant $c$ leaves corrected slopes unchanged and changes the
reporting intercept by $-c^\top\hat\theta^{\rm db}$.

## From marginal errors to one simultaneous critical value

Let $\mathcal J$ be the family of coordinates to cover jointly. Ideally we would
know the distribution of the largest standardized estimation error:

$$
T=\max_{j\in\mathcal J}
\left|\frac{\hat\theta_j^{\rm db}-\theta_j^0}
{\widehat{\mathrm{se}}_j}\right|.
$$

If $c_{1-\alpha_{\rm sim}}$ were its appropriate quantile, intervals
$\hat\theta_j^{\rm db}\pm c_{1-\alpha_{\rm sim}}\widehat{\mathrm{se}}_j$
would target

$$
\Pr\{\theta_j^0\in CI_j^{\rm sim}\text{ for all }j\in\mathcal J\}
\approx1-\alpha_{\rm sim}.
$$

The target is the maximum over the family, not one standard-normal coordinate.
Many 95% marginal intervals do not by themselves provide 95% joint coverage.

### Approximate the maximum with shared Gaussian multipliers

Dropping the small remainder from the de-biasing decomposition suggests the
leading term $MX_w^\top\varepsilon_w/n$. Replace the unknown working errors by
$\hat r_w$, and for draw $b$ sample independent
$\xi_i^{(b)}\sim N(0,1)$. The slope perturbation is

$$
\boxed{S^{*(b)}=\frac{1}{n}MX_w^\top
(\xi^{(b)}\odot\hat r_w),\qquad
Z_j^{*(b)}=S_j^{*(b)}/\widehat{\mathrm{se}}_j.}
$$

Every coordinate in a draw uses the **same multiplier vector**. This retains
cross-coordinate dependence induced by the observations, design and common
precision matrix, instead of bootstrapping coefficients independently.
For a slope-only family, compute

$$
T^{*(b)}=\max_{j=1,\ldots,p}|Z_j^{*(b)}|.
$$

After $B=$ `simultaneous_n_bootstrap` draws, use the empirical quantile

$$
\hat c_{1-\alpha_{\rm sim}}=
\widehat Q_{1-\alpha_{\rm sim}}(T^{*(1)},\ldots,T^{*(B)}),\qquad
CI_j^{\rm sim}=\hat\theta_j^{\rm db}\pm
\hat c_{1-\alpha_{\rm sim}}\widehat{\mathrm{se}}_j.
$$

This max-|Z| procedure is dependence-aware calibration of the maximum, rather than independent
coordinate bootstraps or a mechanical Bonferroni adjustment. It is still an
approximation under de-biased-Lasso and multiplier-bootstrap assumptions, not
an exact finite-sample guarantee for arbitrary designs, errors or weights.
More draws reduce Monte Carlo error; they cannot repair invalid assumptions.

### Put the intercept in the same maximum

With `fit_intercept=True` and `simultaneous_include_intercept=True`, use the
same $h=q/n$ as in the marginal intercept calculation:

$$
S_0^{*(b)}=\sum_i\xi_i^{(b)}\hat r_{w,i}h_i,\qquad
Z_0^{*(b)}=S_0^{*(b)}/\widehat{\mathrm{se}}_0,
$$

$$
T^{*(b)}=\max\{|Z_0^{*(b)}|,\max_{j=1,\ldots,p}|Z_j^{*(b)}|\}.
$$

The intercept participates in calibration itself. It is not appended afterward
using a slope-only critical value. The default
`simultaneous_include_intercept=False` targets slopes only: when an intercept
was fitted, row 0 of `_conf_int_simultaneous` retains its **marginal** interval.
When populated, `_inference_result.simultaneous_target_mask` identifies the
jointly calibrated rows. The current CPU slope-only path leaves this field
`None`; identify its target rows from `fit_intercept` and
`simultaneous_include_intercept` instead. Without an intercept, all rows
represent slopes.

## A complete CPU example

Run the following steps in one session. The penalty is fixed for illustration;
the data are simulated and no observed coverage rate is inferred from one fit.

<!-- learner-example: lasso-inference-companion -->
```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(400, 5)) + np.array([1.0, -0.5, 0.0, 0.8, -0.2])
beta = np.array([1.5, -1.0, 0.0, 0.7, 0.0])
y = 0.4 + X @ beta + rng.normal(size=400)
```

Choose `debiased` explicitly when enabling simultaneous inference; the current
constructor rejects `auto` for this combination. Use a positive integer draw
count and a family error level strictly between 0 and 1.

```python
model = Lasso(
    alpha=0.05, nodewise_alpha=None, device="cpu",
    solver="coordinate_descent", max_iter=5000, tol=1e-10,
    compute_inference=True, inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap", simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=200, simultaneous_random_state=123,
    simultaneous_include_intercept=True,
).fit(X, y)
```

Read prediction and inference estimates separately, then inspect the interval
arrays and the calibrated family:

```python
print("Prediction slopes:", model.coef_)
print("Reporting parameters:", model._params)
print("Node-wise penalty:", model.nodewise_alpha_)
print("Marginal/joint shapes:", model._conf_int.shape,
      model._conf_int_simultaneous.shape)
print("Joint target rows:", model._inference_result.simultaneous_target_mask)
print("Common critical value:", model._simultaneous_critical_value)
```
<!-- example-end: lasso-inference-companion -->

Both interval shapes are `(6, 2)`: an intercept followed by five slopes. All six
target-mask entries are `True` here. The 200 draws keep the example small; choose
a larger count for more stable Monte Carlo calibration. The seed controls draws
within a backend; it does not promise identical CPU and GPU random sequences.

## Devices and limits

`inference_method` chooses the statistical method; `device` chooses execution.
Explicit `device="cuda"` requests CuPy CUDA and `device="torch"` requests
Torch CUDA; unavailable explicit devices raise rather than silently selecting
CPU. Automatic backend selection is reserved for genuine `device="auto"`.
See the [device guide](../../guides/device-and-memory.md) for global defaults and
backend-native input routing.

Marginal debiased inference uses the successful fit's backend. With
`fit_intercept=True`, simultaneous calibration also runs on that concrete
backend/device and returns NumPy reporting arrays. With `fit_intercept=False`,
the simultaneous calculation currently uses a NumPy host helper even after a
GPU fit; that calculation is **not GPU-native**. Do not interpret a GPU fit as
proof that every subsequent helper remains on the GPU.

For marginal p-value adjustment or combination, use the
[multiple-testing guide](../../guides/multiple-testing-combine-pvalues.md).
These operations are a separate layer and require valid input p-values; they
cannot repair biased estimates or unsupported inferential assumptions.

## References

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional
  parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On
  asymptotically optimal confidence regions and tests for high-dimensional
  models. *Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis
  testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional
  linear models. *JASA*, 112(518), 757–768.
