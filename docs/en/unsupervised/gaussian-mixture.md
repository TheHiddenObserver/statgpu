# GaussianMixture

> Language: English
> Last updated: 2026-10-13
> Switch: [Chinese](../../cn/unsupervised/gaussian-mixture.md)

## Overview

Suppose measurements from several operating regimes overlap: a reading near the boundary could plausibly come from either regime, and some regimes vary more widely or along different directions. One Gaussian distribution may smooth over those differences. A hard nearest-center label hides the ambiguity by assigning every reading to exactly one group.

`GaussianMixture` describes the data as a weighted blend of Gaussian components, each with a center and a covariance structure. An observation can be compatible with several components at once. Expectation-maximization (EM) alternates between estimating those fractional memberships and updating the component distributions from them. This provides both a density model and model-based soft grouping; see the [Gaussian mixture guide](https://scikit-learn.org/stable/modules/mixture.html) for the probabilistic formulation.

## When to use it

Use a mixture when overlap itself matters, or when you want to evaluate how much density the fitted model assigns to new observations. Compared with [KMeans](kmeans.md), covariance modeling allows the component's spread to affect membership rather than using only Euclidean distance to its center. The chosen covariance structure determines which shapes are possible; a simple structure can be a useful starting point when there is little data per component.

Separate the goal of describing density from the goal of naming groups. Several Gaussian components may help approximate one non-Gaussian population, so a fitted component need not correspond to a real population. More components or more flexible covariances are not automatically more meaningful. Inspect responsibilities, convergence, and held-out log density; compare candidate fits on the same rows with the same preprocessing.

Soft membership is also not a noise detector. Even a reading far from every component can receive a large relative responsibility for the least unlikely one. Check its density as well as its memberships, and use [DBSCAN](dbscan.md) when the intended definition of a cluster is a dense connected region with unassigned background points. Neither output alone establishes the substantive cause of an unusual observation.

## Path

```python
from statgpu.unsupervised import GaussianMixture
```

## Objective Function / Loss Function

Because the component that generated each observation is unknown, its density is obtained by adding the weighted contributions from all components. For a fixed number of mixture components, the model maximizes the average logarithm of that combined density, or average log likelihood:

$$
\ell(\theta)
= \frac{1}{n}\sum_{i=1}^{n}
\log\left[
\sum_{k=1}^{K}
\pi_k \,
\mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
\right].
$$

Here $x_i$ is observation $i$, $n$ is the number of observations, and $K$ is the
number of components. Each $\pi_k$ is a nonnegative mixing weight, with
$\sum_k\pi_k=1$; $\mu_k$ is the component's mean vector and $\Sigma_k$ its
covariance, describing its spread and within-component correlations. The sum
inside the logarithm is the mixture density at $x_i$.

`covariance_type` controls the shape of `\Sigma_k`: diagonal per component, spherical per component, one tied full covariance, or one full covariance per component. `reg_covar` is a variance floor for diagonal/spherical M-step updates; full/tied updates add it to the covariance diagonal. These conventions differ, so equal values need not match another library.

## A small CPU example

<!-- learner-example: gaussian-mixture -->
```python
import numpy as np
from statgpu.unsupervised import GaussianMixture

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.4, (40, 2)), rng.normal(2, 0.4, (40, 2))])
model = GaussianMixture(n_components=2, covariance_type="full", n_init=2, random_state=0, device="cpu")
model.fit(X)
proba = model.predict_proba(X)
print(proba.shape, model.converged_, model.score(X), model.bic(X))
```

Responsibilities have shape `(80, 2)` and each row sums to one. They are conditional membership probabilities under this fitted model, not confidence levels. Compare AIC/BIC on the same observations; neither corrects a failed fit.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#gaussianmixture) for output ownership and host-side work. An unavailable explicit GPU raises an error.

`lower_bound_` records the monitored E-step value before the final parameter update; use `score(X)` for the final fitted mean log density. The [EM update equations](#advanced-em-updates) explain how these values are computed.

## Parameters

- `n_components`: number of mixture components.
- `covariance_type`: `"diag"`, `"spherical"`, `"tied"`, or `"full"`.
- `tol`, `reg_covar`, `max_iter`, `n_init`.
- `init_params`: `"kmeans"` or `"random"`.
- `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## Approximation and interpretation

GMM has likelihood scores but no parameter standard errors or p-values. EM optimizes a non-convex likelihood and can converge to local optima. Reproducibility depends on initialization, `random_state`, `n_init`, `tol`, and `max_iter`.

## Outputs

- `weights_`
- `means_`
- `covariances_`
- `precisions_cholesky_`
- `converged_`
- `n_iter_`
- `lower_bound_`
- `n_features_in_`

## FAQ

**Which covariance type should I use?**
`"diag"` models separate feature variances but no within-component correlations. `"spherical"` additionally forces all feature variances within each component to be equal, so feature units matter especially strongly. `"tied"` shares one full covariance across components; `"full"` allows a different full covariance per component and needs more data to estimate it reliably. Compare held-out log densities on the same validation rows, or compare AIC/BIC computed on the same training rows. Inspect whether the fitted shapes make sense.

**What do `score`, `score_samples`, `aic`, and `bic` mean?**
`score_samples` returns per-sample log likelihood, `score` returns its mean, and `aic`/`bic` use the covariance-type-specific parameter count.


## Numerical and lifecycle cautions

`reg_covar=0` removes covariance protection and is suitable only when the initial and updated covariances remain positive definite. Constant data, constant features in a diagonal model, or collapsed components can violate that requirement. The current diagonal/spherical paths can then return from `fit` with NaN covariance, density and responsibility values; full/tied paths can raise a linear-algebra error. Keep a positive `reg_covar` for such data, and check that fitted covariances, `score_samples(X)` and `predict_proba(X)` are finite before using the result. A returned estimator alone is not evidence of a usable fit.

Diagonal and spherical covariance updates use raw second moments; the diagonal density formula also subtracts large quadratic terms. Large offsets relative to within-cluster spread can therefore produce wrong covariance and likelihood values even when `converged_` is true. Center features using a training-derived offset and reuse it for later scoring; translation preserves the intended mixture densities. `reg_covar` cannot repair this cancellation.

<a id="estimating-equation"></a>

## Advanced: EM updates

The implementation uses log-domain EM:

- Initialize means with KMeans or random samples.
- E-step: compute weighted component log probabilities
  $$
  a_{ik}
  =
  \log \pi_k
  +
  \log \mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right).
  $$
  Then normalize with log-sum-exp:
  $$
  \log p(x_i)
  =
  \operatorname{logsumexp}_{k=1}^{K}\left(a_{ik}\right)
  =
  \log\left[
    \sum_{k=1}^{K}
    \pi_k \mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
  \right].
  $$
  The responsibility of component `k` for sample `i` is
  $$
  r_{ik}
  =
  \exp\left(a_{ik} - \log p(x_i)\right)
  =
  \frac{
    \pi_k \mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
  }{
    \sum_{\ell=1}^{K}
    \pi_\ell \mathcal{N}\left(x_i \mid \mu_\ell, \Sigma_\ell\right)
  } .
  $$
- M-step: update effective component sizes, weights, means, and covariances:
  $$
  n_k = \sum_{i=1}^{n} r_{ik}.
  $$
  $$
  \pi_k = \frac{n_k}{n}.
  $$
  $$
  \mu_k = \frac{1}{n_k}\sum_{i=1}^{n} r_{ik}x_i.
  $$
  $$
  \Sigma_k^{\text{full}}
  =
  \frac{1}{n_k}\sum_{i=1}^{n}r_{ik}
  (x_i-\mu_k)(x_i-\mu_k)^\top
  +
  \text{reg\_covar}\,I.
  $$
  $$
  \Sigma^{\text{tied}}
  =
  \frac{1}{n}\sum_{k=1}^{K}\sum_{i=1}^{n}r_{ik}
  (x_i-\mu_k)(x_i-\mu_k)^\top
  +
  \text{reg\_covar}\,I.
  $$
  The diagonal and spherical cases use the diagonal or feature-averaged diagonal of the same responsibility-weighted covariance update:
  $$
  \sigma_{kj}^{2}
  =
  \max\left(
    \frac{1}{n_k}\sum_{i=1}^{n} r_{ik} x_{ij}^{2}
    -
    \mu_{kj}^{2},
    \text{reg\_covar}
  \right),
  \qquad
  \sigma_k^2 = \frac{1}{p}\sum_{j=1}^{p}\sigma_{kj}^{2}.
  $$
- The monitored lower bound is
  $$
  \mathcal{L}
  =
  \frac{1}{n}\sum_{i=1}^{n}\log p(x_i).
  $$
  Stop when the absolute change between monitored values is below `tol`, or `max_iter` is reached. `lower_bound_` records the E-step value before the final parameter update; use `score(X)` for the final fitted mean log density.
- Run `n_init` initializations and keep the highest lower bound.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [GaussianMixture API reference](api-reference.md#gaussianmixture).

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley Series in Probability and Statistics. Wiley.
