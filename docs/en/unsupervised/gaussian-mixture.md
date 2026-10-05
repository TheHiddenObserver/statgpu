# GaussianMixture

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/gaussian-mixture.md)

## Overview

`GaussianMixture` fits a Gaussian mixture model with expectation-maximization. It supports `"diag"`, `"spherical"`, `"tied"`, and `"full"` covariance types on CPU, CuPy/CUDA, and Torch CUDA backends.

## When to use it

Use a Gaussian mixture for overlapping groups when soft membership is useful. Start with a small component count and a simple covariance structure; inspect responsibilities, convergence and held-out log density. A fitted component need not correspond to a real population.

## Path

```python
from statgpu.unsupervised import GaussianMixture
```

## Objective Function / Loss Function

For a fixed number of mixture components, the model maximizes average log likelihood:

$$
\ell(\theta)
= \frac{1}{n}\sum_{i=1}^{n}
\log\left[
\sum_{k=1}^{K}
\pi_k \,
\mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
\right].
$$

`covariance_type` controls the shape of `\Sigma_k`: diagonal per component, spherical per component, one tied full covariance, or one full covariance per component. `reg_covar` is a variance floor for diagonal/spherical M-step updates; full/tied updates add it to the covariance diagonal. These conventions differ, so equal values need not match another library.

## Estimating Equation

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

## Parameters

- `n_components`: number of mixture components.
- `covariance_type`: `"diag"`, `"spherical"`, `"tied"`, or `"full"`.
- `tol`, `reg_covar`, `max_iter`, `n_init`.
- `init_params`: `"kmeans"` or `"random"`.
- `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

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

## Approximation and interpretation

GMM has likelihood scores but no strict inference covariance or p-value mode. EM optimizes a non-convex likelihood and can converge to local optima. Reproducibility depends on initialization, `random_state`, `n_init`, `tol`, and `max_iter`.

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
`"diag"` and `"spherical"` are cheaper and work well when features are weakly correlated within components. `"tied"` shares one full covariance across components. `"full"` is the most flexible but also the most expensive and needs more samples per component.

**What do `score`, `score_samples`, `aic`, and `bic` mean?**
`score_samples` returns per-sample log likelihood, `score` returns its mean, and `aic`/`bic` use the covariance-type-specific parameter count.


## Numerical and lifecycle cautions

Diagonal and spherical covariance updates use raw second moments; the diagonal density formula also subtracts large quadratic terms. Large offsets relative to within-cluster spread can therefore produce wrong covariance and likelihood values even when `converged_` is true. Center features using a training-derived offset and reuse it for later scoring; translation preserves the intended mixture densities. `reg_covar` cannot repair this cancellation.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [GaussianMixture API reference](api-reference.md#gaussianmixture).

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley Series in Probability and Statistics. Wiley.
