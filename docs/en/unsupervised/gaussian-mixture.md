# GaussianMixture

> Language: English
> Last updated: 2026-05-07
> Switch: [Chinese](../../cn/unsupervised/gaussian-mixture.md)

## Overview

`GaussianMixture` fits a Gaussian mixture model with expectation-maximization. It supports `"diag"`, `"spherical"`, `"tied"`, and `"full"` covariance types on CPU, CuPy/CUDA, and Torch CUDA backends.

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

`covariance_type` controls the shape of `\Sigma_k`: diagonal per component, spherical per component, one tied full covariance, or one full covariance per component. `reg_covar` adds a small diagonal ridge to covariance estimates for numerical stability.

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
  Stop when its improvement is below `tol` or `max_iter` is reached.
- Run `n_init` initializations and keep the highest lower bound.

## Parameters

- `n_components`: number of mixture components.
- `covariance_type`: `"diag"`, `"spherical"`, `"tied"`, or `"full"`.
- `tol`, `reg_covar`, `max_iter`, `n_init`.
- `init_params`: `"kmeans"` or `"random"`.
- `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import GaussianMixture

X = np.random.default_rng(0).normal(size=(4000, 16))

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0, device="torch")
gmm.fit(X)
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
ll = gmm.score(X)
```

## Strict/Approx Difference

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

## External Validation

- Tests: `dev/tests/test_unsupervised_gmm.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3b.py`.
- Latest remote artifact: `results/unsupervised_phase3b_verify_20260507_003957.json`.
- Baseline: sklearn `GaussianMixture` with aligned `covariance_type`, initialization, and convergence controls.
- Phase 3B validation target: CPU/CuPy/Torch score consistency and sklearn parity for `"diag"`, `"spherical"`, `"tied"`, and `"full"`.

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley Series in Probability and Statistics. Wiley.
