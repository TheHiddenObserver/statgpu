# GaussianMixture

> Language: English
> Last updated: 2026-05-02
> Switch: [Chinese](../../unsupervised/gaussian-mixture.md)

## Overview

`GaussianMixture` fits a diagonal-covariance Gaussian mixture model with expectation-maximization. It supports CPU, CuPy/CUDA, and Torch CUDA backends.

## Path

```python
from statgpu.unsupervised import GaussianMixture
```

## Objective Function / Loss Function

For `K` components and diagonal covariance `sigma_k^2`, the model maximizes average log likelihood:

```text
(1 / n) * sum_i log sum_k pi_k * Normal(x_i | mu_k, diag(sigma_k^2))
```

`reg_covar` lower-bounds diagonal variances for numerical stability.

## Estimating Equation

The implementation uses log-domain EM:

- Initialize means with KMeans or random samples.
- E-step:
  - compute weighted log probabilities `log pi_k + log N(x_i | mu_k, Sigma_k)`;
  - normalize with `logsumexp`;
  - produce responsibilities `r_ik`.
- M-step:
  - `n_k = sum_i r_ik`;
  - `pi_k = n_k / n`;
  - `mu_k = sum_i r_ik x_i / n_k`;
  - `sigma_k^2 = max(sum_i r_ik x_i^2 / n_k - mu_k^2, reg_covar)`.
- Stop when the lower-bound improvement is below `tol` or `max_iter` is reached.
- Run `n_init` initializations and keep the highest lower bound.

## Parameters

- `n_components`: number of mixture components.
- `covariance_type`: only `"diag"` is supported.
- `tol`, `reg_covar`, `max_iter`, `n_init`.
- `init_params`: `"kmeans"` or `"random"`.
- `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import GaussianMixture

X = np.random.default_rng(0).normal(size=(4000, 16))

gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=0, device="torch")
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

**Is full covariance supported?**
No. Phase 2 supports only `covariance_type="diag"`.

**What do `score`, `score_samples`, `aic`, and `bic` mean?**
`score_samples` returns per-sample log likelihood, `score` returns its mean, and `aic`/`bic` use the diagonal-GMM parameter count.

## External Validation

- Tests: `dev/tests/test_unsupervised_gmm.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase2.py`.
- Baseline: sklearn `GaussianMixture(covariance_type="diag")` with aligned initialization and convergence controls.
- Latest remote matrix: CPU/CuPy/Torch scores match at floating-point noise scale; sklearn score diff is about `2.4e-11`.

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
