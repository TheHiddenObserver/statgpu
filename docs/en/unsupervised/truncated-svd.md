# TruncatedSVD

> Language: English
> Last updated: 2026-05-04
> Path: `statgpu.unsupervised.TruncatedSVD`

## Overview

`TruncatedSVD` computes a low-rank projection without centering the input matrix. This makes it different from `PCA` and suitable for dense LSA-style workflows.

## Objective Function / Loss Function

For a rank `k` approximation, Truncated SVD solves:

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

The returned components are the leading right singular vectors of `X`.

## Estimating Equation

The exact path computes:

$$
X = U \Sigma V^\top.
$$

The randomized path projects `X` to a lower-dimensional random subspace, re-orthogonalizes each power iteration, applies a deterministic component sign convention, and computes a small SVD of the projected matrix.

## Parameters

`n_components`, `algorithm`, `n_iter`, `n_oversamples`, `random_state`, and `device` control the decomposition and backend.

## CPU+GPU Examples

```python
from statgpu.unsupervised import TruncatedSVD

svd = TruncatedSVD(n_components=10, device="cpu")
Z = svd.fit_transform(X)

svd_gpu = TruncatedSVD(n_components=10, device="cuda")
Z_gpu = svd_gpu.fit_transform(X_gpu)
```

## Strict/Approx Difference

`algorithm="full"` is exact dense SVD. `algorithm="randomized"` is approximate and should be compared with sign- or subspace-invariant metrics.

## Outputs

`components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `n_components_`, and `n_features_in_`.

## FAQ

Sparse input and ARPACK are not supported in Phase 3A.

## External Validation

Tests: `dev/tests/test_unsupervised_truncated_svd.py`.
Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3.py`.
Baselines: sklearn `TruncatedSVD`, statsmodels PCA-style SVD, and R `svd` where available.

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.
- scikit-learn developers. `sklearn.decomposition.TruncatedSVD` API documentation.
