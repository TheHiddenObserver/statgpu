# IncrementalPCA

> Language: English
> Last updated: 2026-05-07
> Switch: [Chinese](../../unsupervised/incremental-pca.md)

## Overview

`IncrementalPCA` fits principal components from dense mini-batches while maintaining running mean, variance, sample count, and a truncated SVD basis. It supports CPU, CuPy/CUDA, and Torch CUDA backends.

## Path

```python
from statgpu.unsupervised import IncrementalPCA
```

## Objective Function / Loss Function

For `k` components, IncrementalPCA approximates the centered rank-k PCA objective:

$$
\min_{V_k^\top V_k=I}
\left\|X - \bar{X} - (X-\bar{X})V_kV_k^\top\right\|_F^2 .
$$

## Estimating Equation

Each `partial_fit` updates running batch statistics and computes an SVD of a compact matrix containing the previous low-rank basis, the current centered batch, and a mean-correction row. The leading right singular vectors become `components_`.

## Parameters

- `n_components`: number of components; `None` keeps `n_features`.
- `batch_size`: batch size used by `fit`; `partial_fit` accepts caller-provided batches.
- `whiten`: scales transformed components by explained variance.
- `copy`: sklearn-style compatibility flag.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import IncrementalPCA

ipca = IncrementalPCA(n_components=8, batch_size=1024, device="cuda")
ipca.fit(X)
Z = ipca.transform(X)
X_hat = ipca.inverse_transform(Z)
```

## Strict/Approx Difference

IncrementalPCA is an approximate streaming/batch estimator. Its result can differ from full PCA depending on batch order and batch size, but CPU/CuPy/Torch should agree for the same batches.

## Outputs

- `components_`
- `mean_`
- `var_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`
- `n_samples_seen_`

## FAQ

**Does v1 support sparse input?**
No. Phase 3C supports dense 2D float arrays only.

## External Validation

- Tests: `dev/tests/test_unsupervised_incremental_pca.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3c.py`.
- Latest remote artifact: `results/unsupervised_phase3c_opt7_20260507_185500.json`.
- Baseline: sklearn `IncrementalPCA` with aligned `n_components` and `batch_size`.

## References

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
