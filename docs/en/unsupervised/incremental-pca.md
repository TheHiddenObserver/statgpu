# IncrementalPCA

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/incremental-pca.md)

## Overview

`IncrementalPCA` fits principal components from dense mini-batches while maintaining running mean, variance, sample count, and a truncated SVD basis. It supports CPU, CuPy/CUDA, and Torch CUDA backends.

## When to use it

Use IncrementalPCA to learn a linear basis from successive batches. Truncation after each batch can lose directions that full PCA would retain, so compare a representative subset and keep preprocessing fixed. A small first `partial_fit` batch limits the default rank for that incremental fit; a new `fit` resets it.

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

- `n_components`: number of components; `None` uses `min(n_samples, n_features)` in `fit`, or fixes the rank from the first `partial_fit` batch.
- `batch_size`: batch size used by `fit`; `partial_fit` accepts caller-provided batches.
- `whiten`: divides transformed components by the square root of the fitted variance, with a small variance floor.
- `copy`: sklearn-style compatibility flag.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## A small CPU example

<!-- learner-example: incremental-pca -->
```python
import numpy as np
from statgpu.unsupervised import IncrementalPCA

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5))
model = IncrementalPCA(n_components=2, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
Z = model.transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, model.n_samples_seen_, X_hat.shape)
```

The final basis transforms all 60 rows to `(60, 2)`; `n_samples_seen_` is 60. `mean_` and `var_` describe all processed rows, but earlier per-batch coordinates are not automatically rotated into the final basis.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#incrementalpca) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

IncrementalPCA is an approximate streaming/batch estimator. Its result can differ from full PCA depending on batch order and batch size, and floating-point decomposition choices can differ across backends. Compare explained variance, reconstruction and retained subspaces within appropriate tolerances; component signs and bases within repeated-singular-value subspaces are not unique.

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

**Does it support sparse input?**
No. The current implementation supports dense 2D float arrays only.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [IncrementalPCA API reference](api-reference.md#incrementalpca).

## References

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
