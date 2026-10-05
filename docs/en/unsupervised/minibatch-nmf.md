# MiniBatchNMF

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/minibatch-nmf.md)

## Overview

`MiniBatchNMF` fits a non-negative low-rank factorization from dense mini-batches. The current implementation supports Frobenius loss with multiplicative-update style mini-batch updates on CPU, CuPy/CUDA, and Torch CUDA.

## When to use it

Use MiniBatchNMF for nonnegative additive structure learned in batches. `fit` receives all rows; `partial_fit` accepts an external stream. Keep rank and feature meaning fixed, and inspect reconstruction with factors returned by `transform` rather than assuming its solve equals the fitting updates.

## Path

```python
from statgpu.unsupervised import MiniBatchNMF
```

## Objective Function / Loss Function

For non-negative `W` and `H`, the estimator minimizes mini-batch approximations to the Frobenius reconstruction loss:

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

## Estimating Equation

For each batch, `MiniBatchNMF` approximately solves activations `W_batch` with fixed `H`. In `fit`, it holds `H` fixed for a whole epoch, sums `A = sum(W_batch.T @ W_batch)` and `B = sum(W_batch.T @ X_batch)`, then updates `H`. In `partial_fit`, these statistics accumulate across calls. The elementary multiplicative updates have the form:

$$
W \leftarrow W \odot \frac{XH^\top}{WHH^\top + \epsilon},
\qquad
H \leftarrow H \odot \frac{W^\top X}{W^\top WH + \epsilon}.
$$

## Parameters

- `n_components`: factorization rank; `None` uses `min(n_samples, n_features)`.
- `init`: currently supports `"random"`.
- `batch_size`, `max_iter`, `tol`, `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## A small CPU example

<!-- learner-example: minibatch-nmf -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

rng = np.random.default_rng(0)
X = rng.uniform(0.1, 1.0, (60, 2)) @ rng.uniform(0.1, 1.0, (2, 5))
model = MiniBatchNMF(n_components=2, random_state=0, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
W = model.transform(X)
X_hat = model.inverse_transform(W)
print(W.shape, model.components_.shape, np.linalg.norm(X - X_hat))
```

The factors are `(60, 2)` and `(2, 5)`. After `partial_fit`, `reconstruction_err_` describes only that batch with the fitting factors; the independently computed full-data residual above answers a different question.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#minibatchnmf) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

MiniBatchNMF is non-convex; incremental `partial_fit` updates depend on batch order. Ordinary `fit` aggregates statistics over an epoch before updating components. It is intended for scalable approximate factorization, not strict statistical inference.

## Outputs

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**Does it support negative or sparse input?**
No. Inputs must be dense and non-negative.

**Does it support CD solver or other beta losses?**
No. The current implementation supports MU-style updates and Frobenius loss only.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [MiniBatchNMF API reference](api-reference.md#minibatchnmf).

## References

- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. *Advances in Neural Information Processing Systems*, 13.
- Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S.-I. (2009). *Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation*. Wiley.
- scikit-learn Developers. `sklearn.decomposition.MiniBatchNMF`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
