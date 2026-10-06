# MiniBatchNMF

> Language: English
> Last updated: 2026-10-06
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

## Initial batches with zero features

A feature that is zero throughout the first `partial_fit` batch can set its entire dictionary column to zero. Later positive values cannot revive that column under the current multiplicative updates. An all-zero first batch can lock the whole dictionary at zero; changing `max_iter` or `tol` does not repair this state.

Start with a representative buffered batch containing positive observations for every feature you expect to become active. If a later batch contains positive values for a zero dictionary column, create a new estimator and refit representative retained data that include that feature. Do not add arbitrary positive offsets just to bypass this limitation: that changes the factorization problem.

The following small example buffers two batches before initialization. This avoids the zero-column trap, but does not guarantee convergence or an optimal factorization.

<!-- learner-example: minibatch-nmf-warmup -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

first = np.array([[1., 0.], [2., 0.], [3., 0.]])
second = np.array([[1., 1.], [2., 2.], [3., 3.]])
warmup = np.vstack([first, second])
if np.any(np.all(warmup == 0, axis=0)):
    raise ValueError("Buffer more representative rows before initialization")
model = MiniBatchNMF(n_components=1, random_state=0, device="cpu")
model.partial_fit(warmup)
reconstructed = model.inverse_transform(model.transform(second))
print("active dictionary columns:", np.any(model.components_ > 0, axis=0))
print("reconstructed second feature:", reconstructed[:, 1])
```

Both dictionary columns are active, and the reconstructed second feature is positive. Without buffering, fitting `first` then `second` leaves that reconstructed feature identically zero. Always inspect reconstruction feature by feature when the data stream changes.

## Very small input units

Fixed absolute stabilizers can also overwhelm very small positive observations in both `fit` and `partial_fit`. An exactly rank-one matrix expressed at a scale of `1e-12` can reconstruct as essentially zero even when every feature in every batch is positive. This differs from the zero-feature initialization limit above; buffering positive rows alone does not resolve it. Increasing `max_iter` or setting `tol=0` is not sufficient.

Choose a finite positive common scale from representative buffered training rows and retain it for all batches and later `transform` calls. Divide every feature by that same scale, then multiply `inverse_transform` output by it to return to original units. A common scale preserves nonnegativity and changes the Frobenius objective by only a common multiplier; per-feature rescaling changes the objective's feature weighting. Do not add offsets. See [NMF input units](nmf.md#very-small-input-units) for the dictionary interpretation.

<!-- learner-example: minibatch-nmf-units -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

first = 1e-12 * np.array([[1., 2.], [2., 4.]])
second = 1e-12 * np.array([[3., 6.], [4., 8.]])
scale = float(first.max())
if not np.isfinite(scale) or scale <= 0:
    raise ValueError("Choose a finite positive training scale")
model = MiniBatchNMF(n_components=1, random_state=5, device="cpu")
for batch in (first, second):
    model.partial_fit(batch / scale)
X = np.vstack([first, second])
W = model.transform(X / scale)
X_hat = model.inverse_transform(W) * scale
print("relative reconstruction error:", np.linalg.norm(X - X_hat) / np.linalg.norm(X))
```

The relative error is close to zero for these proportional positive batches. The fixed scale works here because both batches have comparable magnitudes; it is not a guarantee for arbitrary streams or dynamic ranges. Inspect relative and feature-wise errors rather than relying only on a small absolute `reconstruction_err_`.

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
Sparse input is unsupported. `fit`, `partial_fit`, `fit_transform`, `transform`, and `predict` require nonnegative observations. `inverse_transform` only multiplies supplied coordinates by `components_`; it accepts negative coordinates and can return negative values. Pass nonnegative factors when a nonnegative reconstruction is required.

**Does it support CD solver or other beta losses?**
No. The current implementation supports MU-style updates and Frobenius loss only.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [MiniBatchNMF API reference](api-reference.md#minibatchnmf).

## References

- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. *Advances in Neural Information Processing Systems*, 13.
- Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S.-I. (2009). *Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation*. Wiley.
- scikit-learn Developers. `sklearn.decomposition.MiniBatchNMF`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
