# MiniBatchNMF

> Language: English
> Last updated: 2026-10-14
> Switch: [Chinese](../../cn/unsupervised/minibatch-nmf.md)

## Overview

MiniBatchNMF approximates a nonnegative data table using a chosen number of shared basic patterns. Each row of the input is a sample and each column is a feature. It learns two nonnegative tables: `H` contains one basic pattern per row, with a value for every feature; `W` contains one row per sample, giving the coefficient that multiplies each pattern. Multiplying each pattern by its sample-specific coefficient and adding the results reconstructs that sample approximately. In matrix form, `X` is approximated by `W @ H`.

For `n` samples, `p` features, and `k` patterns, `X` is `(n, p)`, `H` is `(k, p)`, and `W` is `(n, k)`. Both learned tables contain only zero or positive values: contributions add without positive and negative cancellation. The coefficients need not sum to one. These patterns need not be actual physical parts, and the decomposition need not be unique.

MiniBatchNMF learns the shared patterns from batches rather than updating from all observations together as in [NMF](nmf.md). With `partial_fit`, it estimates the current batch's coefficients, accumulates summary statistics, and updates the patterns without rereading all historical samples. In contrast, `fit` keeps the patterns fixed throughout one full pass over its input batches and updates them after that pass. Earlier coefficients are not updated automatically when the patterns change; after training, call `transform` again on samples you want to compare using the final patterns.

### Example: two basic spectra

Suppose you measure 60 samples at the same five wavelengths. Here `X` has shape `(60, 5)`, and the basic patterns are spectra. Instead of assigning each sample to one type, approximate its five intensities using two basic spectra, A and B. `H` has shape `(2, 5)` and stores A and B as its rows; `W` has shape `(60, 2)` and stores each sample's two coefficients.

For example, let A be `[1, 0, 2, 0, 1]` and B be `[0, 2, 0, 2, 1]`. A sample with coefficients `[2, 0.5]` is approximated by `2*A + 0.5*B = [2, 1, 4, 1, 2.5]`. Compare those five reconstructed intensities with its five measured intensities to judge the approximation. These illustrative values show the multiplication; they are not a promised fitted result.

<!-- worked-example: nonnegative-spectra -->
```python
import numpy as np

H = np.array([[1., 0., 2., 0., 1.], [0., 2., 0., 2., 1.]])
w = np.array([2., 0.5])
x_hat = w @ H
print(x_hat)  # [2.  1.  4.  1.  2.5]
```

The coefficients are multipliers, not the number of basic spectra or probabilities. Here `n_components=2` sets the number of basic spectra; this sample gives them coefficients 2 and 0.5, which need not sum to one. The learned basic spectra need not be actual pure substances. In general, the table of basic patterns is also called a dictionary, and the sample coefficients are called activations.

In this example, each incoming batch contains new samples measured at the same five wavelengths. Updating the shared patterns means updating the current A and B.

Because A and B change, yesterday's “2 times A” can mean different intensities from today's “2 times A”. For example, if A doubles while the sample stays the same, its matching coefficient must halve. This is why samples being compared must use coefficients computed with the same final basic spectra.

## When to use it

Choose MiniBatchNMF when batchwise fitting or incoming dense nonnegative observations are part of the problem. Use [NMF](nmf.md) as a full-data reference when the collection is manageable, and compare reconstructions using the same rank and preprocessing. If the question is about signed variation around a mean rather than additive patterns, [IncrementalPCA](incremental-pca.md) addresses that different objective.

Keep feature meaning, units, and rank fixed across batches. The current incremental updates accumulate historical statistics without a forgetting factor, so new batches do not automatically replace old operating conditions. Begin with representative data, including features expected to become active; the zero-feature initialization limitation below is especially important when early batches cover only a narrow part of the data.

## Path

```python
from statgpu.unsupervised import MiniBatchNMF
```

This estimator supports dense input on CPU, CuPy/CUDA, and Torch CUDA.

## Objective Function / Loss Function

For non-negative `W` and `H`, the estimator minimizes mini-batch approximations to the Frobenius reconstruction loss:

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

<a id="estimating-equation"></a>
## Fitting steps and update formulas

Each batch contributes a compact summary of how its estimated activations relate to the observations; those summaries guide dictionary updates.

For each batch, `MiniBatchNMF` approximately solves activations `W_batch` with fixed `H`. In `fit`, it holds `H` fixed for a whole epoch, sums `A = sum(W_batch.T @ W_batch)` and `B = sum(W_batch.T @ X_batch)`, then updates `H`. In `partial_fit`, these statistics accumulate across calls. Here the statistic matrices `A` and `B` are different objects from the two illustrative spectra called A and B above. The elementary multiplicative updates have the form:

$$
W \leftarrow W \odot \frac{XH^\top}{WHH^\top + \epsilon},
\qquad
H \leftarrow H \odot \frac{W^\top X}{W^\top WH + \epsilon}.
$$

## Parameters

- `n_components`: number of basic patterns (the reconstruction has rank at most this number); `None` uses `min(n_samples, n_features)`.
- `init`: currently supports `"random"`.
- `batch_size`, `max_iter`, `tol`, `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

`fit` receives and converts the whole input array before its internal batch loop. For an external stream, load dense batches yourself and call `partial_fit`; the estimator does not read files or accept sparse input. To compare observations after training, compute their activations with the final dictionary using `transform`, rather than assuming its solve equals the fitting updates.

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

Continuing the spectrum example, `components_[0]` and `components_[1]` are the two learned basic spectra A and B, each containing five intensities. `W[i]` contains sample `i`'s two coefficients. Thus `W` is `(60, 2)`, `components_` is `(2, 5)`, and `W @ model.components_` reconstructs all 60 samples as a `(60, 5)` table. The rows of `W` follow the original order of `X`, including the earlier batches, and all use the final basic spectra. After `partial_fit`, `reconstruction_err_` describes only that batch with the fitting factors; the independently computed full-data residual above answers a different question.

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

MiniBatchNMF is non-convex; incremental `partial_fit` updates depend on batch order. Ordinary `fit` aggregates statistics over an epoch before updating components. It is intended for scalable approximate factorization, and does not provide standard errors or significance tests.

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
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
