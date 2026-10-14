# NMF

> Language: English
> Last updated: 2026-10-14
> Switch: [Chinese](../../cn/unsupervised/nmf.md)

## Overview

Suppose you measure 60 samples at the same five wavelengths. The nonnegative data table `X` has shape `(60, 5)`: one row per sample and one column per wavelength. Instead of assigning each sample to one type, you want to approximate its five intensities using two basic spectra, A and B.

NMF learns two tables together. `H`, with shape `(2, 5)`, stores A and B as its two rows. `W`, with shape `(60, 2)`, stores each sample's two nonnegative coefficients: how much to multiply A and B by before adding them. More generally, for `n` samples, `p` measurements, and `k` basic spectra, `X` is `(n, p)`, `H` is `(k, p)`, and `W` is `(n, k)`.

For example, let A be `[1, 0, 2, 0, 1]` and B be `[0, 2, 0, 2, 1]`. A sample with coefficients `[2, 0.5]` is approximated by `2*A + 0.5*B = [2, 1, 4, 1, 2.5]`. Compare those five reconstructed intensities with its five measured intensities to judge the approximation. These illustrative values show the multiplication; they are not a promised fitted result.

<!-- worked-example: nonnegative-spectra -->
```python
import numpy as np

H = np.array([[1., 0., 2., 0., 1.], [0., 2., 0., 2., 1.]])
w = np.array([2., 0.5])
x_hat = w @ H
print(x_hat)  # [2.  1.  4.  1.  2.5]
```

The coefficients are multipliers, not the number of basic spectra or probabilities. Here `n_components=2` sets the number of basic spectra; this sample gives them coefficients 2 and 0.5, which need not sum to one. Both tables are learned from data and contain only zero or positive values, so reconstruction adds contributions without positive and negative cancellation. The learned basic spectra need not be actual pure substances, and the decomposition need not be unique. The table of basic spectra is also called a dictionary; its per-sample coefficients are called activations.

## When to use it

Use NMF for dense nonnegative measurements when sums of patterns are a useful explanation. PCA can compress the data, but its signed directions can reconstruct observations by cancellation, which is awkward when the intended explanation is additive. Do not mean-center the input; if signed departures from a baseline are the object of interest, [PCA](pca.md) may be more appropriate. [TruncatedSVD](truncated-svd.md) also approximates an uncentered matrix, but permits cancellation through signed factors. Consider [MiniBatchNMF](minibatch-nmf.md) when a shared dictionary must be learned in batches.

`n_components` sets the number of basic spectra, not necessarily the actual matrix rank. Hold out representative samples, fit on the others, then use `transform` and `inverse_transform` to compare held-out reconstructed intensities with their measurements. Repeat with a few random seeds to see whether similar basic spectra appear. Before comparing, match similar spectra and account for scale: doubling A and halving its coefficients leaves the reconstruction unchanged.

The fitting objective adds up squared intensity errors (and multiplies the total by one half); this is the squared Frobenius reconstruction loss below. Changing a column to much larger numerical units gives it more influence on that total. Nonnegative input is required, but does not by itself make this error measure suitable for your problem.

## Path

```python
from statgpu.unsupervised import NMF
```

This estimator supports dense input on CPU, CuPy/CUDA, and Torch CUDA.

## Objective Function / Loss Function

The fitted factors solve the non-convex constrained problem:

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

`components_` stores `H`; `fit_transform` returns `W`.

<a id="estimating-equation"></a>
## Fitting steps and update formulas

The implementation uses multiplicative updates:

$$
W \leftarrow W \odot
\frac{XH^\top}{WHH^\top + \varepsilon}
$$

$$
H \leftarrow H \odot
\frac{W^\top X}{W^\top W H + \varepsilon}
$$

With `init="random"`, the seed controls data-row sampling for the initial dictionary when there are at least as many rows as components; otherwise it uses positive mean-scaled random entries. Initial activations are derived from the data and dictionary. Reconstruction error is checked periodically and at the final iteration; the check cadence depends on the backend. `transform(X)` keeps fitted `H` fixed and updates a new `W` for the new data. It always runs `max_iter` multiplicative updates; `tol` controls stopping during `fit` only and does not stop the transform solve early.

## Parameters

- `n_components`: number of basic spectra (the reconstruction has rank at most this number); `None` uses `min(n_samples, n_features)`.
- `init`: only `"random"` is supported.
- `solver`: only `"mu"` is supported.
- `beta_loss`: only `"frobenius"` is supported.
- `max_iter`, `tol`, `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## A small CPU example

<!-- learner-example: nmf -->
```python
import numpy as np
from statgpu.unsupervised import NMF

rng = np.random.default_rng(0)
X = rng.uniform(0.1, 1.0, (60, 2)) @ rng.uniform(0.1, 1.0, (2, 5))
model = NMF(n_components=2, max_iter=100, random_state=0, device="cpu")
W = model.fit_transform(X)
X_hat = model.inverse_transform(W)
print(W.shape, model.components_.shape, np.linalg.norm(X - X_hat))
```

`components_[0]` and `components_[1]` are the two learned basic spectra A and B, each containing five intensities. `W[i]` contains sample `i`'s two coefficients. Thus `W` is `(60, 2)`, `components_` is `(2, 5)`, and `W @ model.components_` reconstructs all 60 samples as a `(60, 5)` table. `reconstruction_err_` is a Frobenius norm, not its square or a normalized per-row error. A later `transform` solves new activations with the dictionary held fixed.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#nmf) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Very small input units

The fixed absolute stabilizers in the multiplicative updates can dominate very small positive observations. For example, an exactly rank-one positive matrix expressed at a scale of `1e-12` can produce an all-zero reconstruction, even with `tol=0`. More iterations alone do not repair that collapsed state. A small absolute `reconstruction_err_` can be misleading when the observations themselves are tiny; also inspect relative and feature-wise reconstruction errors.

Choose one finite positive scale from representative training data and divide every feature by that same scale before fitting. Use the same scale for later `transform` calls, and multiply reconstructed observations by it to recover the original units. This preserves nonnegativity and changes the Frobenius objective by a common positive multiplier; separate per-feature scaling changes feature weighting. Do not mean-center or add a positive offset to address this limitation.

<!-- learner-example: nmf-units -->
```python
import numpy as np
from statgpu.unsupervised import NMF

X = 1e-12 * np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
scale = float(X.max())
if not np.isfinite(scale) or scale <= 0:
    raise ValueError("Choose a finite positive training scale")
model = NMF(n_components=1, random_state=5, device="cpu")
W = model.fit_transform(X / scale)
X_hat = model.inverse_transform(W) * scale
new_rows = 1e-12 * np.array([[5., 10.]])
W_new = model.transform(new_rows / scale)
new_hat = model.inverse_transform(W_new) * scale
print("relative reconstruction error:", np.linalg.norm(X - X_hat) / np.linalg.norm(X))
print("new reconstruction in original units:", new_hat)
```

The relative error is close to zero for this simple rank-one example, and `new_hat` is close to `[[5e-12, 1e-11]]`. Scaling is a numerical precaution, not a guarantee of convergence on arbitrary data. The fitted dictionary is in scaled units: `W @ (model.components_ * scale)` reports the same original-unit reconstruction without modifying the estimator. Keep the fitted dictionary unchanged for subsequent `transform` calls.

## Approximation and interpretation

NMF does not provide standard errors or significance tests. The objective is non-convex, and multiplicative updates seek a local solution whose quality depends on initialization and stopping criteria; exhausting the iteration budget is not proof of convergence.

## Outputs

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**Can input contain negative values?**
`fit`, `fit_transform`, `transform`, and `predict` reject negative observations. `inverse_transform` only multiplies supplied coordinates by `components_`; it accepts negative coordinates and can return negative values. Pass nonnegative factors when a nonnegative reconstruction is required.

**Is coordinate descent supported?**
No. The current implementation supports only MU with Frobenius loss.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [NMF API reference](api-reference.md#nmf).

## References

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13* (pp. 556-562). MIT Press.
- scikit-learn Developers. [NMF: nonnegative factors and reconstruction objectives](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).
