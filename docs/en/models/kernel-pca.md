# Kernel PCA

> Language: English  
> Last updated: 2026-10-12  
> Switch: [Chinese](../../cn/models/kernel-pca.md)

## Find nonlinear coordinates without a response

`KernelPCA` represents observations by a few coordinates derived from their
kernel similarities. Ordinary PCA finds linear combinations of the input
features; kernel PCA finds directions of variation in the feature space defined
by a kernel. Use it to explore nonlinear structure or to construct a
representation for a later model. No response variable is needed.

The output is an embedding, not a response prediction or a set of original
feature slopes. Use [KernelRidge](kernel-ridge.md) for supervised regression,
or [Nystroem](kernel-nystroem.md) when you want an approximate kernel feature map
with fewer landmarks. Start with ordinary PCA when linear coordinates suffice
and the dense training kernel would be unnecessarily expensive. The
[kernel-methods overview](kernel-methods.md) compares the available tasks.

<a id="theory"></a>

## Center, decompose, and project

Let $K_{ij}=k(x_i,x_j)$ be the training kernel and
$H=I-\mathbf 1\mathbf 1^\top/n$. Centering in feature space gives

$$
\widetilde K=HKH
=K-\mathbf 1\bar k_{\mathrm{col}}^\top
  -\bar k_{\mathrm{row}}\mathbf 1^\top
  +\bar k\mathbf 1\mathbf 1^\top,
$$

where the bars denote training-kernel column means, row means, and the overall
mean. This is kernel centering, not simply subtracting the mean of each input
feature. For the centered kernel, eigendecompose

$$
\widetilde K=V\Lambda V^\top.
$$

Retain up to `n_components` leading eigenvalues greater than `1e-12` and their
unit-norm eigenvectors. The projection coefficients and training coordinates are

$$
A=V\Lambda^{-1/2},\qquad
Z_{\mathrm{train}}=\widetilde K A=V\Lambda^{1/2}.
$$

`alphas_` stores $A$, so its columns are scaled eigenvectors rather than
unit-norm eigenvectors. `lambdas_` stores the retained eigenvalues of the
centered kernel, not explained-variance ratios for the original features.
With an indefinite kernel, negative directions are discarded; do not assume
this is ordinary PCA in a positive-semidefinite RKHS.

For new rows, first compute $K_*=K(X_{\mathrm{new}},X_{\mathrm{train}})$.
Each query is centered using the **training** column and overall means, and
that query's mean similarity to training rows:

$$
\widetilde K_*=
K_*-\mathbf 1\bar k_{\mathrm{col}}^\top
-\bar k_{*,\mathrm{row}}\mathbf 1^\top
+\bar k\mathbf 1\mathbf 1^\top,\qquad
Z_{\mathrm{new}}=\widetilde K_*A.
$$

This is why you reuse the fitted transformer for new data instead of fitting a
second one: separate fits learn different centering and coordinate systems.

<a id="example"></a>

## A complete CPU embedding example

This example runs independently. It learns a coordinate system from 180 training
observations, then applies it unchanged to 60 held-out observations with the
same two input features. No `y` enters the fit.

<!-- example: kernel-pca-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelPCA
```

### Prepare training and held-out rows

The response is generated only to match the other tutorials; it is unused.
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
# Generate the same example dataset used in the regression tutorials.
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
```

### Learn the coordinate system

Fit on training inputs, then reuse the same projection on test inputs.
```python
kpca = KernelPCA(n_components=3, gamma=0.7, device="cpu")
train_coordinates = kpca.fit_transform(X_train)
test_coordinates = kpca.transform(X_test)
```

### Inspect coordinates and retained directions

Check the output before using it in a plot or downstream analysis.
```python
assert np.isfinite(kpca.lambdas_).all()
assert np.isfinite(kpca.alphas_).all()
assert np.isfinite(train_coordinates).all()
assert np.isfinite(test_coordinates).all()
print("Training / test coordinate shapes:",
      train_coordinates.shape, test_coordinates.shape)
print("Retained centered-kernel eigenvalues:", kpca.lambdas_)
print("First two test coordinates:\n", test_coordinates[:2])
```
<!-- example-end: kernel-pca-cpu -->

The shapes are `(180, 3)` and `(60, 3)`. Three nonlinear coordinates can be
extracted from two original features because the kernel's feature space need
not have the same dimension as the input. The first coordinate corresponds to
the largest retained centered-kernel eigenvalue. You can plot the first two
columns to explore the embedding or use all retained columns in a downstream
analysis; neither step makes them response predictions.

If a downstream model or preprocessing step is tuned with cross-validation,
fit the entire preprocessing–KernelPCA–model sequence within each training fold.
Learning the embedding on all rows before evaluating a validation fold leaks
information about that fold's input distribution.

## Choose dimensions and similarity

- `n_components` is a **maximum**, not a promise of that many output columns.
  Only eigenvalues above `1e-12` survive. A low-rank centered kernel can yield
  fewer columns; no positive directions raises an error.
- `gamma` controls the distance scale of the RBF kernel. Large values emphasize
  very local similarity; small values make observations more similar. Choose
  feature units and preprocessing deliberately and validate the embedding for
  its intended use.
- `alpha` is a finite nonnegative diagonal shift applied before eigendecomposition.
  The shift is subtracted from eigenvalues afterward. It is **not** the ridge
  penalty of a supervised predictor and does not regularize an inverse transform.
- `eigen_solver="auto"` and `"dense"` currently both use dense symmetric
  eigendecomposition.

Use the [complete kernel API](../reference/kernel-methods-api.md) for every
constructor argument, fitted field, method signature, kernel choice, and shape.
`fit(X, y=None)` ignores `y`; `predict(X)` is an alias for `transform(X)`.
There is no model-specific score and no `inverse_transform` method. In
particular, this API does not reconstruct inputs or expose coefficient
inference.

## Interpret comparisons carefully

Eigenvector signs are arbitrary: one fit may return a component with the
opposite sign to another equally correct fit. Repeated or nearly repeated
eigenvalues also allow rotations of the corresponding eigenspace. Compare
represented subspaces, pairwise geometry, or downstream quantities when this
ambiguity matters, rather than requiring every coordinate column to match.
When comparing implementations, align kernel settings, training centering,
retained directions, and eigenvector normalization first.

A visually separated embedding is not by itself evidence of predictive
performance, meaningful clusters, or recovery of the original variables.
Evaluate properties relevant to the application, with held-out downstream
performance when prediction is the purpose.

<a id="failed-refit"></a>

## After a failed refit, discard the instance

If `fit` or `fit_transform` raises, create a fresh `KernelPCA` and fit suitable
data successfully before transforming any query. Do not reuse an apparently
retained old fit.

A failed refit can overwrite training-kernel centering while leaving old
eigenvectors and the fitted flag intact. Later `transform` or `predict` calls
can then return **finite but incorrect coordinates**, even for the original
training rows. A finiteness check cannot detect this mixed fitted state.

Finite constant data can trigger the problem because the centered kernel has
no positive eigenvalue directions. A fresh instance will reject the same
degenerate data; recreating the object does not make a zero-rank embedding valid.
Reconsider the data or the analysis. Also refit after any parameter changes:
`set_params` can leave earlier fitted arrays in place.

## Input, numerical, and execution boundaries

Provide finite numeric input with consistent feature order and count. A
one-dimensional `X` is treated as multiple rows of one feature. NaN/Inf input
is rejected during fitting and transformation, but finite inputs do not ensure
finite computed kernels: polynomial kernels can overflow. Check appropriate
kernel controls and learned arrays as well as transformed outputs, and discard
invalid results. Chi-squared kernels require nonnegative features. For RBF
coordinates with a large common offset, subtract the same training-derived
offset from training and query data to reduce squared-distance cancellation.

The training kernel uses quadratic memory, and dense eigendecomposition has
cubic worst-case arithmetic cost. Decomposition normally uses the selected
backend, but a linear-algebra exception triggers NumPy CPU eigendecomposition
without a separate fallback indicator. Stored `X_fit_`, `alphas_`, and
`lambdas_` are NumPy arrays even when numerical work uses a GPU.

Explicit unavailable `device="cuda"` or `"torch"` requests raise. However,
KernelPCA currently can leave NumPy or Torch CPU inputs on CPU after the Torch
CUDA availability check succeeds. Inspect returned features with Torch
`.device`/`.is_cuda` or CuPy `.device`; the selected backend and host fitted
arrays cannot establish output placement. If CUDA placement is required,
reject CPU outputs before use. NumPy input with `device="cpu"` is a predictable
CPU path. See the [device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)
and [API reference](../reference/kernel-methods-api.md).

## Further reading

- [Kernel-methods overview](kernel-methods.md)
- [Nystroem: approximate reusable features](kernel-nystroem.md)
- [Complete kernel API](../reference/kernel-methods-api.md)
- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
