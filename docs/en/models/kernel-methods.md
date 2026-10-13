# Kernel Methods

> Language: English  
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/kernel-methods.md)

This page explains the shared ideas before you choose a method. Each linked tutorial has its own complete CPU example; the [complete API reference](../reference/kernel-methods-api.md) collects signatures, parameters, outputs and execution limits.

## Choose a tool for your question

A kernel measures similarity between observations. Instead of specifying every
nonlinear interaction as an input column, a model can use these similarities to
learn a curved response or a new representation of the data.

- For **predicting a numeric response**, use `KernelRidge`. `KernelRidgeCV` selects
  its regularization strength from training folds, then refits on the training
  data you supplied. RBF kernels are useful when nearby observations should have
  similar responses without a known parametric curve.
- For **nonlinear coordinates without a response**, use `KernelPCA`. Its output
  is an embedding for visualization or a later model, not a response prediction.
- For **a reusable, smaller kernel feature map**, use `Nystroem`. It approximates
  a full kernel using a chosen number of training landmarks; feed its columns to
  a downstream model when an exact training kernel is too large.

Start with a [linear model](linear-regression.md) when a straight-line effect is adequate or
interpretable slopes matter. A [GAM](semiparametric.md) is often easier to interpret
when the nonlinearity is a sum of separate feature effects. Exact kernel methods
can model interactions more flexibly, but store a quadratic-size training kernel.
Choose feature units/scales deliberately: RBF similarity depends on distance.
Any learned preprocessing must be fitted within training folds during CV.

## From a similarity to a Gram matrix

For observations $x_i$ and $x_j$, a kernel $k(x_i,x_j)$ assigns a similarity. A positive-semidefinite kernel can be interpreted as an inner product in a possibly high-dimensional feature space:

$$
k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle.
$$

The map $\phi$ need not be built explicitly. Algorithms can use its pairwise inner products instead: this is the kernel trick. This interpretation requires a symmetric positive-semidefinite kernel; accepting a callable or a sigmoid kernel does not by itself guarantee that property for your data and parameters.

For $n$ training rows, the Gram matrix is $K_{ij}=k(x_i,x_j)$, with shape $(n,n)$. For $q$ new rows, the query-to-training kernel has shape $(q,n)$. Its columns refer to training observations, not original features; keep the training order and reuse the fitted model. Similarity is not a probability, and it is not a correlation coefficient in general.

### A three-point example

The RBF kernel is $k(x,y)=\exp(-\gamma\lVert x-y\rVert_2^2)$. With one-dimensional points 0, 1 and 2 and $\gamma=\log 2$, distances of 1 and 2 give similarities $1/2$ and $1/16$:

$$
K=\begin{pmatrix}
1 & 1/2 & 1/16\\
1/2 & 1 & 1/2\\
1/16 & 1/2 & 1
\end{pmatrix}.
$$

<!-- example: kernel-gram-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import pairwise_kernels

X = np.array([[0.0], [1.0], [2.0]])
```

Compute the training Gram matrix and compare with the hand calculation.

```python
K = pairwise_kernels(X, metric="rbf", gamma=np.log(2.0), xp=np)
expected = np.array([[1.0, 0.5, 0.0625],
                     [0.5, 1.0, 0.5],
                     [0.0625, 0.5, 1.0]])
np.testing.assert_allclose(K, expected)
```

New rows use similarities to those same three training observations.

```python
query = np.array([[0.0], [2.0]])
K_query = pairwise_kernels(query, X, metric="rbf", gamma=np.log(2.0), xp=np)
np.testing.assert_allclose(K_query, expected[[0, 2]])
print(K.shape, K_query.shape)
```
<!-- example-end: kernel-gram-cpu -->

The shapes are `(3, 3)` and `(2, 3)`. The diagonal is one because each point has zero distance from itself. The first and third training points are less similar than adjacent points. This example computes similarities only; it has not fitted a response model or an embedding.

## Feature units and kernel scale

For RBF, increasing `gamma` makes similarity decay over shorter distances. Very small `gamma` can make many rows look nearly identical; very large `gamma` can make the training matrix almost diagonal and leave new rows weakly related to every training row. Neither extreme guarantees useful predictions. Polynomial `degree` and `coef0` change a different notion of similarity; do not interpret them as RBF distance controls.

Rescaling a feature changes distances and therefore the question the kernel asks. Decide whether original units or comparable standardized scales are appropriate. Fit any means and scales only on training rows, reuse them on query rows, and learn preprocessing again inside each validation fold. Centering the Gram matrix in KernelPCA is feature-space centering; it does not standardize the original input columns.

`gamma=None` generally uses `1 / n_features` for the kernels with that default; it is not an adaptive bandwidth search. Chi-squared uses a different default and requires nonnegative features, so blindly standardizing it can invalidate its domain. See [kernel controls and forwarding](../reference/kernel-methods-api.md#kernel-parameter-forwarding).

## Method tutorials

<a id="kernel-cpu-workflow"></a>
<a id="a-training-and-held-out-cpu-workflow"></a>
<a id="prepare-and-split-the-data"></a>
<a id="fit-one-kernel-ridge-model"></a>
<a id="predict-and-interpret"></a>

### [Kernel ridge: fit and predict](kernel-ridge.md#example)

Fit a numeric response on training data, then predict held-out rows. The tutorial explains the objective, kernel coefficients versus raw-feature slopes, and numerical checks.

<a id="kernel-cv-workflow"></a>
<a id="kernel-ridge-cross-validation"></a>
<a id="select-alpha-within-training-folds"></a>

### [Kernel ridge cross-validation: select alpha](kernel-ridge-cv.md#selection)

Built-in CV selects only alpha using MSE, while best_score_ reports R². The standalone tutorial explains why preprocessing, grouped splits and time splits need an external loop.

<a id="kernel-pca"></a>
<a id="learn-and-reuse-nonlinear-coordinates"></a>

### [KernelPCA: nonlinear coordinates](kernel-pca.md#example)

Learn and reuse coordinates of the centered kernel without a response. The tutorial covers new-data projection, retained dimensions, sign ambiguity and failed-refit handling.

<a id="nystroem-approximation"></a>
<a id="build-a-smaller-kernel-feature-matrix"></a>
<a id="optional-check-the-approximation-on-small-data"></a>

### [Nystroem: low-rank kernel features](kernel-nystroem.md#example)

Select training landmarks and produce fewer feature columns for a downstream model. The tutorial includes prediction and a small-data kernel diagnostic, separating approximation error from prediction error.

<a id="optional-gpu-examples"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>

### [Optional GPU examples](kernel-ridge-cv.md#gpu-examples)

Change devices after completing the CPU cross-validation example. KernelPCA and Nystroem have the decomposition and output limits described below.

## Built-In Kernels

| Kernel | Definition |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

The chi-squared kernel requires non-negative input features. A callable kernel
may be supplied where accepted by the estimator; it must return an array on the
requested backend and obey the expected pairwise-kernel shape.

<a id="complete-estimator-api"></a>
<a id="constructor-controls"></a>
<a id="methods-and-shapes"></a>
<a id="fitted-attributes-and-outputs"></a>

## Look up the complete API

The [kernel methods API reference](../reference/kernel-methods-api.md) collects all four estimators and eight functions: full signatures, defaults, methods, fitted attributes, array shapes, callable kernels and backend requirements.

Find constructor controls, method shapes and fitted attributes under [KernelRidge](../reference/kernel-methods-api.md#kernelridge), [KernelRidgeCV](../reference/kernel-methods-api.md#kernelridgecv), [KernelPCA](../reference/kernel-methods-api.md#kernelpca), and [Nystroem](../reference/kernel-methods-api.md#nystroem).

<a id="reading-the-cross-validation-results"></a>

The [CV result fields and shapes](../reference/kernel-methods-api.md#kernelridgecv) explain candidate arrays, fold tables, fold means, and how best_score_ differs from the selection loss.

<a id="complete-pairwise-function-api"></a>

The [pairwise-function API](../reference/kernel-methods-api.md#pairwise-kernels) covers pairwise_kernels and the seven kernels, including the callable example, xp, dtype, and parameter forwarding.

<a id="how-the-methods-work"></a>
<a id="kernel-ridge-regression"></a>

The [kernel ridge objective and prediction equations](kernel-ridge.md#theory) explain the regularized linear system, summed loss, and kernel coefficients.

<a id="computing-kernel-principal-components"></a>

[KernelPCA centering and projection](kernel-pca.md#theory) derives new-data coordinates from the training Gram matrix.

<a id="the-nystroem-feature-map"></a>

The [Nystroem SVD feature map](kernel-nystroem.md#theory) explains the landmark kernel, normalization, and low-rank approximation.

## Backend and Execution Boundaries

All four estimators accept `device`, but kernel construction, decomposition,
and outputs do not necessarily stay on one device:

- **KernelRidge / KernelRidgeCV** construct kernels and solve on the selected backend.
- **KernelPCA** normally decomposes on the selected backend. A linear-algebra
  exception triggers a NumPy CPU eigendecomposition without a separate fallback
  indicator. Stored training data and projection coefficients are also NumPy
  arrays, so do not assume a fully device-resident fit.
- **Nystroem** copies selected landmarks to NumPy, constructs their kernel and
  performs its SVD on CPU, then uses the selected array library for query-to-landmark
  kernels and returned features, subject to the placement limitation below.
  Custom kernels must also accept NumPy landmark inputs. Landmark indices,
  normalization, and singular values remain host arrays; selecting a GPU does
  not move the landmark decomposition there.

Explicit unavailable device requests raise.

KernelPCA and Nystroem still reject `device="torch"` when Torch CUDA is unavailable. Passing that check does not guarantee CUDA execution: NumPy or Torch CPU input can remain on CPU during fitting and transformation because the requested-device conversion is not consistently applied.

Inspect the actual `fit_transform`, `transform`, and `predict` outputs with Torch `.device`/`.is_cuda` (or CuPy `.device`), rather than relying on the configured device or selected backend/library.

Their public fitted arrays are deliberately NumPy and cannot prove numerical-device placement. This also affects Nystroem's query kernel and output, separately from its intentional CPU landmark SVD.

If CUDA placement is required, reject CPU outputs before using them. For a predictable CPU alternative, pass NumPy input with `device="cpu"`. See the [device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions).

`KernelPCA` and `Nystroem` reject NaN/Inf in the supplied input arrays during
fitting and transformation. This input check does not guarantee finite computed
kernels or learned arrays; see [finite-result checks](#check-for-nonfinite-kernel-fits).
Kernel-specific domain checks, such as non-negative inputs for the chi-squared
kernel, fail explicitly.

<a id="inference-semantics"></a>

## Evaluate the model

Kernel methods do not currently expose coefficient-level standard errors,
hypothesis tests, or confidence intervals. Model quality is evaluated through
prediction scores, cross-validation loss, embedding properties, kernel-matrix reconstruction
or approximation diagnostics, and application-specific validation. KernelPCA does not provide an inverse transform to reconstruct original input rows.

There is no strict/approximate inference mode distinction in this module.
Nystroem is an explicit low-rank kernel approximation, not a silent fallback for
an exact kernel estimator.

## Complexity and Performance Notes

- Exact kernel methods materialize an $n\times n$ training kernel and therefore
  have quadratic memory cost.
- Direct kernel-ridge solves and dense eigendecompositions have cubic worst-case
  arithmetic cost in the number of training observations.
- `KernelRidgeCV` may reuse decompositions across alpha values, but CV still
  multiplies work across folds.
- `Nystroem` reduces kernel storage to $O(nm)$ plus landmark linear algebra.
- GPU speed depends on sample size, dtype, kernel, synchronization, and available
  memory; small problems may be faster on CPU.

For example, an $n=10{,}000$ float64 square matrix alone occupies about 800 MB, before factorization workspace and copies. With $m=100$, a Nystroem $(n,m)$ feature array takes about 8 MB, but still needs landmark storage and $O(nm^2+m^3)$ dense normalization/SVD work; downstream fitting adds its own cost. The memory saving trades against approximation accuracy, and fewer columns alone say nothing about prediction quality.

<a id="limitations-and-failure-modes"></a>

## Numerical issues and practical precautions

### After a failed refit

If `KernelPCA.fit` or `fit_transform` raises, discard that instance. Create a
fresh `KernelPCA` and fit it successfully before transforming query points; do
not continue using the apparently retained old fit.

A failed refit can overwrite training-kernel centering information while leaving
old eigenvectors and the fitted flag intact. Subsequent `transform` or `predict`
calls can then return finite but incorrect coordinates, even for the original
training data. Checking only output finiteness cannot detect this mixed state.

Finite constant data can trigger this failure because its centered kernel has
no positive eigenvalue directions. A fresh instance will reject the same
degenerate data. Recreating the object is not a way to produce a valid embedding
from a zero-rank kernel; reconsider the data or analysis instead.

### Check for nonfinite kernel fits

Finite training data can still overflow a computed polynomial kernel. Current
`KernelRidge` and `Nystroem` fits can then return successfully with NaN learned
arrays and predictions/features; invalid kernel controls such as NaN `gamma`
can also produce this result. A linear-algebra warning or the absence of an
exception is not a reliable fit-status check.

Use finite applicable kernel controls and check learned arrays and query
outputs before use. For NumPy CPU fits, check `np.isfinite(model.dual_coef_).all()`
for KernelRidge, and `np.isfinite(model.normalization_).all()` plus
`np.isfinite(model.eigenvalues_).all()` for Nystroem. Also check predictions or
transformed features; use equivalent backend-native checks on device arrays.
Discard nonfinite fits. Review feature units/scales and kernel settings, then
fit a fresh model. Increasing `alpha` cannot repair a kernel that has already
overflowed. Scaling changes polynomial similarity, so choose and validate that
preprocessing within training folds. Finiteness is necessary, not sufficient,
for a well-conditioned or scientifically useful model.

### Large coordinate offsets with RBF kernels

For RBF kernels, subtract the same training-derived offset from training and query features when coordinates have a large common offset. Current squared-distance arithmetic can otherwise lose the differences between nearby points: coordinates near `1e9` can produce an almost all-ones kernel and badly changed predictions. Centering preserves the intended RBF kernel; it is not a general preprocessing rule for every kernel, especially chi-squared.

### Other practical precautions

If the kernel matrix is poorly conditioned, consider increasing `alpha` or
adjusting the kernel scale; distance-based kernels such as RBF are particularly
sensitive to `gamma`. Custom kernels must satisfy the selected estimator’s
backend, dtype, shape, and symmetry requirements. See the complexity discussion
above for large-sample memory costs and cross-validation overhead.

<a id="validation-guidance"></a>

## Compare with other implementations

For kernel ridge, verify the regularized linear system or compare with scikit-learn using the same kernel, alpha, data, and unweighted objective. For CV, inspect finite per-candidate MSE and verify the final refit uses the selected alpha. For Nystroem, align selected landmarks and the SVD singular-value floor before comparing normalization or features. CPU checks do not establish GPU accuracy or performance; validate those on the intended device.

## FAQ

### Should I use KernelRidge or Nystroem plus a linear model?

Use exact `KernelRidge` when the training kernel fits comfortably in memory and the
exact kernel representation is important. Use Nystroem when a controlled low-rank
feature approximation is preferable for scale or downstream reuse.

### Why can a GPU kernel method be slower than CPU?

Kernel construction and linear algebra must be large enough to amortize device
launch, synchronization, and memory-transfer overhead.

### Does `device="auto"` silently change an explicit request?

No. `"auto"` is itself an automatic selection request. An explicit `"cuda"` or
`"torch"` request fails if that backend is unavailable. KernelPCA/Nystroem can still leave CPU inputs on CPU after that check passes; inspect actual returned features as described above.

### Are KernelPCA components directly comparable across separate fits?

Eigenvector signs and bases within repeated or nearly repeated eigenspaces are not
uniquely identified. Compare represented subspaces or downstream quantities when
that ambiguity matters.

## References

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.

