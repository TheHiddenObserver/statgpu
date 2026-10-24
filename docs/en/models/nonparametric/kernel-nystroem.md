# Nystroem Kernel Approximation

> Language: English  
> Last updated: 2026-10-23<br>
> Switch: [Chinese](../../../cn/models/nonparametric/kernel-nystroem.md)

## Replace a full kernel with reusable features

A full kernel compares every training observation with every other one.
`Nystroem` instead compares observations with a subset of training rows called
landmarks, then normalizes those similarities to form an approximate kernel
feature map. With $m\ll n$ landmarks, the resulting $n\times m$ feature matrix
can be reused by a downstream model when the full $n\times n$ training kernel
would be too large.

Use [KernelRidge](kernel-ridge.md) when you can afford the exact kernel and want
a direct regression model. Use [KernelPCA](kernel-pca.md) for a centered
nonlinear coordinate system based on leading kernel principal components.
Nystroem's output is a set of approximate kernel features; it is not itself a
response prediction.

<a id="theory"></a>

## How landmarks define the map

The map needs both the similarities to the landmarks and the similarities
among the landmarks themselves. The latter provide the normalization that
turns raw similarity columns into approximate kernel features.

Choose $m=\min(\texttt{n\_components},n)$ training observations without replacement.
Let $K_{mm}$ be their landmark-to-landmark kernel and $K_{nm}$ the similarities
of all input rows to those landmarks. For a positive-semidefinite landmark
kernel with decomposition

$$
K_{mm}=V\Lambda V^\top,
$$

the symmetric inverse-square-root feature orientation has the form

$$
Z=K_{nm}V\Lambda^{-1/2}V^\top.
$$

Ignoring the numerical floor and using the appropriate pseudoinverse when
needed, $ZZ^\top$ approximates the full kernel as
$K_{nm}K_{mm}^{\dagger}K_{mn}$. Landmarks provide a smaller set of similarity
comparisons through which to represent the observations.

The implementation uses **selected-backend SVD**, rather than directly using signed
eigenvalues:

$$
K_{mm}=U\Sigma V^\top,\qquad
N=U\operatorname{diag}\!\left(
\frac{1}{\sqrt{\max(\sigma_j,10^{-12})}}\right)V^\top,\qquad
Z=K_{nm}N.
$$

`normalization_` stores $N$. Despite its name, `eigenvalues_` stores the
**floored singular values**, not signed eigenvalues. Small singular values are
floored, not removed, so the map keeps all $m$ selected landmarks and returns
$m$ columns. For a positive-semidefinite kernel this corresponds to the
symmetric inverse-square-root orientation, with the floor for small values.
An indefinite kernel does not imply an exact Euclidean representation: any
$ZZ^\top$ is positive-semidefinite and cannot reproduce a genuinely indefinite
kernel exactly.

<a id="example"></a>

## A complete CPU map and prediction example

The following self-contained example selects 40 landmarks only from the 180
training rows, maps both training and held-out rows with the same object, then
fits an unweighted linear ridge predictor on the mapped features. The small
NumPy solve makes the downstream model explicit and needs no additional model
library. Its penalty `downstream_alpha` uses summed squared loss and no separate
intercept; it is distinct from the landmark count and kernel scale.

<!-- example: kernel-nystroem-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import Nystroem, pairwise_kernels
```

### Prepare training and held-out rows

Keep held-out inputs out of landmark selection.
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### Fit and reuse the feature map

The test rows use the training landmarks and normalization.
The example uses the default RBF kernel,
$k(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$. Its `gamma=0.7` sets the distance
scale: larger `gamma` makes similarity decay faster with distance.
`n_components=40` limits the number of training landmarks used in the map.

```python
nystroem = Nystroem(
    n_components=40, gamma=0.7, random_state=42, device="cpu",
)
train_features = nystroem.fit_transform(X_train)
test_features = nystroem.transform(X_test)
```

### Check the learned map

Inspect both the normalization and transformed features.
```python
assert np.isfinite(nystroem.normalization_).all()
assert np.isfinite(nystroem.eigenvalues_).all()
assert np.isfinite(train_features).all()
assert np.isfinite(test_features).all()
print("Training / test feature shapes:", train_features.shape, test_features.shape)
```

### Fit a downstream predictor

Learn response weights from the training feature matrix.
```python
downstream_alpha = 1.0
weights = np.linalg.solve(
    train_features.T @ train_features
    + downstream_alpha * np.eye(train_features.shape[1]),
    train_features.T @ y_train,
)
prediction = test_features @ weights
assert np.isfinite(prediction).all()
print("Prediction shape:", prediction.shape)
print("Held-out prediction MSE:", np.mean((y_test - prediction) ** 2))
```

### Compare the approximate kernel on small data

To see what the features approximate, compare their pairwise inner products
with the exact kernel on the same test rows. This optional diagnostic
deliberately forms a dense kernel, so keep it to small data.
```python
# Small-data diagnostic only: this explicitly forms a dense test kernel.
K_test = pairwise_kernels(X_test, metric="rbf", gamma=0.7, xp=np)
approximation_error = np.linalg.norm(test_features @ test_features.T - K_test)
relative_kernel_error = approximation_error / np.linalg.norm(K_test)
print("Held-out relative kernel approximation error:", relative_kernel_error)
```
<!-- example-end: kernel-nystroem-cpu -->

The map produces `(180, 40)` training features and `(60, 40)` held-out features.
Only after fitting `weights` using `y_train` do we obtain `(60,)` response
predictions. `Nystroem.predict(X)` is merely an alias for `transform(X)`; it
would return the 40 features, not those response predictions. The class does
not provide a model-specific score or an inverse transform.

<a id="approximation-check"></a>

## Separate kernel approximation from prediction error

The example's small-data diagnostic measures relative Frobenius error:

$$
\frac{\lVert Z_{\mathrm{test}}Z_{\mathrm{test}}^\top-K_{\mathrm{test,test}}\rVert_F}
     {\lVert K_{\mathrm{test,test}}\rVert_F}.
$$

A smaller value means the learned map more closely approximates the exact RBF
kernel on these rows. It does **not** necessarily imply smaller response
prediction error: that also depends on the target, downstream model,
regularization, and data distribution. Prediction MSE uses `y_test` and answers
a different question. These are evaluation diagnostics, not coefficient-level
standard errors or confidence intervals.

This diagnostic explicitly allocates a dense test kernel and is suitable only
for small data. Do not use it at a scale where materializing a full kernel is
what you are trying to avoid. Do not repeatedly tune landmarks or kernel
settings using the held-out diagnostics; keep model selection within training
data.

## Choose landmarks and fit within folds

The main approximation choice is `n_components`, which trades capacity against
memory and computation. The actual landmark count is capped at the number of
training rows. More landmarks do not by themselves guarantee better downstream
predictions. Since the map also depends on which rows are selected, fix
`random_state` for reproducible landmark sampling and assess sensitivity to
landmark choices within the training procedure when the application warrants it.

The kernel determines which similarities the map approximates. For RBF, `gamma`
sets the distance scale and defaults to `1 / n_features`; feature units and
scaling change these similarities. Other kernels have their own controls and
domain restrictions. Choose the downstream model's regularization as part of
the same training procedure: a good kernel approximation is not a substitute
for validating that model.

For cross-validation, fit scaling or imputation **and the Nystroem map separately
within each training fold**. Use that fold's map to transform its training and
validation rows, fit the downstream model on the training features, and evaluate
it on the validation features. Selecting landmarks on the whole
dataset before CV leaks validation-input information. After selecting settings,
refit preprocessing, landmarks, and the downstream model on all outer training
rows and apply them unchanged to held-out data. Use group- or time-aware splits
when the observations require them. The
[external-loop pattern](kernel-ridge-cv.md#fold-local-preprocessing) illustrates
fold-local preprocessing; a Nystroem pipeline must also learn its map inside
that loop.

The [complete kernel API](../../reference/kernel-methods-api.md) covers the remaining
controls, along with every method, fitted array, kernel function, and output shape.

## Numerical and execution boundaries

Finite numeric inputs are required for `fit` and `transform`; NaN/Inf inputs
are rejected. A one-dimensional `X` means multiple observations of one feature.
Keep feature order and count unchanged for queries. Chi-squared requires
nonnegative input; a custom kernel must obey pairwise shape and backend
requirements for both landmark and query inputs.

Finite inputs do not ensure finite kernels. Polynomial overflow or invalid
kernel controls can produce a successful return with nonfinite normalization,
singular values, or features. Check `normalization_`, `eigenvalues_`, and mapped
outputs as above; discard nonfinite fits and fit a fresh model after correcting
scaling or settings. More downstream regularization cannot repair an already
overflowed kernel. Finiteness is necessary, not sufficient, for a useful map.
For RBF data with large common coordinate offsets, apply the same
training-derived offset to training and query rows to reduce distance
cancellation. This is not a general rule for every kernel. Refit after parameter
changes; earlier fitted arrays can otherwise remain present.

The representation uses $O(nm)$ feature/kernel storage plus $O(m^2)$ landmark
storage and dense landmark linear algebra, rather than a full $n\times n$
training kernel. The normalization is dense, so applying it also has a cost.
Nystroem is an explicit low-rank approximation, not a silent fallback for an
exact estimator or an approximate inference mode.

Explicit unavailable device requests raise. KernelPCA and Nystroem convert fit and transform inputs through the selected backend, including host inputs explicitly requesting Torch CUDA. KernelPCA propagates decomposition failures rather than retrying on CPU. Nystroem computes its landmark kernel and SVD on the selected backend; seeded landmark-index selection remains on the host. Custom kernels must support the selected numerical backend.

Public fitted arrays remain NumPy reporting/storage boundaries. Inspect returned features with Torch `.device`/`.is_cuda` or CuPy `.device`; host fitted arrays alone do not prove where numerical work ran. Input transfers and public-array exports are part of end-to-end cost. Use `device="cpu"` for CPU execution. These contracts do not imply GPU performance or physical validation.

When comparing other implementations, align selected landmarks, kernel settings,
feature orientation, and the `1e-12` singular-value floor. CPU comparisons do
not establish GPU accuracy or performance.

## Further reading

- [KernelRidge: exact kernel regression](kernel-ridge.md)
- [Kernel-methods overview](kernel-methods.md)
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
