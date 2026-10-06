# Kernel Methods

> Language: English  
> Last updated: 2026-10-06  
> Switch: [Chinese](../../cn/models/kernel-methods.md)

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

## A training and held-out CPU workflow

This seeded example creates a nonlinear response. It reserves held-out rows
before fitting, fixes the RBF scale `gamma`, and selects `alpha` using only the
training data. If you also tune `gamma` or a feature map for a prediction task,
keep that selection and the fitted preprocessing inside the training procedure;
do not repeatedly choose settings from the held-out scores.

<!-- example: kernel-methods-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import (
    KernelRidge, KernelRidgeCV, KernelPCA, Nystroem, pairwise_kernels,
)

rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

# Fix the kernel scale, and tune alpha only within the training rows.
kr = KernelRidge(alpha=1.0, gamma=0.7, device="cpu").fit(X_train, y_train)
kr_cv = KernelRidgeCV(
    alphas=[0.01, 0.1, 1.0, 10.0], gamma=0.7,
    cv=5, random_state=42, device="cpu",
).fit(X_train, y_train)
assert np.isfinite(kr_cv.cv_results_["mean_mse"]).all()
assert np.isfinite(kr_cv.best_score_)
prediction = kr_cv.predict(X_test)
held_out_r2 = kr_cv.score(X_test, y_test)
held_out_mse = np.mean((y_test - prediction) ** 2)
print("Fixed-alpha held-out R2:", kr.score(X_test, y_test))
print("Selected alpha:", kr_cv.alpha_)
print("CV model held-out R2 / MSE:", held_out_r2, held_out_mse)

# Learn feature maps on training rows; reuse them for held-out rows.
kpca = KernelPCA(n_components=3, gamma=0.7, device="cpu")
train_coordinates = kpca.fit_transform(X_train)
test_coordinates = kpca.transform(X_test)
nystroem = Nystroem(
    n_components=40, gamma=0.7, random_state=42, device="cpu",
)
train_features = nystroem.fit_transform(X_train)
test_features = nystroem.transform(X_test)
print("Held-out coordinate / feature shapes:",
      test_coordinates.shape, test_features.shape)

# Small diagnostic only: how well do these features approximate this kernel?
K_test = pairwise_kernels(X_test, metric="rbf", gamma=0.7, xp=np)
approximation_error = np.linalg.norm(test_features @ test_features.T - K_test)
relative_kernel_error = approximation_error / np.linalg.norm(K_test)
print("Held-out relative kernel approximation error:", relative_kernel_error)
```

R-squared compares response predictions with the held-out response mean: 1 is
perfect, 0 matches that constant predictor, and negative values are worse. MSE
is mean squared prediction error in squared response units; lower is better.
`best_score_` summarizes the training CV folds, so it is not the held-out result.
The two held-out scores illustrate evaluation, not a reason to keep tuning on
these same test rows.

KernelPCA's `(60, 3)` output contains coordinates, while Nystroem's `(60, 40)`
output contains approximate kernel features. Neither predicts `y` on its own.
The relative kernel error compares `Z @ Z.T` with the exact held-out RBF kernel;
smaller means a closer kernel approximation on these rows, not necessarily
better response prediction. This small dense diagnostic is unsuitable when
materializing that kernel would itself be too expensive. For a downstream
predictor, fit the feature map separately within each training fold.

## Kernel Ridge Regression

Given a training kernel matrix $K$, kernel ridge regression solves the dual
system

$$
(K+\alpha I)c=y.
$$

For a positive-semidefinite kernel, the RKHS-penalized objective is

$$
\min_c \lVert y-Kc\rVert_2^2+\alpha c^\top Kc.
$$

The loss is a sum, not an average. For positive-definite $K$, its first-order condition gives the displayed linear system. With singular positive-semidefinite $K$, the same system (for $\alpha>0$) selects a coefficient representation of the minimizing fitted function. A penalty on $\lVert c\rVert^2$ alone would define a different estimator. Indefinite kernels do not have this convex RKHS interpretation.

Predictions for test observations are

$$
\hat y_{test}=K(X_{test},X_{train})c.
$$

`KernelRidge` solves the regularized linear system directly. If the solve raises a linear-algebra error, it retries with increasing diagonal jitter; that stabilized result need not solve the original displayed system exactly. Prefer a positive, well-scaled `alpha`. Multi-output responses are supported with a compatible response matrix.

`fit(X, y, sample_weight=None)` currently ignores supplied `sample_weight`; it does not fit a weighted objective. Omit that argument and use this estimator only for an unweighted analysis. Removing zero-weight rows is useful when the intended operation is simply exclusion, but does not implement arbitrary weighting.

## Kernel Ridge Cross-Validation

`KernelRidgeCV` evaluates a grid of regularization parameters across CV folds and
refits the selected value on the complete dataset. Backend-specific
implementations may reuse a kernel eigendecomposition or vectorize the alpha
sweep rather than solving every system independently.

Selection minimizes mean validation MSE, averaged equally over folds and response columns. `alpha_` is the chosen value; `best_score_` is the mean fold R-squared at that value, not the MSE used for selection. `cv_results_` contains `alphas`, `mean_mse`, `mse_table`, `mean_r2`, `r2_table`, `best_alpha`, and `best_score`. Tables have shape `(n_alphas, n_folds, n_targets)`; means have shape `(n_alphas, n_targets)`.

With a singular training-fold kernel, an explicit zero alpha can produce nonfinite CV scores and still be selected. Use strictly positive candidates and check that `mean_mse` and `best_score_` are finite before interpreting the selection. A returned estimator alone does not establish a valid search. The automatic grid uses positive values.

## Kernel PCA

For the centered kernel matrix $\widetilde K$, Kernel PCA eigendecomposes

$$
\widetilde K = V\Lambda V^\top.
$$

The leading eigenvectors define nonlinear components. Transforming new data
requires computing the test-to-training kernel, applying the training centering
quantities, and projecting onto the retained components.

### After a failed refit

A rejected KernelPCA refit can overwrite the training centering information
while keeping the previous eigenvectors and fitted flag. This can happen with
finite constant data when the centered kernel has no positive directions.
Subsequent `transform` or `predict` calls can then return finite but incorrect
coordinates, even for the original training data. Checking output finiteness
alone does not detect this mixed state.

If `fit` or `fit_transform` raises, discard that instance. Create a new
KernelPCA and fit suitable data successfully before transforming queries;
do not continue with its apparently retained earlier fit. A fresh instance
also rejects the same degenerate data, so recreating the object is not a way
to make an unsupported zero-rank embedding valid.

## Nystroem Approximation

Nystroem selects $m$ landmark observations and forms an explicit approximate
feature map. If the landmark kernel is

$$
K_{mm}=V\Lambda V^\top,
$$

then the transformed features have the form

$$
Z=K_{nm}V\Lambda^{-1/2}V^\top.
$$

This is the symmetric inverse-square-root feature orientation used for a positive-semidefinite landmark kernel. The implementation actually computes $K_{mm}=U\Sigma V^\top$ by NumPy SVD and uses $U\max(\Sigma,10^{-12}I)^{-1/2}V^\top$ as its normalization. It keeps all $m$ selected landmarks; `eigenvalues_` contains the floored singular values, including for an indefinite kernel. Such a kernel does not imply an exact Euclidean feature representation.

This replaces a full $n\times n$ kernel representation with an $n\times m$
feature matrix when $m\ll n$.

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

## Complete estimator API

All four classes are imported from `statgpu.nonparametric.kernel_methods`.
Constructors and defaults are:

```text
KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, device='auto', n_jobs=None)
KernelRidgeCV(alphas=None, cv=5, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, random_state=None, device='auto', n_jobs=None)
KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0, eigen_solver='auto', device='auto', n_jobs=None)
Nystroem(kernel='rbf', n_components=100, gamma=None, degree=3, coef0=1, random_state=None, device='auto', n_jobs=None)
```

### Constructor controls

| Argument | Meaning and restrictions |
|---|---|
| `kernel` | A built-in name or a matrix-valued callable; see the function API below. Use the lowercase canonical names with estimator-specific controls. |
| `gamma` | RBF/Laplacian/polynomial/sigmoid coefficient; `None` uses `1 / n_features`. Chi-squared defaults to `1.0`. KernelRidge/CV currently ignore this constructor argument for chi-squared; use `kernel_params={"gamma": value}` instead. |
| `degree`, `coef0` | Polynomial degree and additive constant; sigmoid uses `coef0`. Supply these only for applicable kernels. |
| `kernel_params` | KernelRidge/CV only: extra built-in or callable kernel arguments. Explicit applicable nondefault constructor controls can override dictionary entries. Pass callable-specific controls in this dictionary. |
| `alpha` | KernelRidge: finite nonnegative strength in the summed-loss objective. KernelPCA: finite nonnegative diagonal shift used during eigendecomposition; the shift is removed from reported eigenvalues and does not regularize an inverse transform. |
| `alphas` | KernelRidgeCV: finite nonempty nonnegative candidates; prefer strictly positive values. `None` builds 100 positive log-spaced values from the training kernel spectrum. |
| `cv` | KernelRidgeCV: integer between 2 and the number of rows; shuffled row-wise K-fold splitting. No custom/grouped-fold argument is available. |
| `n_components` | Positive integer. KernelPCA retains at most this many positive eigenvalue directions and can return fewer; Nystroem selects `min(n_components, n_training_rows)` landmarks without replacement. |
| `eigen_solver` | KernelPCA: `"auto"` or `"dense"`; both currently use dense symmetric eigendecomposition. |
| `random_state` | Integer seed or `None` for CV row shuffling or Nystroem landmark selection. |
| `device` | `"cpu"`, `"cuda"` (CuPy), `"torch"` (Torch CUDA), or `"auto"`. KernelPCA/Nystroem have a current Torch CPU-placement exception after the availability check; see backend boundaries below. |
| `n_jobs` | Shared estimator setting; these estimators do not use it to parallelize fitting. |

### Methods and shapes

Use a finite numeric design `X` of shape `(n,p)` with the same feature order at prediction. A one-dimensional `X` is interpreted as multiple rows of one feature, not a multivariate query. Estimator calculations convert numeric input to float64. Let `q` be the number of query rows and `r` the response-column count. KernelPCA retains `k <= min(n_components,n)` positive directions; Nystroem uses `m=min(n_components,n)` landmarks.

| Class | Complete model-specific method calls | Return / restrictions |
|---|---|---|
| KernelRidge | `fit(X, y, sample_weight=None)`, `predict(X)`, `score(X, y)` | `fit` returns `self`; `y` is `(n,)` or `(n,r)`. Weights are currently ignored. Predictions are `(q,)` for one target, including a one-column training response, otherwise `(q,r)`. |
| KernelRidgeCV | `fit(X, y)`, `predict(X)`, `score(X, y)` | Same response/prediction shapes; `fit` returns `self`. No sample-weight argument. Prediction and scoring delegate to `estimator_`. |
| KernelPCA, Nystroem | `fit(X, y=None)`, `transform(X)`, `fit_transform(X, y=None)`, `predict(X)` | `fit` returns `self`; `y` is unused. Other calls return feature arrays (Torch placement caveat below): `(q,k)` for KernelPCA, `(q,m)` for Nystroem; `predict` is a transform alias, not a response prediction. |

Ridge `score` is a Python float: per-target R-squared uniformly averaged over response columns. Constant targets use 1 for effectively exact predictions and 0 otherwise. KernelPCA/Nystroem have no model-specific `score` or inverse-transform method. All four also expose `get_params(deep=True)` and `set_params(**params)` plus the [shared inference utilities](../reference/estimator-api.md). Refit after parameter changes; KernelPCA/Nystroem can retain previous fitted arrays until refitting, so do not rely on automatic invalidation. These utilities do not provide kernel-coefficient inference.

### Fitted attributes and outputs

| Class | Fields and interpretation |
|---|---|
| KernelRidge | `X_fit_`: backend array `(n,p)`; `dual_coef_`: backend array `(n,r)`, including `(n,1)` for a vector response; `n_features_in_`: integer. These are kernel coefficients, not raw-feature slopes. |
| KernelRidgeCV | `alpha_`, `best_score_`, `cv_results_` as described above; `estimator_`: final KernelRidge; `dual_coef_` and `X_fit_`: its arrays. Read `n_features_in_` on `estimator_`. CV result arrays are NumPy. |
| KernelPCA | NumPy `lambdas_` `(k,)`: retained positive centered-kernel eigenvalues; NumPy `alphas_` `(n,k)`: eigenvectors divided by their square roots; NumPy `X_fit_` `(n,p)`; integers `n_samples_`, `n_features_in_`. Only eigenvalues above the numerical cutoff are retained; no positive directions raises an error. |
| Nystroem | NumPy `components_` `(m,p)`, `component_indices_` `(m,)`, `normalization_` `(m,m)`, `eigenvalues_` `(m,)`; integer `n_features_in_`. The misleading historical name `eigenvalues_` stores floored singular values. `m=min(n_components,n_training_rows)`. |

## Complete pairwise-function API

All functions below are public exports from the same module:

```text
pairwise_kernels(X, Y=None, metric='rbf', xp=None, **params)
rbf_kernel(X, Y=None, gamma=None, xp=None)
polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None)
linear_kernel(X, Y=None, xp=None)
laplacian_kernel(X, Y=None, gamma=None, xp=None)
sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None)
cosine_kernel(X, Y=None, xp=None)
chi2_kernel(X, Y=None, gamma=1.0, xp=None)
```

`X` and `Y` are compatible finite backend arrays `(n,p)` and `(q,p)`; `Y=None` means `Y=X`. The result is the pairwise matrix `(n,q)`, or `(n,n)` when `Y` is omitted. For built-in kernels, `xp=None` means NumPy, not automatic input-backend detection. Set `xp` to the matching NumPy/CuPy/Torch module; this argument alone does not transfer inputs to a GPU. The low-level functions do not share a uniform dtype-conversion or input-validation policy; prepare valid arrays explicitly. Chi-squared requires nonnegative values and finite nonnegative `gamma`.

Names are case/whitespace normalized by `pairwise_kernels`; aliases are `gaussian` for `rbf`, `poly` for `polynomial`, and `chi-squared` for `chi2`. A callable receives the complete `X, Y` arrays, additional `params`, and `xp` if its signature accepts that keyword. Callable dispatch forwards the supplied `xp` unchanged, including `None`. The callable must choose its own default or be called with explicit `xp=np`; it must also handle `Y=None` when used that way and return the complete pairwise matrix, not one scalar per pair. An unknown name raises `ValueError`.

A NumPy-default callable can handle both omitted and explicit modules:

<!-- example: kernel-callable-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import pairwise_kernels

def custom_linear(X, Y=None, xp=None):
    xp = np if xp is None else xp
    Y = X if Y is None else Y
    return xp.asarray(X) @ xp.asarray(Y).T

X = np.array([[-1.0], [0.5], [2.0]])
implicit = pairwise_kernels(X, metric=custom_linear)
explicit = pairwise_kernels(X, metric=custom_linear, xp=np)
np.testing.assert_allclose(implicit, X @ X.T)
np.testing.assert_allclose(explicit, implicit)
```

The cosine implementation adds `1e-10` to its norm-product denominator, so zero-vector pairs return zero and tiny-norm inputs differ from exact cosine normalization. For chi-squared, zero/zero feature contributions are zero on NumPy; CuPy/Torch use a denominator floor of `1e-10`, which can change results for very small nonnegative features.

Implementation references: [KernelRidge](../../../statgpu/nonparametric/kernel_methods/_krr.py), [KernelRidgeCV](../../../statgpu/nonparametric/kernel_methods/_krr_cv.py), [KernelPCA](../../../statgpu/nonparametric/kernel_methods/_kpca.py), [Nystroem](../../../statgpu/nonparametric/kernel_methods/_nystroem.py), and [pairwise kernels](../../../statgpu/nonparametric/kernel_methods/_kernels.py).

## Optional GPU examples

### CuPy

```python
import cupy as cp
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = cp.random.randn(500, 10, dtype=cp.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="cuda").fit(X, y)
```

### Torch CUDA

```python
import torch
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = torch.randn(500, 10, device="cuda", dtype=torch.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="torch").fit(X, y)
```

`device="cuda"` requests CuPy CUDA; `device="torch"` requests Torch CUDA. For KernelPCA/Nystroem, successful backend selection alone does not establish output placement; see below.

## Backend and Execution Boundaries

KernelRidge/CV kernel construction and solves use the selected backend. KernelPCA normally decomposes on that backend, but a linear-algebra exception triggers a NumPy CPU eigendecomposition without a separate fallback indicator. Its stored training data and projection coefficients are also NumPy arrays. Do not assume a fully device-resident KernelPCA fit. Nystroem is a hybrid implementation: it copies selected landmarks to NumPy, constructs the landmark kernel and performs its SVD on CPU, then uses the selected array library for query-to-landmark kernels and returned features, subject to the placement limitation below. Custom Nystroem kernels must therefore also accept NumPy landmark inputs. Landmark indices, normalization, and singular values are host arrays. Do not interpret a GPU Nystroem fit as an entirely device-resident decomposition. Explicit unavailable device requests raise.

KernelPCA and Nystroem still reject `device="torch"` when Torch CUDA is unavailable. Passing that check does not guarantee CUDA execution: NumPy or Torch CPU input can remain on CPU during fitting and transformation because the requested-device conversion is not consistently applied. Inspect the actual `fit_transform`, `transform`, and `predict` outputs with Torch `.device`/`.is_cuda` (or CuPy `.device`), rather than relying on the configured device or selected backend/library. Their public fitted arrays are deliberately NumPy and cannot prove numerical-device placement. This also affects Nystroem's query kernel and output, separately from its intentional CPU landmark SVD. If CUDA placement is required, reject CPU outputs before using them. For a predictable CPU alternative, pass NumPy input with `device="cpu"`. See the [device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions).

`KernelPCA` and `Nystroem` reject NaN/Inf in the supplied input arrays during
fitting and transformation. This input check does not guarantee finite computed
kernels or learned arrays; see [finite-result checks](#check-for-nonfinite-kernel-fits).
Kernel-specific domain checks, such as non-negative inputs for the chi-squared
kernel, fail explicitly.

## Inference Semantics

Kernel methods do not currently expose coefficient-level standard errors,
hypothesis tests, or confidence intervals. Model quality is evaluated through
prediction scores, cross-validation loss, embedding properties, reconstruction
or approximation diagnostics, and application-specific validation.

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

## Limitations and Failure Modes

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



For RBF kernels, subtract the same training-derived offset from training and query features when coordinates have a large common offset. Current squared-distance arithmetic can otherwise lose the differences between nearby points: coordinates near `1e9` can produce an almost all-ones kernel and badly changed predictions. Centering preserves the intended RBF kernel; it is not a general preprocessing rule for every kernel, especially chi-squared.


- Kernel matrices can become poorly conditioned; increase `alpha` or adjust the
  kernel scale when necessary.
- RBF-like kernels are sensitive to `gamma`.
- Chi-squared kernels require non-negative inputs.
- Dense exact kernel methods may exhaust device memory for large $n$.
- User-supplied kernels are responsible for backend, dtype, shape, and symmetry
  contracts required by the selected estimator.
- `KernelRidgeCV` can be expensive for large fold and alpha grids.

## Validation guidance

For kernel ridge, verify the regularized linear system or compare with scikit-learn using the same kernel, alpha, data, and unweighted objective. For CV, inspect finite per-candidate MSE and verify the final refit uses the selected alpha. For Nystroem, align selected landmarks and the SVD singular-value floor before comparing normalization or features. CPU checks do not establish GPU accuracy or performance; validate those on the intended device.

## FAQ

### Should I use KernelRidge or Nystroem plus a linear model?

Use exact Kernel Ridge when the training kernel fits comfortably in memory and the
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
