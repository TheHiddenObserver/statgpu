# Kernel methods API

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../../cn/reference/kernel-methods-api.md)

This reference lists the signatures, parameters, methods, fitted attributes, and
execution limits of the four estimators and eight pairwise functions in
`statgpu.nonparametric.kernel_methods`. Start with the
[kernel-methods overview](../models/nonparametric/kernel-methods.md) to choose a method.
For complete worked examples, see [KernelRidge](../models/nonparametric/kernel-ridge.md),
[KernelRidgeCV](../models/nonparametric/kernel-ridge-cv.md),
[KernelPCA](../models/nonparametric/kernel-pca.md), and [Nystroem](../models/nonparametric/kernel-nystroem.md).

## Complete estimator API

```text
KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, device='auto', n_jobs=None)
KernelRidgeCV(alphas=None, cv=5, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, random_state=None, device='auto', n_jobs=None)
KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0, eigen_solver='auto', device='auto', n_jobs=None)
Nystroem(kernel='rbf', n_components=100, gamma=None, degree=3, coef0=1, random_state=None, device='auto', n_jobs=None)
```

The displayed device default `'auto'` is the string equivalent of the runtime `Device.AUTO` default. All four classes accept a string or `statgpu.Device` value. Import the estimators with:

```python
from statgpu.nonparametric.kernel_methods import (
    KernelRidge, KernelRidgeCV, KernelPCA, Nystroem,
)
```

### Shared constructor controls

| Argument | Meaning and restrictions |
|---|---|
| `kernel='rbf'` | Built-in name or a callable returning the full pairwise matrix. See [pairwise dispatch](#pairwise-kernels) and [parameter forwarding](#kernel-parameter-forwarding). There is no `precomputed` kernel mode. |
| `gamma=None` | Kernel coefficient. `None` normally uses `1 / n_features` for RBF, Laplacian, polynomial, and sigmoid; chi-squared defaults to `1.0`. Estimator-specific forwarding matters, especially for chi-squared and callables. |
| `degree=3` | Polynomial degree. Use a valid finite kernel setting; the general interface does not comprehensively validate kernel controls. |
| `coef0=1` | Additive constant for polynomial and sigmoid kernels. |
| `device='auto'` | `'cpu'`: NumPy; `'cuda'`: CuPy CUDA; `'torch'`: Torch CUDA; `'auto'`: follows the global device setting, with automatic selection preferring CuPy, then Torch CUDA, then NumPy when that setting is also automatic. Explicit unavailable devices raise. KernelPCA/Nystroem have important [CPU execution and placement boundaries](#backend-and-execution-boundaries). |
| `n_jobs=None` | Shared estimator option; these estimators do not use it to parallelize fitting. |

### Common input, output, and lifecycle contract

- Supply finite numeric `X` of shape `(n,p)`, with nonzero training row and feature counts. A one-dimensional `X` is reshaped to `(n,1)`, not interpreted as one multivariate query. Queries have shape `(q,p)` and must keep the training feature order.
- Estimator numeric inputs are converted to float64. Compatible callable kernels must also preserve the required backend, dtype, shape, and device. This estimator conversion is distinct from the low-level functions' [dtype rules](#pairwise-inputs-and-dtypes).
- Supervised `y` is `(n,)` or `(n,r)`. A vector is stored internally as one response column. Ridge predictions for a single target are `(q,)`, including when training `y` was `(n,1)`; multiple targets return `(q,r)`.
- `fit` returns `self`. Calling prediction or transformation before fitting raises `RuntimeError`. Public numeric-input checks reject NaN/Inf; this does **not** validate computed kernels, learned arrays, or all scalar kernel parameters.
- All four have `get_params(deep=True)` and `set_params(**params)`. The former returns constructor configuration, with nested settings when applicable; the latter returns `self`. Unknown constructor names raise `ValueError`. Refit after changes. KernelRidge/CV use the base reconstruction path; KernelPCA/Nystroem overrides can retain fitted arrays after changes to their own parameters, so do not rely on automatic invalidation.
- All four inherit `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test`. Their complete signatures, restrictions, and returns are in the [shared estimator API](estimator-api.md). They do not supply kernel-coefficient inference or automatically create valid model-specific resampling procedures.

<a id="kernelridge"></a>

### KernelRidge

[Tutorial: fit and evaluate nonlinear response predictions](../models/nonparametric/kernel-ridge.md).

In addition to the shared controls, `alpha=1.0` is a finite nonnegative scalar, and `kernel_params=None` is an optional dictionary of kernel keyword arguments. The regularized system is

$$
(K+\alpha I)c=y.
$$

For a positive-semidefinite kernel, this corresponds to minimizing $\lVert y-Kc\rVert_2^2+\alpha c^\top Kc$: **summed** squared loss, not mean loss and not a penalty on $\lVert c\rVert^2$. For positive-definite $K$, the first-order conditions give that system; for singular positive-semidefinite $K$ and positive alpha, it selects coefficients representing an optimal fitted function. An indefinite kernel does not have this convex RKHS interpretation. There is no separate fitted intercept; predictions are $K(X_{query},X_{train})c$.

| Method | Return and restrictions |
|---|---|
| `fit(X, y, sample_weight=None)` | Returns `self`. Fits an **unweighted** objective: the accepted `sample_weight` is currently ignored. Omit it. Removing zero-weight rows can exclude observations but does not implement arbitrary weighted fitting. |
| `predict(X)` | Backend response array `(q,)` or `(q,r)`. |
| `score(X, y)` | Python float: uniform average of per-target R-squared. Response shape must match predictions after vector-to-column conversion. Constant targets score 1 when residual sum of squares is at most `1e-15`, otherwise 0. |
| `get_params(deep=True)`, `set_params(**params)` | Parameter-management behavior described above. |

| Fitted attribute | Shape / meaning |
|---|---|
| `X_fit_` | Backend float64 array `(n,p)`, training design used for predictions. |
| `dual_coef_` | Backend array `(n,r)`, including `(n,1)` for one target. These are kernel coefficients, not original-feature slopes. |
| `n_features_in_` | Integer input feature count. |

If the direct solve raises a linear-algebra exception, fitting retries with progressively increased diagonal jitter. If stabilization still fails, it raises `ValueError` advising a larger alpha. Stabilized coefficients need not solve the original unperturbed system exactly; prefer a suitably scaled positive alpha. Nonfinite results that do not raise can escape this mechanism; follow the [finite-result checks](#check-for-nonfinite-kernel-fits).

<a id="kernelridgecv"></a>

### KernelRidgeCV

[Tutorial: select alpha inside training folds](../models/nonparametric/kernel-ridge-cv.md).

| Additional argument | Meaning and restrictions |
|---|---|
| `alphas=None` | Finite, nonempty, nonnegative candidate array; supplied values are flattened. Prefer strictly positive candidates. `None` builds 100 positive log-spaced values from the full training-kernel spectrum. |
| `cv=5` | Integer satisfying `2 <= cv <= n_samples`; booleans are rejected. Uses shuffled row-wise K-fold splitting. Custom, grouped, and time-ordered splitters are not accepted. |
| `kernel_params=None` | Optional kernel-parameter dictionary, with the same forwarding rules as KernelRidge. |
| `random_state=None` | Integer seed or `None` for NumPy `RandomState` row shuffling. |

| Method | Return and restrictions |
|---|---|
| `fit(X, y)` | Returns `self`; searches alpha and refits a final KernelRidge on all supplied training rows. No sample-weight argument. |
| `predict(X)` | Delegates to `estimator_`; backend `(q,)` or `(q,r)` predictions. |
| `score(X, y)` | Delegates to `estimator_`; Python float R-squared as above. |
| `get_params(deep=True)`, `set_params(**params)` | Parameter-management behavior described above. |

Selection minimizes validation MSE, averaged **equally over folds and response columns**. It selects one alpha shared by all targets. It does not tune the kernel, gamma, or preprocessing. `best_score_` is the mean fold R-squared at that selected alpha, a different metric from the selection loss; it is not a held-out test score. Constant-target CV R-squared uses 1 when fold MSE is at most `1e-15`, otherwise 0.

The full training kernel is constructed once; each fold's symmetric eigendecomposition is reused across candidates. The automatic grid adapts to the full training-kernel spectrum and contains only positive values; pass `alphas` explicitly when you need a fixed, reproducible search range across datasets.

| Fitted attribute | Shape / meaning |
|---|---|
| `alpha_` | Python float, selected regularization strength. |
| `best_score_` | Python float, selected mean fold R-squared uniformly averaged across targets. |
| `cv_results_` | Dictionary detailed below; result arrays are NumPy. |
| `estimator_` | Final fitted KernelRidge with selected alpha and the same kernel controls/device. Read feature count at `estimator_.n_features_in_`. |
| `dual_coef_`, `X_fit_` | Shortcuts to the final estimator's backend arrays, with KernelRidge shapes. |

For `a` candidates, `f` folds, and `r` response columns, `cv_results_` contains:

| Key | Value |
|---|---|
| `alphas` | Candidate array `(a,)`. |
| `mse_table`, `r2_table` | Per-candidate, per-fold, per-target arrays `(a,f,r)`. |
| `mean_mse`, `mean_r2` | Fold means `(a,r)`, including a one-column array for a vector response. |
| `best_alpha`, `best_score` | Same values as `alpha_` and `best_score_`. |

A singular fold kernel with an explicit zero alpha can produce nonfinite CV scores and still be selected. The CV eigenvalue division does not use KernelRidge's jitter-retry solve. Check that `cv_results_["mean_mse"]` and `best_score_` are finite before interpreting the result; a returned estimator alone does not establish a valid search. Positive candidates reduce this singularity risk, but do not cure nonfinite or badly scaled kernels.

For grouped/time-ordered splits or learned preprocessing within folds, use an external splitting loop with preprocessing and KernelRidge fitted separately inside each training fold. Do not preprocess all rows before passing them to this internal CV. Keep final evaluation data outside selection.

<a id="kernelpca"></a>

### KernelPCA

[Tutorial: learn nonlinear coordinates](../models/nonparametric/kernel-pca.md).

| Additional argument | Meaning and restrictions |
|---|---|
| `n_components=2` | Positive integer; maximum retained positive directions. Output can have fewer columns. |
| `alpha=1.0` | Finite nonnegative diagonal shift added before eigendecomposition and removed from reported eigenvalues. It is **not** inverse-transform regularization. |
| `eigen_solver='auto'` | `'auto'` or `'dense'`; both currently perform dense symmetric eigendecomposition. |

| Method | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Returns `self`; `y` is unused. |
| `transform(X)` | Backend array `(q,k)` of coordinates, where `k <= min(n_components,n)` is the retained positive count. See the Torch placement limitation below. |
| `fit_transform(X, y=None)` | Fits, then calls the same transformation/centering path on `X`; returns `(n,k)`. |
| `predict(X)` | Alias for `transform`, not a response prediction. |
| `get_params(deep=True)`, `set_params(**params)` | Parameter-management behavior described above. |

There is no model-specific `score` or `inverse_transform` method. Training forms the centered kernel $\widetilde K$, decomposes $\widetilde K+\alpha I$, subtracts alpha from its eigenvalues, and retains at most `n_components` eigenvalues **strictly above `1e-12`**. No positive directions raises `ValueError`. Negative directions of an indefinite kernel are discarded; alpha does not make them valid positive components after its shift is removed.

| Fitted attribute | Shape / meaning |
|---|---|
| `lambdas_` | NumPy `(k,)`, retained positive eigenvalues of the unshifted centered kernel. |
| `alphas_` | NumPy `(n,k)`, eigenvectors divided by square roots of retained eigenvalues. These scaled columns are not generally unit-norm eigenvectors. |
| `X_fit_` | NumPy `(n,p)`, training data, even when fitting uses a GPU backend. |
| `n_samples_`, `n_features_in_` | Integer training row and feature counts. |

Query centering uses the **training** kernel's column means and grand mean, and each query row's mean cross-kernel value; do not center a query kernel independently. Coordinates are the centered query-to-training kernel multiplied by `alphas_`. Eigenvector signs, and bases within repeated or nearly repeated eigenspaces, are not unique; compare subspaces or downstream quantities when appropriate.

If `fit` or `fit_transform` raises, discard the instance and fit suitable data on a fresh one. A failed refit can mix new centering information with old eigenvectors and return finite but wrong coordinates. See [failed-refit handling](#after-a-failed-refit) and [backend boundaries](#backend-and-execution-boundaries).

<a id="nystroem"></a>

### Nystroem

[Tutorial: build reusable approximate kernel features](../models/nonparametric/kernel-nystroem.md).

| Additional argument | Meaning and restrictions |
|---|---|
| `n_components=100` | Positive integer requested landmark count; actual count is `m=min(n_components,n_training_rows)`. |
| `random_state=None` | Integer seed or `None` for NumPy `RandomState` sampling without replacement. Landmark order is preserved, not sorted. |

| Method | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Returns `self`; `y` is unused. Selects landmarks and learns their normalization. |
| `transform(X)` | Backend feature array `(q,m)`; see the Torch placement limitation below. |
| `fit_transform(X, y=None)` | Fits and transforms the training rows, returning `(n,m)`. |
| `predict(X)` | Alias for `transform`, not a response prediction. |
| `get_params(deep=True)`, `set_params(**params)` | Parameter-management behavior described above. |

There is no model-specific `score` or `inverse_transform` method. If the CPU landmark kernel has SVD $K_{mm}=U\Sigma V^\top$, the implementation uses

$$
N=U\max(\Sigma,10^{-12}I)^{-1/2}V^\top,\qquad Z=K_{qm}N.
$$

For a positive-semidefinite landmark kernel, this is the symmetric inverse-square-root feature orientation. All `m` landmark columns are kept: small singular values are **floored, not removed**. The SVD also runs on indefinite kernels, but this does not give an exact Euclidean representation of an indefinite similarity.

| Fitted attribute | Shape / meaning |
|---|---|
| `components_` | NumPy `(m,p)`, selected landmarks. |
| `component_indices_` | NumPy integer array `(m,)`, their indices in the training rows. |
| `normalization_` | NumPy `(m,m)`, SVD normalization `N`. |
| `eigenvalues_` | NumPy `(m,)`, singular values floored at `1e-12`, **not signed eigenvalues** despite the historical name. |
| `n_features_in_` | Integer input feature count. |

Landmark kernel construction and SVD always run on CPU. Custom kernels must therefore accept NumPy landmark inputs as well as the selected query backend. Fit the feature map only on training data and refit it inside each fold when selecting a downstream predictor. An approximation diagnostic such as `Z @ Z.T` versus the exact kernel requires materializing that dense comparison kernel; use it only for small data. A lower approximation error does not necessarily imply better response prediction.

<a id="kernel-parameter-forwarding"></a>

## Kernel parameter forwarding

Constructor controls are not forwarded identically by all estimators:

| Estimator | Forwarding rules |
|---|---|
| KernelRidge / KernelRidgeCV | Start with a copy of `kernel_params` or `{}`. For normalized names `rbf`, `gaussian`, `polynomial`, `poly`, `laplacian`, and `sigmoid`, non-`None` constructor `gamma` overrides the dictionary. For polynomial aliases, nondefault `degree != 3` overrides dictionary `degree`. For polynomial aliases and sigmoid, nondefault `coef0 != 1` overrides dictionary `coef0`. Defaults leave dictionary entries intact. |
| KernelRidge / KernelRidgeCV with chi-squared | Constructor `gamma` is currently ignored; use `kernel_params={"gamma": value}`. Otherwise the function default is `1.0`. |
| KernelRidge / KernelRidgeCV with a callable | Constructor `gamma`, `degree`, and `coef0` are not automatically passed. Put callable-specific keywords in `kernel_params`. |
| KernelPCA / Nystroem | No `kernel_params` argument. Every non-`None` `gamma` is forwarded, including to chi-squared and callables. For exact lowercase `poly`/`polynomial`, both `degree` and `coef0` are forwarded, including their defaults. For exact lowercase `sigmoid`, `coef0` is forwarded. |

Pairwise dispatch normalizes names, but KernelPCA/Nystroem's degree/coef0 selection does not. Use canonical lowercase names when configuring these estimators; a name such as `'Polynomial'` can dispatch successfully while omitting your intended degree/coef0. For their linear/cosine kernels, leave `gamma=None`: a non-`None` gamma is forwarded and raises `TypeError` because those functions do not accept it. Extra dictionary keywords are not filtered against a built-in function's signature and may also raise `TypeError`.

A custom kernel must return a full `(n,q)` matrix on the expected backend, with compatible dtype and device, and provide the symmetry/positive-semidefiniteness needed by the selected method. A callable's own errors propagate. Each call returns an entire pairwise matrix; some other libraries instead call
a custom kernel once per pair and expect a scalar.

## Complete pairwise-function API

All eight functions are public exports from `statgpu.nonparametric.kernel_methods`:

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

<a id="pairwise-inputs-and-dtypes"></a>

### Inputs, shapes, dtype, and placement

For built-ins, prepare compatible finite two-dimensional backend arrays `X` `(n,p)` and `Y` `(q,p)` with the same feature count. `Y=None` uses `X`. The return is `(n,q)`, or `(n,n)` when `Y` is omitted. Unlike estimators, these functions do not reshape a vector to a column and do not generally accept nested lists; convert them before calling. The NumPy chi-squared path also accepts two-dimensional array-like inputs.

`xp=None` selects NumPy for built-ins; it does **not** automatically detect the input backend. Set `xp` to the matching NumPy, CuPy, or Torch module and prepare compatible input arrays on the intended device yourself. Passing `xp` alone does not move arrays to a GPU or impose float64. There is no uniform dtype-conversion or finite-input-validation policy across these low-level functions:

- NumPy RBF uses `np.result_type(X.dtype, Y.dtype, np.float32)` and preserves explicit float32/float64 inputs under that promotion rule. Torch/CuPy RBF preserve compatible floating input dtypes; if either input is nonfloating, both are promoted to float64. RBF rejects complex input.
- NumPy Laplacian uses SciPy `cdist` and returns its floating distance-array dtype, normally float64.
- NumPy chi-squared uses scikit-learn when available, otherwise a chunked NumPy calculation with at least float64 accumulation. Do not assume one identical output dtype across all optional-dependency/backend paths.
- Other kernels use backend arithmetic and its promotion rules. Integer dot products or powers can overflow before a later floating operation; use appropriate floating arrays. Torch matrix operands must have compatible dtypes and devices.

<a id="pairwise-kernels"></a>

### pairwise_kernels

`pairwise_kernels(X, Y=None, metric='rbf', xp=None, **params)` dispatches to the selected function and returns its complete pairwise matrix. Built-in names are stripped and lowercased. Canonical names are `rbf`, `polynomial`, `linear`, `laplacian`, `sigmoid`, `cosine`, and `chi2`; aliases are `gaussian`, `poly`, and `chi-squared`, respectively. Unknown names, including `precomputed`, raise `ValueError`. `**params` are passed through unchanged, without automatic filtering.

For a callable `metric`, the original `X` and `Y` objects are passed unchanged, including `Y=None`. The supplied `xp`, including `None`, is forwarded if the signature accepts `xp` or `**kwargs`; if signature inspection is unavailable, dispatch also attempts `xp`. Otherwise it omits that keyword. The callable must handle its own default module, input conversion, and omitted `Y`, and must return the full matrix. Errors raised inside the callable are not retried under a different signature.

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
<!-- example-end: kernel-callable-cpu -->

This NumPy example handles both an omitted `xp` and an explicit `xp=np`. A
custom GPU kernel must also handle dtype and device placement explicitly.

<a id="rbf-kernel"></a>

### rbf_kernel

`rbf_kernel(X, Y=None, gamma=None, xp=None)` returns $\exp(-\gamma\lVert x-y\rVert_2^2)$ for each pair. `gamma=None` means `1/p`. Shape compatibility and real-valued inputs are checked; gamma is not comprehensively validated, so supply a finite meaningful coefficient.

Squared distances use squared norms minus twice a dot product, with negative roundoff clamped to zero. This can lose nearby-point differences under a large common coordinate offset: values near `1e9` can produce an almost all-ones kernel and badly changed predictions. Subtract the same **training-derived** offset from training and query features when needed. Translation preserves the intended RBF kernel; it is not a universal rule for other kernels, especially chi-squared. NumPy computes in chunks when necessary but still returns the full dense matrix.

<a id="polynomial-kernel"></a>

### polynomial_kernel

`polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None)` returns $(\gamma x^\top y+\mathrm{coef0})^{\mathrm{degree}}$. `gamma=None` means `1/p`. Large finite inputs, scale, or degree can overflow; invalid controls can produce NaN without a clean failure. Feature scaling changes this similarity, so choose it inside the training procedure. See [finite-result checks](#check-for-nonfinite-kernel-fits).

<a id="linear-kernel"></a>

### linear_kernel

`linear_kernel(X, Y=None, xp=None)` returns $x^\top y$. It accepts no gamma, degree, or coef0 argument and performs no automatic centering or intercept addition. Its matrix product uses the input arrays' arithmetic; `xp` does not convert them.

<a id="laplacian-kernel"></a>

### laplacian_kernel

`laplacian_kernel(X, Y=None, gamma=None, xp=None)` returns $\exp(-\gamma\lVert x-y\rVert_1)$; `gamma=None` means `1/p`. NumPy uses SciPy city-block distances; CuPy/Torch use a broadcasted `(n,q,p)` difference array, which can consume substantially more temporary memory than the returned kernel.

<a id="sigmoid-kernel"></a>

### sigmoid_kernel

`sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None)` returns $\tanh(\gamma x^\top y+\mathrm{coef0})$; `gamma=None` means `1/p`. It accepts no degree. Sigmoid kernels are not positive-semidefinite for arbitrary parameters/data; the PSD interpretation of ridge/Nystroem and KernelPCA's retained-positive-direction rule still matter.

<a id="cosine-kernel"></a>

### cosine_kernel

`cosine_kernel(X, Y=None, xp=None)` returns $x^\top y/(\lVert x\rVert\lVert y\rVert+10^{-10})$. It accepts no gamma, degree, or coef0. The fixed denominator addition makes zero-vector pairs zero and changes exact cosine normalization for tiny-norm inputs.

<a id="chi2-kernel"></a>

### chi2_kernel

`chi2_kernel(X, Y=None, gamma=1.0, xp=None)` returns

$$
\exp\left\{-\gamma\sum_j\frac{(x_j-y_j)^2}{x_j+y_j}\right\}.
$$

This is the exponentiated chi-squared kernel, not the additive chi-squared kernel. It requires non-negative input features, matching two-dimensional shapes, and finite nonnegative gamma; violations raise `ValueError`. The default gamma is `1.0`, not `1/p`.

NumPy treats zero/zero feature contributions as zero. CuPy/Torch floor each denominator at `1e-10`, which changes results for very small nonnegative features and uses broadcasted `(n,q,p)` temporaries. Do not center histogram features into negative values. KernelRidge/CV require a gamma override through `kernel_params`, as described above.

## Backend and Execution Boundaries

All four estimators accept `device`, but kernel construction, decomposition, fitted arrays, and outputs do not necessarily stay on one device:

| Estimator | Execution and storage |
|---|---|
| KernelRidge / KernelRidgeCV | Construct kernels and solve on the selected backend. Ridge fitted arrays and predictions are backend arrays. CV result tables are copied to NumPy. |
| KernelPCA | Normally decomposes on the selected array backend. A linear-algebra exception triggers NumPy CPU eigendecomposition **without a separate fallback indicator**. Training data, eigenvalues, and projection coefficients are stored as NumPy. |
| Nystroem | Copies selected landmarks to NumPy, builds their kernel and performs SVD on CPU, then uses the selected array library for query-to-landmark kernels and returned features, subject to the placement limitation below. Public fitted arrays remain NumPy. A callable must accept NumPy landmark inputs. |

Explicit unavailable device requests raise. KernelPCA and Nystroem still reject `device="torch"` when Torch CUDA is unavailable, but passing that availability check does not guarantee CUDA execution: NumPy or Torch CPU inputs can remain on CPU during fit and transform because requested-device conversion is not consistently applied. This also affects Nystroem's query kernel and output, separately from its intentional CPU landmark SVD.

Inspect actual `fit_transform`, `transform`, and `predict` outputs with Torch `.device`/`.is_cuda` or CuPy `.device`. The configured device and selected library do not prove placement. Public fitted NumPy arrays cannot establish where numerical work ran. If CUDA placement is required, reject CPU outputs before using them. For a predictable CPU path, pass NumPy input with `device="cpu"`. See the [device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions). Neither KernelPCA nor Nystroem guarantees that all computation stays on GPU.

<a id="numerical-failures"></a>
<a id="limitations-and-failure-modes"></a>

## Numerical issues and failure handling

<a id="after-a-failed-refit"></a>

### After a failed refit

If `KernelPCA.fit` or `fit_transform` raises, **discard that instance**. A failed refit can overwrite training-kernel centering information while leaving old eigenvectors and the fitted flag intact. Subsequent `transform` or `predict` can return finite but incorrect coordinates, even on the original training data. Finiteness checks do not detect this mixed state.

Finite constant data can trigger the failure because the centered kernel has no positive directions. A fresh instance will reject the same degenerate data; reconsider the data or analysis, then fit suitable data successfully on a fresh model before transforming queries.

<a id="check-for-nonfinite-kernel-fits"></a>

### Check for nonfinite kernel fits

Finite input does not guarantee a finite computed kernel. Polynomial kernels can overflow, and current KernelRidge and Nystroem fits can return successfully with NaN learned arrays and predictions/features. Invalid scalar kernel settings such as NaN gamma can also produce nonfinite results. Warnings or the absence of an exception are not reliable fit-status checks.

Use finite applicable kernel controls and check learned arrays and query outputs before use. On NumPy CPU, check `np.isfinite(model.dual_coef_).all()` for KernelRidge, and `np.isfinite(model.normalization_).all()` plus `np.isfinite(model.eigenvalues_).all()` for Nystroem. Check predictions/transformed features too; use equivalent backend-native checks for device arrays. CV also needs finite `cv_results_["mean_mse"]` and `best_score_` checks.

Discard nonfinite fits, review feature units/scales and kernel parameters, then fit a fresh instance. Increasing alpha cannot repair an already-overflowed kernel. Scaling changes polynomial similarity; choose and validate preprocessing inside training folds. Finiteness is necessary, not sufficient, for a well-conditioned or useful model. Ill-conditioned finite kernels may require a larger positive alpha or a better kernel scale; neither substitutes for fixing the [RBF offset problem](#rbf-kernel).

<a id="inference-semantics"></a>

## Inference and validation boundaries

These estimators do not expose coefficient-level standard errors, hypothesis tests, or confidence intervals. There is no strict/approximate inference mode. Nystroem is an explicit low-rank kernel approximation, not a silent fallback for an exact estimator. Evaluate prediction scores, CV loss, embeddings, approximation diagnostics, and application-specific behavior.

For kernel ridge, verify the regularized linear system or compare with scikit-learn using the same kernel, alpha, data, and **unweighted summed-loss** objective. A jitter-stabilized solve may differ from the original system. For CV, inspect finite per-candidate MSE and verify that the final refit uses selected alpha. For Nystroem comparisons, align selected landmarks, their order, and the singular-value floor. Compare KernelPCA subspaces when eigenvector sign/basis ambiguity matters. CPU checks do not establish GPU numerical accuracy or speed; validate on the intended device.

## Complexity

Exact kernel methods materialize an `n x n` training kernel: quadratic memory and, for direct solves/dense eigendecompositions, cubic worst-case arithmetic. KernelRidgeCV reuses each fold decomposition across alpha values but adds fold work and candidate-dependent temporaries. Nystroem replaces the main representation with `n x m` features plus `m x m` landmark linear algebra. GPU speed depends on problem size, dtype, kernel, synchronization, transfers, and available memory; small problems may be faster on CPU. No speedup follows from setting `device` alone.

## Source and statistical references

Implementation: [KernelRidge](../../../statgpu/nonparametric/kernel_methods/_krr.py), [KernelRidgeCV](../../../statgpu/nonparametric/kernel_methods/_krr_cv.py), [KernelPCA](../../../statgpu/nonparametric/kernel_methods/_kpca.py), [Nystroem](../../../statgpu/nonparametric/kernel_methods/_nystroem.py), [pairwise functions](../../../statgpu/nonparametric/kernel_methods/_kernels.py), and [shared estimator behavior](../../../statgpu/_base.py).

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
