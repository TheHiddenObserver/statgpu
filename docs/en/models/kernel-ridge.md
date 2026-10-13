# Kernel Ridge Regression

> Language: English  
> Last updated: 2026-10-12  
> Switch: [Chinese](../../cn/models/kernel-ridge.md)

## Predict a curved response from similarities

Use `KernelRidge` to predict a continuous response when nearby or otherwise
similar observations should have similar outcomes, but a straight-line model is
too restrictive. An RBF kernel lets the fitted response bend without specifying
every nonlinear feature or interaction yourself. The model combines similarities
to training observations, so its coefficients are not slopes for the original
features.

Start with a [linear model](linear-regression.md) when linear effects suffice or
interpretable slopes matter. A [GAM](semiparametric.md) is often easier to
interpret when separate smooth feature effects add together. Exact kernel ridge
is useful when interactions matter and the dense training kernel fits in memory;
[Nystroem features](kernel-nystroem.md) provide a smaller approximate alternative.
For the wider choice of tools, see the [kernel-methods overview](kernel-methods.md).

<a id="theory"></a>

## From similarity to the fitted function

Let the training rows be $x_1,\ldots,x_n$, their responses be $y$, and the kernel
matrix be $K_{ij}=k(x_i,x_j)$. The default RBF kernel is

$$
k(x,z)=\exp\{-\gamma\lVert x-z\rVert_2^2\}.
$$

Larger `gamma` makes similarity decay over a shorter distance. Feature units
therefore matter: a feature with a much larger scale can dominate the distance.
For a positive-semidefinite kernel, kernel ridge minimizes squared error with an
RKHS penalty:

$$
\min_{f\in\mathcal H}\sum_{i=1}^n(y_i-f(x_i))^2
    +\alpha\lVert f\rVert_{\mathcal H}^2.
$$

The fitted function has a representation $f(x)=\sum_i c_i k(x,x_i)$. Substitution
gives the coefficient objective

$$
\min_c\ \lVert y-Kc\rVert_2^2+\alpha c^\top Kc.
$$

The loss is a **sum**, not an average. Penalizing $\lVert c\rVert^2$ instead
would give a different estimator. For symmetric $K$, the gradient is

$$
\nabla_c=2K(Kc-y)+2\alpha Kc
       =2K\{(K+\alpha I)c-y\}.
$$

For positive-definite $K$, setting this to zero gives

$$
(K+\alpha I)c=y.
$$

When $K$ is singular positive-semidefinite and $\alpha>0$, the same invertible
system selects one coefficient representation of the minimizing fitted function;
coefficients themselves need not be unique in the original objective. An
indefinite kernel does not have this convex RKHS interpretation. For several
response columns $Y$, the same system is solved with a matrix right-hand side;
the objective sums the squared losses and RKHS penalties over columns.

For $q$ new rows, predictions use the $q\times n$ query-to-training kernel:

$$
\hat y_{\mathrm{new}}=K(X_{\mathrm{new}},X_{\mathrm{train}})c.
$$

No separate intercept is fitted by this API. If you choose response centering,
learn the response mean only from training data, fit the centered responses, and
add that mean back to predictions. Repeat this process inside each training fold
when selecting settings.

<a id="example"></a>
<a id="kernel-cpu-workflow"></a>

## A complete training and held-out CPU example

This example runs independently. `X` has 240 rows and two continuous features;
`y` is one continuous response per row. We reserve 60 rows for final evaluation
and fix `alpha=1.0` and `gamma=0.7` before looking at them. To choose `alpha` from
training folds instead, continue with the separate
[KernelRidgeCV tutorial](kernel-ridge-cv.md).

<!-- example: kernel-ridge-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge
```

### Prepare and split the data

Reserve test rows before choosing settings.
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### Fit the fixed-setting model

Fit only on the 180 training observations.
```python
kr = KernelRidge(alpha=1.0, gamma=0.7, device="cpu").fit(X_train, y_train)
assert np.isfinite(kr.dual_coef_).all()
```

### Predict and evaluate

Apply the fitted model to test rows with the same feature order.
```python
fixed_prediction = kr.predict(X_test)
assert np.isfinite(fixed_prediction).all()
print("Prediction shape:", fixed_prediction.shape)
print("Fixed-alpha held-out R2:", kr.score(X_test, y_test))
print("Held-out MSE:", np.mean((y_test - fixed_prediction) ** 2))
```
<!-- example-end: kernel-ridge-cpu -->

The `(60,)` output contains one prediction for each held-out observation, in the
same response units as `y`. MSE is in squared response units; lower is better.
R-squared is 1 for perfect predictions, 0 for the held-out response-mean baseline,
and negative for predictions worse than that baseline. Multi-output `score`
uniformly averages the per-response R-squared values; constant responses use 1
for effectively exact predictions and 0 otherwise.

`dual_coef_` has shape `(180, 1)` here, even though predictions are flattened.
It weights training observations in kernel space. It cannot be interpreted as
the effect of increasing one input feature, and this module does not provide
coefficient-level standard errors, hypothesis tests, or confidence intervals.

### Check the defining equation on this small example

Continue with the CPU fit above. Forming another dense training kernel is useful
for a small diagnostic, but duplicates memory and is unnecessary for prediction.

<!-- example-requires: kernel-ridge-cpu -->
<!-- example: kernel-ridge-system-check -->
```python
from statgpu.nonparametric.kernel_methods import pairwise_kernels

K = pairwise_kernels(X_train, metric="rbf", gamma=0.7, xp=np)
np.testing.assert_allclose(
    (K + kr.alpha * np.eye(len(X_train))) @ kr.dual_coef_,
    y_train[:, None], rtol=1e-9, atol=1e-9,
)
```
<!-- example-end: kernel-ridge-system-check -->

## Choose the kernel and regularization together

- `alpha` controls shrinkage for the summed-loss objective. Use a finite,
  strictly positive value as a practical starting point; zero is accepted but
  is fragile with singular kernels. An averaged-loss formulation needs a
  corresponding penalty-scale conversion before comparison.
- `gamma` controls the RBF similarity scale, separately from `alpha`. Its default
  is `1 / n_features`. Choose both within the training procedure; the built-in
  CV class selects **only alpha**.
- `kernel` chooses the similarity definition. Polynomial kernels also use
  `degree` and `coef0`; chi-squared requires nonnegative inputs. For KernelRidge,
  chi-squared `gamma` must be passed through `kernel_params={"gamma": value}`,
  because that kernel currently ignores the constructor's `gamma` argument.
- Learned scaling, imputation, or response centering must be fitted within each
  training fold. Use an external splitting loop when preprocessing must be
  learned, or when observations are grouped or time-ordered; see the
  [CV tutorial](kernel-ridge-cv.md#fold-local-preprocessing).

These are teaching highlights. The [complete kernel API](../reference/kernel-methods-api.md)
lists every constructor argument, public method, fitted attribute, kernel
function, and backend boundary.

<a id="numerical-precautions"></a>

## Numerical and practical precautions

`KernelRidge` solves the regularized system directly. If linear algebra raises,
it retries with increasing diagonal jitter. A stabilized result need not solve
the original displayed system exactly. Prefer a positive, well-scaled `alpha`;
when checking against scikit-learn or the linear system, align the kernel,
regularization, data, and unweighted summed-loss objective.

`fit(X, y, sample_weight=None)` currently **ignores sample weights**. Omit that
argument and use this estimator for unweighted analysis. Removing rows is a
valid way to exclude observations, but it does not implement arbitrary weights.

Finite training data does not guarantee a finite computed kernel. Polynomial
kernels can overflow, and invalid controls such as NaN `gamma` can produce a
returned estimator with nonfinite coefficients or predictions. Check applicable
controls, learned arrays, and outputs, as in the example; discard nonfinite fits
and fit a fresh model after revising scaling or kernel settings. Increasing
`alpha` cannot repair an already overflowed kernel. Finiteness alone does not
establish good conditioning or predictive quality.

RBF squared-distance arithmetic can lose nearby differences when coordinates
have a large common offset, for example near `1e9`. Subtract the same
training-derived offset from training and query features. This preserves the
intended RBF similarities; it is not a general preprocessing rule for every
kernel, especially chi-squared. Fit any learned preprocessing inside CV folds.

Keep feature order and count consistent at prediction. A one-dimensional `X`
means multiple rows of one feature, not one multivariate query; represent one
multivariate query as a two-dimensional row. Refit after changing parameters.
Do not repeatedly use the held-out score to choose settings.

## Computation and devices

Exact fitting stores an $n\times n$ kernel, giving quadratic memory use and a
cubic worst-case direct solve. Kernel construction and solving run on the
selected backend. Explicit unavailable `"cuda"` or `"torch"` requests raise;
`"auto"` requests automatic selection. Small examples can be faster on CPU
because transfer, launch, and synchronization overheads matter.

For cross-validated GPU fits, see the dependent [CuPy and Torch CUDA examples](kernel-ridge-cv.md#gpu-examples). CPU checks do not establish GPU accuracy or speed. KernelPCA and Nystroem have different execution and output-placement boundaries; see their tutorials and the [device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions).

## Further reading

- [Select alpha with KernelRidgeCV](kernel-ridge-cv.md)
- [Nystroem features for a smaller representation](kernel-nystroem.md)
- [Kernel-methods overview](kernel-methods.md) and [complete API](../reference/kernel-methods-api.md)
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
