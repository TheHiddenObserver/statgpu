# Kernel Ridge Cross-Validation

> Language: English  
> Last updated: 2026-10-12  
> Switch: [Chinese](../../cn/models/kernel-ridge-cv.md)

## Select regularization without using the test set

`KernelRidgeCV` chooses the regularization strength `alpha` for kernel ridge
regression from training folds, then fits a final `KernelRidge` on all rows
supplied to `fit`. It is useful when the kernel and its settings are already
fixed and observations can be split into shuffled row-wise folds.

Use [KernelRidge](kernel-ridge.md) when `alpha` is fixed or when you need to
control the splitting and preprocessing yourself. `KernelRidgeCV` selects
**only alpha**, not the kernel type, `gamma`, or a preprocessing recipe. For
other kernel tasks, see the [overview](kernel-methods.md).

<a id="selection"></a>

## What the search actually compares

For each candidate $\alpha$ and training fold $T$, the method solves

$$
(K_{TT}+\alpha I)c_{\alpha,T}=y_T
$$

and predicts the corresponding validation fold $V$ with
$K_{VT}c_{\alpha,T}$. If there are $F$ folds and $r$ response columns, selection
minimizes

$$
L(\alpha)=\frac{1}{Fr}\sum_{f=1}^{F}\sum_{j=1}^{r}
\frac{1}{|V_f|}\sum_{i\in V_f}
  (y_{ij}-\hat y_{ij}^{(-f,\alpha)})^2.
$$

Folds and response columns receive equal weight; this is not a pooled
observation-weighted loss when fold sizes differ. With several responses,
columns with larger response units can dominate MSE even though columns are
averaged uniformly.

`alpha_` is the selected value. **`best_score_` is the mean fold R-squared at that
value, not the MSE minimized during selection.** After selection, `estimator_`
is the final model refitted on all supplied training rows. Neither its training
score nor `best_score_` is an independent held-out evaluation.

For a fixed fold kernel $K_{TT}=Q\Lambda Q^\top$, coefficients for each candidate
can be computed as

$$
c_{\alpha,T}=Q\,\operatorname{diag}\!\left(\frac{1}{\lambda_i+\alpha}\right)
Q^\top y_T.
$$

Reusing the fold eigendecomposition avoids a new linear solve for every
candidate. The kernel-ridge objective and its summed-loss normalization are
explained in the [KernelRidge tutorial](kernel-ridge.md#from-similarity-to-the-fitted-function).

<a id="example"></a>
<a id="kernel-cv-workflow"></a>

## A complete CPU selection and evaluation example

This example runs independently and keeps 60 of 240 observations outside model
selection. The two features are generated on the same scale, so no learned
preprocessing is used. The kernel and `gamma=0.7` are fixed in advance.

<!-- example: kernel-ridge-cv-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidgeCV
```

### Prepare training and held-out rows

Keep the held-out set outside the search.
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### Select alpha on training folds

All candidates share the fixed kernel and scale.
```python
kr_cv = KernelRidgeCV(
    alphas=[0.01, 0.1, 1.0, 10.0], gamma=0.7,
    cv=5, random_state=42, device="cpu",
).fit(X_train, y_train)
```

### Check the search and predict

Reject nonfinite search or model results before interpreting the fit.
```python
assert np.isfinite(kr_cv.cv_results_["mean_mse"]).all()
assert np.isfinite(kr_cv.best_score_)
assert np.isfinite(kr_cv.dual_coef_).all()
prediction = kr_cv.predict(X_test)
assert np.isfinite(prediction).all()
held_out_r2 = kr_cv.score(X_test, y_test)
held_out_mse = np.mean((y_test - prediction) ** 2)
```

### Read the selected model and held-out results

Report the selection metric separately from the final evaluation.
```python
print("Selected alpha:", kr_cv.alpha_)
print("Mean validation MSE by alpha:", kr_cv.cv_results_["mean_mse"][:, 0])
print("Mean fold R2 at selected alpha:", kr_cv.best_score_)
print("Prediction shape:", prediction.shape)
print("CV model held-out R2 / MSE:", held_out_r2, held_out_mse)
```
<!-- example-end: kernel-ridge-cv-cpu -->

The prediction shape is `(60,)`. Lower MSE means smaller squared prediction
error; R-squared compares predictions with the response mean of the evaluated
rows, with 1 perfect, 0 equal to that baseline, and negative values worse.
`best_score_` summarizes the training folds. The final held-out R-squared and
MSE evaluate the chosen, refitted model. Do not keep adjusting the candidates,
kernel, or `gamma` using these same test rows.

### Read the result arrays

For $a$ candidates, $F$ folds, and $r$ response columns:

- `cv_results_["alphas"]` records the candidates.
- `mse_table` and `r2_table` have shape `(a, F, r)`.
- `mean_mse` and `mean_r2` have shape `(a, r)` after averaging folds.
- `best_alpha` and `best_score` mirror `alpha_` and `best_score_`.

These result arrays are NumPy arrays even for a GPU fit. For multi-output
selection, average each row of `mean_mse` over responses to recover the selection
loss. `predict` and `score` delegate to `estimator_`; the latter uniformly
averages per-response R-squared. See the [complete API](../reference/kernel-methods-api.md)
for all attributes and shapes.

## Choose candidates and folds

- `alphas` accepts a nonempty finite nonnegative array. Prefer **strictly
  positive** values on a logarithmic scale. With `alphas=None`, the class builds
  100 positive log-spaced candidates from the training kernel spectrum.
- `cv` must be an integer from 2 through the number of supplied rows. It creates
  shuffled row-wise K-fold splits. `random_state` makes this shuffling
  reproducible; it does not turn the splitter into a grouped or chronological
  one.
- Kernel type and kernel controls remain fixed throughout a search. Choosing
  `gamma` or other controls requires a separate training-only selection
  procedure that evaluates the complete fitting pipeline.

An explicit zero alpha can cause division by zero in a singular training-fold
kernel. Nonfinite CV scores can still be selected, so merely obtaining a fitted
object does not establish a valid search. Use positive candidates and check
`mean_mse`, `best_score_`, final coefficients, and predictions for finiteness.
A successful final fit cannot repair an invalid earlier selection.

<a id="fold-local-preprocessing"></a>

## When you need an external splitting loop

Custom splitters, grouped folds, and time-ordered splits are not supported by
the built-in `cv` argument. Nor does `KernelRidgeCV` fit a preprocessing step
separately in each fold. Do not fit a scaler on all supplied training rows and
then pass the scaled data to this class: its inner validation rows would have
influenced that scaler.

For learned preprocessing, use an external loop: split the raw training data,
fit preprocessing on each fold's training rows, apply it to that fold's
validation rows, and fit `KernelRidge` for each candidate. Reuse the same splits
across candidates. After selecting settings, fit preprocessing and the model on
all outer training rows, then evaluate the untouched test set once. For grouped
or temporal data, use splits that preserve the grouping or time direction.

The independent example below demonstrates the fold-local scaling pattern with
row-wise folds. Its splitter is for independent observations; replace those
splits for grouped or time-ordered data.

<!-- example: kernel-ridge-fold-local-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge

```

Prepare raw data and reserve the test set.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(160, 2)) * np.array([1.0, 10.0])
y = np.sin(X[:, 0]) + 0.02 * X[:, 1] ** 2 + rng.normal(0, 0.1, 160)
order = rng.permutation(len(X))
X_train, X_test = X[order[:120]], X[order[120:]]
y_train, y_test = y[order[:120]], y[order[120:]]
folds = np.array_split(rng.permutation(len(X_train)), 4)
alphas = np.array([0.01, 0.1, 1.0, 10.0])
losses = np.empty((len(alphas), len(folds)))

```

Fit scaling and each candidate using only the current training fold.

```python
for fi, validation in enumerate(folds):
    training = np.concatenate([fold for j, fold in enumerate(folds) if j != fi])
    mean = X_train[training].mean(axis=0)
    scale = X_train[training].std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    X_fit = (X_train[training] - mean) / scale
    X_validation = (X_train[validation] - mean) / scale
    for ai, alpha in enumerate(alphas):
        model = KernelRidge(alpha=alpha, gamma=0.7, device="cpu")
        model.fit(X_fit, y_train[training])
        pred = model.predict(X_validation)
        losses[ai, fi] = np.mean((y_train[validation] - pred) ** 2)

```

Select alpha, then learn fresh preprocessing on all outer training rows.

```python
assert np.isfinite(losses).all()
selected_alpha = alphas[np.argmin(losses.mean(axis=1))]
mean = X_train.mean(axis=0)
scale = X_train.std(axis=0)
scale = np.where(scale > 0, scale, 1.0)
final_model = KernelRidge(alpha=selected_alpha, gamma=0.7, device="cpu")
final_model.fit((X_train - mean) / scale, y_train)
held_out_prediction = final_model.predict((X_test - mean) / scale)
assert np.isfinite(held_out_prediction).all()
print("Selected alpha:", selected_alpha)
print("Held-out MSE:", np.mean((y_test - held_out_prediction) ** 2))
```
<!-- example-end: kernel-ridge-fold-local-cpu -->

Here each validation loss assesses a scaler and model that never used that
fold's validation rows for fitting. The final scaler is learned from all 120
outer training rows and reused unchanged on the 40 test rows.

## Boundaries and computation

Cross-validation does not provide coefficient inference or confidence intervals.
The underlying model is unweighted; this class has no sample-weight argument.
The [KernelRidge precautions](kernel-ridge.md#numerical-and-practical-precautions)
also apply: finite data can yield an overflowed kernel, poorly scaled RBF
coordinates can lose distances, and a refit may use numerical jitter.

The full kernel takes quadratic memory. Each training fold requires a dense
eigendecomposition; reuse across alpha values saves work, but folds still add
cost. Kernels and fold linear algebra use the selected backend; unavailable
explicit device requests raise. See the [optional GPU examples](#gpu-examples) and the [complete API](../reference/kernel-methods-api.md) for execution details.

<a id="gpu-examples"></a>

## Optional GPU examples

These optional cross-validated GPU examples depend on the [CPU example above](#kernel-cv-workflow),
which defines `KernelRidgeCV`, `X_train`, `y_train`, `X_test`, and `kr_cv`.
Run that example first, then choose one snippet for a working CUDA installation.
Only the device changes; these are not independent CPU examples.

<a id="cupy"></a>

### CuPy CUDA

<!-- example-requires: kernel-ridge-cv-cpu -->
<!-- example: kernel-methods-cupy -->
```python
gpu_options = {**kr_cv.get_params(), "device": "cuda"}
kr_cupy = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_cupy = kr_cupy.predict(X_test)
```
<!-- example-end: kernel-methods-cupy -->

<a id="torch-cuda"></a>

### Torch CUDA

<!-- example-requires: kernel-ridge-cv-cpu -->
<!-- example: kernel-methods-torch -->
```python
gpu_options = {**kr_cv.get_params(), "device": "torch"}
kr_torch = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_torch = kr_torch.predict(X_test)
```
<!-- example-end: kernel-methods-torch -->

Use backend-native finite checks on device predictions and coefficients, and
check the NumPy CV results as in the CPU tutorial. CPU checks do not establish
GPU accuracy or speed. KernelPCA and Nystroem have different execution and output
placement boundaries; consult their tutorials and the
[device guide](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions).

## Further reading

- [KernelRidge: objective, predictions, and numerical checks](kernel-ridge.md)
- [Kernel-methods overview](kernel-methods.md) and [complete API](../reference/kernel-methods-api.md)
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
