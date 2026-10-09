# GAM (Generalized Additive Model)

> Language: English  
> Last updated: 2026-10-09
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/semiparametric.md)

## When a straight line is too restrictive

Suppose a continuous response rises, levels off, or curves with a predictor. A linear regression gives that predictor one slope; `GAM` learns a smooth curve for each feature and adds those curves:

$$
y = \alpha + \sum_j f_j(x_j) + \epsilon.
$$

Use it when the effects are plausibly **smooth and additive** and you want more flexibility than a linear model. This implementation fits **penalized least squares for a continuous response**. Despite the general class name, it has no `family` or `link` API for binary/count responses, and it does not construct interactions automatically. See [GLMs](generalized-linear-model.md) for response-family models, [kernel regression](nonparametric.md) for local smoothing, and [spline bases](splines.md) for building your own terms.

## Intuition and objective

Each curve is a weighted combination of B-spline basis functions. More basis functions allow finer detail; the smoothness penalty discourages unnecessary wiggles. After centering the basis columns on the training data, the intercept describes the overall response level.

$$
\min_\beta \|y-B\beta\|_2^2 + \lambda\beta^\top S\beta,
\qquad (B^\top B+\lambda S)\hat\beta=B^\top y.
$$

Here $B$ concatenates an intercept and the centered basis for every feature. $S$ is block diagonal with $D^\top D$ for each curve and zero intended penalty on the intercept. $D$ is the difference matrix, with its order set by `penalty_order`. The loss is a **sum**, not an average. The implementation uses a stabilized Cholesky solve with general-solve/least-squares fallbacks; numerical stabilization also enters its reported effective degrees of freedom (EDF).

`degree=3` means piecewise **cubic** basis functions. `penalty_order=2` penalizes second differences of adjacent **basis coefficients**. These settings do different jobs: changing the penalty to order 1 does not turn cubic splines into piecewise-linear splines. Use `degree=1` if piecewise-linear basis functions are intended.

<a id="gam-cpu-workflow"></a>

## A complete CPU workflow

Fit and choose smoothing on the training sample only. The held-out sample below stays inside the training range, where the smooths have support.

<!-- example: gam-cpu -->
```python
import numpy as np
from statgpu.semiparametric import GAM
```

### Prepare features and a continuous response

Each row is an observation and the two columns are continuous features. `X_train` has shape `(240, 2)` and `y_train` has shape `(240,)`; the 80 test rows keep the same feature order. Run the blocks in this section in order.

```python
rng = np.random.default_rng(42)
X_train = rng.uniform(-2, 2, size=(240, 2))
y_train = (np.sin(2 * X_train[:, 0]) + 0.4 * X_train[:, 1] ** 2
           + rng.normal(0, 0.15, 240))
X_test = rng.uniform(-1.9, 1.9, size=(80, 2))
y_test = (np.sin(2 * X_test[:, 0]) + 0.4 * X_test[:, 1] ** 2
          + rng.normal(0, 0.15, 80))
```

### Fit the smooth curves

Use 12 spline basis functions per feature. `lam=None` selects smoothing by generalized cross-validation (GCV) on training data without using the test responses.

```python
gam = GAM(n_splines=12, lam=None, device="cpu").fit(X_train, y_train)
```

### Check the fit before using predictions

The current implementation can return an object even when no GCV candidate is valid, and stabilization at large penalties can alter the training response mean. These checks are necessary but do not prove curve accuracy; see “Large smoothing penalties and the intercept” below.

```python
if not np.isfinite(gam.gcv_score_):
    raise RuntimeError("No finite GCV candidate; revise the model before prediction.")
training_prediction = gam.predict(X_train)
if not np.isfinite(training_prediction).all() or not np.isclose(
    training_prediction.mean(), y_train.mean(), rtol=1e-8, atol=1e-10,
):
    raise RuntimeError("Stabilization changed the unpenalized mean; validate the solver.")
```

### Predict held-out responses

Continue with this section's `gam`, `X_test`, and `y_test`. `predict` returns one continuous-response prediction per row; the constant baseline uses only the training responses.

```python
prediction = gam.predict(X_test)
mse = np.mean((prediction - y_test) ** 2)
baseline_mse = np.mean((y_train.mean() - y_test) ** 2)
print(prediction.shape)
print(f"Test MSE: {mse:.4f}; mean baseline: {baseline_mse:.4f}")
print(f"lambda: {gam.lam_:.4f}; EDF: {gam.edf_:.2f}; GCV: {gam.gcv_score_:.4f}")
```

Typical output (rounded):

```text
(80,)
Test MSE: 0.0214; mean baseline: 0.6278
lambda: 0.1963; EDF: 17.78; GCV: 0.0269
```

### Read the result

- Test MSE is in squared response units. Beating the training-mean baseline on unseen observations is more informative than an excellent training fit. Check residuals and performance across the predictor ranges, not just one score.
- `edf_` measures effective model flexibility after smoothing. It need not be an integer and is not the raw number of coefficients.
- `gcv_score_` is the generalized cross-validation (GCV) score used to select smoothing within the training data, not the held-out MSE or a p-value. Lower is better when comparing candidates on the same data and with the same `gamma`.
- `coef_` contains an intercept and spline-basis coefficients. These are **not raw-feature slopes**; inspect predictions while varying a feature to understand a fitted curve. The intended unpenalized `intercept_` equals the training response mean because the smooth bases are centered, but large-lambda stabilization can substantially violate that property; see the [stabilization limitation](#large-smoothing-penalties-and-the-intercept).

## Choose the amount of smoothing

With `lam=None`, the model searches 100 log-spaced values from $10^{-10}$ to $10^{10}$ and minimizes

$$
\operatorname{GCV}(\lambda)=\frac{n\operatorname{RSS}}{(n-\gamma\operatorname{edf})^2},
\qquad
\operatorname{edf}=\operatorname{clip}\!\left(\operatorname{tr}\!\left((A+\delta I)^{-1}B^\top B\right),0,m\right).
$$

Here $A=B^\top B+\lambda S$, $m$ is the number of basis coefficients including the intercept, and $\delta=10^{-10}\operatorname{tr}(A)/m$. This is the reported EDF on the usual stabilized Cholesky path. Centering complete spline blocks can leave $A$ singular, so an ordinary inverse of the unstabilized matrix must not be assumed. If the numerical EDF solve fails, the implementation reports $m$; inspect diagnostics rather than interpreting that fallback as an independently validated model complexity. Standard GCV has `gamma=1`. Larger `gamma` penalizes effective complexity more strongly. When the correction term $1-\gamma\operatorname{edf}/n$, before squaring, is nonpositive or too close to zero, the candidate is assigned infinite GCV; check that the selected score is finite. If every candidate has infinite GCV, the current implementation still returns the first grid value and a fitted object. That is not a valid smoothing selection: treat a nonfinite `gcv_score_` as failure, inspect the sample size, basis design, and `gamma`, and use a scientifically justified revised design or externally validated fixed `lam` before interpreting predictions.

Start with cubic splines and order-2 penalty. Increase `n_splines` only if the fitted shape appears too restricted, then reassess held-out error. Increasing `lam` usually smooths more strongly. Quantile knots place more knots where data are dense; uniform knots are equally spaced across the observed range. Use a validation split or CV to choose these design settings, keeping a final test set untouched.

GCV is a discrete parameter search and may miss an optimum between grid points; a fixed `lam` skips selection but still uses the same numerical solver. `GAM` has no constructor option for a custom lambda grid. For a finer search, fit candidate fixed values using training/validation data, then refit the chosen setting.

### Reuse a selected smoothing value

After “A complete CPU workflow,” reuse its `GAM`, `X_train`, `y_train`, and `gam.lam_`. Passing that value explicitly skips GCV selection and refits the same training data:

```python
fixed = GAM(n_splines=12, lam=gam.lam_, device="cpu").fit(X_train, y_train)
print(fixed.gcv_score_)
```
<!-- example-end: gam-cpu -->

The output is `None` because this fit did not search, not because fitting failed. Reusing the selected value gives the same predictions as `gam`.

## Large smoothing penalties and the intercept

The diagonal stabilization term above grows with `lam` and currently acts on
all coefficients, including directions that the model penalty leaves alone.
It can therefore change the fitted target rather than just prevent roundoff.
For 50 equally spaced points, a constant response of 5 and `n_splines=8`,
`lam=1e10` predicts about 4.630 and `lam=1e12` about 0.556. The intended
unpenalized-intercept optimum is exactly 5 for every nonnegative lambda.
The first value is within the built-in GCV search range; this fixed-lambda
counterexample does not mean GCV necessarily selects it.

Check that fitted training predictions preserve the response mean, as in the
CPU workflow. A material mean shift is a failure of the intended objective,
not evidence of useful smoothing. This check is necessary, not a full proof
of coefficient or curve accuracy: other penalty-nullspace directions can also
be affected. Finite predictions or finite GCV are insufficient. Use an
independently validated, nullspace-preserving solver when such penalties are
needed; changing `lam` changes the statistical problem and is not a repair.

## Inputs, boundaries, and limitations

- `fit(X, y)` converts real numeric inputs to float64 and takes finite `X` of shape `(n_samples, n_features)` and one continuous target per row, normally `(n_samples,)`. A 1D `X` is treated as one feature; `y` is flattened, so a single-column target also works. Empty data, mismatched lengths, nonfinite values, and constant feature columns raise `ValueError`. Do not add your own all-ones intercept column.
- Use numeric continuous predictors. Quantile knots may coincide for heavily tied/discrete features; duplicates are removed, and knots at the training boundary can cause a `ValueError`. Do not assume a large basis makes categorical data suitable for smoothing.
- `predict(X)` requires the same feature count and order. Prefer an explicit `(n_query, n_features)` array. For one fitted feature a 1D vector means several queries; for several fitted features a length-`n_features_` vector means one query. Predictions are a **NumPy array** of shape `(n_query,)`, including after GPU fitting.
- For multi-feature Torch models, keep queries explicitly two-dimensional, including `(1, n_features_)` for one row. The current vector shape check raises `TypeError` on Torch; use `X[:1]` rather than `X[0]`.
- The training knots and boundaries are reused at prediction time. Outside a feature's training range its B-spline basis is zero before centering; this is not a reliable linear or smooth extrapolation rule. Restrict interpretation to supported ranges.
- The additive model can miss interactions. Highly correlated predictors can make separate smooth effects hard to interpret even when predictions are useful.
- No coefficient standard errors, p-values, confidence bands, `cov_type`, sample-weight fit, or family/link likelihood is implemented here. EDF and GCV do not supply uncertainty intervals.
- If a refit fails while constructing the basis, the instance can retain old coefficients alongside partially replaced knots and feature metadata. Discard that instance and fit a fresh model before predicting or reading its summary.

## Complete constructor and output reference

Import: `from statgpu.semiparametric import GAM`.

| Parameter | Default | Meaning |
|---|---:|---|
| `n_splines` | `20` | Requested basis count per feature; integer greater than `degree + 1`. Duplicate knots can reduce the actual count. |
| `degree` | `3` | Nonnegative integer spline degree. |
| `lam` | `None` | GCV selection, or a finite nonnegative fixed penalty strength. |
| `penalty_order` | `2` | Positive integer difference order, smaller than each feature's actual basis count. |
| `knot_method` | `"quantile"` | `"quantile"` or `"uniform"`; use these lowercase spellings. |
| `gamma` | `1.0` | Positive finite EDF multiplier in GCV; does not change a fixed-lambda objective. |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"`; see device guidance below. |
| `n_jobs` | `None` | Provided for compatibility with the common estimator interface; the current GAM fit does not use it for parallel computation. |

| Output | Shape/type | Interpretation |
|---|---|---|
| `coef_` | Backend array, `(1 + sum(n_basis_j),)` | Intercept followed by coefficients for each feature's centered basis. |
| `intercept_` | `float` | Intercept coefficient. |
| `edf_` | `float` | Total effective degrees of freedom. |
| `gcv_score_` | `float` or `None` | Best searched GCV, or `None` for fixed `lam`. |
| `lam_` | Scalar | Smoothing parameter used by the fitted model. |
| `knots_` | List of backend arrays | Interior knots for each feature. |
| `n_features_` | `int` | Training feature count. |

Methods: `fit(X, y)` returns `self`; `predict(X)` returns predictions; `summary()` prints diagnostics and returns a dictionary (the `gcv_score` key is omitted for fixed `lam`); `get_params(deep=True)` / `set_params(**params)` provide estimator parameter access. Refit after changing parameters. There is no GAM-specific `score()` method; compute a held-out metric as above.

The [complete method and output reference](../reference/survival-smoothing-api.md#gam)
includes exact call signatures, summary dictionary keys, inherited-helper
boundaries, and parameter-change behavior. Only `X` and `y` are used by `fit`;
extra fit keywords currently do not enable additional capabilities and should
be omitted. In particular, passing `sample_weight` does not produce a weighted fit.

Algorithm sources: [GAM](../../../statgpu/semiparametric/_gam.py), [penalized least squares and GCV](../../../statgpu/nonparametric/splines/_penalized.py), [basis construction](../../../statgpu/nonparametric/splines/_bspline_basis.py), and [shared estimator methods](../../../statgpu/_base.py).

## Optional GPU execution

After running “A complete CPU workflow,” reuse its `GAM`, `X_train`, `y_train`, and `X_test` below. This snippet requires a working CuPy/CUDA installation. `device="torch"` selects the Torch CUDA route; neither explicit GPU request silently falls back to CPU. `device="auto"` permits automatic selection. See [device and memory](../guides/device-and-memory.md). GPU benefit depends on basis size, transfers, and hardware; measure your workload rather than assuming a speedup.

<!-- example-requires: gam-cpu -->
<!-- example: gam-gpu -->
```python
# Optional: reuses X_train, y_train, X_test from the CPU example.
gam_gpu = GAM(n_splines=12, device="cuda").fit(X_train, y_train)
prediction_gpu = gam_gpu.predict(X_test)  # NumPy output
```
<!-- example-end: gam-gpu -->

## External comparisons and references

For comparisons with pyGAM or another smoother, align knots, basis degree, difference penalty, loss normalization, lambda, and GCV `gamma`; similar class names alone do not imply identical fits or inference. A test of one CPU example is not a GPU/performance benchmark.

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman & Hall/CRC.
