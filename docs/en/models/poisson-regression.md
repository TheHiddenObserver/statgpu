# PoissonRegression

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/poisson-regression.md)

## When to use this model

`PoissonRegression` models a nonnegative count through a log-linked conditional
mean. It is useful for event counts observed over comparable exposure periods.
A slope describes a multiplicative association with the expected count; it does
not establish a causal effect.

The Poisson model assumes the conditional variance equals the conditional mean.
Marked overdispersion can motivate `NegativeBinomialRegression`; excess zeros
or dependent counts may require a different model or covariance analysis.
Positive continuous outcomes usually call for a family such as Gamma rather
than treating them as counts. See [choosing a GLM](generalized-linear-model.md).

This typed class fixes the Poisson family and shares the ordinary GLM API.
Its default `C=1` with `solver="auto"` applies ridge regularization.
Use `C=0` for explicitly unpenalized IRLS, or choose an explicit unpenalized
solver as described below. Use `PenalizedPoissonRegression` for an alpha-based
penalty interface. There is no offset/exposure argument: adding exposure as an
ordinary predictor estimates its coefficient rather than fixing it at one.

<a id="cpu-example"></a>

## A complete CPU example

Run the steps in order: prepare counts, fit, predict, then optionally add coefficient inference. No GPU or formula dependencies are needed.

### 1. Import

<!-- learner-example: poisson-unpenalized -->
```python
import numpy as np
from statgpu import PoissonRegression
```

### 2. Prepare inputs and held-out data

`X` has shape `(200, 2)`: rows are observations and columns are two numeric features. `y` is a length-200 vector of nonnegative integer counts. The log link makes the conditional mean `exp(0.3 + X @ beta)` positive; this is not an event probability. Handle missing values in real data and retain the training feature order for prediction.

```python
rng = np.random.default_rng(8)
X = rng.normal(size=(200, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
X_test = rng.normal(size=(50, 2))
y_test = rng.poisson(np.exp(0.3 + X_test @ np.array([0.4, -0.2])))
```

The separately generated 50 rows in `X_test`, `y_test` are never used for fitting.

### 3. Fit an unpenalized model

`C=0` makes the unpenalized intent explicit. This example requests Newton, which does not use C. The default auto/IRLS path with positive C instead applies ridge shrinkage; the objective section explains that distinction.

```python
model = PoissonRegression(
    C=0, solver="newton", device="cpu",
    max_iter=200, tol=1e-9, compute_inference=False,
).fit(X, y)
print("Slopes:", np.round(model.coef_, 3))
print("Mean multipliers:", np.round(np.exp(model.coef_), 3))
```

For this seed, slopes round to `[0.377, -0.154]` and mean multipliers to `[1.459, 0.857]`. Holding the other feature fixed, a unit increase in the first feature corresponds to about a 45.9% higher expected count; a unit increase in the second to about a 14.3% decrease. These are associations, not causal effects.

### 4. Predict and evaluate held-out data

```python
mean_prediction = model.predict(X_test)
heldout_loss = np.mean(mean_prediction - y_test * np.log(mean_prediction))
print("Predicted means:", np.round(mean_prediction[:3], 3))
print("Held-out Poisson loss:", round(float(heldout_loss), 3))
```

Predictions have shape `(50,)`, are positive, and may be fractional even though observations are counts. The held-out loss is about `0.963` and omits the response-only log-factorial term. Lower is better only on the same held-out responses and weighting; it is neither accuracy nor ordinary R². This ordinary GLM class has no `score` or `predict_proba` method.

### 5. Optional: coefficient intervals

Reuse the training data from step 3 and refit with inference enabled. `cov_type="nonrobust"` uses the Poisson model's mean–variance assumption. Covariance choice does not change the fitted mean; assumptions and alternatives are explained in the inference section below.

```python
model.set_params(compute_inference=True, cov_type="nonrobust")
model.fit(X, y)
print("Interval shape:", model._conf_int.shape)
```

The interval array has shape `(3, 2)`: intercept first, then the two slopes. These are asymptotic marginal coefficient intervals, not intervals for future counts. `summary()` returns a string; display it with `print(model.summary())`.
<!-- example-end: poisson-unpenalized -->

## Choosing settings and checking results

- Use `C=0` for an unpenalized fit. For prediction with positive-C IRLS,
  choose C by training-only validation; larger positive C weakens shrinkage.
  The special value zero removes it exactly rather than increasing it.
- Learn any scaling inside each training fold and apply the same transform to
  evaluation data. Predictor units affect ridge shrinkage. Keep a final test
  set separate from all tuning.
- Examine observed versus predicted counts, exposure comparability and
  overdispersion. A finite fit or small training loss does not validate the
  Poisson variance assumption. HC covariance changes uncertainty, not the mean
  model or predictions.
- Use `max_iter` and `tol` to assess numerical stability. `n_iter_` is an
  iteration count, not a convergence certificate. Poor scaling or collinearity
  cannot always be repaired by increasing the iteration budget.

## Objective Function

With log link, the model assumes:

$$
\mu_i = \exp(x_i^\top\beta)
$$

and minimizes the average Poisson negative log-likelihood, up to constants independent of the parameters:

$$
\min_\beta \frac{1}{n}\sum_i \left[\mu_i - y_i \log(\mu_i)\right]
$$

With an intercept, replace $x_i^\top\beta$ by $b+x_i^\top\beta$.
With analytic `sample_weight=w`, replace the average by a weighted sum divided
by `sum(w)`. For positive C, ordinary auto/IRLS adds
$\|\beta\|_2^2/(4C)$, excluding the intercept. `C=0` removes that term;
ordinary explicit `newton`, `lbfgs`, and `fista` ignore C and optimize the
unpenalized loss. Changing solvers with positive C can therefore change the
statistical model. This differs from standalone LogisticRegression's C scale.
See the [ordinary GLM objective](generalized-linear-model.md).

## Estimating Equation

The score equation for the unpenalized Poisson GLM is:

$$
\sum_i x_i(y_i - \mu_i)=0
$$

For positive C on auto/IRLS, the average-loss slope equation additionally contains the ridge gradient `beta/(2*C)`.

`PoissonRegression` defaults to `solver="auto"`, which currently dispatches to IRLS. Explicit `solver="newton"` and `solver="lbfgs"` are also available for smooth Poisson GLM objectives and run on the selected backend. The model inherits the GLM formula interface, so formula intercept semantics follow patsy/R conventions.

## Covariance/Inference

Set `compute_inference=True` for coefficient uncertainty, as in step 5 of the [CPU example](#cpu-example). Use `cov_type="hc0"` or `"hc1"` when the corresponding score-robust
covariance assumptions fit the application; they do not change the estimates.

**Covariance types**:
- ``'nonrobust'`` (default): model-based, φ·I(β)⁻¹ using expected Fisher information in the unpenalized case; positive-C IRLS adds penalty curvature.
- ``'hc0'``: sandwich H⁻¹·J·H⁻¹ with observed Hessian.
- ``'hc1'``: HC0 × n/(n−k) degrees-of-freedom correction.
- ``'hc2'``, ``'hc3'``, ``'hac'``: not yet implemented for Poisson (raises ``NotImplementedError``).

**Distribution**: z-statistics (asymptotic normal).  P-values are two-sided.
**Dispersion**: φ = 1.0 (Poisson variance = mean).  Pearson dispersion available via metadata.

These are marginal, asymptotic normal-reference intervals. With positive C on
IRLS, the covariance includes penalty curvature and describes the penalized fit;
it does not remove shrinkage bias or account for choosing C. For comparisons
with an unpenalized `statsmodels.GLM`, align C/solver, design, weights, covariance
and convergence settings. A comparison on one dataset is not a universal
precision guarantee.
**GPU**: fully supported (CuPy, Torch) — backend-agnostic sandwich engine runs on the same device as fitting.  No silent CPU fallback.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `100` | Maximum IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `C` | `1.0` | Positive C: auto/IRLS adds slope penalty `sum(beta**2)/(4*C)`; zero removes it. Explicit Newton/L-BFGS/FISTA ignore C. |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs` |
| `n_jobs` | `None` | Shared configuration; this class does not promise parallel fitting |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy memory pool cleanup after fit |
| `compute_inference` | `False` | Compute supported coefficient uncertainty after fitting |
| `cov_type` | `"nonrobust"` | `nonrobust`, `hc0`, or `hc1` |

`formula` and `data` are arguments to
`fit(X=None, y=None, sample_weight=None, formula=None, data=None)`, not constructor
parameters. This class fixes the Poisson family and otherwise inherits the
[ordinary GLM methods and fitted fields](../reference/linear-model-api.md#generalizedlinearmodel),
including `predict`, `summary`, likelihood diagnostics and shared inference
helpers. It has no `score` or `predict_proba` method. `summary()` returns a string;
use `print(model.summary())`. On current auto/IRLS/FISTA paths, create a fresh
estimator after a failed refit, as explained in the [failed-refit warning](../reference/linear-model-api.md#failed-ordinary-glm-refits).

## Optional GPU and formula inputs

Reuse the imports and data from the completed [CPU example](#cpu-example). The same unpenalized fit can request CuPy CUDA:

```python
model_gpu = PoissonRegression(
    C=0, solver="newton", device="cuda", max_iter=200, tol=1e-9,
).fit(X, y)
mean_prediction_gpu = model_gpu.predict(X_test)
```

Use `device="torch"` for Torch CUDA. Explicit GPU requests require an installed,
usable CUDA backend and raise when it is unavailable. The result is a native
prediction array on the resolved backend; coefficient reporting arrays are
NumPy. Formula parsing itself runs on CPU.

The following formula example is independently runnable after installing
`statgpu[formula]`. Formula syntax controls the intercept and categorical coding:

<!-- learner-example: poisson-formula -->
```python
import numpy as np
import pandas as pd
from statgpu import PoissonRegression
```

In this independent example, each row of `df` is an observation: `count` is the response, `x` is numeric, and `group` is categorical.

```python
rng = np.random.default_rng(18)
df = pd.DataFrame({"x": rng.normal(size=80), "group": ["a", "b"] * 40})
df["count"] = rng.poisson(np.exp(0.2 + 0.3 * df["x"]))
```

The formula constructs the intercept and categorical coding. For prediction, supply a DataFrame with the same column names and category levels.

```python
model = PoissonRegression(C=0, device="cpu")
model.fit(formula="count ~ x + C(group)", data=df)
prediction = model.predict(df.iloc[:5])
assert prediction.shape == (5,)
```
<!-- example-end: poisson-formula -->


Pass formula/data without simultaneous array X/y. Prediction DataFrames rebuild
the training columns and categorical levels; unseen levels raise. Missing
predictor rows are currently dropped, producing a shorter unlabelled array.
Resolve missing values and check output length before aligning predictions;
see the [ordinary GLM missing-row warning](../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms). Weights may cover original rows or retained rows, matched by
position. See the [complete formula contract](../reference/linear-model-api.md#formula-inputs).
For large GPU workloads, already prepared arrays avoid formula parsing overhead.

## strict/approx difference

There is no public strict/approx inference switch for `PoissonRegression`. Supported inference follows the chosen covariance convention; changing a solver can change the penalty as described above.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Iterations: `n_iter_`
- Methods: `fit`, `predict`
- Formula metadata is stored internally when fitting with `formula` and `data`

`predict` returns the inverse-link mean response, so for Poisson it returns estimated counts/rates \(\hat\mu\), not the linear predictor.

## FAQ

- When should I use `PoissonRegression` instead of `GeneralizedLinearModel(family="poisson")`? Use `PoissonRegression` when you want the explicit model class and clearer public API. Both share the GLM implementation.
- When should I use `PenalizedPoissonRegression`? Use it when you need L1, L2, ElasticNet, group, or adaptive penalty support.
- Does `PoissonRegression` provide standard errors and p-values? Yes — set ``compute_inference=True``.  Supports model-based (``cov_type='nonrobust'``) and sandwich (``hc0``, ``hc1``) covariance.  For unpenalized comparisons, explicitly align the objective and covariance settings.
- Does `device="cuda"` always guarantee GPU execution? For supported Poisson GLM solver paths, yes: core computation stays on CuPy or raises a clear error. `device="torch"` similarly requires Torch CUDA.

## Comparing with another implementation

For an unpenalized comparison with `statsmodels.GLM`, align the log link,
intercept/design, responses, weights, covariance and convergence settings; use
C=0 with IRLS or an explicit unpenalized solver. For positive-C IRLS versus
sklearn `PoissonRegressor`, match the average-loss penalty using
`sklearn_alpha = 1 / (2*C)`. Equal parameter names do not imply equal objectives.

Check prediction means and score-equation residuals as well as coefficients.
A comparison on one dataset or CPU backend does not establish GPU precision or
a universal accuracy guarantee. See the [GLM comparison guidance](generalized-linear-model.md#comparing-with-another-implementation).

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- scikit-learn PoissonRegressor documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
