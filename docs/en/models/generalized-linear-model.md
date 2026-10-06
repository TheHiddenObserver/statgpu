# GeneralizedLinearModel and Penalized GLM

> Language: English  
> Last updated: 2026-10-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/generalized-linear-model.md)

## Choosing a response model

A GLM connects a linear predictor to the mean of an outcome through a link function. Use Gaussian/identity for continuous outcomes with a linear mean, binomial/logit for 0/1 outcomes, or Poisson/log for counts. The log link keeps a predicted mean positive; a coefficient then describes a multiplicative change in that mean after exponentiation. It is not a causal effect without additional assumptions.

Poisson assumes conditional variance equals the mean. Strong count overdispersion can motivate a Negative Binomial model; strictly positive continuous outcomes can motivate Gamma or Inverse Gaussian. Choose the family from the response and scientific assumptions, not simply the lowest training error. These APIs do not expose an offset/exposure argument; adding exposure as an ordinary predictor estimates its coefficient rather than fixing it at one.

## A complete CPU example

This simulation fits an unpenalized log-link Poisson model on 180 rows with analytic weights. The final 60 rows are held out. `C=0` makes the unpenalized intent explicit; Newton does not use C. No GPU or optional formula package is needed.

<!-- learner-example: glm-poisson -->
```python
import numpy as np
from statgpu import GeneralizedLinearModel

rng = np.random.default_rng(59)
X = rng.normal(size=(240, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
weights = np.linspace(0.5, 2.0, 180)
model = GeneralizedLinearModel(
    family="poisson", C=0, solver="newton", device="cpu",
    max_iter=1000, tol=1e-8, compute_inference=True, cov_type="hc1",
).fit(X[:180], y[:180], sample_weight=weights)
mean_prediction = model.predict(X[180:])
heldout_loss = np.mean(mean_prediction - y[180:] * np.log(mean_prediction))
print("Slopes:", np.round(model.coef_, 3))
print("Mean multipliers:", np.round(np.exp(model.coef_), 3))
print("Predicted means:", np.round(mean_prediction[:3], 3))
print("Held-out Poisson loss:", round(float(heldout_loss), 3))
print("Interval shape:", model._conf_int.shape)
```

Rounded CPU output for this seed:

```text
Slopes: [ 0.402 -0.327]
Mean multipliers: [1.495 0.721]
Predicted means: [1.139 1.24  2.107]
Held-out Poisson loss: 0.556
Interval shape: (3, 2)
```

The coefficient multipliers describe changes in the conditional mean, holding the other feature fixed. The first slope corresponds to about a 49.5% higher conditional mean per unit increase in the first feature; the second corresponds to about a 27.9% decrease.  Predicted means may be fractional even when observations are integer counts. The held-out loss omits the response-only log-factorial constant; compare it only on the same held-out responses and weighting. Smaller is better. It is not a probability, accuracy or ordinary R².

Inference arrays have three entries here: intercept first, then the two slopes; `_conf_int` has shape `(3,2)`. HC1 changes covariance, not the fitted mean. These are marginal coefficient intervals, not intervals for future counts. This generic ordinary class returns mean probabilities for binomial but has no `predict_proba` or `score`; choose evaluation appropriate to the family. `print(model.summary())` displays its returned summary string.

## Ordinary and penalized entry points

`GeneralizedLinearModel` is the common entry point for core ordinary GLMs such as Gaussian, binomial, and Poisson models. Typed ordinary estimators such as `GammaRegression`, `InverseGaussianRegression`, `NegativeBinomialRegression`, and `TweedieRegression` use the same shared GLM implementation for their family-specific behavior.

For regularized models, use `PenalizedGeneralizedLinearModel` or a typed wrapper such as `PenalizedLinearRegression`, `PenalizedLogisticRegression`, or `PenalizedPoissonRegression`. `Ridge`, `Lasso`, and `ElasticNet` are sklearn-style thin wrappers over penalized Gaussian regression.

If you are here for weighted GLMs, the main rules are:

- `sample_weight` is an **analytic objective weight** on supported GLM paths;
- the weighted loss is normalized by `sum(sample_weight)`, so multiplying all weights by one positive constant does not change the fitted optimum;
- explicit `solver="newton"` and `solver="lbfgs"` remain explicit—weights do not silently switch the solver;
- explicit `device="cuda"` and `device="torch"` remain on the requested GPU backend when that route is supported;
- unsupported combinations fail clearly instead of silently falling back to a different solver or CPU.

For penalized coefficient inference, see [Penalized GLM inference](../guides/penalized-glm-inference.md). For the complete solver table, see [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md).

Complete [typed GLM constructors and methods](../reference/linear-model-api.md#typed-glm-constructors)
cover family-specific link/dispersion/power controls and the different penalized
wrapper defaults. Shared names do not make constructor controls interchangeable.

## Public paths

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.GammaRegression`
- `statgpu.linear_model.InverseGaussianRegression`
- `statgpu.linear_model.NegativeBinomialRegression`
- `statgpu.linear_model.TweedieRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`

The internal GLM objective layer is `statgpu.glm_core`.

## Objective function and `sample_weight`

Let n be the number of observations, x_i the predictor vector, beta the slope vector and b the unpenalized intercept. Write $\eta_i=b+x_i^\top\beta$; set b=0 when `fit_intercept=False`. The **unpenalized** data-fit objective uses the family's average negative log-likelihood (with the family's dispersion convention):

$$
L(b,\beta)=\frac{1}{n}\sum_{i=1}^n\ell(y_i,\eta_i).
$$

On a supported weighted path, analytic `sample_weight=w` replaces it by

$$
L_w(b,\beta)=\frac{\sum_i w_i\ell(y_i,\eta_i)}{\sum_i w_i}.
$$

Weights must be finite and nonnegative with positive sum. Zero-weight rows contribute no data-fit loss, and multiplying all weights by one positive constant does not change this objective.

**Ordinary GLM is not automatically unpenalized.** Its default `solver="auto"` uses IRLS for the ordinary families listed here. Ordinary IRLS with positive `C` adds a slope ridge term:

$$
\min_{b,\beta} L_w(b,\beta)+\frac{1}{4C}\lVert\beta\rVert_2^2.
$$

Use L instead of L_w without weights. The corresponding ridge gradient is $\beta/(2C)$; the intercept is excluded. The default `C=1` therefore shrinks slopes. In this ordinary IRLS implementation, nonpositive C disables that term; use `C=0` for an explicit unpenalized IRLS fit. Explicit ordinary `newton`, `lbfgs` and `fista` paths minimize the unpenalized loss and do not use C. Changing the ordinary solver can therefore change the statistical problem if positive C was active. This C convention is distinct from standalone LogisticRegression and from the alpha interface below.

The **penalized GLM** classes instead add their declared penalty:

$$
\min_{b,\beta}L(b,\beta)+\alpha P(\beta),\qquad
\text{or}\quad\min_{b,\beta}L_w(b,\beta)+\alpha P(\beta).
$$

The intercept remains unpenalized. `statgpu.glm_core` is GLM-specific; Cox partial likelihood, robust losses and quantile losses retain their own statistical definitions and documentation.

## Solver selection

Smooth GLMs use supported first- or second-order solvers. Non-smooth penalized objectives use proximal methods such as FISTA and FISTA-BB. In current direct sparse Gaussian fits, `stopping="kkt"` does not switch the stopping check; see the [direct-fit control limits](../reference/linear-model-api.md#elasticnet).

The following table describes the unified **penalized-estimator** automatic dispatch. Ordinary GLM defaults to IRLS as described above; it does not inherit this table merely because C adds a ridge term.

| Setting | `solver="auto"` behavior |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` on NumPy/CPU | exact closed-form L2 path |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton for these supported penalized fits |
| non-convex SCAD/MCP | FISTA + LLA continuation |

`solver="auto"` is a request for the model's automatic dispatch policy. An explicit solver name is different: statgpu either runs that supported solver or raises an error; it does not silently substitute another solver.

The same principle applies to devices. Explicit `device="cuda"` uses CuPy CUDA and explicit `device="torch"` uses Torch CUDA on supported routes. Formula parsing may run on CPU, but fit/predict numerical work follows the selected numerical backend.

### Explicit Newton and L-BFGS with analytic weights

For supported ordinary GLM combinations, explicit `solver="newton"` and `solver="lbfgs"` accept non-uniform analytic weights when that family/link combination supports the solver.

Both solvers optimize the normalized weighted objective shown above. The same weight vector is used consistently throughout the optimization:

- Newton uses it in objective, gradient, Hessian, and Armijo line-search evaluations;
- L-BFGS uses it in the initial gradient, current objective, every line-search candidate, and the accepted-point gradient.

This consistency is important: a weighted search direction is never paired with an unweighted line-search objective.

Adding `sample_weight` does **not** replace an explicit Newton/L-BFGS request with IRLS or FISTA. Likewise, an explicit CUDA/Torch request does not fall back to CPU. Uniform and historically effectively-uniform weights retain the historical unweighted objective.

#### Initializing `inverse_power` Gamma

`GammaRegression(link="inverse_power")` uses the inverse link, so fitting requires

$$
\eta_i=b+x_i^\top\beta>0.
$$

Before the first objective evaluation, explicit Newton/L-BFGS constructs an interior starting point that satisfies this condition. With an intercept, the intercept column provides an immediate feasible direction. Without an intercept, statgpu searches the positive-weight training rows for a direction $d$ with $Xd>0$ and scales that direction to a valid interior point. If such a start cannot be numerically certified, fitting fails before optimization begins. Subsequent updates are kept inside the valid inverse-link domain by solver-owned feasibility checks and step caps; these are numerical safeguards rather than additional statistical assumptions on the Gamma model.

#### Scope of weighted L-BFGS support

Weighted L-BFGS is a **GLM loss capability**, not a blanket promise for non-GLM `LossBase` implementations. Other model families retain their own weight and solver semantics; consult their model pages and the compatibility matrix. Ordered GLMs also retain their separate weight policy.

Weighted penalized smooth GLMs use the same analytic-weight convention. Their existing direct-fit/CV dispatch remains authoritative: a supported L2 row may use Newton or L-BFGS according to that policy, not merely because weights are present.

## Covariance and inference

Generic and typed penalized GLM estimators use `inference_method="auto"` as the recommended public request. Supported M-estimation, debiased and residual-bootstrap results report the procedure, target and tuning/selection conditions. The current `post_selection_ols` path can leave `inference_method_` and `inference_target_` unset despite returning inference; read `_inference_result.method` and its metadata as described in [post-selection reporting](../guides/inference-modes.md#post_selection_ols).

For supported smooth non-Gaussian L2/no-penalty models, `auto` resolves to fixed-penalty `m_estimation`. Positive L2 fits target the penalized estimating equation; no-penalty aliases are canonicalized to zero-strength L2 and target the unpenalized population parameter. Current covariance support is `nonrobust`, `hc0`, and `hc1`; HC2/HC3/HAC are not available on this penalized non-Gaussian path and raise instead of being substituted silently.

Supported M-estimation and Gaussian residual-bootstrap calculations reuse the fitted backend and device. Ordinary GLM inference uses the same fitted analytic-weight convention as estimation; weight support still depends on the chosen inference method. Non-Gaussian L1/ElasticNet coefficient inference is unsupported. SCAD/MCP `oracle` is explicit, but its current non-Gaussian child reconstruction can silently reset family settings and regularization and use an automatic child device. Do not interpret those results as the intended unpenalized active-set inference; see the [oracle limitation and diagnostic-refit alternative](../guides/penalized-glm-inference.md#current-non-gaussian-oracle-limitation). Group-penalty paths remain estimation-only where noted.

For supported Gaussian sparse penalties, `inference_method="bootstrap"` selects an unweighted residual bootstrap with `cov_type="nonrobust"`. The design matrix and fitted tuning configuration remain fixed: each draw resamples residuals, constructs a bootstrap response by adding those residuals to the fitted values, and refits the same penalized model. `n_bootstrap` controls the number of refits and `bootstrap_random_state` controls reproducibility.

Bootstrap execution follows the successful fit's backend and concrete device. CPU fits use NumPy; CuPy and Torch CUDA fits keep bootstrap refits on the same GPU device. Final inference arrays use the standard NumPy reporting boundary. Weighted residual bootstrap, robust/HC or HAC/block bootstrap, and non-Gaussian bootstrap are not supported.

See [Penalized GLM inference](../guides/penalized-glm-inference.md) and [Inference Modes](../guides/inference-modes.md) for the complete support matrix, resampling boundaries, and statistical interpretation.

## Parameters

This is a selection of controls across different classes, not one shared constructor. `family` belongs to ordinary GLM; the generic penalized class uses `loss`. `formula` and `data` are fit arguments. The complete live constructors and methods are in the [ordinary GLM](../reference/linear-model-api.md#generalizedlinearmodel), [generic penalized GLM](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel), and [generic CV](../reference/linear-model-api.md#penalizedglm_cv) references.

| Parameter | Default | Description |
|---|---:|---|
| `family` | model-specific | GLM family, for example `"gaussian"`, `"binomial"`, or `"poisson"` |
| `penalty` | `"l1"` on generic direct GLM; `"l2"` on generic CV | `none`, `l1`, `l2`, `elasticnet`, and reserved structured penalties |
| `alpha` | `1.0` or model-specific | Penalty strength in statgpu objective scale |
| `C` | `1.0` on ordinary GLMs | Ordinary IRLS slope penalty is `sum(beta**2)/(4*C)` for positive C; C=0 disables it. Ordinary explicit Newton/L-BFGS/FISTA ignore C. Not the penalized-GLM alpha control. |
| `l1_ratio` | `0.5` where accepted | ElasticNet mixing parameter on generic/typed penalized GLMs; specialized wrappers can omit this parameter. |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `solver` | `"auto"` | Solver dispatch; see the solver-selection section above |
| `device` | `"auto"` | `cpu`, `cuda`, `torch`, or `auto` depending on estimator support |
| `max_iter` | model-specific | Maximum optimizer iterations |
| `tol` | model-specific | Convergence tolerance |
| `compute_inference` | `False` on generic/typed penalized GLM | Whether to compute coefficient inference |
| `inference_method` | `"auto"` on generic/typed penalized GLM | Public inference request; specialized sparse-Gaussian wrappers retain their established defaults |
| `cov_type` | `"nonrobust"` | Covariance convention; non-Gaussian penalized M-estimation currently supports nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | Retained control; does not imply HAC support for non-Gaussian penalized M-estimation |
| `formula` | `None` | Optional patsy-style formula used with `data` |
| `data` | `None` | DataFrame used with `formula` |

Alpha scaling is explicit. Do not compare same-named parameters across frameworks without conversion:

- Ridge: `sklearn_alpha = n_samples * statgpu_alpha`
- Logistic L2: `sklearn_C = 1 / (n_samples * statgpu_alpha)`
- Poisson L2: align against sklearn `PoissonRegressor(alpha=...)`
- Poisson L1/ElasticNet: align against statsmodels `fit_regularized`

## Optional GPU and formula inputs

After the CPU data-generation example, the same ordinary weighted Poisson fit can request CuPy CUDA as follows. Use device="torch" for Torch CUDA; an unavailable explicit backend raises.

```python
model_gpu = GeneralizedLinearModel(
    family="poisson", C=0, solver="newton", device="cuda",
    max_iter=1000, tol=1e-8, compute_inference=False,
).fit(X[:180], y[:180], sample_weight=weights)
```

Formula input requires `pip install statgpu[formula]`. This example is independently runnable once those optional dependencies are installed:

<!-- learner-example: glm-formula -->
```python
import numpy as np
import pandas as pd
from statgpu import GeneralizedLinearModel

rng = np.random.default_rng(18)
df = pd.DataFrame({"x": rng.normal(size=80), "group": ["a", "b"] * 40})
df["count"] = rng.poisson(np.exp(0.2 + 0.3 * df["x"]))
model = GeneralizedLinearModel(family="poisson", C=0, device="cpu")
model.fit(formula="count ~ x + C(group)", data=df)
prediction = model.predict(df.iloc[:5])
assert prediction.shape == (5,)
```

Formula parsing runs on CPU. Pass formula/data without simultaneous array X/y. Weights may cover original rows or exactly the rows retained after formula/missing-data processing; alignment is positional. Prediction reconstructs the training columns and categorical levels. For large data, explicit arrays avoid formula parsing overhead.

Ordinary formula prediction currently drops rows with missing predictors and
returns a shorter unlabelled array. Resolve missing values and check output
length before aligning predictions; see the [missing-row limitation](../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms).

### Failed-refit limitation

On ordinary auto/IRLS/FISTA paths, a failed refit can mix old coefficients with new row counts or formula/intercept settings while the object still appears fitted. Predictions and likelihood diagnostics may then change. Create a fresh estimator after any such failure and require a successful fit before using outputs; see the [full failed-refit contract](../reference/linear-model-api.md#failed-ordinary-glm-refits). Explicit Newton/L-BFGS currently preserve the previous fit after failure, which is still not a fit to the new data.

## Strict and approximate CV

Supported non-Gaussian L2/no-penalty combinations expose fixed-penalty M-estimation with nonrobust/HC0/HC1 covariance. Unsupported loss × penalty × method combinations raise. Residual bootstrap has the narrower scope described above; an accepted non-Gaussian SCAD/MCP oracle request still has the reconstruction limitation, so acceptance alone does not establish the intended target.

`solver="auto"` follows the model's direct-fit dispatch. Analytic weights do not rewrite a public solver request: explicit smooth solvers remain explicit, while `auto` continues to use the dispatch for the corresponding fit.

`PenalizedGLM_CV` defaults to `cv_strategy="strict"`. In strict mode every fold/alpha is evaluated with the requested `max_iter` and `tol`, and GPU optimizations are limited to caching, fused kernels, and batched validation-score transfers.

The optional `cv_strategy="two_stage"` mode first screens the alpha grid with relaxed CV solves, then strictly refines candidate alphas and performs a strict final refit. Because screening can change the ranking when CV curves are close, two-stage mode emits `ApproximateCVWarning` unless `acknowledge_approx=True` is passed.

```python
from statgpu.linear_model import PenalizedGLM_CV

# Default: strict CV.
strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

# Opt-in approximate screening; candidate refinement and final refit stay strict.
fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

## Reading CV inference results

The generic `PenalizedGLM_CV` tunes alpha at a fixed l1_ratio and refits on all training rows. It is different from `ElasticNetCV`, which can also search l1_ratio. The generic CV class has no public fit_intercept option for scalar responses and fits an intercept. Its final score is response-scale R², whereas best_score_ is negative validation loss.

The current `summary()` method cannot render successful generic final-refit inference: it delegates to a final estimator without that method and raises AttributeError. Read the result container instead:

<!-- learner-example: glm-cv-inference -->
```python
import numpy as np
from statgpu import PenalizedGLM_CV

rng = np.random.default_rng(9)
X = rng.normal(size=(60, 2))
y = rng.poisson(np.exp(0.2 + 0.3 * X[:, 0]))
model = PenalizedGLM_CV(
    loss="poisson", penalty="l2", alpha_grid=[0.05, 0.2],
    cv=2, device="cpu", compute_inference=True,
).fit(X, y)
report = model.estimator_._inference_result.to_dict()
print("Selected alpha:", model.alpha_)
print("Method:", report["method"])
print("Standard errors:", np.round(model.estimator_._bse, 3))
assert model.cv_results_["all_scores"].shape == (2, 2)
assert np.isclose(model.best_score_, -np.min(model.cv_results_["mean_score"]))
```

This produces method `m_estimation` and three finite standard errors. Inference is conditional on the selected alpha and does not account for tuning uncertainty. The final generic estimator has no predict_proba or summary method. Its logistic prediction is a 0/1 label rather than a mean probability; use a suitable typed estimator when probability methods are needed. Shapes, default values and complete CV result keys are in the [API reference](../reference/linear-model-api.md#penalizedglm_cv).

## See also

- [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) — full dispatch table for loss × penalty × solver combinations, CV paths, and inference support.
- [Solver Algorithms](../guides/solver-algorithms.md) — algorithm-level description of Newton, L-BFGS, FISTA, and other solvers.
- [Loss Functions](losses.md) — low-level loss interfaces and their capability boundaries.

## FAQ

- **What does `sample_weight` mean here?** On supported ordinary and penalized GLM paths it is an analytic objective weight, normalized by `sum(weights)`. It is not a survey-bootstrap weight and does not request weighted residual bootstrap.
- **Does adding `sample_weight` change my solver?** No. Explicit solver requests stay explicit; `solver="auto"` continues to use the dispatch policy.
- **Does `device="cuda"` force every GLM solver onto GPU?** On a supported route, yes: CuPy performs the numerical work. If the route is unsupported or the device is unavailable, statgpu raises instead of silently falling back to CPU.
- **Should I use formulas for very large GPU workloads?** Usually not. Formula parsing is a CPU-side convenience; explicit arrays are preferable for large GPU jobs.
- **Are `Ridge`, `Lasso`, and `ElasticNet` aliases?** No. They are thin wrappers with sklearn-style constructors.

## Comparing with another implementation

Align the response family/link, intercept, input columns, analytic weights, penalty definition, solver and convergence settings before comparing coefficients or uncertainty. For an unpenalized ordinary Poisson comparison, use `C=0` with IRLS or an explicit unpenalized solver; the default ordinary `C=1` IRLS fit is a different target. For penalized fits, align average-versus-summed loss and the penalty scale rather than assuming matching parameter names imply matching objectives.

Check held-out predictions and objective/optimality diagnostics as well as coefficients. Inference comparisons also need the same covariance convention and the same conditioning on tuning or selection. A numerical comparison on one backend does not establish results on another device.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
