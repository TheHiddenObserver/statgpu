# SCAD regression

> Language: English
> Last updated: 2026-09-04
> Switch: [简体中文](../../cn/models/scad.md)

## What problem does it solve?

`SCADRegression` fits a sparse linear model using the smoothly clipped absolute deviation (SCAD) penalty. Like Lasso, it can set unimportant coefficients to zero. Unlike Lasso, it reduces the penalty applied to sufficiently large coefficients, which can reduce shrinkage bias.

## When to use it

SCAD is useful when:

- the response is continuous and the predictor set may contain many irrelevant variables;
- you want a sparse, interpretable model;
- Lasso selects useful variables but shrinks their large coefficients too strongly;
- you can tune the penalty using held-out data or cross-validation.

Prefer Lasso or Elastic Net when you need a convex objective and a more stable global optimization path. Prefer Ridge when most variables may have small effects or predictors are strongly correlated. Do not use automatic selection as a substitute for a causal identification strategy.

## Intuition

SCAD changes how hard a coefficient is pulled toward zero:

1. small coefficients receive an L1-like penalty and can become exactly zero;
2. medium coefficients receive progressively less shrinkage;
3. sufficiently large coefficients receive no additional shrinkage.

This shape aims to combine sparsity with lower bias for strong signals. The tradeoff is non-convexity: different initial values or tuning choices can lead to different local solutions.

## Model and penalty

SCAD minimizes

$$
\frac{1}{2n}\lVert y-\beta_0\mathbf 1-X\beta\rVert_2^2
+
\sum_{j=1}^{p}p_{\lambda,a}(|\beta_j|),
$$

where $\lambda$ is exposed as `alpha` and

$$
p_{\lambda,a}(\theta)=
\begin{cases}
\lambda\theta,
&0\leq\theta\leq\lambda,\\
\dfrac{2a\lambda\theta-\theta^2-\lambda^2}{2(a-1)},
&\lambda<\theta\leq a\lambda,\\
\dfrac{(a+1)\lambda^2}{2},
&\theta>a\lambda.
\end{cases}
$$

The usual value is $a=3.7$. Under regularity conditions and suitable tuning, SCAD can have selection-consistency and oracle-style asymptotic properties. These are theoretical conditions, not a guarantee for every finite dataset.

## Minimal runnable example

This example contains four true signals among twenty standardized predictors.

```python
import numpy as np
from statgpu.linear_model import SCADRegression

rng = np.random.default_rng(2)
X = rng.normal(size=(600, 20))
X = (X - X.mean(axis=0)) / X.std(axis=0)

true_coef = np.zeros(20)
true_coef[:4] = [2.5, -1.8, 1.2, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.8, size=600)

model = SCADRegression(
    alpha=0.08,
    a=3.7,
    device="cpu",
    compute_inference=False,
).fit(X, y)

selected = np.flatnonzero(np.abs(model.coef_) > 1e-8)
print("selected feature indices:", selected)
print("coefficients:", model.coef_)
print("training R²:", model.score(X, y))
```

The first four features should usually be retained, but the exact selected set depends on the noise and `alpha`. Selection on the training set is not evidence of out-of-sample usefulness.

## How to read the result

- A coefficient equal or very close to zero is not selected.
- A nonzero coefficient is interpreted like a linear-regression coefficient, conditional on the selected model and the feature scaling.
- `intercept_` is not penalized.
- `score(X, y)` returns $R^2$.
- The selected set can change across resamples. Examine stability rather than reporting a single fit as certain.

## Key parameters and how to choose them

| Parameter | Default | Guidance |
|---|---:|---|
| `alpha` | `1.0` | Larger values create more sparsity; choose from a log-spaced grid using cross-validation or a validation set |
| `a` | `3.7` | Keep the literature default unless you have a documented sensitivity analysis |
| `fit_intercept` | `True` | Usually keep enabled |
| `max_iter` | `1000` | Increase if the solver has not converged |
| `tol` | `1e-4` | Tighten when coefficient accuracy matters and accept the extra work |
| `device` | `"auto"` | `cpu`, `cuda`, or `torch` according to scale and backend availability |
| `solver` | `"auto"` | Let statgpu choose the compatible optimization path unless reproducing a specific setup |
| `compute_inference` | `False` | Keep false for the typed beginner path; see the inference section below |

Standardize features before tuning `alpha`. Otherwise, the same penalty strength treats measurement units rather than scientific importance equally.

## Solver support

SCAD has a model-specific path: `solver="auto"` (recommended) resolves to FISTA dispatch, then the estimator runs a continuation path with local linear approximation (FISTA-LLA). `solver="fista"` selects the same public route explicitly.

| Choice | Support | Actual computation |
|---|:---:|---|
| `auto` (default) | yes | FISTA-LLA continuation |
| `fista` | yes | FISTA-LLA continuation |
| `newton`, `lbfgs`, `irls`, `exact` | no | Rejected as incompatible with a non-convex, non-smooth penalty |
| `fista_bb`, `admm`, `coordinate_descent` | no distinct SCAD path | Current wrapper still routes SCAD through FISTA-LLA; do not use these names for solver comparisons |

`fista_lla_path` is an internal subpath, not a value to pass as `solver=`. The LLA weights and continuation logic belong on this model page; the inner FISTA update is described in the [general solver guide](../guides/solver-algorithms.md#1-fista).

## Compare with alternatives

| Property | Lasso | Elastic Net | SCAD |
|---|---|---|---|
| Objective | convex | convex | non-convex |
| Exact zeros | yes | yes | yes |
| Shrinkage of large signals | persistent | persistent | reduced after the SCAD threshold |
| Correlated predictors | may pick one unstably | often more stable | may still be unstable |
| Global optimum guarantee | yes for the convex objective | yes for the convex objective | no |

## CPU, GPU, and inference support

SCAD estimation supports CPU, CuPy CUDA, and Torch CUDA on their compatible paths:

```python
gpu_model = SCADRegression(
    alpha=0.08,
    device="cuda",
    compute_inference=False,
).fit(X, y)
```

The typed `SCADRegression` constructor exposes `compute_inference` but not an `inference_method` selector. Therefore `SCADRegression(compute_inference=True)` alone is **not** the documented inference route: the inherited default is incompatible with SCAD.

For device-resident LLA/FISTA behavior, synchronization, precision, and the CPU active-set refit boundary, see [CPU/GPU acceleration internals](../guides/acceleration-internals.md).

For reviewed active-set inference, use the general penalized estimator explicitly:

```python
from statgpu.linear_model import PenalizedLinearRegression

inferential_model = PenalizedLinearRegression(
    penalty="scad",
    penalty_kwargs={"a": 3.7},
    alpha=0.08,
    device="cpu",
    compute_inference=True,
    inference_method="oracle",
).fit(X, y)

print(inferential_model._bse)
print(inferential_model._pvalues)
```

This procedure refits an unpenalized model on the active set. It is conditional on successful selection and relies on oracle-property assumptions; it is not generic finite-sample selective inference. The active-set refit runs on CPU even if estimation began on a GPU.

## Common pitfalls

- Tune `alpha` without using the final test data.
- Check selection stability across folds or bootstrap samples.
- Do not state that SCAD always has the oracle property; the result requires assumptions and an appropriate tuning sequence.
- Because the objective is non-convex, inspect convergence and sensitivity to tuning.
- A sparse predictive model does not prove that omitted coefficients are scientifically zero.

## API and validation

Main path: `statgpu.linear_model.SCADRegression`

Advanced path: `statgpu.linear_model.PenalizedLinearRegression(penalty="scad")`

Main estimation outputs are `coef_`, `intercept_`, `fit`, `predict`, and `score`. The general oracle-inference path additionally populates `_bse`, `_zvalues`, `_pvalues`, and `_conf_int` for the active-set refit.

Validation covers non-convex solver behavior, CPU/GPU parity, convergence contracts, and inference dispatch. Benchmark results should be read from the versioned [dashboard](/dashboard/) rather than copied as timeless speed claims.

## References

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
