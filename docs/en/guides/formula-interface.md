# Formula interface and support matrix

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/guides/formula-interface.md)

Formula input is a convenience layer over pandas and Patsy. It builds a named design matrix on CPU, stores its encoding metadata, and then sends the numerical arrays to the selected statgpu backend.

## Install and first fit

```bash
pip install "statgpu[formula]"
```

```python
import pandas as pd
from statgpu.linear_model import LinearRegression

data = pd.DataFrame({
    "y": [1.2, 2.1, 2.8, 4.2, 5.0],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0],
    "group": ["a", "a", "b", "b", "b"],
})

model = LinearRegression(device="cpu").fit(
    formula="y ~ x + C(group)",
    data=data,
)
prediction = model.predict(data.iloc[:2])
```

Use either `formula + data` or `X + y` in normal code. A Formula intercept is included by default; use `y ~ x - 1` or `y ~ 0 + x` to remove it.

## Useful syntax

| Syntax | Meaning |
|---|---|
| `y ~ x1 + x2` | additive main effects |
| `y ~ C(group)` | categorical coding |
| `y ~ x1:x2` | interaction only |
| `y ~ x1 * x2` | both main effects and their interaction |
| `y ~ np.log(x)` | evaluated transformation available to Patsy |
| `y ~ x - 1` | no intercept |

The fitted design information is reused when `predict` receives a DataFrame, preserving category and interaction columns. New data must contain compatible source columns and category levels.

## Verified support matrix

The matrix reflects current `fit` signatures and inheritance, not a promise that every estimator supports Formula input.

| Area | Formula-capable estimators | Special syntax or note |
|---|---|---|
| ordinary linear | `LinearRegression` | standard Patsy formula |
| regularized linear | `Ridge`, `Lasso`, `ElasticNet`, `AdaptiveLasso`, `SCADRegression`, `MCPRegression`, `PenalizedLinearRegression` | standard Patsy formula |
| ordinary GLM | `GeneralizedLinearModel`, `PoissonRegression`, `GammaRegression`, `InverseGaussianRegression`, `NegativeBinomialRegression`, `TweedieRegression` | standard Patsy formula |
| penalized GLM/loss | `PenalizedGeneralizedLinearModel` and typed linear, logistic, Poisson, Gamma, inverse-Gaussian, negative-binomial, Tweedie, quantile, and robust wrappers | standard Patsy formula |
| panel models | `PooledOLS`, `PanelOLS`, `BetweenOLS`, `RandomEffects`, `FirstDifferenceOLS`, `FamaMacBeth` | supports `y ~ x \| entity + time` and documented panel tokens |
| survival | `CoxPH` and `PenalizedCoxPHModel` | `Surv(time, event) ~ x` or `Surv(start, stop, event) ~ x` |

Important array-only surfaces include standalone `LogisticRegression`, `QuantileRegression`, ordered GLMs, `PenalizedGLM_CV`, most nonparametric/covariance/unsupervised estimators, and preprocessing utilities. Check the actual `fit` signature when an estimator is not listed above.

### Logistic Formula alternative

```python
from statgpu.linear_model import GeneralizedLinearModel

model = GeneralizedLinearModel(
    family="binomial",
    solver="newton",
    device="cpu",
).fit(
    formula="outcome ~ age + C(group)",
    data=data,
)
probability = model.predict(data)
```

Use standalone `LogisticRegression` instead when you need its classification-specific methods and can provide an explicit design matrix.

### Panel Formula example

```python
from statgpu.panel import PanelOLS

model = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=panel_frame,
)
```

The pipe separates regressors from fixed-effect identifiers. Standard formulas, `EntityEffects`, and `TimeEffects` are also supported where documented on the panel pages.

### Survival Formula example

```python
from statgpu.survival import CoxPH

model = CoxPH().fit(
    formula="Surv(time, event) ~ age + C(treatment)",
    data=survival_frame,
)
```

Cox models remove an intercept because the partial likelihood does not identify one.

## Missing data and side arrays

Patsy removes rows containing missing values in referenced terms. statgpu records the retained positional rows and aligns supported side arrays such as `sample_weight` before validation. Panel, Cox, and model-specific side arrays have additional alignment rules; read the relevant model page rather than pre-dropping different rows independently.

Inspect Formula expansion directly when debugging:

```python
from statgpu.core.formula import FormulaParser

parser = FormulaParser("y ~ x + C(group)")
y_array, X_array, design_info = parser.eval(data)
print(parser.column_names)
print(parser.summary())
```

## CPU/GPU boundary

Formula parsing and DataFrame/category handling occur on CPU. The resulting dense arrays are converted to the explicit `cpu`, `cuda`, or `torch` backend for model computation. This is preprocessing, not silent device fallback. For repeated very large GPU fits, building and reusing explicit backend arrays can avoid repeated parsing and host-to-device transfer.

## Common failures

- Missing `data=` with `formula=` raises an error.
- Missing columns, unseen category levels, or incompatible transformations fail during Patsy evaluation or prediction.
- Formula intercept syntax takes priority over the constructor's `fit_intercept` setting.
- Formula support does not imply Formula-aware cross-validation; `PenalizedGLM_CV` remains array-only.
- Do not assume an estimator supports Formula because another estimator in the same module does.

## Implementation and tests

- Parser: `statgpu/core/formula/_parser.py`
- Side-array alignment: `statgpu/core/formula/_alignment.py`
- Panel extension: `statgpu/panel/_formula.py`
- Core tests: `statgpu/core/formula/tests/`
- Integration tests: `dev/tests/test_panel_formula.py` and `dev/tests/test_cox_phase1_completion.py`
