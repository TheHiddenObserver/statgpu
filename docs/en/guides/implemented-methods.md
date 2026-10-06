# Implemented Methods

> Last updated: 2026-10-06  
> Switch: [Chinese](../../cn/guides/implemented-methods.md)

This page is the public inventory of models, functions, and major solver families available in statgpu. Detailed mathematics, inference scope, and compatibility rules live on the linked model and guide pages.

## Regression and Generalized Linear Models

| Class | Description | Backends |
|---|---|---|
| `LinearRegression` | OLS with classical, HC0–HC3, and HAC inference | NumPy, CuPy, Torch |
| `Ridge` | L2-penalized linear regression | NumPy, CuPy, Torch |
| `Lasso` | L1 regression with debiased/bootstrap inference paths | NumPy, CuPy, Torch |
| `ElasticNet` | L1+L2 penalized regression | NumPy, CuPy, Torch |
| `LogisticRegression` | Binary logistic/probit regression | NumPy, CuPy, Torch |
| `PoissonRegression` | Poisson GLM | NumPy, CuPy, Torch |
| `GammaRegression` | Gamma GLM | NumPy, CuPy, Torch |
| `InverseGaussianRegression` | Inverse Gaussian GLM | NumPy, CuPy, Torch |
| `NegativeBinomialRegression` | Negative-binomial GLM | NumPy, CuPy, Torch |
| `TweedieRegression` | Tweedie GLM | NumPy, CuPy, Torch |
| `QuantileRegression` | Quantile regression with kernel/bootstrap inference | NumPy, CuPy, Torch |
| `OrderedLogitRegression` | Ordered logit with analytical-Hessian inference | NumPy, CuPy, Torch |
| `OrderedProbitRegression` | Ordered probit with analytical-Hessian inference | NumPy, CuPy, Torch |

## Penalized Models

The penalty registry includes L1, L2, Elastic Net, SCAD, MCP, adaptive L1, group Lasso, adaptive group Lasso, group MCP, and group SCAD implementations. Aliases are accepted for selected penalties; use the registry/compatibility references rather than assuming every penalty supports every solver.

| Class | Loss or model family | Backends |
|---|---|---|
| `PenalizedGeneralizedLinearModel` | Unified penalized GLM interface | NumPy, CuPy, Torch |
| `PenalizedLinearRegression` | Penalized Gaussian regression | NumPy, CuPy, Torch |
| `PenalizedLogisticRegression` | Penalized binary regression | NumPy, CuPy, Torch |
| `PenalizedPoissonRegression` | Penalized Poisson regression | NumPy, CuPy, Torch |
| `PenalizedQuantileRegression` | Quantile loss with supported proximal/FISTA/IRLS routes | NumPy, CuPy, Torch |
| `PenalizedRobustRegression` | Huber, bisquare, and fair losses where supported | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | Penalized Cox partial likelihood | NumPy, CuPy, Torch |

Solver availability depends on the selected loss and penalty. Consult the [Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md) and [Solver × Penalty Matrix](solver-penalty-matrix.md) before choosing an explicit solver.

### Example

<!-- api-example: inventory-poisson -->
```python
import numpy as np
from statgpu.linear_model import PenalizedGeneralizedLinearModel

rng = np.random.default_rng(42)
X = rng.normal(size=(80, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.5, -0.2])))
model = PenalizedGeneralizedLinearModel(
    loss="poisson", penalty="l1", alpha=0.05, solver="fista",
    device="cpu", compute_inference=False, max_iter=2000, tol=1e-8,
).fit(X, y)
print(np.round(model.predict(X[:3]), 6))
```

The fitted values are expected event counts, about `[1.458758, 1.547328, 0.775672]`. This L1-Poisson example is estimation-only; coefficient inference for non-Gaussian L1/ElasticNet fits is not implemented. Choose the penalty by a separate validation procedure for real data.

## Cross-Validation

| Class | Description | Backends |
|---|---|---|
| `RidgeCV` | Ridge alpha selection | NumPy, CuPy, Torch |
| `LassoCV` | Lasso alpha-path selection and full-data refit | NumPy, CuPy, Torch |
| `ElasticNetCV` | Joint `l1_ratio` and alpha search | NumPy, CuPy, Torch |
| `LogisticRegressionCV` | Logistic-regression CV | NumPy, CuPy, Torch |
| `PenalizedGLM_CV` | Unified penalized-GLM CV | NumPy, CuPy, Torch |
| `CoxPHCV` | Cox penalty search and final refit | NumPy, CuPy, Torch |

See [Cross-Validation](cross-validation.md) for fold, selection, refit, weight, and inference-after-selection semantics.

## ANOVA

- `f_oneway`
- `f_twoway`
- `f_welch`
- `tukey_hsd`
- `bonferroni`
- `cohens_f`
- `partial_eta_squared`

See [ANOVA](../models/anova.md) for design restrictions and scalar distribution boundaries.

## Covariance Estimation

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

See [Covariance Estimation](../models/covariance.md).

## Panel Data

- `PanelOLS`
- `RandomEffects`
- `PooledOLS`
- `BetweenOLS`
- `FirstDifferenceOLS`
- `FamaMacBeth`

See [Panel Data Models](../models/panel.md) for model choice, covariance, rank-deficiency, and prediction behavior.

## Nonparametric and Semiparametric Methods

- `KernelDensityEstimator` / `KDE`
- `KernelRegression` / `KernelRegressionRegressor`
- `KernelRidge` and `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `SplineTransformer`
- B-spline, natural cubic, cyclic cubic, and thin-plate spline bases
- `GAM`

## Unsupervised Learning

- `PCA`, `TruncatedSVD`, `IncrementalPCA`
- `NMF`, `MiniBatchNMF`
- `KMeans`, `MiniBatchKMeans`, `DBSCAN`
- `GaussianMixture`, `AgglomerativeClustering`
- `UMAP`, `TSNE`

UMAP can use an internal approximate NNDescent search through `nn_method="nndescent"`. NNDescent is not a separate public estimator export; see the [UMAP support limits](../unsupervised/umap.md).

## Survival Analysis

| Class | Description | Backends |
|---|---|---|
| `CoxPH` | Breslow/Efron/Exact ties, delayed entry, `(start, stop]` rows, strata, robust/cluster inference, backend-aware prediction | NumPy, CuPy, Torch |
| `CoxPHCV` | L2 grid selection with the same risk-set semantics and subject-preserving folds | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | Standard right-censored Cox partial likelihood with convex/non-convex penalties where supported | NumPy, CuPy, Torch |

The base installation contains Cox fitting. The optional `statgpu[survival]` extra installs statsmodels for external comparison; it is not required for statgpu's Cox estimator itself. See [Cox Proportional Hazards](../models/coxph.md) for the precise support matrix.

## Feature Selection and Diagnostics

- `StepwiseSelector` and `stepwise_selection`
- fixed-X and model-X knockoff filters and selector wrappers
- `RegressionDiagnostics` and `diagnose_model`

## Multiple Testing and Resampling

- `adjust_pvalues`
- `combine_pvalues`
- `permutation_test`
- bootstrap utilities exposed by the inference API

For detailed semantics, continue to the relevant model/reference page rather than inferring solver, inference, or device support from this inventory alone.
