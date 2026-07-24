# Implemented Methods

> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/guides/implemented-methods.md)

This page is the maintained inventory of public models, functions, and major solver
families in statgpu. Detailed mathematical and backend contracts live on the linked
model and guide pages.

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

The penalty registry includes L1, L2, Elastic Net, SCAD, MCP, adaptive L1,
group Lasso, adaptive group Lasso, group MCP, and group SCAD implementations.
Aliases are accepted for selected penalties; the registry and compatibility matrix are
the source of truth rather than a hard-coded count.

| Class | Loss or model family | Backends |
|---|---|---|
| `PenalizedGeneralizedLinearModel` | Unified penalized GLM interface | NumPy, CuPy, Torch |
| `PenalizedLinearRegression` | Penalized Gaussian regression | NumPy, CuPy, Torch |
| `PenalizedLogisticRegression` | Penalized binary regression | NumPy, CuPy, Torch |
| `PenalizedPoissonRegression` | Penalized Poisson regression | NumPy, CuPy, Torch |
| `PenalizedQuantileRegression` | Quantile loss with proximal/FISTA paths | NumPy, CuPy, Torch |
| `PenalizedRobustRegression` | Huber, bisquare, and fair losses where supported | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | Penalized Cox partial likelihood | NumPy, CuPy, Torch |

Solver availability depends on the selected loss and penalty. Consult the
[Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md) and
[Solver × Penalty Matrix](solver-penalty-matrix.md) before choosing an explicit
solver.

### Example

```python
from statgpu.linear_model import PenalizedGeneralizedLinearModel

# L1 is non-smooth, so use FISTA or solver="auto".
model = PenalizedGeneralizedLinearModel(
    loss="poisson",
    penalty="l1",
    alpha=0.05,
    solver="fista",
)
model.fit(X, y)
```

## Cross-Validation

| Class | Description | Backends |
|---|---|---|
| `RidgeCV` | Ridge alpha selection | NumPy, CuPy, Torch |
| `LassoCV` | Warm-start Lasso path | NumPy, CuPy, Torch |
| `ElasticNetCV` | Joint `l1_ratio` and alpha search | NumPy, CuPy, Torch |
| `LogisticRegressionCV` | Logistic-regression CV | NumPy, CuPy, Torch |
| `PenalizedGLM_CV` | Unified penalized-GLM CV | NumPy, CuPy, Torch |
| `CoxPHCV` | Cox penalty search and final refit | NumPy, CuPy; see CoxPH docs |

## ANOVA

- `f_oneway`
- `f_twoway`
- `f_welch`
- `tukey_hsd`
- `bonferroni`
- `cohens_f`
- `partial_eta_squared`

See [ANOVA](../models/anova.md) for design restrictions and scalar distribution
boundaries.

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

See [Panel Data Models](../models/panel.md) for covariance, rank-deficiency, and
backend-preserving prediction contracts.

## Nonparametric and Semiparametric Methods

- `KernelDensity` and kernel regression
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
- `UMAP`, `TSNE`, `NNDescent`

## Survival Analysis

| Class | Description | Backends |
|---|---|---|
| `CoxPH` | Breslow/Efron ties, delayed entry, robust/cluster inference contracts, backend-native prediction | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | Cox partial likelihood with convex/non-convex penalties where supported | NumPy, CuPy, Torch |

Optional CPU dependencies for exact Efron robust inference and delayed-entry reference
paths are installed with `pip install statgpu[survival]`. See
[Cox Proportional Hazards](../models/coxph.md) for the precise support matrix.

## Feature Selection and Diagnostics

- `StepwiseSelector` and `stepwise_selection`
- fixed-X and model-X knockoff filters and selector wrappers
- `RegressionDiagnostics` and `diagnose_model`

## Multiple Testing and Resampling

- `adjust_pvalues`
- `combine_pvalues`
- `permutation_test`
- bootstrap utilities exposed by the inference API

## Validation Scope

Backend support in this inventory means the public execution path exists. Numerical,
performance, and physical-GPU claims remain scoped to the exact model, backend,
hardware, and commit recorded by the corresponding tests or validation artifact.
