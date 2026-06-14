# Implemented Methods

> Last updated: 2026-06-14

Complete list of all implemented models, functions, and classes in statgpu.

## Regression & GLM

| Class | Description | Link Functions | Backends |
|---|---|---|---|
| `LinearRegression` | OLS with HC0-HC3/HAC inference | identity | CPU, CuPy, Torch |
| `Ridge` | L2 penalty, exact/irls solver | identity | CPU, CuPy, Torch |
| `Lasso` | L1 penalty, debiased inference | identity | CPU, CuPy, Torch |
| `ElasticNet` | L1+L2 penalty | identity | CPU, CuPy, Torch |
| `LogisticRegression` | Binary logistic, L2 penalty | logit, probit | CPU, CuPy, Torch |
| `PoissonRegression` | Poisson GLM | log | CPU, CuPy, Torch |
| `GammaRegression` | Gamma GLM | log, inverse_power | CPU, CuPy, Torch |
| `InverseGaussianRegression` | Inverse Gaussian GLM | log, inverse_squared | CPU, CuPy, Torch |
| `NegativeBinomialRegression` | NB GLM (configurable α) | log | CPU, CuPy, Torch |
| `TweedieRegression` | Tweedie GLM (configurable p) | log | CPU, CuPy, Torch |
| `OrderedLogitRegression` | Ordered logit model | logit | CPU, CuPy, Torch |
| `OrderedProbitRegression` | Ordered probit model | probit | CPU, CuPy, Torch |

## Penalized GLM

All 7 GLM families support penalties through `PenalizedGeneralizedLinearModel` or typed wrappers:

| Class | Loss | Solvers | Penalties | Backends |
|---|---|---|---|---|
| `PenalizedGeneralizedLinearModel` | Any of 7 families | exact, irls, newton, lbfgs, fista, fista_bb | l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad | CPU, CuPy, Torch |
| `PenalizedLinearRegression` | squared_error | exact, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedLogisticRegression` | logistic | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedPoissonRegression` | poisson | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |

For Gamma, InverseGaussian, NegativeBinomial, and Tweedie with penalties, use `PenalizedGeneralizedLinearModel(loss=..., penalty=...)`:

```python
from statgpu.linear_model import PenalizedGeneralizedLinearModel

# Gamma + SCAD
model = PenalizedGeneralizedLinearModel(loss="gamma", penalty="scad", alpha=0.1)
model.fit(X, y)

# NegativeBinomial + ElasticNet
model = PenalizedGeneralizedLinearModel(loss="negative_binomial", penalty="elasticnet",
                                        loss_kwargs={"alpha": 2.0}, alpha=0.1)
model.fit(X, y)

# Tweedie + group_lasso
model = PenalizedGeneralizedLinearModel(loss="tweedie", penalty="group_lasso",
                                        loss_kwargs={"power": 1.5}, alpha=0.1)
model.fit(X, y)
```

## Cross-Validation

| Class | Description | Backends |
|---|---|---|
| `RidgeCV` | GPU-accelerated Ridge CV | CPU, CuPy, Torch |
| `LassoCV` | Warm-start alpha path | CPU, CuPy, Torch |
| `ElasticNetCV` | l1_ratio + alpha grid | CPU, CuPy, Torch |
| `LogisticRegressionCV` | GPU-accelerated logistic CV | CPU, CuPy, Torch |
| `PenalizedGLM_CV` | Unified CV for all 7 losses × 10 penalties | CPU, CuPy, Torch |
| `CoxPHCV` | CV penalty search + refit | CPU, CuPy |

## ANOVA

| Function | Description |
|---|---|
| `f_oneway` | GPU-accelerated one-way ANOVA |

## Covariance Estimation

| Class | Description | Backends |
|---|---|---|
| `EmpiricalCovariance` | Sample covariance with jitter-stabilized inversion | CPU, CuPy, Torch |
| `LedoitWolf` | Ledoit-Wolf shrinkage estimator | CPU, CuPy, Torch |
| `OAS` | Oracle Approximating Shrinkage estimator | CPU, CuPy, Torch |

## Panel Data

| Class | Description | Backends |
|---|---|---|
| `PanelOLS` | Fixed effects with nonrobust/robust/clustered SE | CPU, CuPy, Torch |
| `RandomEffects` | Swamy-Arora feasible GLS random effects | CPU, CuPy, Torch |

## Nonparametric Methods

| Class/Function | Description |
|---|---|
| `KernelRidge` | Kernel ridge regression |
| `KernelRidgeCV` | Cross-validated kernel ridge regression |
| `pairwise_kernels` | 6 kernel functions (RBF, polynomial, linear, Laplacian, sigmoid, cosine) |
| `bspline_basis` | B-spline basis (De Boor algorithm, vectorized on GPU) |
| `natural_cubic_spline_basis` | Natural cubic spline basis |

## Semiparametric Models

| Class | Description | Backends |
|---|---|---|
| `GAM` | Generalized additive model with penalized B-splines + GCV | CPU, CuPy, Torch |

## Unsupervised Learning

**Dimensionality Reduction & Factorization:**

| Class | Description | Backends |
|---|---|---|
| `PCA` | Principal component analysis | CPU, CuPy, Torch |
| `TruncatedSVD` | Dense truncated SVD | CPU, CuPy, Torch |
| `IncrementalPCA` | Incremental PCA for large datasets | CPU, CuPy, Torch |
| `NMF` | Non-negative matrix factorization (multiplicative updates) | CPU, CuPy, Torch |
| `MiniBatchNMF` | Mini-batch NMF for large datasets | CPU, CuPy, Torch |
| `UMAP` | Uniform Manifold Approximation and Projection | CPU, CuPy, Torch |
| `TSNE` | t-distributed Stochastic Neighbor Embedding | CPU, CuPy, Torch |

**Clustering & Mixture Models:**

| Class | Description | Backends |
|---|---|---|
| `KMeans` | Lloyd K-Means clustering (k-means++ init) | CPU, CuPy, Torch |
| `MiniBatchKMeans` | Mini-batch K-Means for large datasets | CPU, CuPy, Torch |
| `DBSCAN` | Density-based spatial clustering | CPU, CuPy, Torch |
| `GaussianMixture` | Gaussian mixture model (log-domain EM) | CPU, CuPy, Torch |
| `AgglomerativeClustering` | Exact agglomerative hierarchical clustering | CPU, CuPy, Torch |

## Survival

| Class | Description | Backends |
|---|---|---|
| `CoxPH` | Cox proportional hazards | CPU, CuPy, Torch |

## Feature Selection

| Function | Description |
|---|---|
| `fixed_x_knockoff_filter` | Fixed-X knockoff filter |
| `model_x_knockoff_filter` | Model-X knockoff filter |

## Multiple Testing

| Function | Description |
|---|---|
| `adjust_pvalues` | BH/BY/Holm/Bonferroni/Hochberg correction |
| `combine_pvalues` | Fisher/Cauchy/Stouffer combination |
| `permutation_test` | Permutation-based hypothesis testing |
