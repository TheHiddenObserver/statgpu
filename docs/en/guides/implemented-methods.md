# Implemented Methods

> Last updated: 2026-07-12

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

## Penalized Models

All seven GLM families support penalties through
`PenalizedGeneralizedLinearModel` or typed wrappers. Specialized `LossBase`
wrappers add quantile, robust, and Cox objectives:

| Class | Loss | Solvers | Penalties | Backends |
|---|---|---|---|---|
| `PenalizedGeneralizedLinearModel` | Any of 7 families | exact, irls, newton, lbfgs, fista, fista_bb | l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad | CPU, CuPy, Torch |
| `PenalizedLinearRegression` | squared_error | exact, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedLogisticRegression` | logistic | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedPoissonRegression` | poisson | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedQuantileRegression` | quantile | proximal_irls_cd, fista | scad, mcp, l2 | CPU, CuPy, Torch |
| `PenalizedRobustRegression` | huber, bisquare | proximal_newton, irls | scad, mcp, l2 | CPU, CuPy, Torch |
| `PenalizedCoxPHModel` | cox_ph | fista/newton; FISTA-LLA for SCAD/MCP | l1, l2, elasticnet, scad, mcp | CPU, CuPy, Torch |

For Gamma, InverseGaussian, NegativeBinomial, and Tweedie with penalties, use `PenalizedGeneralizedLinearModel(loss=..., penalty=...)`:

```python
import numpy as np
from statgpu.inference import norm, poisson, uniform
from statgpu.linear_model import PenalizedGeneralizedLinearModel

# Default: numpy backend (scipy-compatible: rvs, cdf, sf, ppf)
X = norm.rvs(size=(2000, 20))
y = poisson.rvs(mu=3.0, size=2000).astype(float)

# GPU backend via backend= keyword
X_torch = norm.rvs(size=(2000, 20), backend="torch")   # torch tensor on CUDA
X_cupy = norm.rvs(size=(2000, 20), backend="cupy")     # CuPy array on GPU

# Auto-detect from input type
import torch
x = torch.tensor([0.0, 1.96]).cuda()
p = norm.cdf(x)  # automatically uses torch backend

# Gamma + SCAD with auto solver selection
model = PenalizedGeneralizedLinearModel(loss="gamma", penalty="scad", alpha=0.1, solver="auto")
model.fit(X, y)

# NegativeBinomial + ElasticNet with custom dispersion
model = PenalizedGeneralizedLinearModel(
    loss="negative_binomial", penalty="elasticnet",
    loss_kwargs={"alpha": 2.0},  # custom dispersion parameter
    alpha=0.1, l1_ratio=0.5,
    solver="fista",  # explicit solver choice
)
model.fit(X, y)

# Tweedie + group_lasso with sample_weight
sw = uniform.rvs(size=len(y)) * 0.5 + 0.5  # uniform(0.5, 1.5)
model = PenalizedGeneralizedLinearModel(
    loss="tweedie", penalty="group_lasso",
    loss_kwargs={"power": 1.5},
    alpha=0.1, solver="fista",
)
model.fit(X, y, sample_weight=sw)

# Poisson + L1 with IRLS solver (smooth penalty)
model = PenalizedGeneralizedLinearModel(
    loss="poisson", penalty="l1", alpha=0.05,
    solver="irls",  # IRLS for smooth penalties
)
model.fit(X, y)
```

**Solver selection guide:**

| Solver | When to use | Penalties |
|---|---|---|
| `exact` | squared_error + L2 (closed-form) | l2 only |
| `irls` | Smooth penalties (L2, ElasticNet) | l2, elasticnet |
| `newton` / `lbfgs` | Smooth penalties with Hessian | l2, elasticnet |
| `fista` | Non-smooth penalties (L1, SCAD, MCP) | l1, scad, mcp, adaptive_l1 |
| `fista_bb` | Non-smooth with BB step acceleration | l1, elasticnet |
| `auto` | Automatic selection based on penalty | all |

**`sample_weight` support:** All GLM families and solvers support `sample_weight` parameter for weighted regression. Pass a 1D array of weights to `fit(X, y, sample_weight=sw)`.

## Cross-Validation

| Class | Description | Backends |
|---|---|---|
| `RidgeCV` | GPU-accelerated Ridge CV | CPU, CuPy, Torch |
| `LassoCV` | Warm-start alpha path | CPU, CuPy, Torch |
| `ElasticNetCV` | l1_ratio + alpha grid | CPU, CuPy, Torch |
| `LogisticRegressionCV` | GPU-accelerated logistic CV | CPU, CuPy, Torch |
| `PenalizedGLM_CV` | Unified CV for all 7 losses × 10 penalties | CPU, CuPy, Torch |
| `CoxPHCV` | L2 penalty-grid search + refit; start/strata/subject-aware folds; Breslow/Efron/Exact | CPU, CuPy, Torch |

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
| `UMAP` | Uniform Manifold Approximation and Projection (sparse COO edges, NNDescent NN) | CPU, CuPy, Torch |
| `TSNE` | t-distributed Stochastic Neighbor Embedding | CPU, CuPy, Torch |
| `NNDescent` | Approximate nearest neighbor descent (standalone) | CPU, CuPy, Torch |

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
| `CoxPH` | Breslow/Efron/Exact Cox PH; delayed entry, start-stop, strata, subject-aware concordance; nonrobust/HC0/HC1/cluster covariance | CPU, CuPy, Torch |
| `CoxPHCV` | L2 grid selection and final refit with start, strata, subject-preserving folds, and all three tie methods | CPU, CuPy, Torch |
| `PenalizedCoxPHModel` | Estimation-only L1/L2/ElasticNet/SCAD/MCP Cox PH; no intercept; FISTA-LLA for SCAD/MCP | CPU, CuPy, Torch |

Survival-specific boundaries:

- Exact-tie inference is model-based (`cov_type="nonrobust"`) only.
- Baseline hazards use a unified Breslow convention for coefficients fitted by
  Breslow, Efron, or Exact partial likelihood.
- `PenalizedCoxPHModel(compute_inference=True)` raises `NotImplementedError`.
- `subject_id` controls time-varying concordance and automatic CV grouping;
  `cluster` separately defines cluster-robust covariance groups.

The audited quick/full RTX 5880 Ada artifacts are
`results/survival_completion_2026-07-12.json` and
`results/survival_completion_full_2026-07-12.json`. Full delayed-entry speedups
were 1.044x (CuPy) and 1.374x (Torch), but full stratified start-stop speedups
were 0.241x and 0.411x; Exact and standard heavy-tie target scenarios were also
slower than NumPy. No general crossover threshold is claimed.

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
