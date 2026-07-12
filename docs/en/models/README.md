# Models Overview

> Language: English
>
> Last updated: 2026-07-12
>
> Switch: [Chinese](../../cn/models/README.md)

---

## Core Framework

| Page | Content |
|------|---------|
| [Loss Functions (LossBase)](losses.md) | Architecture overview: 12 loss types, per-sample formulas |
| [Solver Algorithms](../guides/solver-algorithms.md) | 10 solvers: algorithm steps, convergence, backend support |
| [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) | Complete dispatch logic and coverage matrix |
| [Solver × Penalty Matrix](../guides/solver-penalty-matrix.md) | Solver routing and penalty constraints |

---

## Loss Functions

| Loss | Page | Penalized Model | Key Solver |
|------|------|-----------------|------------|
| Quantile | [quantile.md](quantile.md) | `PenalizedQuantileRegression` | Proximal IRLS-CD |
| Huber | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Bisquare | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Fair | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Cox PH | [coxph.md](coxph.md) | `PenalizedCoxPHModel` | FISTA / FISTA-LLA |
| GLM (7 families) | [losses.md](losses.md) | `PenalizedGeneralizedLinearModel` | IRLS / Newton / FISTA |

---

## Regression & GLM

| Model | Page | Penalty |
|-------|------|---------|
| LinearRegression | [linear-regression.md](linear-regression.md) | — |
| Ridge | [ridge.md](ridge.md) | L2 |
| Lasso | [lasso.md](lasso.md) | L1 |
| ElasticNet | [elastic-net.md](elastic-net.md) | L1 + L2 |
| SCAD | [scad.md](scad.md) | SCAD (non-convex) |
| MCP | [mcp.md](mcp.md) | MCP (non-convex) |
| AdaptiveLasso | [adaptive-lasso.md](adaptive-lasso.md) | Weighted L1 |
| LogisticRegression | [logistic-regression.md](logistic-regression.md) | L2 |
| PoissonRegression | [poisson-regression.md](poisson-regression.md) | — |
| GeneralizedLinearModel | [generalized-linear-model.md](generalized-linear-model.md) | All penalties |
| Ordered (Logit/Probit) | [ordered.md](ordered.md) | — | Newton-Raphson + analytical Hessian inference |

---

## Survival Analysis

| Model | Page | Features |
|-------|------|----------|
| `CoxPH` | [coxph.md](coxph.md) | Breslow/Efron/Exact ties; entry/start-stop, strata, subject-aware concordance; nonrobust/HC0/HC1/cluster inference |
| `CoxPHCV` | [coxph.md](coxph.md) | L2 grid selection and refit; supports start, strata, subject-preserving folds, and Exact ties |
| `PenalizedCoxPHModel` | [coxph.md](coxph.md) | Estimation-only L1/L2/ElasticNet/SCAD/MCP; no intercept; FISTA-LLA for SCAD/MCP |

`CoxPH` and penalized Cox use native NumPy, CuPy, and Torch model operations.
`CoxPHCV` keeps fold bookkeeping and held-out scoring on the host while fitting
every candidate and the final refit on the requested backend. Exact-tie robust
covariance is not implemented, and penalized Cox inference is explicitly
unavailable.

---

## Unsupervised Learning

| Model | Page | Notes |
|-------|------|-------|
| PCA | [unsupervised.md](unsupervised.md) | Linear dimensionality reduction |
| KMeans | [unsupervised.md](unsupervised.md) | Lloyd k-means++ |
| DBSCAN | [unsupervised.md](unsupervised.md) | Torch CUDA on-device, CuPy + host syncs |
| GaussianMixture | [unsupervised.md](unsupervised.md) | Log-domain EM |
| NMF / MiniBatchNMF | [unsupervised.md](unsupervised.md) | Multiplicative updates |
| IncrementalPCA | [unsupervised.md](unsupervised.md) | Batch-wise |
| TruncatedSVD | [unsupervised.md](unsupervised.md) | Uncentered low-rank |
| UMAP | [unsupervised.md](unsupervised.md) | Sparse COO graph, backend-aware neg-sampling |
| NNDescent | [unsupervised.md](unsupervised.md) | Approximate NN, per-point candidates |
| TSNE | [unsupervised.md](unsupervised.md) | KL divergence |
| AgglomerativeClustering | [unsupervised.md](unsupervised.md) | Hierarchical |

---

## Specialized Modules

| Domain | Page |
|--------|------|
| ANOVA | [anova.md](anova.md) |
| Covariance Estimation | [covariance.md](covariance.md) |
| Panel Data | [panel.md](panel.md) |
| Nonparametric (KDE, Kernel Reg) | [nonparametric.md](nonparametric.md) |
| Kernel Ridge Regression | [kernel-methods.md](kernel-methods.md) |
| Spline Basis Functions | [splines.md](splines.md) |
| GAM (Semiparametric) | [semiparametric.md](semiparametric.md) |
| Knockoff (Feature Selection) | [knockoff.md](knockoff.md) |
| Multiple Testing | [multiple-testing.md](multiple-testing.md) |

---

## v0.2.1 Coverage Summary

| Category | Details |
|----------|---------|
| Loss types | 12 total: 7 GLM + quantile + huber + bisquare + fair + cox_ph |
| Penalties | 10: l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad |
| Solvers | 10: exact, irls, newton, lbfgs, fista, fista_bb, fista_lla, proximal_irls_cd, proximal_newton, admm |
| Backends | numpy, cupy, torch — all core solvers support all three |
| GPU fallback | Explicit GPU devices do not silently fall back to CPU |
| sample_weight | Supported by IRLS/FISTA paths; not supported by Ordered models, Cox partial likelihood, and GLM Newton/LBFGS |
| CV | LassoCV, RidgeCV, LogisticRegressionCV, CoxPHCV (L2; all tie methods), PenalizedGLM_CV |
| Inference | nonrobust/HC0/HC1 (sandwich), Cox cluster covariance, HC2/HC3/HAC (Gaussian only), bootstrap, debiased Lasso, analytical Hessian (ordered); Exact Cox is nonrobust-only |
