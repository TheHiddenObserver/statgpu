# 已实现方法

> 最后更新：2026-06-14

statgpu 已实现的所有模型、函数和类的完整列表。

## 回归与广义线性模型

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

## 惩罚 GLM

所有 7 个 GLM family 都支持惩罚，通过 `PenalizedGeneralizedLinearModel` 或类型化 wrapper：

| Class | Loss | Solvers | Penalties | Backends |
|---|---|---|---|---|
| `PenalizedGeneralizedLinearModel` | 7 个 family 通用 | exact, irls, newton, lbfgs, fista, fista_bb | l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad | CPU, CuPy, Torch |
| `PenalizedLinearRegression` | squared_error | exact, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedLogisticRegression` | logistic | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |
| `PenalizedPoissonRegression` | poisson | irls, fista | l1, l2, elasticnet, scad, mcp, adaptive_l1 | CPU, CuPy, Torch |

对于 Gamma、InverseGaussian、NegativeBinomial 和 Tweedie 的惩罚，使用 `PenalizedGeneralizedLinearModel(loss=..., penalty=...)`：

```python
from statgpu.linear_model import PenalizedGeneralizedLinearModel
from statgpu.inference import get_distribution

# 使用 statgpu 分布 API 生成数据
norm = get_distribution("norm", backend="numpy")
pois = get_distribution("poisson", backend="numpy")

X = norm.rvs(size=(2000, 20))
y = pois.rvs(mu=3.0, size=2000).astype(float)

# Gamma + SCAD，自动选择 solver
model = PenalizedGeneralizedLinearModel(loss="gamma", penalty="scad", alpha=0.1, solver="auto")
model.fit(X, y)

# NegativeBinomial + ElasticNet，自定义离散参数
model = PenalizedGeneralizedLinearModel(
    loss="negative_binomial", penalty="elasticnet",
    loss_kwargs={"alpha": 2.0},  # 自定义离散参数
    alpha=0.1, l1_ratio=0.5,
    solver="fista",  # 显式指定 solver
)
model.fit(X, y)

# Tweedie + group_lasso，带 sample_weight
uniform = get_distribution("uniform", backend="numpy")
sw = uniform.rvs(size=len(y)) * 0.5 + 0.5  # uniform(0.5, 1.5)
model = PenalizedGeneralizedLinearModel(
    loss="tweedie", penalty="group_lasso",
    loss_kwargs={"power": 1.5},
    alpha=0.1, solver="fista",
)
model.fit(X, y, sample_weight=sw)

# Poisson + L1，使用 IRLS solver（光滑惩罚）
model = PenalizedGeneralizedLinearModel(
    loss="poisson", penalty="l1", alpha=0.05,
    solver="irls",  # IRLS 用于光滑惩罚
)
model.fit(X, y)
```

**Solver 选择指南：**

| Solver | 使用场景 | 支持的 Penalties |
|---|---|---|
| `exact` | squared_error + L2（闭式解） | 仅 l2 |
| `irls` | 光滑惩罚（L2、ElasticNet） | l2, elasticnet |
| `newton` / `lbfgs` | 需要 Hessian 的光滑惩罚 | l2, elasticnet |
| `fista` | 非光滑惩罚（L1、SCAD、MCP） | l1, scad, mcp, adaptive_l1 |
| `fista_bb` | BB 步加速的非光滑惩罚 | l1, elasticnet |
| `auto` | 根据 penalty 自动选择 | 所有 |

**`sample_weight` 支持：** 所有 GLM family 和 solver 都支持 `sample_weight` 参数。传入 1D 权重数组即可：`fit(X, y, sample_weight=sw)`。

## 交叉验证

| Class | Description | Backends |
|---|---|---|
| `RidgeCV` | GPU-accelerated Ridge CV | CPU, CuPy, Torch |
| `LassoCV` | Warm-start alpha path | CPU, CuPy, Torch |
| `ElasticNetCV` | l1_ratio + alpha grid | CPU, CuPy, Torch |
| `LogisticRegressionCV` | GPU-accelerated logistic CV | CPU, CuPy, Torch |
| `PenalizedGLM_CV` | Unified CV for all 7 losses × 10 penalties | CPU, CuPy, Torch |
| `CoxPHCV` | CV penalty search + refit | CPU, CuPy |

## 方差分析

| Function | Description |
|---|---|
| `f_oneway` | GPU-accelerated one-way ANOVA |

## 协方差估计

| Class | Description | Backends |
|---|---|---|
| `EmpiricalCovariance` | Sample covariance with jitter-stabilized inversion | CPU, CuPy, Torch |
| `LedoitWolf` | Ledoit-Wolf shrinkage estimator | CPU, CuPy, Torch |
| `OAS` | Oracle Approximating Shrinkage estimator | CPU, CuPy, Torch |

## 面板数据

| Class | Description | Backends |
|---|---|---|
| `PanelOLS` | Fixed effects with nonrobust/robust/clustered SE | CPU, CuPy, Torch |
| `RandomEffects` | Swamy-Arora feasible GLS random effects | CPU, CuPy, Torch |

## 非参数方法

| Class/Function | Description |
|---|---|
| `KernelRidge` | Kernel ridge regression |
| `KernelRidgeCV` | Cross-validated kernel ridge regression |
| `pairwise_kernels` | 6 kernel functions (RBF, polynomial, linear, Laplacian, sigmoid, cosine) |
| `bspline_basis` | B-spline basis (De Boor algorithm, vectorized on GPU) |
| `natural_cubic_spline_basis` | Natural cubic spline basis |

## 半参数模型

| Class | Description | Backends |
|---|---|---|
| `GAM` | Generalized additive model with penalized B-splines + GCV | CPU, CuPy, Torch |

## 无监督学习

**降维与分解：**

| Class | Description | Backends |
|---|---|---|
| `PCA` | Principal component analysis | CPU, CuPy, Torch |
| `TruncatedSVD` | Dense truncated SVD | CPU, CuPy, Torch |
| `IncrementalPCA` | Incremental PCA for large datasets | CPU, CuPy, Torch |
| `NMF` | Non-negative matrix factorization (multiplicative updates) | CPU, CuPy, Torch |
| `MiniBatchNMF` | Mini-batch NMF for large datasets | CPU, CuPy, Torch |
| `UMAP` | Uniform Manifold Approximation and Projection | CPU, CuPy, Torch |
| `TSNE` | t-distributed Stochastic Neighbor Embedding | CPU, CuPy, Torch |

**聚类与混合模型：**

| Class | Description | Backends |
|---|---|---|
| `KMeans` | Lloyd K-Means clustering (k-means++ init) | CPU, CuPy, Torch |
| `MiniBatchKMeans` | Mini-batch K-Means for large datasets | CPU, CuPy, Torch |
| `DBSCAN` | Density-based spatial clustering | CPU, CuPy, Torch |
| `GaussianMixture` | Gaussian mixture model (log-domain EM) | CPU, CuPy, Torch |
| `AgglomerativeClustering` | Exact agglomerative hierarchical clustering | CPU, CuPy, Torch |

## 生存分析

| Class | Description | Backends |
|---|---|---|
| `CoxPH` | Cox proportional hazards | CPU, CuPy, Torch |

## 特征选择

| Function | Description |
|---|---|
| `fixed_x_knockoff_filter` | Fixed-X knockoff filter |
| `model_x_knockoff_filter` | Model-X knockoff filter |

## 多重检验

| Function | Description |
|---|---|
| `adjust_pvalues` | BH/BY/Holm/Bonferroni/Hochberg correction |
| `combine_pvalues` | Fisher/Cauchy/Stouffer combination |
| `permutation_test` | Permutation-based hypothesis testing |
