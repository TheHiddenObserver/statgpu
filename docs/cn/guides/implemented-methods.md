# 已实现方法

> 最后更新：2026-09-17  
> 切换：[English](../../en/guides/implemented-methods.md)

本页是 statgpu 当前公开模型、函数与主要求解器族的用户侧清单。详细数学定义、推断范围与兼容性规则以对应模型页和指南为准。

## 回归与广义线性模型

| Class | 说明 | 后端 |
|---|---|---|
| `LinearRegression` | OLS，支持经典、HC0–HC3 与 HAC 推断 | NumPy, CuPy, Torch |
| `Ridge` | L2 惩罚线性回归 | NumPy, CuPy, Torch |
| `Lasso` | L1 回归，含 debiased/bootstrap 推断路径 | NumPy, CuPy, Torch |
| `ElasticNet` | L1+L2 惩罚回归 | NumPy, CuPy, Torch |
| `LogisticRegression` | 二元 logistic/probit 回归 | NumPy, CuPy, Torch |
| `PoissonRegression` | Poisson GLM | NumPy, CuPy, Torch |
| `GammaRegression` | Gamma GLM | NumPy, CuPy, Torch |
| `InverseGaussianRegression` | Inverse Gaussian GLM | NumPy, CuPy, Torch |
| `NegativeBinomialRegression` | 负二项 GLM | NumPy, CuPy, Torch |
| `TweedieRegression` | Tweedie GLM | NumPy, CuPy, Torch |
| `QuantileRegression` | 分位数回归，支持 kernel/bootstrap 推断 | NumPy, CuPy, Torch |
| `OrderedLogitRegression` | Ordered logit 与解析 Hessian 推断 | NumPy, CuPy, Torch |
| `OrderedProbitRegression` | Ordered probit 与解析 Hessian 推断 | NumPy, CuPy, Torch |

## 惩罚模型

Penalty registry 包含 L1、L2、Elastic Net、SCAD、MCP、adaptive L1、group Lasso、adaptive group Lasso、group MCP 与 group SCAD。部分 penalty 接受别名；solver 支持应以兼容性 reference 为准，不能仅由 penalty 名称推断。

| Class | Loss 或模型族 | 后端 |
|---|---|---|
| `PenalizedGeneralizedLinearModel` | 统一惩罚 GLM 接口 | NumPy, CuPy, Torch |
| `PenalizedLinearRegression` | 惩罚 Gaussian 回归 | NumPy, CuPy, Torch |
| `PenalizedLogisticRegression` | 惩罚二元回归 | NumPy, CuPy, Torch |
| `PenalizedPoissonRegression` | 惩罚 Poisson 回归 | NumPy, CuPy, Torch |
| `PenalizedQuantileRegression` | Quantile loss 与受支持的 proximal/FISTA/IRLS 路径 | NumPy, CuPy, Torch |
| `PenalizedRobustRegression` | 支持范围内的 Huber、bisquare 与 fair loss | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 惩罚 Cox partial likelihood | NumPy, CuPy, Torch |

显式 solver 的可用性取决于 loss 与 penalty。使用前请查看 [Loss × Penalty × Solver 框架](loss-penalty-solver-framework.md)和 [Solver × Penalty 矩阵](solver-penalty-matrix.md)。

### 示例

```python
from statgpu.linear_model import PenalizedGeneralizedLinearModel

model = PenalizedGeneralizedLinearModel(
    loss="poisson",
    penalty="l1",
    alpha=0.05,
    solver="fista",
)
model.fit(X, y)
```

## 交叉验证

| Class | 说明 | 后端 |
|---|---|---|
| `RidgeCV` | Ridge alpha 选择 | NumPy, CuPy, Torch |
| `LassoCV` | Lasso alpha-path 选择与全数据 refit | NumPy, CuPy, Torch |
| `ElasticNetCV` | 联合搜索 `l1_ratio` 与 alpha | NumPy, CuPy, Torch |
| `LogisticRegressionCV` | Logistic 回归 CV | NumPy, CuPy, Torch |
| `PenalizedGLM_CV` | 统一惩罚 GLM CV | NumPy, CuPy, Torch |
| `CoxPHCV` | Cox penalty 搜索与最终 refit | NumPy, CuPy, Torch |

fold、selection、refit、weights 与 inference-after-selection 语义见 [交叉验证](cross-validation.md)。

## 方差分析

- `f_oneway`
- `f_twoway`
- `f_welch`
- `tukey_hsd`
- `bonferroni`
- `cohens_f`
- `partial_eta_squared`

设计限制与标量分布边界见 [ANOVA](../models/anova.md)。

## 协方差估计

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

详见 [协方差估计](../models/covariance.md)。

## 面板数据

- `PanelOLS`
- `RandomEffects`
- `PooledOLS`
- `BetweenOLS`
- `FirstDifferenceOLS`
- `FamaMacBeth`

模型选择、协方差、秩亏与预测行为见 [面板模型](../models/panel.md)。

## 非参数与半参数方法

- `KernelDensity` 与核回归
- `KernelRidge` 与 `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `SplineTransformer`
- B-spline、natural cubic、cyclic cubic 与 thin-plate spline basis
- `GAM`

## 无监督学习

- `PCA`、`TruncatedSVD`、`IncrementalPCA`
- `NMF`、`MiniBatchNMF`
- `KMeans`、`MiniBatchKMeans`、`DBSCAN`
- `GaussianMixture`、`AgglomerativeClustering`
- `UMAP`、`TSNE`、`NNDescent`

## 生存分析

| Class | 说明 | 后端 |
|---|---|---|
| `CoxPH` | Breslow/Efron/Exact ties、delayed entry、`(start, stop]` 行、strata、robust/cluster 推断与 backend-aware prediction | NumPy, CuPy, Torch |
| `CoxPHCV` | 使用同一风险集语义的 L2 网格选择，并保持受试者完整分组 | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 支持范围内的标准右删失凸/非凸 Cox 惩罚 | NumPy, CuPy, Torch |

基础安装已经包含 Cox 拟合。可选 `statgpu[survival]` extra 安装 statsmodels，用于外部比较；statgpu 的 Cox estimator 本身不依赖该 extra。精确支持矩阵见 [Cox 比例风险模型](../models/coxph.md)。

## 特征选择与诊断

- `StepwiseSelector` 与 `stepwise_selection`
- fixed-X/model-X knockoff filter 与 selector wrapper
- `RegressionDiagnostics` 与 `diagnose_model`

## 多重检验与重抽样

- `adjust_pvalues`
- `combine_pvalues`
- `permutation_test`
- inference API 暴露的 bootstrap 工具

需要详细语义时，请继续查看对应模型/reference 页，不要仅根据本清单推断 solver、inference 或 device 支持。
