# 已实现方法

> 最后更新：2026-09-17  
> 切换：[English](../../en/guides/implemented-methods.md)

本页汇总 statgpu 当前公开的模型、函数与主要求解器族。详细的数学定义、推断范围与兼容性规则，请以对应模型页和指南为准。

## 回归与广义线性模型

| 类 | 说明 | 后端 |
|---|---|---|
| `LinearRegression` | OLS，支持经典、HC0–HC3 与 HAC 推断 | NumPy, CuPy, Torch |
| `Ridge` | L2 惩罚线性回归 | NumPy, CuPy, Torch |
| `Lasso` | L1 回归，含去偏推断和自助法推断路径 | NumPy, CuPy, Torch |
| `ElasticNet` | L1+L2 惩罚回归 | NumPy, CuPy, Torch |
| `LogisticRegression` | 二元 logistic/probit 回归 | NumPy, CuPy, Torch |
| `PoissonRegression` | Poisson GLM | NumPy, CuPy, Torch |
| `GammaRegression` | Gamma GLM | NumPy, CuPy, Torch |
| `InverseGaussianRegression` | Inverse Gaussian GLM | NumPy, CuPy, Torch |
| `NegativeBinomialRegression` | 负二项 GLM | NumPy, CuPy, Torch |
| `TweedieRegression` | Tweedie GLM | NumPy, CuPy, Torch |
| `QuantileRegression` | 分位数回归，支持核方法和自助法推断 | NumPy, CuPy, Torch |
| `OrderedLogitRegression` | Ordered logit 与解析 Hessian 推断 | NumPy, CuPy, Torch |
| `OrderedProbitRegression` | Ordered probit 与解析 Hessian 推断 | NumPy, CuPy, Torch |

## 惩罚模型

惩罚项注册表包含 L1、L2、Elastic Net、SCAD、MCP、自适应 L1、Group Lasso、自适应 Group Lasso、Group MCP 与 Group SCAD。部分惩罚项接受别名；求解器支持范围应以兼容性参考为准，不能只根据惩罚项名称推断。

| 类 | 损失函数或模型族 | 后端 |
|---|---|---|
| `PenalizedGeneralizedLinearModel` | 统一惩罚 GLM 接口 | NumPy, CuPy, Torch |
| `PenalizedLinearRegression` | 惩罚 Gaussian 回归 | NumPy, CuPy, Torch |
| `PenalizedLogisticRegression` | 惩罚二元回归 | NumPy, CuPy, Torch |
| `PenalizedPoissonRegression` | 惩罚 Poisson 回归 | NumPy, CuPy, Torch |
| `PenalizedQuantileRegression` | 分位数损失与受支持的近端/FISTA/IRLS 路径 | NumPy, CuPy, Torch |
| `PenalizedRobustRegression` | 支持范围内的 Huber、Bisquare 与 Fair 损失 | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 惩罚 Cox 部分似然 | NumPy, CuPy, Torch |

显式求解器的可用性取决于损失函数与惩罚项。使用前请查看 [损失函数 × 惩罚项 × 求解器框架](loss-penalty-solver-framework.md) 和 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

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

| 类 | 说明 | 后端 |
|---|---|---|
| `RidgeCV` | 选择 Ridge 的 `alpha` | NumPy, CuPy, Torch |
| `LassoCV` | 沿 Lasso 的 `alpha` 路径选择参数，并进行全数据最终重拟合 | NumPy, CuPy, Torch |
| `ElasticNetCV` | 联合搜索 `l1_ratio` 与 `alpha` | NumPy, CuPy, Torch |
| `LogisticRegressionCV` | Logistic 回归交叉验证 | NumPy, CuPy, Torch |
| `PenalizedGLM_CV` | 统一惩罚 GLM 交叉验证 | NumPy, CuPy, Torch |
| `CoxPHCV` | 搜索 Cox 惩罚强度并进行最终重拟合 | NumPy, CuPy, Torch |

数据折、参数选择、最终重拟合、权重以及选择后推断的语义见 [交叉验证](cross-validation.md)。

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
- B 样条、自然三次样条、周期三次样条与薄板样条基
- `GAM`

## 无监督学习

- `PCA`、`TruncatedSVD`、`IncrementalPCA`
- `NMF`、`MiniBatchNMF`
- `KMeans`、`MiniBatchKMeans`、`DBSCAN`
- `GaussianMixture`、`AgglomerativeClustering`
- `UMAP`、`TSNE`、`NNDescent`

## 生存分析

| 类 | 说明 | 后端 |
|---|---|---|
| `CoxPH` | Breslow/Efron/Exact 并列事件、延迟进入、`(start, stop]` 数据、`strata`、稳健/聚类推断与后端感知的预测 | NumPy, CuPy, Torch |
| `CoxPHCV` | 使用同一风险集语义进行 L2 网格选择，并保持受试者完整分组 | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 支持范围内的标准右删失凸/非凸 Cox 惩罚 | NumPy, CuPy, Torch |

基础安装已经包含 Cox 拟合。可选的 `statgpu[survival]` 扩展会安装 statsmodels，用于外部比较；statgpu 的 Cox 估计器本身不依赖该扩展。精确支持矩阵见 [Cox 比例风险模型](../models/coxph.md)。

## 特征选择与诊断

- `StepwiseSelector` 与 `stepwise_selection`
- fixed-X / model-X knockoff filter 与选择器封装
- `RegressionDiagnostics` 与 `diagnose_model`

## 多重检验与重抽样

- `adjust_pvalues`
- `combine_pvalues`
- `permutation_test`
- 推断 API 提供的自助法工具

需要详细语义时，请继续查看对应模型页或参考文档，不要只根据本清单推断求解器、推断方法或设备支持范围。
