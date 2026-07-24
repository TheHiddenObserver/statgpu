# 已实现方法

> 最后更新：2026-07-24  
> 切换：[English](../../en/guides/implemented-methods.md)

本页是 statgpu 当前公开模型、函数与主要求解器族的维护中清单。详细数学定义、
推断范围与后端契约以对应模型页和指南为准。

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

Penalty registry 包含 L1、L2、Elastic Net、SCAD、MCP、adaptive L1、
group Lasso、adaptive group Lasso、group MCP 与 group SCAD。部分 penalty
还接受别名；应以 registry 与兼容矩阵为准，不再在文档中维护容易漂移的固定数量。

| Class | Loss 或模型族 | 后端 |
|---|---|---|
| `PenalizedGeneralizedLinearModel` | 统一惩罚 GLM 接口 | NumPy, CuPy, Torch |
| `PenalizedLinearRegression` | 惩罚 Gaussian 回归 | NumPy, CuPy, Torch |
| `PenalizedLogisticRegression` | 惩罚二元回归 | NumPy, CuPy, Torch |
| `PenalizedPoissonRegression` | 惩罚 Poisson 回归 | NumPy, CuPy, Torch |
| `PenalizedQuantileRegression` | Quantile loss 与 proximal/FISTA 路径 | NumPy, CuPy, Torch |
| `PenalizedRobustRegression` | 支持范围内的 Huber、bisquare 与 fair loss | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 惩罚 Cox partial likelihood | NumPy, CuPy, Torch |

显式 solver 的可用性取决于 loss 与 penalty。使用前请查看
[Loss × Penalty × Solver 框架](loss-penalty-solver-framework.md)和
[Solver × Penalty 矩阵](solver-penalty-matrix.md)。

### 示例

```python
from statgpu.linear_model import PenalizedGeneralizedLinearModel

# L1 是非光滑惩罚，应使用 FISTA 或 solver="auto"。
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
| `LassoCV` | Warm-start Lasso path | NumPy, CuPy, Torch |
| `ElasticNetCV` | 联合搜索 `l1_ratio` 与 alpha | NumPy, CuPy, Torch |
| `LogisticRegressionCV` | Logistic 回归 CV | NumPy, CuPy, Torch |
| `PenalizedGLM_CV` | 统一惩罚 GLM CV | NumPy, CuPy, Torch |
| `CoxPHCV` | Cox penalty 搜索与最终 refit | NumPy, CuPy；见 CoxPH 文档 |

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

协方差、秩亏与后端保持预测契约见 [面板模型](../models/panel.md)。

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
| `CoxPH` | Breslow/Efron ties、delayed entry、robust/cluster 推断契约与后端原生预测 | NumPy, CuPy, Torch |
| `PenalizedCoxPHModel` | 支持范围内的凸/非凸 Cox 惩罚 | NumPy, CuPy, Torch |

Exact Efron robust inference 与 delayed-entry reference path 所需的可选 CPU
依赖可通过 `pip install statgpu[survival]` 安装。精确支持矩阵见
[Cox 比例风险模型](../models/coxph.md)。

## 特征选择与诊断

- `StepwiseSelector` 与 `stepwise_selection`
- fixed-X/model-X knockoff filter 与 selector wrapper
- `RegressionDiagnostics` 与 `diagnose_model`

## 多重检验与重抽样

- `adjust_pvalues`
- `combine_pvalues`
- `permutation_test`
- inference API 暴露的 bootstrap 工具

## 验证范围

本清单中的后端支持表示公开执行路径存在。数值、性能与物理 GPU 结论仍应限定到
对应测试或验证 artifact 所记录的具体模型、后端、硬件与 commit。
