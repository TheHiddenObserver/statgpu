# 模型总览

> 语言: 中文  
> 最后更新: 2026-05-28  
> 页面定位: 模型索引  
> 切换: [English](../en/models/README.md)

语言切换：[English](../en/models/README.md)

本节按方法维度组织文档，便于后续新增统计方法时持续扩展。

## 线性与 GLM 模型

- [LinearRegression](linear-regression.md)
- [GeneralizedLinearModel 与 Penalized GLM](generalized-linear-model.md)
- [PoissonRegression](poisson-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [ElasticNet](elastic-net.md)
- [LogisticRegression](logistic-regression.md)
- [Ordered Generalized Linear Models (Logit/Probit)](ordered.md)

## 方差分析

- [单因素方差分析 (One-Way ANOVA)](anova.md)

## 协方差估计

- [经验协方差、LedoitWolf、OAS](covariance.md)

## 面板数据

- [固定效应与随机效应 (PanelOLS / RandomEffects)](panel.md)

## 非参数方法

- [核密度估计与核回归](nonparametric.md)
- [核岭回归 (KernelRidge / KernelRidgeCV)](nonparametric/kernel-methods.md)
- [样条基函数](nonparametric/splines.md)

## 半参数模型

- [GAM（广义可加模型）](semiparametric.md)

## 生存分析

- [CoxPH](coxph.md)

## 特征选择

- [Knockoff](knockoff.md)

## 新增模型文档流程

新增一个估计器时：

1. 创建 `docs/models/<model-name>.md`
2. 将新页面加入本索引
3. 将入口链接加入 `USAGE_CN.md` 与 `USAGE.md`
4. 若包含基准测试，同步在 `docs/benchmarks.md` 添加脚本引用

## 当前覆盖说明

- 当前核心模型均支持 `device="cpu"` / `device="cuda"` / `device="auto"`。
- 当前核心模型均支持 `gpu_memory_cleanup`。
- `GeneralizedLinearModel` 与 typed penalized GLM 见 [GeneralizedLinearModel 与 Penalized GLM](generalized-linear-model.md)。
- `PoissonRegression` 作为普通 Poisson GLM estimator 单独记录。
- 推断能力较完整的模型：
  - `LinearRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`：CPU/GPU OLS 风格推断 + bootstrap
  - `LogisticRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` 支持 Breslow/Efron ties，并提供 CPU/GPU 拟合路径。
- `OrderedLogitRegression` / `OrderedProbitRegression` 支持 CPU/CuPy/Torch 三后端，跨后端精度已修复（coef diff < 1e-2）。
- `CoxPH` 的 `entry`（delayed entry）路径已支持：
  - `entry + breslow`：CPU/CUDA/Torch
  - `entry + efron`：CPU/CUDA/Torch
- 特征选择：
  - `Knockoff`：fixed-X / model-X 统一 API + selector 封装
- `LassoCV` 已实现并可直接训练使用。
- 已导出的 CV 类中：
  - `RidgeCV`、`LogisticRegressionCV`、`CoxPHCV` 均已可直接训练使用。
  - `CoxPHCV` 当前边界：GPU 下 `entry` 目前仅支持 `ties='breslow'`；`cluster` 口径暂未支持，会抛出 `NotImplementedError`。
- 新增模块（Tesla P100 验证 38/38 ALL PASS）：
  - `ANOVA`：`f_oneway` — 可替代 `scipy.stats.f_oneway`
  - `Covariance`：`EmpiricalCovariance`、`LedoitWolf`、`OAS` — 等价于 `sklearn.covariance`
  - `KernelMethods`：`KernelRidge`、`KernelRidgeCV` — 等价于 `sklearn.kernel_ridge`
  - `Panel`：`PanelOLS`、`RandomEffects` — 等价于 `linearmodels.panel`
  - `Splines`：`bspline_basis`、`natural_cubic_spline_basis`、`GAM` — 惩罚 B 样条 GAM + GCV
