# 模型总览

> 语言: 中文  
> 最后更新: 2026-04-22  
> 页面定位: 模型索引  
> 切换: [English](../en/models/README.md)

语言切换：[English](../en/models/README.md)

本节按方法维度组织文档，便于后续新增统计方法时持续扩展。

## 线性模型

- [LinearRegression](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [ElasticNet](elastic-net.md)
- [LogisticRegression](logistic-regression.md)

## 生存分析

- [CoxPH](coxph.md)

## 特征选择

- [Knockoff](knockoff.md)

## 非参数方法

- [Nonparametric](nonparametric.md)

## 新增模型文档流程

新增一个估计器时：

1. 创建 `docs/models/<model-name>.md`
2. 将新页面加入本索引
3. 将入口链接加入 `USAGE_CN.md` 与 `USAGE.md`
4. 若包含基准测试，同步在 `docs/benchmarks.md` 添加脚本引用

## 当前覆盖说明

- 当前核心模型均支持 `device="cpu"` / `device="cuda"` / `device="auto"`。
- 当前核心模型均支持 `gpu_memory_cleanup`。
- 推断能力较完整的模型：
  - `LinearRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`：CPU/GPU OLS 风格推断 + bootstrap
  - `LogisticRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` 支持 Breslow/Efron ties，并提供 CPU/GPU 拟合路径。
- `CoxPH` 的 `entry`（delayed entry）路径已支持：
  - `entry + breslow`：CPU/CUDA/Torch
  - `entry + efron`：CPU/CUDA/Torch
- 特征选择：
  - `Knockoff`：fixed-X / model-X 统一 API + selector 封装
- `LassoCV` 已实现并可直接训练使用。
- 已导出的 CV 类中：
  - `RidgeCV`、`LogisticRegressionCV`、`CoxPHCV` 均已可直接训练使用。
  - `CoxPHCV` 当前边界：GPU 下 `entry` 目前仅支持 `ties='breslow'`；`cluster` 口径暂未支持，会抛出 `NotImplementedError`。
