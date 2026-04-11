# 模型总览

> 语言: 中文  
> 最后更新: 2026-04-11  
> 页面定位: 模型索引  
> 切换: [English](../en/models/README.md)

语言切换：[English](../en/models/README.md)

本节按方法维度组织文档，便于后续新增统计方法时持续扩展。

## 线性模型

- [LinearRegression](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [LogisticRegression](logistic-regression.md)

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
- 推断能力较完整的模型：
  - `LinearRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`：CPU/GPU OLS 风格推断 + bootstrap
  - `LogisticRegression`：经典协方差 + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` 支持 Breslow/Efron ties，并提供 CPU/GPU 拟合路径。
- 特征选择：
  - `Knockoff`：fixed-X / model-X 统一 API + selector 封装
- `LassoCV` 已实现并可直接训练使用。
- 已导出的 CV 类（`RidgeCV`、`LogisticRegressionCV`、`CoxPHCV`）当前仅为接口骨架；CV 训练/搜索逻辑尚未完成。
