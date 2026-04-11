# Changelog

> 语言: 中文  
> 最后更新: 2026-04-11  
> 页面定位: 变更记录  
> 切换: [English](en/changelog.md)

语言切换：[English](en/changelog.md)

## 2026-04

### 新增

- Knockoff 特征选择 API（fixed-X + model-X 高斯二阶路径）：
  - `statgpu.knockoff_filter`
  - `statgpu.fixed_x_knockoff_filter`
  - `statgpu.model_x_knockoff_filter`
  - `statgpu.KnockoffSelector` / `statgpu.FixedXKnockoffSelector`
  - Knockoff 统计量新增 `method='corr_diff'` 与 `method='ols_coef_diff'`
  - model-X 校准新增协方差收缩与多次 knockoff 聚合（W 平均），提升跨 seed 稳定性
- Lasso 推断方法语义化重命名：
  - `cpu_ols_inference`（兼容旧名：`naive_ols`）
  - `gpu_ols_inference`（兼容旧名：`gpu_naive_ols`）
- 全模型显存管理开关 `gpu_memory_cleanup`：
  - `LinearRegression`
  - `Ridge`
  - `Lasso`
  - `LogisticRegression`
  - `CoxPH`
- `LinearRegression(cov_type=...)`：
  - `nonrobust`
  - `hc0`
  - `hc1`
  - `hc2`
  - `hc3`
  - `hac`（支持 `hac_maxlags`）
  并支持 CPU + GPU 推断路径
- `Ridge(cov_type=...)`：
  - `nonrobust`
  - `hc0`
  - `hc1`
  - `hc2`
  - `hc3`
  - `hac`（支持 `hac_maxlags`）
  并支持 CPU + GPU 推断路径
- `LogisticRegression(cov_type=...)`：
  - `nonrobust`
  - `hc0`
  - `hc1`
  - `hc2`
  - `hc3`
  - `hac`（支持 `hac_maxlags`）
  并支持 CPU + GPU 推断路径
- `CoxPH(cov_type=...)`：
  - `nonrobust`
  - `hc0`
  - `hc1`
  （当前为稳健协方差近似路径）
- `CoxPH(cov_type='cluster')`：
  - 支持按 cluster 分组的 sandwich 协方差（CPU 路径）
- 导出 CV 估计器接口骨架：
  - `RidgeCV`
  - `LogisticRegressionCV`
  - `CoxPHCV`
  - 当前状态：仅提供接口骨架；CV 训练逻辑尚未实现，当前会抛出 `NotImplementedError`。
- 新增外部框架统一对标脚本：
  - `dev/benchmarks/benchmark_external_frameworks.py`
- 新增全方法大规模 benchmark：
  - `dev/benchmarks/benchmark_all_methods_large_scale.py`
- 新增 knockoff 基准脚本：
  - `dev/benchmarks/benchmark_knockoff_fixedx.py`
  - `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
  - `benchmark_knockoff_vs_baselines.py` 新增可选 `knockpy` 基线对比能力（环境可用时）
- 新增多重检验指南：
  - `docs/guides/multiple-testing-combine-pvalues.md`

### 改进

- Lasso `gpu_ols_inference` 路径将更多推断步骤放在 GPU 侧，减少 CPU 传输与 SciPy 依赖。
- `LinearRegression` 的 CPU HAC 路径新增自适应精度选择（mixed/float64 快速探测 + 形状分桶缓存），用于降低大规模场景回摆风险。
- 文档体系拆分为：
  - `docs/getting-started`
  - `docs/guides`
  - `docs/models`
  - `docs/benchmarks`
  - `docs/en/*`（英文文档）

### 修复

- 修复 `LogisticRegression.fit()` 在 `y` 为 CuPy 数组时的隐式 NumPy 转换问题。

### 验证

- 新增与 `statsmodels` 的一致性验证：
  - `LinearRegression` 的 `HC0/HC1`
  - `LogisticRegression` 的 `HC0/HC1`（CPU + GPU）
  - `CoxPH` 与 `statsmodels.PHReg`（`breslow/efron`）系数一致性
- 新增/刷新统一三方协方差对比产物（同设定、可审计）：
  - `results/remote_covariance_full_compare_2026-04-10.json`
  - 覆盖 `statsmodels` / `statgpu CPU` / `statgpu GPU` 的 `hc2/hc3/hac` 时间与精度对比
