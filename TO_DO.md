# statgpu TO DO

## 开发门禁（必须遵守）

- 每次新增功能，必须同时提供：
  - 全 CPU 实现
  - 全 GPU 实现
  - 两条路径都可独立验证
- 每次新增统计功能（推断/停止准则/显存机制影响数值路径）后，必须补外部框架对标验证：
  - `statsmodels`（推断与统计量优先）
  - `sklearn`（估计量与预测一致性优先）
  - `R`（关键方法补充验证）
- 外部对标时必须显式统一口径：
  - 同一特征集合（禁止隐式 `y ~ .` 把目标列带入特征）
  - 同一 `ties/solver` 配置
  - 同一正则与收敛设置（`alpha/C/max_iter/tol`）

---

## 已完成（2026-04）

- Lasso 推断方法语义化重命名：
  - `cpu_ols_inference`（兼容旧名：`naive_ols`）
  - `gpu_ols_inference`（兼容旧名：`gpu_naive_ols`）
- Lasso 的 `gpu_ols_inference` 路径增强为 GPU 侧推断计算，减少 `scipy.stats` 依赖与大块 CPU 传输。
- 新增 `gpu_memory_cleanup`，覆盖：
  - `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
- 修复 `LogisticRegression.fit()` 在 CUDA 输入 `cupy.ndarray` 时的隐式 `np.asarray` 转换报错。
- `LinearRegression` 新增 `cov_type=nonrobust/hc0/hc1`，并补 CPU/GPU 推断路径。
- `LogisticRegression` 新增 `cov_type=nonrobust/hc0/hc1`，并补 CPU/GPU 推断路径。
- `CoxPH` 新增 `cov_type=nonrobust/hc0/hc1`（稳健协方差近似）并补 CPU/GPU 路径可用性。
- 新增并验证对标测试（`statsmodels`）：
  - `LinearRegression` HC0/HC1（CPU+GPU）
  - `LogisticRegression` HC0/HC1（CPU+GPU）
- 新增 benchmark 脚本：
  - `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
  - `examples/benchmark_gpu_memory_cleanup.py`
  - `examples/benchmark_all_methods_large_scale.py`
  - `examples/benchmark_external_frameworks.py`

---

## 进行中（P0）

- 完善推断严谨性：
  - 扩展稳健协方差到 `cluster-robust` / `HAC`
  - 提升跨设备一致性（`SE/t/z/p/CI`、`AIC/BIC/LLF`）
- Lasso 推断增强：
  - 引入更严谨的 post-selection inference（如 de-biased lasso）
  - 继续推进 bootstrap 的 GPU 化与大规模 benchmark
- CoxPH 推断与评估增强：
  - robust/cluster sandwich 方差
  - `C-index` 提供严格 pairwise 与近似版本可切换

---

## 计划中（P1-P3）

### P1：API parity / 功能补齐

- Lasso：`ElasticNet(l1_ratio)`、`positive`、`warm_start`、alpha path
- Ridge：`warm_start`、path、更完整推断输出
- LogisticRegression：multinomial/softmax、L1/elastic-net、更完整诊断
- CoxPH：strata、frailty、time-varying covariates、penalized Cox
- 稀疏输入支持：CSR/CSC

### P2：模型选择与预处理

- `path / cv / grid-search / warm_start`
- `center/standardize/normalize` 等预处理开关

### P3：Benchmark 框架化

- 统一“数据构造 / fit / inference”拆分计时
- 统一等价 stopping（KKT）标定脚本与口径
- 统一结果差异指标模板（`L_inf`、`L2_rel`、`bse/t/p/CI`）
- 统一 `gpu_memory_cleanup` 报告模板

---

## 功能差距速览（对比 sklearn / statsmodels / R）

- 通用：
  - 稳健协方差类型仍不完整（cluster/HAC 待补）
  - 稀疏矩阵与模型选择工具（CV/path）待完善
  - 预处理开关待补
- LinearRegression：
  - 公式接口、GLS/更完整诊断仍弱于 statsmodels
- Ridge：
  - solver/path/warm_start 与推断体系待增强
- Lasso：
  - 缺 ElasticNet、positive、路径工具与严格 post-selection 推断
- LogisticRegression：
  - 缺多分类与 L1/elastic-net 路径
- CoxPH：
  - 缺 strata/frailty/time-varying、robust/cluster、penalized Cox

