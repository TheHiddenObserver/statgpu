# statgpu TO DO

> Canonical merged planning entry: `PLAN_UNIFIED.md` in workspace root.
> This file is retained for detailed execution history and checklist context.

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

- **RidgeCV 和 LogisticRegressionCV 完整实现** (2026-04-21):
  - 从接口骨架升级为完整功能实现
  - 支持 K-fold 交叉验证（自定义 folds 或 folds 生成器）
  - Alpha 网格自动生成（log-spaced grid）
  - 交叉验证结果缓存（Blake2b hash key, LRU cache maxsize=64）
  - 支持 `sample_weight` 和 `scoring` 参数
  - 后端支持：CPU (NumPy), GPU (CuPy), GPU (PyTorch)

- Lasso 推断方法语义化重命名：
  - `cpu_ols_inference`（兼容旧名：`naive_ols`）
  - `gpu_ols_inference`（兼容旧名：`gpu_naive_ols`）
- Lasso 的 `gpu_ols_inference` 路径增强为 GPU 侧推断计算，减少 `scipy.stats` 依赖与大块 CPU 传输。
- 新增 `gpu_memory_cleanup`，覆盖：
  - `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
- 修复 `LogisticRegression.fit()` 在 CUDA 输入 `cupy.ndarray` 时的隐式 `np.asarray` 转换报错。
- `LinearRegression` 新增 `cov_type=nonrobust/hc0/hc1`，并补 CPU/GPU 推断路径。
- `LinearRegression / Ridge / LogisticRegression` 新增 `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac`，并补 `hac_maxlags` 与 CPU/GPU 路径。
- `LogisticRegression` 新增 `cov_type=nonrobust/hc0/hc1`，并补 CPU/GPU 推断路径。
- 新增三方统一协方差对比产物：
  - `results/remote_covariance_full_compare_2026-04-10.json`（`statsmodels` / `statgpu CPU` / `statgpu GPU`）
- `CoxPH` 新增 `cov_type=nonrobust/hc0/hc1`（稳健协方差近似）并补 CPU/GPU 路径可用性。
- `CoxPH` 新增 `cov_type=cluster`（按 cluster 分组的 sandwich 协方差，CPU 路径）。
- 新增并验证对标测试（`statsmodels`）：
  - `LinearRegression` HC0/HC1（CPU+GPU）
  - `LogisticRegression` HC0/HC1（CPU+GPU）
  - `CoxPH` 与 `statsmodels.PHReg`（`breslow/efron`）系数一致性
  - CoxPH 综合性能对比 (2026-04-20)：Torch GPU 15.44x 加速 (n=5000, p=20)
- 新增 benchmark 脚本：
  - `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
  - `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
  - `dev/benchmarks/benchmark_all_methods_large_scale.py`
  - `dev/benchmarks/benchmark_external_frameworks.py`
- `Ridge` 新增完整推断体系，与 `LinearRegression` 对齐：
  - `cov_type=nonrobust/hc0/hc1`（CPU + GPU 双路径）
  - `compute_inference` 开关
  - 新增属性：`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic`
  - 新增 `summary()` 方法（R/statsmodels 风格）
  - 修复 `fit_intercept=False` 时 `_fit_cpu` 的 `y_centered` 未赋值 bug
  - 新增专项推断测试：`dev/tests/test_ridge_inference.py`（24 tests）

---

## 进行中（P0）

- 完善推断严谨性：
  - 扩展稳健协方差到 `cluster-robust`（Linear/Ridge/Logistic 的 `HC2/HC3/HAC` 已完成）
  - 提升跨设备一致性（`SE/t/z/p/CI`、`AIC/BIC/LLF`）
- Lasso 推断增强：
  - 引入更严谨的 post-selection inference（如 de-biased lasso）
  - 继续推进 bootstrap 的 GPU 化与大规模 benchmark
- CoxPH 推断与评估增强：
  - ✅ robust/cluster sandwich 方差已完成
  - ✅ C-index 已修复 (2026-04-20)，使用精确分块向量化算法
  - ✅ Efron ties 实现已修复数值溢出问题 (clipping 保护)
  - ⚠️ Cython 编译版本存在正确性问题，待调试；当前使用 Python fallback

---

## 计划中（P1-P3）

### P1：API parity / 功能补齐

- Lasso：`ElasticNet(l1_ratio)`、`positive`、`warm_start`、alpha path
- Ridge：~~`warm_start`、path、更完整推断输出~~ 推断体系已完成；待补 `warm_start`、path
- LogisticRegression：multinomial/softmax、L1/elastic-net、更完整诊断
- CoxPH：strata、frailty、time-varying covariates、penalized Cox
- 稀疏输入支持：CSR/CSC
- CV 估计器：
  - ~~`RidgeCV`~~ ✅ 已完成 (2026-04-21)
  - ~~`LogisticRegressionCV`~~ ✅ 已完成 (2026-04-21)
  - `CoxPHCV` - 待实现

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
  - 稳健协方差类型仍不完整（多模型 `cluster-robust` 待补；`HAC` 已覆盖 Linear/Ridge/Logistic）
  - 稀疏矩阵与模型选择工具（CV/path）待完善
  - 预处理开关待补
- LinearRegression：
  - 公式接口、GLS/更完整诊断仍弱于 statsmodels
- Ridge：
  - solver/path/warm_start 待增强；推断体系（cov_type/summary）已完成
- Lasso：
  - 缺 ElasticNet、positive、路径工具与严格 post-selection 推断
- LogisticRegression：
  - 缺多分类与 L1/elastic-net 路径
- CoxPH：
  - 缺 strata/frailty/time-varying、robust/cluster、penalized Cox

---

## CV 框架后续改进项

### P1: FISTA/BB solver 支持非均匀 sample_weight

**现状**：`fista_solver` 和 `fista_bb_solver` 拒绝非均匀 `sample_weight`。只有 `irls` 支持，但 IRLS 仅适用于 L2 惩罚。

**影响**：L1/ElasticNet/SCAD/MCP + 非均匀权重 → `ValueError`。

**方案**：在 FISTA 梯度计算中改为 `X' diag(w) residual / sum(w)`，需要：
1. `_solver.py` 的 `fista_solver` 和 `fista_bb_solver` 支持 `sample_weight`
2. Armijo 回溯使用加权 loss
3. Lipschitz 使用加权 Hessian `X' diag(w) X`

### ~~P2: NB alpha / Tweedie power 参数化~~ ✅ 已完成

NB alpha 和 Tweedie power 现在从 loss 对象默认值动态读取（`_resolve_loss_name`），不再硬编码。

### ~~P2: 添加新 loss 的改动点统一~~ ✅ 已完成

已实现 loss formula registry（`_LOSS_RESIDUAL_FNS`、`_LOSS_VALLOSS_FNS`、`_FOLD_BATCH_CONFIGS`、`_LOSS_EVAL_DISPATCH`）。添加新 loss 只需 5 处改动（从 8-10 处减少）。

### ~~P2: CV 策略可扩展性~~ ✅ 已完成

已添加 `cv_splits` 参数，支持自定义 fold 生成器（TimeSeriesSplit、StratifiedKFold 等）。

### P3: Backend 可扩展性（方案 C：混合抽象）

**目标**：消除热循环中 ~12 处 `if is_torch` 分支，统一到 `_array_ops.py` 的 `_xp` 分发。

**方案**：简单操作（where、sign、clip、sum、any、full_like）扩展到 `_array_ops.py`，复杂操作（solve、lstsq、inference）留在 Backend 类。

```python
# _array_ops.py — 统一入口
def where(cond, a, b):
    return _xp(cond).where(cond, a, b)
def sign(x):
    return _xp(x).sign(x)
def clip(arr, lo, hi):
    xp = _xp(arr)
    if xp.__name__ == "torch":
        return xp.clamp(arr, min=lo, max=hi)
    return xp.clip(arr, lo, hi)

# 热循环中直接调用，无 if is_torch
coef_new = sign(w) * clip(abs(w) - thresh, 0, None) / denom
coef = where(active, coef_new, coef)
```

**添加新 backend 的改动量**：只需在 `_xp()` 中加一行检测，~1 处改动。

**状态**：未实现，记录为后续 PR。

### P3: 非 Ridge 模型的 inference

**现状**：`_refit_best` 对非 Ridge 模型设置 `compute_inference=False`，用户无法直接获取标准误和 p 值。

**方案**：实现 debiased inference for L1/ElasticNet（Zhang-Zhang / Javanmard-Montanari 方法），或提供 bootstrap inference 接口。

### P3: `_penalized_cv.py` 文件拆分

**现状**：`_penalized_cv.py` 2800+ 行，包含数值常量、loss 函数、CV path 函数、PenalizedGLM_CV 类等。

**方案**：拆分为 3 个文件：
- `_cv_loss_registry.py` — 数值常量 + `_ps_squared_error` + `_LOSS_EVAL_DISPATCH` + `_LOSS_RESIDUAL_FNS`/`_LOSS_VALLOSS_FNS` + `_register_loss_fns` + `_weighted_mean` + `_evaluate_loss_numpy`
- `_cv_paths.py` — `_logistic_sparse_cv_path`、`_squared_error_sparse_cv_path`、`_glm_sparse_cv_path`、`_scad_mcp_cv_path`、`_FeatureOnlySparsePenalty`
- `_penalized_cv.py` — PenalizedGLM_CV 类 + dispatch table + `_glm_sparse_cv_folds` + fold-batch helpers

**改动量**：~3 处 import 调整，无逻辑变更。
**状态**：未实现，记录为后续 PR。

### P3: CV path 函数 backend 重复代码消除

**现状**：`_logistic_sparse_cv_path`、`_squared_error_sparse_cv_path`、`_glm_sparse_cv_path` 内部各有 3 路 backend 分支（torch/cupy/numpy），大量重复的 `if backend == "torch": ... elif backend == "cupy": ... else: ...` 代码。

**方案**：
1. 将热循环中的 backend 操作统一到 `_fb_*` helpers（已有 `_fb_ones`、`_fb_zeros`、`_fb_cat` 等）
2. 对于 `sign`、`clamp`、`maximum` 等操作，扩展 `_array_ops.py` 的 `_xp` 分发（与 P3 Backend 可扩展性合并）
3. 保留 `_glm_sparse_cv_folds` 的直接 API 调用（性能关键路径），但用 `_fb_*` 减少 boilerplate

**状态**：未实现，记录为后续 PR。

---

## PR #49 Code Review 剩余 P3 项 (2026-06-10)

> 所有 P1/P2 bug 已修复。以下为需要较大重构的 P3 改进项。

### P3: `_elasticnet_cv.py` Alpha Grid 函数合并

**现状**：`_default_elasticnet_alpha_grid`（numpy）和 `_default_elasticnet_alpha_grid_backend`（GPU）功能重复。numpy 版本对 `l1_ratio=0` 有不同逻辑，可能导致 CPU/GPU 结果不一致。
**方案**：合并为单一 backend-agnostic 函数，使用 `_xp()` helpers。
**难度**：中 | **风险**：中 | **状态**：未实现

### P3: `_logistic_cv.py` Log-Loss 函数合并

**现状**：`_batch_log_loss`（numpy）和 `_batch_log_loss_backend`（GPU）几乎相同。
**方案**：使用 backend dispatch 合并。
**难度**：中 | **风险**：中 | **状态**：未实现

### P3: `_hash_logistic_data` 共享

**现状**：`_logistic_cv.py:44-62` 和 `_elasticnet_cv.py:93-114` 有近乎相同的 hash 函数。
**方案**：提取到 `_cv_base.py` 作为共享工具函数。
**难度**：低 | **风险**：低 | **状态**：未实现

### P3: `_solver.py` Loss Name 硬编码

**现状**：`_solver.py:626` 使用硬编码的 loss name 列表检查 fused 路径。新增 loss 时需手动更新。
**方案**：改为 `if _loss_name in _GLM_FUSED_REGISTRY:`。
**难度**：低 | **风险**：低 | **状态**：未实现

### P3: `_penalized.py` Local SelectivePenalty 重复

**现状**：`_penalized.py:4558-4623` 在 `_fit_cpu_loss()` 内定义了本地 `SelectivePenalty` 类，与模块级类重复。
**方案**：复用 `_get_selective_penalty_singleton().configure()`。
**难度**：低 | **风险**：低 | **状态**：未实现

### P3: `_elasticnet_cv.py` XtX 按 l1_ratio 重复计算

**现状**：`_elasticnet_cv.py:621-625` 中 `XtX_fold` 和特征值在每个 l1_ratio 循环内重新计算，但它们只依赖 fold。
**方案**：将 XtX/特征值计算移到 l1_ratio 循环外。
**难度**：中 | **风险**：中 | **状态**：未实现

### P3: `_penalized.py` `_fit_initial` 始终在 CPU

**现状**：`_fit_initial()` 始终转换为 numpy 在 CPU 运行，即使主 fit 在 GPU。
**方案**：在相同 backend 上运行 init。
**难度**：中 | **风险**：中 | **状态**：未实现

### P3: `_logistic_cv.py` GPU 概率未跨 C 向量化

**现状**：概率计算逐个 C 值循环，可通过堆叠 coef 做单次矩阵乘法。
**方案**：类似 `_batch_mse_elasticnet` 的批量处理方式。
**难度**：中 | **风险**：中 | **状态**：未实现

### P3: `_penalized.py` Debiased Inference 代码重复

**现状**：`_compute_inference_debiased_gpu()` 和 `_compute_inference_torch()` 约 280 行几乎相同代码。
**方案**：重构为单一 backend-agnostic 方法。
**难度**：高 | **风险**：高 | **状态**：未实现

### P3: `_penalized.py` Node-wise Lasso 循环优化

**现状**：debiased inference 的 node-wise Lasso 循环为每个特征创建新模型实例（最多 p 次）。
**方案**：复用单个实例（`warm_start=True`）或使用底层 solver。
**难度**：高 | **风险**：高 | **状态**：未实现

### P3: `_cv_base.py` `folds_are_complements` 死代码

**现状**：函数定义但未被任何生产代码导入。
**方案**：删除或标记为测试工具。
**难度**：低 | **风险**：无 | **状态**：未实现

### P3: `_cv_engine.py` 为 Reference Implementation

**现状**：`run_cv` 仅被测试文件调用。模块文档承认"The production CV paths use their own optimized loops"。
**方案**：移至 `dev/` 或标记为 reference。
**难度**：低 | **风险**：无 | **状态**：未实现

