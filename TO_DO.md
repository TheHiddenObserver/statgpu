# statgpu TO DO

> Canonical merged planning entry: `PLAN_UNIFIED.md` in workspace root.
> This file is retained for detailed execution history and checklist context.

## 开发门禁（必须遵守）

> 来源：PLAN_UNIFIED.md §1 Hard Gates + PR #49 Code Review 经验总结

### 功能门禁

- 每次新增功能，必须同时提供：
  - NumPy 实现（CPU）
  - CuPy 实现（GPU）
  - Torch 实现（GPU）
  - 三条路径都可独立验证
- 每次新增统计功能（推断/停止准则/显存机制影响数值路径）后，必须补外部框架对标验证：
  - `statsmodels`（推断与统计量优先）
  - `sklearn`（估计量与预测一致性优先）
  - `R`（关键方法补充验证）
- 外部对标时必须显式统一口径：
  - 同一特征集合（禁止隐式 `y ~ .` 把目标列带入特征）
  - 同一 `ties/solver` 配置
  - 同一正则与收敛设置（`alpha/C/max_iter/tol`）

### 推断门禁

- Ridge/Lasso strict 模式必须通过外部对齐阈值：
  - coef: `1e-6`
  - bse: `1e-3`
  - p-value: `5e-2`
- strict 失败策略：默认 raise error，仅在显式启用时降级

### 设备一致性门禁

- strict 模式输出在 CPU/GPU 上对齐
- CUDA 使用分层规则：model 层不应散布直接 cupy import

### 工程门禁

- 每次提交：lint + type + test
- 每月稳定版：外部矩阵 + benchmark 非回退 + 文档同步
- 性能敏感推断方法必须包含远程 CUDA 重跑产物

---

## 编码规范（PR #49 总结）

### 数值正确性

- 浮点比较不用 `==`，用容差：`abs(a - b) < 1e-10`
- 除法前检查分母：`sw_sum = max(sw_sum, 1e-15)` 或 `if sw_sum > 0`
- sqrt 前检查非负：`np.sqrt(max(x, 0.0))`
- log 前检查正数：`np.log(max(x, 1e-10))`
- 统一 ddof：全部用 `ddof=1`（样本标准差）
- alpha_max 考虑 l1_ratio：`max(|X'y|) / (n * l1_ratio)`
- 初始化循环变量：`iteration = -1` 在 `for` 循环前（防止 `max_iter=0` 时 NameError）
- 未知参数抛 TypeError：移除 `**kwargs` 或验证

### 后端一致性

- 三端公式必须一致：检查 numpy/cupy/torch 路径的数学公式
- 不用 `backend.exp()`，用 `xp.exp()`：BackendBase 无 exp 方法
- `_to_numpy()` 用于验证，保留原数组：避免破坏 GPU 流水线
- `float()` 调用是 GPU sync：批量 sync，避免逐坐标调用
- `np.asarray()` 强制转 CPU：用 `_to_numpy()` 或保留原 backend

### 缓存与线程安全

- 缓存键用内容哈希，不用内存指针：`blake2b(tobytes())` 而非 `data_ptr`
- 缓存存副本，不存引用：`cache.put(key, (val.copy(),))`
- 共享缓存加 `threading.Lock`：多线程数据竞争
- eviction 在锁内：锁外 eviction 是竞态条件

### 接口设计

- `get_params` 包含所有构造参数：sklearn clone/GridSearchCV 丢失参数
- `_fitted` 在 `fit()` 末尾设置：`summary()` 等方法提前调用报错
- `best_score_` 用负 MSE：sklearn 惯例 higher=better
- `cv >= 2` 验证：单折 CV 无意义

### 代码组织

- 共享函数放 `_cv_base.py`：避免多文件重复
- Magic numbers 用命名常量：`_EIGVAL_FLOOR = 1e-15` 而非 `1e-15`
- 死代码及时删除：删除 `and False`、未使用的 import
- backend 分支用 helper：`_xp()`, `_clip()`, `_sigmoid()`
- 单一职责：大文件应拆分

### 测试规范

- 每个修复对应测试：回归无法检测
- 三端精度对比：同一 `random_state`，比较 `best_score_`
- 性能基准测试：记录 before/after 耗时
- 远程 GPU 测试：本地无 GPU 无法验证
- 边界情况测试：空输入、零权重、max_iter=0

### 提交规范

- 每个 fix 单独 commit：方便 revert 和 review
- commit message 包含测试结果：`Tested: 428 passed, 0 failed on Tesla P100`
- TO_DO.md 同步更新：修复后标记 ✅，新发现后添加

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

### ~~P1: FISTA/BB solver 支持非均匀 sample_weight~~ ✅ 已修复

**现状**：`fista_solver`、`fista_bb_solver`、`fista_lla_path` 现已支持非均匀 `sample_weight`。
**方案**：
- `_base.py`：`GLMLoss.value()` 和 `gradient()` 添加 `sample_weight` 参数
- `_squared.py`/`_logistic.py`：实现加权 value/gradient/hessian/lipschitz
- `_solver.py`：添加 `_weighted_loss_and_grad` helper，修改 solver 使用加权 loss/gradient/Lipschitz
- `fista_bb_solver`/`fista_lla_path`：替换 `_validate_uniform_sample_weight` 为基本验证，传递 `sample_weight` 到 loss 函数
**状态**：✅ 已修复 (2026-06-11)。`newton_solver`/`lbfgs_solver`/`admm_solver` 暂不支持（保留 ValueError）。
**三端验证**：numpy ✅ cupy ✅ torch ✅（Tesla P100）

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

## PR #49 Code Review 剩余项 (2026-06-10，最后更新: 2026-06-11)

> P1 全部修复（22 个）。P2 大部分修复。以下为剩余项。

### 已修复但 TO_DO.md 未更新的项 (20 项)

- ✅ elasticnet alpha grid 合并（已简化）
- ✅ logistic log-loss 合并（已合并）
- ✅ hash_logistic_data 共享（已提取到 _cv_base.py）
- ✅ solver loss name 硬编码（已改为 _GLM_FUSED_REGISTRY 检查）
- ✅ SelectivePenalty 重复（已复用 singleton）
- ✅ elasticnet XtX 按 l1_ratio 重复（已移到 fold 循环外）
- ✅ _fit_initial GPU（已支持 backend_name）
- ✅ logistic GPU 概率向量化（已用批量矩阵乘法）
- ✅ folds_are_complements 死代码（已删除）
- ✅ _cv_engine.py reference impl（已标记文档）
- ✅ elasticnet predict/score（已委托给 estimator_）
- ✅ 两个 LassoCV 类（已统一委托 _select_lasso_alpha_cv）
- ✅ array_identity_token（已改为 GPU 端采样）
- ✅ batch_mse 内存峰值（已改为分块计算 chunk_size=256）
- ✅ sample_weight 检查重复（已提取 _is_uniform_weight）
- ✅ populate_refit df_resid（已检查 fit_intercept）
- ✅ Lasso **kwargs（已移除）
- ✅ ddof=1 一致性（已统一）
- ✅ dead code obj_old/yty（已删除）
- ✅ n_cont SCAD/MCP（已统一为 20 步）

---

### 剩余 P2 (7 项，本轮 Code Review 新发现)

### P2: `_solver.py` fista_bb BB state 用 backtracking 前的 coef_new

**现状**：已修复。BB state 现在使用实际接受的迭代 `coef`。
**状态**：✅ 已修复

### P2: `_solver.py` admm_solver 计算 Cholesky 但用 `np.linalg.solve`

**现状**：已修复。现在使用预计算的 `_L` 做前向/回代。
**状态**：✅ 已修复

### P2: `_penalized.py` `_irls_cd_gpu` 逐坐标 `float()` 调用

**现状**：已修复。CD 循环从逐坐标 `for j` 改为向量化 block CD，GPU sync 从 O(pp) 降为 O(1)/sweep。
**方案**：`rho_all = X_work.T @ (d * r) + XDX_diag * beta` 批量梯度 + `xp.where` 向量化 thresholding。
**状态**：✅ 已修复 (2026-06-11)

### P2: `_penalized_cv.py` SCAD/MCP 内层 FISTA 无 CV 迭代上限

**现状**：已修复。添加了 `_FISTA_MAX_ITER_CV` 上限。
**状态**：✅ 已修复

### P2: `_elasticnet_cv.py` alpha_max 公式未考虑 l1_ratio

**现状**：已修复。公式改为 `max(|X'y|) / (n * l1_ratio)`。
**状态**：✅ 已修复

### P2: `_lasso_cv.py` `_fit_cv` 方法 340 行死代码

**现状**：已修复。标记为 deprecated 并添加警告。
**状态**：✅ 已修复

### P2: `_fixed_effects.py` `xp.unique` GPU sync 优化

**现状**：已修复。unique 计算移到 numpy 路径。
**状态**：✅ 已修复

---

### 剩余 P3 (5 项，大规模重构)

### P3: Backend 可扩展性（3-way 分支消除）

**现状**：从原始 ~241 个分支消除到 ~153 个（消除 88 个，37%）。
- `_penalized.py`：61→20（消除 41）
- `_solver.py`：58→10（消除 48）
- `_irls.py`：12→4（消除 8）
- `_glm_base.py`：24→15（消除 9）
- `_penalized_cv.py`：69→11（消除 58）
- penalties：17→14（消除 3）

**剩余分支构成**：fused kernel 路径（~15）、torch dtype promotion（~10）、linalg API 差异（~10）、GPU cleanup dispatch（~10）、OrderedGLM 3 个独立 solver（~30）、debiased inference（~30）、其他（~48）。大部分为 intentionally separate 设计。
**状态**：基本完成，剩余分支多为性能关键路径或 API 差异，强行统一会损害性能或引入 bug

### P3: 添加新 loss 需改 6+ 文件

**现状**：✅ 已修复。每个 loss 类实现 `per_sample_value`/`per_sample_gradient`/`_mu_from_eta` 作为单一真实来源，base class 自动派生 `value()`/`gradient()`/`fused_value_and_gradient()`。添加新 loss 只需 2 步。
**状态**：✅ 已修复 (2026-06-11)

### P3: `_penalized_cv.py` 文件拆分

**现状**：2800+ 行。
**方案**：拆为 3 个文件。
**难度**：高 | **风险**：高 | **状态**：未实现

### P3: CV path 函数 backend 重复代码消除

**现状**：5+ 函数的后端分支重复。
**方案**：扩展 `_fb_*` helpers。
**难度**：高 | **风险**：高 | **状态**：未实现

### P3: `_penalized.py` Debiased Inference 代码重复

**现状**：~280 行 CuPy/Torch 几乎相同代码。
**方案**：重构为单一 backend-agnostic 方法。
**难度**：高 | **风险**：高 | **状态**：未实现

### P3: `_penalized.py` Node-wise Lasso 循环优化

**现状**：为每个特征创建新模型实例（最多 p 次）。
**方案**：复用单个实例或使用底层 solver。
**难度**：高 | **风险**：高 | **状态**：未实现

---

## 第九轮 Code Review 剩余项 (2026-06-11)

> 以下为需要中等重构的项，不影响正确性。

### ~~P2: `_lasso.py` + `_lasso_cv.py` 重复缓存定义~~ ✅ 已解决

**现状**：`_lasso_cv.py` 不再包含缓存定义，已统一到 `_lasso.py`。
**状态**：✅ 已解决

### ~~P2: 多文件重复 `_batch_mse` 实现~~ ✅ 已修复

**现状**：已统一到 `_cv_base.py` 的 `batch_mse`。删除 5 个重复定义（~200 行），5 个调用点改为 `batch_mse`。
**状态**：✅ 已修复 (2026-06-11)

### ~~P2: `_irls.py` deviance 计算 3-way 后端重复~~ ✅ 已修复

**现状**：`_dev_val` 中 2 个 `if backend == "torch"` 分支已替换为 `_clip()` helper。
**状态**：✅ 已修复 (2026-06-11)

---

## Code Review 发现项 (2026-06-11)

### 已修复的 Bug

- ✅ `_penalized_cv.py` `_logistic_sparse_cv_path`：非 torch 后端 FISTA 循环缺少 soft-thresholding（coef 永不更新）
- ✅ `_penalized_cv.py` `_glm_sparse_cv_folds`：使用未定义的 `xp` 变量
- ✅ `_penalized_cv.py` `_scad_mcp_cv_path`：使用未定义的 `torch`/`cp` 变量
- ✅ `_penalized_cv.py` `_squared_error_sparse_cv_path`：使用未定义的 `torch` 变量
- ✅ `_glm_base.py` probit PDF：`xp.asarray(2π)` 在 torch+CUDA 上设备不匹配
- ✅ `_irls.py` dtype promotion：逻辑反转，将 y downcast 到 X.dtype 而非 promote
- ✅ `_array_ops.py` `_max_eigval_power`：numpy 路径 `n_iter=0` 时 `lambda_new` 未初始化
- ✅ `_array_ops.py` `_max_eigval_power`：GPU 路径循环结束后缺少 `return`（返回 None）
- ✅ `_solver.py` `_fused_inverse_gaussian`：value 公式与 `InverseGaussianLoss.value` 不一致
- ✅ `_penalized.py` `_fit_initial`：缺少 `backend_name` 参数
- ✅ `_penalized.py` `np.asarray(X)`：CuPy 数组在 `device="auto"` 时无法隐式转换
- ✅ `_scad.py`/`_mcp.py` `lla_weights`：未使用的 `mod` 变量（死代码）
- ✅ `_solver.py`：未使用的 `_abs_max` import（死代码）
- ✅ `_penalized_cv.py`：`_torch_available`/`_cupy_available` 不存在（ImportError）
- ✅ `_penalized_cv.py`：`_soft_threshold` 在 4 个热循环内 import（移至模块顶部）
- ✅ `_utils.py`：`xp_astype` 传 numpy dtype 给 torch `.to()` 失败（添加 `_np_dtype_to_torch`）
- ✅ `_penalized_cv.py`：`xp_asarray` 参数顺序错误（`Xb` 被传为 `xp`）
- ✅ `_irls.py`：`np.result_type` 无法处理 torch dtype（转为字符串）

### 已知但未修复的问题（低优先级）

- `_solver.py` `_NO_MOMENTUM_LOSSES = frozenset()` 及对应分支是死代码（保留为将来预留）
- `_solver.py` Poisson 动量策略在 `fista_solver` vs `fista_lla_path` 中不一致
- `_xp_copy` 与 `_copy_arr` 功能重复（命名不一致）
- `_xp_zeros(...) + 1.0` 应改为 `_xp_ones`
- `score()` 方法将数据不必要地传回 GPU 做简单比较
- `_penalized_cv.py` 多处 `warnings.warn` 缩进不一致
- `_penalized_cv.py` `_is_uniform_weight` 警告模式重复 4 次，应提取为 helper
- `_penalized.py` `_use_fista` 条件可简化（两个分支覆盖所有情况）
- `_penalized.py` `_use_irls_cd` 标志名与实际使用的 `fista_lla_path` 矛盾
- `_solver.py` `_fused_logistic` 使用字符串模块检测而非 `_xp()`

### 可读性/可维护性问题（Code Review Round 2）

- `_array_ops.py`：`_copy_arr`/`_xp_copy`、`_zeros`/`_xp_zeros`、`_eye_like`/`_xp_eye` 命名重复，应统一
- `_solver.py`：4 种不同的标量提取模式（`float()`、`.item()`、`_to_float_scalar`、`_to_numpy`），应统一为 `_to_float_scalar`
- `_solver.py`：`fista_bb_solver`（470 行）、`fista_lla_path`（560 行）过长，需拆分
- `_penalized.py`：`_fit_cpu`（260 行）、`_fit_lla`（255 行）过长，需拆分
- `_penalized_cv.py`：6 个几乎相同的 FISTA 循环实现，应提取为共享 `_fista_cv_step`
- `_penalized_cv.py`：`_FeatureOnlySparsePenalty` 与 `_penalized.py` 的 `SelectivePenalty` 功能重复
- `_glm_base.py`：3 种不同的 `_xp` 解析函数（`_xp_arr`、`_xp`、`_get_xp`），语义不同
- `_family.py`：第 3 个独立的 `_xp` 实现，应统一
- ✅ `_scad.py`/`_mcp.py`：`_xp` 和 `_to_float_scalar` 已移至模块级 import
- `_solver.py`：`_fused_logistic` 等函数在热循环内 import `_sigmoid`/`_clip`（应移至模块级）

### 性能优化项（Code Review Round 2）

- `_penalized.py` `_irls_cd_gpu`：`_to_numpy` 在 GPU 方法中传输完整 X 矩阵（应改为 GPU 端计算 lambda_max）
- `_penalized_cv.py` `_scad_mcp_cv_path`：每 alpha 一次 D2H transfer（应改为批量 sync）
- `_penalized_cv.py`：FISTA 循环中 `_copy_arr` 每次迭代分配新内存（应改为交替缓冲区）
- ✅ `_solver.py` `fista_bb_solver`：已通过 `fused_value_and_gradient` 统一，`_fused_*` 函数变为死代码
- ✅ `_array_ops.py` `_soft_threshold`：已改用 `xp.where` 融合（2 个中间数组，~15% 性能提升）
- ✅ `fista_bb_solver`：修复 post-divergence reset 缺失 sample_weight
- ✅ `NegativeBinomialLoss.lipschitz`：修复双重 safety factor（2.0×2.0=4.0x → 2.0x）
- ✅ `_fused_logistic`：用 `_softplus(eta)` 替代 15 行重复代码
- ✅ `fista_bb_solver`：循环内常量提升到循环外
- ✅ Lipschitz safety factor 已统一到 loss 类属性（`_lipschitz_safety`），solver 通过 `getattr(loss, '_lipschitz_safety', 1.0)` 读取
- ✅ `torch_compile_supported()` 失败时返回 `False`（避免不必要的编译尝试）
- ✅ `_logistic_sparse_cv_path` 验证路径添加 `Xv is not None` guard
- `_array_ops.py` 与 `_utils.py` helper 重复：`_xp_copy`/`_xp_zeros`/`_xp_asarray`/`_xp_eye`（自动推断 backend）与 `xp_copy`/`xp_zeros`/`xp_asarray`/`xp_eye`（显式传 `xp`）功能重叠，应统一为一套 API

---

## Code Review Round 4 建议项（2026-06-11，最后更新: 2026-06-11）

### 已修复

- ✅ `_irls_cd` step-halving 目标函数漏掉 intercept（`X_work[:, :p]` → `X_work`）
- ✅ `_irls_cd_gpu` `n_iter = it + 1` 未定义 guard
- ✅ `_torch_compile_supported()` 重复（移到 `_utils.py`）
- ✅ `_solver.py` 魔法数字提取为命名常量（`_SLACK_TOLERANCE`、`_DIVERGE_COEF_NORM_CAP` 等）

### 需大重构的项（后续 PR）

- `_fit_cpu`/`_fit_gpu`/`_fit_torch` 代码重复（~80% 逻辑相同）：需提取共享 FISTA 循环逻辑
- `_irls_cd` 和 `_irls_cd_gpu` 近似重复（同一算法的 CPU/GPU 版本）：需统一为 backend-agnostic 实现
- ✅ 添加新 loss 已简化：只需实现 loss class 的 3 个 per-sample 方法 + 注册到 `_base.py` registry。solver 和 CV 自动派生。
- `fista_solver` 400 行、`fista_lla_path` 550 行、`fista_bb_solver` 470 行：需拆分为小函数
- ✅ loss formula 已统一为单一 registry：每个 loss 类实现 `per_sample_value`/`per_sample_gradient`/`_mu_from_eta`，base class 派生 `value()`/`gradient()`/`fused_value_and_gradient()`。添加新 loss 只需实现 3 个 per-sample 方法 + 注册。

---

## 性能优化项（2026-06-11）

> 基于 n=1000 p=500 的 profiling 数据（Tesla P100 CPU 路径）

### 瓶颈分析

| 组件 | 耗时/iter | 占比 | 说明 |
|------|----------|------|------|
| matvec (X@coef + X'@resid) | ~1.3ms | 85% | BLAS-2 操作，CPU 已优化 |
| Lipschitz 重计算 | ~0.8ms (amortized) | 5% | `_max_eigval_power` 每 5 iter 调用 |
| per_sample loss/grad | <0.2ms | <5% | sigmoid/exp 已优化 |
| penalty proximal | <0.05ms | <5% | `_soft_threshold` 已融合 |

### 可优化项

- **Lipschitz 缓存**：当 IRLS 权重 W 变化小于阈值时跳过重计算。当前每 5 次迭代强制重计算，可改为变化检测
- **GPU 加速**：matvec 是 BLAS-2 操作，GPU 上 O(n*p) 的 matvec 比 CPU 快 10-100x。当前 FISTA solver 在 CPU 上运行，GPU 路径使用 `_use_gpu_loop` 但仅限 non-smooth penalties
- ✅ **`_fused_*` 函数清理**：已简化 dispatch 逻辑，`fista_solver` 和 `fista_lla_path` 统一使用 `_fused_glm_value_and_gradient`，legacy registry 函数保留为 fallback
- ✅ **Lipschitz 缓存**：已添加相对系数变化检测，变化 <0.1% 时跳过 eigvalsh 重计算

