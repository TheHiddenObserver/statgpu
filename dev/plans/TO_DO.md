# statgpu TO DO

> Primary planning document. Last updated: 2026-06-14.
> See also `archive/PLAN_UNIFIED.md` for historical context.

## 开发门禁（必须遵守）

### 功能门禁

- 每次新增功能，必须同时提供：NumPy (CPU)、CuPy (GPU)、Torch (GPU) 三条路径
- 每次新增统计功能后，必须补外部框架对标验证（statsmodels、sklearn、R）
- 外部对标时必须显式统一口径（同一特征集合、ties/solver、正则设置）

### 推断门禁

- Ridge/Lasso strict 模式必须通过外部对齐阈值：coef 1e-6, bse 1e-3, p-value 5e-2
- strict 失败策略：默认 raise error

### 设备一致性门禁

- strict 模式输出在 CPU/GPU 上对齐

### 工程门禁

- 每次提交：lint + type + test
- 每月稳定版：外部矩阵 + benchmark 非回退 + 文档同步

---

## 模块完成度 (2026-06-14)

| 模块 | 完成度 | 已实现 | 关键缺失 |
|------|--------|--------|----------|
| **linear_model/** | ~85% | Ridge, Lasso, ElasticNet, AdaptiveLasso, SCAD, MCP, Logistic, 7 GLM, Penalized, Ordered, CV | multinomial, sparse input, SCAD/MCP/AdaptiveLasso 推断, BIC 超参选择 |
| **glm_core/** | ~90% | 6 solvers, 7 families, 5 links | — |
| **penalties/** | ~95% | 12 penalties (L1/L2/EN/SCAD/MCP/Adaptive/Group) | — |
| **survival/** | ~40% | CoxPH (Breslow/Efron, robust SE, cluster) | strata, frailty, time-varying |
| **inference/** | ~80% | 15 distributions, p-value adjustment, bootstrap, permutation | — |
| **unsupervised/** | ~95% | 12 estimators (PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM...) | — |
| **kernel_methods/** | ~60% | 6 kernels, KernelRidge, KernelRidgeCV | Nystroem, KernelPCA, chi2 |
| **panel/** | ~45% | PanelOLS, RandomEffects, clustered SE | FamaMacBeth, HAC, IV |
| **splines/** | ~35% | bspline, natural_cubic, penalized regression, GAM | SplineTransformer API |
| **covariance/** | ~30% | EmpiricalCovariance, LedoitWolf, OAS | GraphicalLasso, MinCovDet |
| **anova/** | ~15% | f_oneway (单因素) | 二因素, 重复测量, 事后检验 |
| **nonparametric/** | ~70% | KDE, kernel regression, bandwidth selection | — |
| **feature_selection/** | ~80% | KnockoffSelector, StepwiseSelector | — |
| **metrics/** | ~60% | ROC, AUC, confusion matrix | — |
| **diagnostics/** | ~50% | RegressionDiagnostics | — |

---

## 待完成项

### P0: 进行中

- [ ] 完善推断严谨性：跨设备一致性（SE/t/z/p/CI、AIC/BIC/LLF）
- [ ] CoxPH Cython 编译版本调试（当前使用 Python fallback）

### P1: API parity / 功能补齐

- [ ] LogisticRegression: multinomial/softmax, L1/elastic-net
- [ ] CoxPH: strata, frailty, time-varying covariates
- [ ] 稀疏输入支持：CSR/CSC
- [ ] CoxPHCV 完整实现（当前为骨架）
- [ ] Ridge: warm_start, alpha path

### P1: Penalized Regression 推断与超参选择

**背景**：Adaptive Lasso、SCAD、MCP 三种非凸惩罚均具有 oracle property，其非零系数的估计量具有渐近正态分布。当前 statgpu 只对 L1/ElasticNet 实现了 debiased inference，Adaptive Lasso/SCAD/MCP 的 `compute_inference=True` 会报错。

**文献基础**：

| 惩罚 | Oracle Property 论文 | 渐近分布 | BIC 选择 |
|---|---|---|---|
| Adaptive Lasso | Zou (2006) JASA | √n(β̂ - β₀) → N(0, Σ) for true non-zero | Wang, Li & Tsai (2007) Biometrika |
| SCAD | Fan & Li (2001) JASA | 同上，oracle asymptotic normality | Wang, Li & Tsai (2007) Biometrika |
| MCP | Zhang (2010) AOS | 同上 | Zhang (2010) AOS, Wang & Zhu (2011) |

**实现方案**：

- [ ] **Adaptive Lasso/SCAD/MCP 的 oracle 推断**
  - 基于 oracle property：对选中的非零系数，用 unpenalized OLS/GLM 重新拟合，得到无偏估计和标准误
  - 实现方式：`inference_method="oracle"` — 先 fit penalized model 得到 support set，再对 support set 做 unpenalized fit
  - 参考：Fan & Li (2001) Section 4, Zou (2006) Theorem 2
  - 输出：coef (无偏), bse, z, p_value, conf_int for selected variables
  - 注意：只对 non-zero 系数有效，zero 系数不做推断

- [ ] **Debiased SCAD/MCP 推断**
  - 类似 debiased Lasso (van de Geer et al. 2014)，构造 bias-corrected estimator
  - β̂_debias = β̂ + M(X'X/n)(y - Xβ̂)/n
  - 适用于 high-dimensional (p > n) 场景
  - 参考：Zheng, Gallagher & Kulasekera (2018) 对 SCAD/MCP 的 debiased 扩展

- [ ] **BIC 超参选择（alpha 网格搜索）**
  - 标准 BIC：BIC(λ) = n·log(RSS/n) + df(λ)·log(n)
  - df(λ) = 非零系数个数（对 SCAD/MCP/Adaptive Lasso）
  - EBIC（高维）：BIC(λ) = n·log(RSS/n) + df(λ)·log(n) + 2·γ·log(C(p, df))
  - 适用：选择 alpha（正则化强度）和 l1_ratio（ElasticNet）、a（SCAD）、gamma（MCP）
  - 参考：Wang, Li & Tsai (2007) Biometrika — 证明 BIC 对 SCAD 具有选择一致性
  - 实现：`PenalizedGLM_CV` 新增 `scoring="bic"` 选项，或独立的 `BICSelector` 类

**优先级**：
1. BIC 超参选择（最实用，用户不需要跑完整 CV）
2. Oracle 推断（对选中变量做 unpenalized refit，实现简单）
3. Debiased 推断（high-dimensional 场景，实现复杂度高）

### P2: SCAD/MCP GPU 性能优化

- [ ] LLA 内层循环 fused kernel：将 `fused_value_and_gradient + proximal + momentum` 合并为单个 kernel，减少 kernel launch 次数
- [ ] 参考 skglm 的 working set 策略：只优化活跃特征子集，减少计算量
- [ ] 参考 skglm 的 Anderson 加速：加速不动点迭代收敛

### P2: Penalized 子类 API 完善

- [ ] `_penalized_poisson.py` / `_penalized_gamma.py` / `_penalized_inverse_gaussian.py` / `_penalized_negative_binomial.py` / `_penalized_tweedie.py`：补充缺失的构造函数参数（`cpu_solver`, `lipschitz_L`, `gpu_memory_cleanup`, `inference_method`, `stopping`, `lla`, `max_lla_iters`, `lla_tol`），当前这些参数只能通过父类默认值使用
- [ ] `_penalized_logistic.py`：补充 `inference_method` 参数（当前只有 `compute_inference`，无法选择 debiased/bootstrap）
- [ ] `_predict_mixin.py` `score()` 与 `predict()` 的设备解析逻辑不一致：`score()` 用 `_get_compute_device()`，`predict()` 用 `_prediction_backend_name()`，应统一
- [ ] `_predict_mixin.py` `_prepare_predict_X()` 强制转 numpy 导致 GPU→CPU→GPU 不必要的往返
- [ ] `_fit_mixin.py` `_fit_irls_backend()` 中首次 `init_coef` 创建后立即被覆盖（dead allocation）
- [ ] `_fit_mixin.py` `_fit_cpu()` 中 `solver_name == "admm"` 分支存在 dead code

### P2: 新模块扩展

**anova/** (15% → 目标 60%):
- [ ] 二因素 ANOVA (with/without interaction)
- [ ] Welch ANOVA (unequal variances)
- [ ] 事后检验: Tukey HSD, Bonferroni
- [ ] 效果量: Cohen's f, partial eta-squared

**covariance/** (30% → 目标 60%):
- [ ] GraphicalLasso / GraphicalLassoCV (稀疏逆协方差)
- [ ] MinCovDet (稳健估计)
- [ ] ShrunkCovariance (通用收缩)

**panel/** (45% → 目标 70%):
- [ ] FamaMacBeth
- [ ] HAC/Newey-West 协方差
- [ ] PooledOLS, BetweenOLS, FirstDifferenceOLS

**splines/** (35% → 目标 60%):
- [ ] sklearn SplineTransformer API (fit/transform)
- [ ] 循环样条 (cyclic cubic)
- [ ] 薄板样条 (thin plate)

**kernel_methods/** (60% → 目标 80%):
- [ ] Nystroem 近似
- [ ] KernelPCA
- [ ] chi2_kernel

### P3: 大规模重构

- [x] `_penalized.py` Mixin 拆分 → `penalized/_base.py` + `_fit_mixin.py` + `_inference_mixin.py` + `_predict_mixin.py` ✅
- [x] `_solver.py` 拆分 → `solvers/` 顶级模块 (6 solver 独立文件) ✅
- [x] `_cv_base.py` / `_cv_engine.py` 提取 → `cross_validation/` ✅
- [x] wrapper 整理 → `linear_model/wrappers/` (13 个文件) ✅
- [x] CV wrapper 整理 → `linear_model/cv/` (4 个文件) ✅
- [x] legacy 文件整理 → `linear_model/legacy/` ✅
- [ ] `_fit_cpu`/`_fit_gpu`/`_fit_torch` 代码重复消除
- [ ] `_irls_cd` 和 `_irls_cd_gpu` 统一为 backend-agnostic 实现
- [ ] `_penalized_cv.py` 6 个 FISTA 循环提取为共享 `_fista_cv_step`

### P3: 性能优化

- [ ] Panel 双向 demeaning 批量化（减少 GPU kernel launch）
- [ ] KernelRidgeCV CuPy 路径实现（当前回退到 NumPy）
- [ ] 加权 CV 快速路径

### P3: 代码质量

- [ ] `_array_ops.py` 与 `_utils.py` helper 统一（`_xp_copy`/`xp_copy` 等重复）
- [ ] `_solver.py` 标量提取模式统一（4 种不同方式）
- [ ] `_solver.py` 异常捕获收窄（已部分完成）
- [x] Panel summary() 返回 PanelSummary 结构化对象 ✅
- [x] PanelOLS.predict() 包含固定效应 (entity_ids/time_ids) ✅
- [x] ANOVA float32 支持 (dtype 参数) ✅

---

## 已完成历史 (2026-04 ~ 2026-06)

> 详细记录见 `archive/PLAN_UNIFIED.md` 和 git history。

- RidgeCV / LogisticRegressionCV 完整实现 (2026-04-21)
- CoxPH C-index / Efron ties 修复 (2026-04-20)
- 完整推断体系：LinearRegression / Ridge / Logistic / CoxPH (HC0-HC3/HAC)
- 12 个 Unsupervised estimator
- 5 个新模块：ANOVA, Covariance, Kernel Methods, Panel Data, Splines/GAM
- PR #49: 110+ bug fixes, 428 tests
- PR #48: Panel, ANOVA, Covariance review fixes
- Async FISTA (v22e): 最高 5.41x 加速
- v23c: 1043/1043 ALL PASS
