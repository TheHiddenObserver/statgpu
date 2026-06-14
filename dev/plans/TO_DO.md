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
| **linear_model/** | ~90% | Ridge, Lasso, ElasticNet, Logistic, 7 GLM, Penalized, Ordered, CV | multinomial, sparse input |
| **glm_core/** | ~85% | 6 solvers, 7 families, 5 links | solver 拆分优化 |
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

- [ ] `_penalized_cv.py` 文件拆分 (2800+ 行)
- [ ] `_solver.py` 函数拆分 (fista_bb_solver 470 行)
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
