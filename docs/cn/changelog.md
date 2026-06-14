# Changelog

> 语言：中文  
> 最后更新：2026-06-14  
> 页面定位：变更记录  
> 切换：[English](en/changelog.md)

语言切换：[English](en/changelog.md)

## 2026-06

### 新增 (2026-06-13 ~ 2026-06-14)

> PR #55~#58 由原始 PR #36（GLM+Penalty 完整模块）拆分而来。PR #36 实现了完整的 GLM + 惩罚系统，在完整矩阵基准测试中达到 1043/1043 ALL PASS (100%)。

- **PR #55 — 核心 GLM 求解器、后端、惩罚、推断 (PR-A, 来自 PR #36)**:
  - 7 个 GLM 族：squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
  - 10 个惩罚：none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
  - 6 个求解器：irls, fista, fista_bb, admm, lbfgs, newton — 按族+惩罚组合调度
  - 3 个后端：NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) 自动设备选择
  - 统一推断：15 个分布、p 值校正、bootstrap、permutation test
  - 关键技术：LLA 路由处理非凸惩罚（SCAD/MCP）、对数链接 GLM 增广截距、迭代相关 Lipschitz 计算、损失+梯度核融合

- **PR #56 — 惩罚模型 + CV 框架 (PR-B, 来自 PR #36)**:
  - 7 个惩罚估计量：PenalizedLinearRegression, PenalizedLogisticRegression, PenalizedPoissonRegression, PenalizedGammaRegression, PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
  - PenalizedGLM_CV：完整 CV（7 族 × 10 惩罚 × 6 求解器）
  - Lasso, Ridge, ElasticNet 完整推断
  - LogisticRegression, LinearRegression GPU 支持

- **PR #57 — 新模块 (PR-C, 来自 PR #36)**:
  - ANOVA：`f_oneway` — GPU 加速单因素方差分析，支持 float32/float64
  - 协方差：`EmpiricalCovariance`, `LedoitWolf`, `OAS` — 收缩协方差估计
  - 面板数据：`PanelOLS`（单/双向固定效应）, `RandomEffects`（Swamy-Arora）, `PanelSummary`, 聚类协方差
  - 样条：`bspline_basis`, `natural_cubic_spline_basis`, 惩罚回归 + GCV
  - 半参数：`GAM`（惩罚 B 样条 + GCV 平滑参数选择）
  - 核方法：`KernelRidge`, `KernelRidgeCV`, 6 个核函数（rbf, polynomial, linear, laplacian, sigmoid, cosine）

- **PR #58 — 基础设施、导出、向后兼容 (PR-D, 来自 PR #36)**:
  - 统一 `statgpu/__init__.py` 导出（~60 个公共名称）
  - `BaseEstimator` 设备管理 + sklearn 兼容 `get_params`/`set_params`
  - `Device` 枚举（CPU/CUDA/TORCH/AUTO）自动检测
  - `kernel_methods/` 和 `splines/` 旧路径向后兼容

- **PR #48 — 模块重组**:
  - 将 kernel_methods/ 和 splines/ 移至 nonparametric/ 子包
  - 创建 kernel_smoothing/ 子包用于 KDE + 核回归
  - 将 GAM 提取到 semiparametric/ 包
  - 旧导入路径向后兼容

- **PR #59 — 文档、changelog、指南 (PR-E)**:
  - 所有新模块的完整模型文档
  - 更新 docs/en/ 和 docs/cn/ 索引

- **PR #60, #61 — README 清理**:
  - 用表格清理 README 实现方法
  - 压缩 README GLM 部分 + 去除冗余

- **PR #62 — Dev 文件夹重组**:
  - 归档 241 个旧/临时文件到 _archive/
  - 更新 remote_config.py：环境变量优先于本地配置

- **PR #63 — Dev 工作区文档**:
  - 新增 dev/README.md（目录结构、远程 GPU 测试配置）
  - 新增 dev/tests/TESTING.md（测试分类、远程工作流）
  - 新增 dev/benchmarks/RESULTS.md（GPU 加速数据、版本历史）
  - 新增 dev/design/ARCHITECTURE.md（后端抽象、GLM 求解器架构）

- **PR #64 — 计划和 changelog 更新**:
  - 重组根文件（USAGE.md → docs/, AGENTS.md → dev/, plans → dev/plans/）
  - TO_DO.md 添加模块完成度百分比
  - 更新 plan 文件实现状态

- **GPU 性能：Async FISTA (v22e)**:
  - 消除 FISTA 循环中每次迭代的 GPU->CPU 同步
  - logistic + L1: 2.22x → **5.41x**（n=5000, p=500）
  - logistic + ElasticNet: 2.18x → **5.17x**
  - Poisson + L1: 1.90x → **4.55x**
  - 小规模：logistic + Adaptive L1 现在超过 CPU（0.56x → **1.12x**）

- **GPU 性能：v23c 完整矩阵（1043/1043 全通过）**:
  - 7 族 × 13 惩罚 × 5 求解器 × 3 后端
  - L-BFGS 融合惩罚梯度修复
  - Section A 计时：CPU 平均 953ms/3995ms/2875ms，Torch n=5000: **2.19x** 加速
  - Section B: 13/13 vs sklearn 全通过
  - Section D: 68/68 vs statsmodels 全通过
  - Section E: 146/146 跨求解器全通过
  - 报告：`dev/tests/_bench_v23c_report.md`

### 修复 (2026-06-10 ~ 2026-06-12)

- **PR #49 Code Review: 110+ fixes across 16 files**:
  - 修复 26 个 P1 bug（merge conflict、NameError、数值公式错误、GPU 路径崩溃等）
  - 修复 55 个 P2 bug（缓存线程安全、后端一致性、边界情况、接口兼容性等）
  - 修复 ~30 个 P3 改进项（死代码清理、magic numbers、性能优化等）
  - 新增 428 个测试用例（远程 GPU Tesla P100 全部通过）
  - 三端精度偏差 < 0.02%（同一 random_state 下）
  - 性能无回退（RidgeCV CuPy 6.8x 加速，PenalizedGLM_CV Torch 3.1x 加速）
  - 删除 ~1300 行死代码
  - 统一 `best_score_` 为负 MSE（sklearn 惯例）

### 新增 (2026-06-07 ~ 2026-06-09)

- **PR #50 — GLM 稀疏 CV 路径添加 val_sample_weight**:
  - 稀疏 GLM 交叉验证的验证样本权重支持
  - 支持不平衡数据集的加权 CV 折

- **PR #53 — 修复加权 Ridge 推断**:
  - 修复加权 Ridge 回归的尺度计算
  - 保留 sample weights 下的 bse/pvalues/conf_int

- **PR #54 — 重构 CV 调度表**:
  - _compute_cv_scores 调度表
  - 更清晰的 CV 评分散逻辑分离

### 优化 (2026-06-05)

- **Strict sparse GLM CV GPU 优化**:
  - 减少 `fista_bb_solver` CV 路径中的 GPU 同步
  - 批量 CuPy 验证评分
  - Torch fold-batched strict logistic sparse CV 路径
  - CuPy fold-batched strict logistic sparse CV 路径
  - 远程 P100 验证：所有 alpha 选择与 CPU 匹配

### 新增 (2026-06-04)

- **Strict-first PenalizedGLM_CV 策略控制**:
  - `PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`
  - 两阶段 CV 使用松弛筛选、严格候选优化
  - 新增 `ApproximateCVWarning`、CV 诊断

### 修复 (2026-06-04)

- **Poisson 稀疏 PenalizedGLM_CV 跨后端精度**:
  - Strict GPU FISTA 不再使用异步 CV-only 更新循环
  - Poisson L1/ElasticNet CV 使用确定性近平局规则

### 优化 (2026-06-04)

- **小规模 sparse CV GPU 传输优化**:
  - squared-error sparse CV 跳过不必要的 coefficient-path 主机传输
  - `squared_error+l1` strict CV 从 820ms 降至 190ms（CuPy）

- **GPU sparse GLM CV 求解器策略**:
  - `solver="auto"` 使用后端感知的 strict-CV 选择

### 优化 (2026-06-01)

- **后端传输 helper 和基准测试解析器**:
  - CuPy <-> Torch CUDA 转换优先使用 DLPack 零拷贝
  - NumPy -> Torch CUDA 传输尝试 pinned host memory

## 2026-05

### 新增 (2026-05-03 ~ 2026-05-11)

- **PR #27~#29 — 无监督学习 Phase 3/3B/3C**:
  - 新增 12 个估计量：PCA, KMeans, DBSCAN, GaussianMixture, NMF, AgglomerativeClustering, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF, IncrementalPCA, TruncatedSVD
  - 凝聚聚类 GPU 精确路径
  - 所有估计量的文档和验证基准测试

- **PR #30, #32 — 凝聚聚类 GPU 精确路径**:
  - GPU 加速精确 linkage（所有距离度量）

- **PR #33 — 非参数模块审查**:
  - KDE GPU 内存修复
  - 带宽选择 GPU 化
  - Log-sum-exp 数值稳定性修复

- **PR #34, #35 — 文档**:
  - 明确运行时设备选择
  - 明确 Torch 后端文档

### 新增 (2026-05-24 ~ 2026-05-29)

- **PR #37 — GLM 惩罚正确性 + 自动 GPU 路由**:
  - 修复惩罚 GLM predict() 返回逆链接均值尺度预测
  - 基于问题规模的惩罚模型自动 GPU 路由

- **PR #38 — Gamma 逆幂 FISTA**:
  - 链接感知 Gamma FISTA 支持（CPU/CuPy/Torch）

- **PR #39~#42 — GLM 求解器重构**:
  - 修复 GLM GPU dtype 和审查回归
  - 重构 GLM 求解器后端 helper

- **PR #43, #44 — 线性推断结果修复**:
  - 重构线性推断结果容器

- **PR #47 — CuPy cummin/cummax 修复**:
  - 修复 CuPy cummin/cummax CUDA 核在非连续数组上的问题
  - adjust_pvalues BH/BY/Hochberg 现在返回正确结果

### 修复 (2026-05-20)

- **v23c: L-BFGS fused penalty gradient 修复**:
  - L-BFGS 收敛到无正则化解而非 loss_grad + α·coef = 0
  - 1043/1043 全通过

### 优化 (2026-05-20)

- **v22g: Async FISTA 与 GPU 优化**:
  - Async FISTA: GLM+非光滑惩罚在 n=5000 时 2-5.5x 加速
  - CuPy/Torch 后端 GPU sync 优化

- **v23c: 完整矩阵基准测试 (1043 tests)**:
  - 7 families x 10 penalties x 3 scales x 多求解器 x 3 backends

## 2026-04

### 新增 (2026-04-26)

- **PR #24 — 精度修复、hochberg/stouffer、包重组**:
  - Ordered 模型跨后端精度修复
  - 新增 hochberg (adjust_pvalues) + stouffer (combine_pvalues) 跨 3 后端
  - 包结构审计与重组

- **PR #26 — README 刷新**:
  - 重组功能、添加模型、推荐可编辑安装
  - 导出 combine_pvalues

### 新增 (2026-04-21)

- **PR #19 — Cython Efron 优化**:
  - Cython 优化 Efron 梯度和 Hessian 计算
  - CoxPH 精度和运行时综合基准测试

- **PR #21 — 分布后端统一**:
  - 将分布模块合并为单一 `_distributions_backend.py`
  - 3 后端 15 个分布

- **PR #22 — 后端工具整合**:
  - 整合重复的后端工具函数

- **CoxPHCV 从接口骨架升级为可训练版本**:
  - 已实现 penalty 网格搜索（K-fold）与最佳 penalty 全量重训

- **RidgeCV 和 LogisticRegressionCV 完整实现**:
  - 从接口骨架升级为完整功能实现

### 新增 (2026-04-20)

- **PR #18 — 远程配置 + 后端增强**:
  - 移除硬编码 SSH 凭据
  - 新增远程配置模块

- **PR #20 — CoxPHCV CuPy 优化**:
  - 优化 CoxPHCV CuPy Hessian 路径

- **CoxPH Efron 实现修复**:
  - 修复数值溢出，添加 clipping 保护
  - statgpu-Torch GPU 实现 **15.44x** 加速（vs statsmodels）

### 新增 (2026-04-18)

- **PR #16 — Torch 后端支持**:
  - 全面 PyTorch 后端集成
  - 与 NumPy 和 CuPy 后端功能对齐

- **PR #17 — Elastic Net 实现**:
  - 新增 `ElasticNet` 类，结合 L1 和 L2 正则化
  - 最大加速：**4.36x** vs sklearn（n=100k, p=500）

- **PyTorch 后端修复**:
  - 修复多个 Torch 兼容性问题
  - 数值精度：系数与 NumPy 差异 < 1e-14

### 新增 (2026-04-17)

- **PyTorch 后端（Phase 1-5 完成）**:
  - 使用 PyTorch 2.0+ 的新 GPU 后端替代方案
  - Ridge, LogisticRegression, Lasso, CoxPH 全部支持
  - 大规模性能：Ridge HC3 Torch 0.067s vs CuPy 0.064s

### 新增 (2026-04-15)

- Knockoff 特征选择 API
- Lasso 推断重命名
- 所有模型的 `gpu_memory_cleanup`
- 稳健协方差：所有模型支持 `nonrobust/hc0/hc1/hc2/hc3/hac`
- CV 估计量接口骨架：RidgeCV, LogisticRegressionCV, CoxPHCV
- 非参数导出和 API 覆盖

### 验证

- 与 `statsmodels` 的稳健协方差一致性测试
- Cox 与 `statsmodels.PHReg` 一致性检查
- 非参数验证：KDE, KernelRegression

### 改进

- LinearRegression CPU HAC 路径使用自适应精度选择
- 核回归 local-linear 多维路径使用批量向量化求解
- KDE 1D Numba 快速路径改进
