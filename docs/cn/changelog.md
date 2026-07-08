# Changelog

> 语言：中文  
> 最后更新：2026-07-07  
> 页面定位：变更记录  
> 切换：[English](en/changelog.md)

语言切换：[English](en/changelog.md)

## 2026-07

### 新增 (2026-07-07)

- **统一推断框架 — Loss × Penalty Sandwich 引擎**:
  - 新模块 `statgpu/inference/_sandwich.py`：`compute_bread_avg`、`compute_meat_avg`、
    `assemble_cov_avg`、`m_estimation_inference` — 平均尺度 M-estimation sandwich，
    支持 `nonrobust`（模型协方差 φ·H⁻¹/n）和 `hc0`/`hc1`（稳健 sandwich
    H⁻¹·J·H⁻¹/n），适用于所有具有 Hessian 的损失函数
  - 新模块 `statgpu/inference/_dispersion.py`：`glm_pearson_dispersion` 用于非典型连接
    GLM（Gamma、IG、Tweedie），`robust_scale_dispersion` 用于 M-估计器
  - **Expected Fisher 接口**：`loss.fisher_information(X, coef, sample_weight)`
    添加到 LossBase（默认 `NotImplementedError`）；在 `GammaLoss`
    （log-link W=1, inverse_power W=1/η²）和 `TweedieLoss`（log-link W=μ^(2-p)）上实现
  - **Penalty 曲率 API**：`Penalty.curvature_diag(coef)` — 返回 P'' 对角线，
    默认 zeros；`L2Penalty` 覆写为 `α·ones`。SCAD/MCP 抛出 `NotImplementedError`
  - **惩罚推断路由**（`penalized/_inference_mixin.py` +256 行）：
    - 具有 Hessian 的损失 + L2/ElasticNet 惩罚 → sandwich
    - SCAD/MCP → oracle active-set refit（Fan & Li 2001，以选中模型为条件）
    - Bootstrap 推断入口（分阶段推出）
  - **GLM 推断管线**（`_glm_base.py` +300 行）：
    - `GeneralizedLinearModel` 上的 `compute_inference`、`cov_type` 参数
    - `_compute_inference()` 读取拟合时元数据（惩罚、求解器、目标尺度）
    - 对齐设计矩阵：intercept 第一列，匹配 statsmodels `sm.add_constant(X, prepend=True)`
    - GLM 基类上的 `summary()`、`aic`、`bic`、`loglikelihood` 属性
  - **Loss 基元**（`losses/_base.py` +79 行）：
    - `per_sample_score(X, y, coef)` — (n, p) 逐样本得分，用于 HC2/HC3/HAC
    - `score_outer(X, y, coef, sample_weight=None)` — 内存高效得分外积，
      含 w_i² 分析权重缩放用于 sandwich meat
  - **GLM wrapper 暴露**：`PoissonRegression`、`GammaRegression`、
    `InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression`
    均暴露 `compute_inference`、`cov_type`
  - **QuantileRegression**（`wrappers/_quantile.py` +329 行）：独立类，
    含 kernel 推断（Powell 1991, Epanechnikov 核 + Hall-Sheather 带宽）
    和 bootstrap 推断
  - 涉及文件：`statgpu/inference/_sandwich.py`、`_dispersion.py`；`statgpu/losses/_base.py`；
    `statgpu/penalties/_base.py`、`_l2.py`；`statgpu/glm_core/_gamma.py`、`_tweedie.py`；
    `statgpu/linear_model/_glm_base.py`、`penalized/_base.py`、`penalized/_inference_mixin.py`；
    `wrappers/_poisson.py`、`_gamma.py`、`_inverse_gaussian.py`、`_negative_binomial.py`、
    `_tweedie.py`、`_quantile.py`；`_ordered_logit.py`、`_ordered_probit.py`

- **Loss × Penalty × Solver 框架指南**（中英文）:
  - 新文档：`docs/en/guides/loss-penalty-solver-framework.md` (+206)、
    `docs/cn/guides/loss-penalty-solver-framework.md` (+205)
  - 完整调度逻辑：12 损失 × 10 惩罚 × 10 求解器
  - 自动求解器选择规则、惩罚约束、求解器-惩罚矩阵

- **有序 Logit/Probit — Newton-Raphson + 解析 Hessian + 推断**:
  - 三端全部从 L-BFGS 替换为 Newton-Raphson + 信赖域优化
    - NumPy: 向量化解析 Hessian + `numpy.linalg.solve`
    - CuPy: 原生 GPU Newton-Raphson，logit 下零 CPU 往返
    - Torch: 原生 `torch.linalg.solve`，正确设备/dtype 处理
  - 典型问题 5–23 次迭代收敛；信赖域内层循环（每次迭代最多 20 次 ridge 尝试）保证 NLL 下降
  - 标准化：X 内部标准化，收敛后系数和阈值转回原始尺度
    （`β_raw = β_fit / X_std`, `θ_raw = θ_fit + X_mean @ β_raw`）
  - 修改文件：`statgpu/linear_model/_glm_base.py`（重大重写）
    - 删除方法：`_ordered_nll_grad_fn`, `_ordered_gradient_vec`（死代码）
    - 新增方法：`_ordered_hessian_analytical`, `_compute_ordered_inference`,
      `_ordered_F_and_f`, `_ordered_gradient_torch`
    - 重写方法：`_fit_scipy_ordered`, `_fit_cupy_ordered`,
      `_fit_torch_ordered`, `_ordered_category_probs`, `predict_proba`
  - 修改文件：`statgpu/glm_core/_gamma.py`, `statgpu/inference/_sandwich.py`
    （设备感知张量创建修复）

- **有序模型推断** (`compute_inference=True`):
  - MLE 处的解析观测 Hessian，分块结构（β-β, β-θ, θ-θ），与 R `MASS::polr`
    和 `ordinal::clm` 一致
  - 标准误：`sqrt(diag(H^{-1}))`；Wald z 统计量，双侧 p 值，95% 置信区间
  - 独立属性组：`_bse_ordered`/`_pvalues_ordered`（系数）
    和 `_bse_thresholds`/`_pvalues_thresholds`（阈值）
  - `loglikelihood`、`aic`、`bic` 属性
  - `summary()` 方法通过 `ParameterInferenceResult`
  - GPU 守卫：显式 `device='cuda'` 或 `device='torch'` 抛出
    `NotImplementedError`（不静默回退 CPU）
  - 当前限制：仅 `cov_type='nonrobust'`、不支持 `sample_weight`

### 修复 (2026-07-07)

- **有序模型 9 项 Bug 修复**（来自 code review）:
  - **probit 梯度**: `_ordered_gradient_torch` 硬编码 `torch.sigmoid`；
    修复为使用 `_ordered_link_derivative(family)` 正确派发 probit
  - **probit f'(z)**: `_compute_ordered_inference` 丢弃正确的 probit f'(z)
    并用 logit 公式重新计算；修复为直接使用返回值
  - **GPU 静默回退**: `_to_numpy()` 静默将 GPU 数组转为 CPU；
    添加 `_resolve_backend` 守卫抛出 `NotImplementedError`
  - **pinv 降级**: `np.linalg.pinv` 在奇异 Hessian 时静默降级推断；
    替换为 `LinAlgError` 抛出
  - **predict_proba 双重除法**: `X_scaled @ coef` 中两者均已缩放
    （coef 已除以 `X_std`）；修复为原始尺度 `X @ coef`
  - **y.dtype 守卫**: `_fit_torch_ordered` 未处理非 int64 的 torch 张量；
    添加 `elif y.dtype != torch.int64` 检查
  - **死代码**: 删除 `_ordered_nll_grad_fn` 和 `_ordered_gradient_vec`
    （约 70 行，零调用者）
  - **loglikelihood**: 有序模型 `loglikelihood` 返回 `nan` 因为
    `_loss`/`_X_design` 未设置；添加 `_final_nll` 存储 + 属性覆盖

### 改进 (2026-07-07)

- **有序模型文档**（中英文）：完整重写，包含 Newton-Raphson 算法、
  解析 Hessian、推断 API、参数表、CPU+GPU 示例、strict vs approximate、
  外部验证和当前限制
- **文档文件**：`docs/en/models/ordered.md`，`docs/cn/models/ordered.md`

### 验证 (2026-07-07)

- 三端有序 logit benchmark：NumPy vs CuPy vs Torch 单步 Hessian 差异
  在机器精度级别（~1e-14）；24 轮迭代累积 BSE 差异 ~4.5e-04，
  源于数学库差异（`libm` vs NVIDIA `libdevice`）
- R `ordinal::clm` 对比：NLL 一致，相同解析 Hessian 结构
- 所有现有有序模型测试通过（4/4 CPU，6 GPU 跳过）

## 2026-06

### 新增 (2026-06-28) — PR #73

- **Loss 架构重构 — LossBase 提取**：
  - 从 `GLMLoss` 提取 `LossBase` 基类，用于 quantile/robust/survival 损失
  - `LossBase`：抽象基类，`per_sample_value()`、`per_sample_gradient()` 为唯一真实来源；自动派生 `value()`、`gradient()`、`fused_value_and_gradient()`
  - `GLMLoss` 继承 `LossBase`，保留 GLM 特有功能（canonical link、IRLS）
  - 新增损失类：`QuantileLoss`、`HuberLoss`、`BisquareLoss`、`CoxPartialLikelihoodLoss`
  - 新增模块：`PenalizedQuantileRegression`、`PenalizedRobustRegression`、`PenalizedCoxPHModel`

- **Proximal IRLS-CD 求解器**：quantile + SCAD/MCP 的新求解器
  - 算法：IRLS 二次上界逼近 + LLA 非凸惩罚 + 并行对角化
  - CPU（numpy）：比 FISTA-LLA 快 ~3 倍（60-120 次迭代 vs 1800+）
  - GPU（torch-CUDA）：大规模问题（n=10K, p=500）比 CPU numpy 快 ~36 倍
  - 三端支持：numpy、cupy、torch — 核心数组操作 GPU 原生；标量收敛检查同步到 Host
  - Benchmark 产物：`results/loss_functions_bench_2026-06-23.json`、`results/penalized_glm_bench_2026-06-22.json`

- **CoxPH Efron 优化**：
  - 向量化 Efron：基于前缀和的梯度/Hessian 计算（无 Python 循环）
  - 多块 CUDA kernel：Efron 的 fused loglik+grad+hess
  - DLPack 桥接：torch-CUDA 通过 DLPack 使用 CuPy Efron kernel
  - 性能：n=5000 时比 statsmodels 快 3-6 倍；GPU 比 CPU 快 6 倍
  - 移除 Numba 依赖，纯 numpy 实现
  - Benchmark 产物：`results/coxph_efron_bench_2026-06-22.json`（精度对比 statsmodels，GPU 加速 47-102x）

- **GLM Fused Value+Gradient**：集成 `_fused.py` 到 `GLMLoss.fused_value_and_gradient()`

- **FISTA GPU 同步优化**：批量 GPU 同步（convergence+divergence+lipschitz 一次传输）

- **Quantile IRLS 求解器**：`QuantileLoss.irls()` 方法，光滑惩罚（L2）下 5-15 次迭代收敛

- **Huber Hessian 支持**：`has_hessian = True`，支持 proximal Newton（5-10 次迭代）

- **Bisquare + SCAD/MCP 修复**：alpha >= 0.1 时返回空活跃集的问题

- **重构**：
  - 提取 `_compute_lla_path()` 共享方法
  - `_NON_IRLS_LOSSES` → `_SPECIAL_LLA_LOSSES` 命名修正
  - `_cd_sweep_batch` → `_parallel_majorization_step` 命名修正
  - 新增 `_dispatch_irls()` 方法路由 IRLS 到正确后端

- **数值稳定性**：IRLS 权重钳制、SCAD 分母零保护、CoxPH Efron `inv_d1_sq` 钳制

- **Bug 修复**：
  - Group penalties cupy 兼容性和 device-aware cache
  - Huber `per_sample_value` 公式修正
  - Quantile IRLS 跳过 intercept 列惩罚
  - Proximal Newton 传递 `sample_weight`
  - DBSCAN `min_samples` off-by-one（sklearn 包含自身）
  - DBSCAN `indices/distances` 返回值顺序修正
  - DBSCAN GPU label propagation 改为收敛即停
  - NNDescent 排除自身候选
  - Cox C-index 排除 censored 短时间
  - CV scoring 传递 loss kwargs
  - ANOVA torch device mismatch

- **UMAP 稀疏图**：
  - 稠密 n×n 图构造改为稀疏 COO 边（O(n·k) 内存）
  - Spectral initialization 使用 `scipy.sparse.linalg.eigsh`
  - 优化循环和负采样使用 backend-native RNG
  - 负采样 RNG 从 `random_state` 种子化

- **NNDescent**：
  - 新增近似最近邻模块（numpy/torch/cupy）
  - 逐点候选集避免 O(n²) 退化
  - 修复收敛返回值顺序

- **Sample Weight 全局后端化**：
  - 统一 `sample_weight` 在 solver 入口转换为 backend-native
  - 防止 torch GPU 路径 CPU/CUDA mismatch
  - 影响：FISTA、FISTA-LLA、quantile IRLS、proximal Newton

- **GPU 收敛检查优化**：
  - IRLS-CD：在 device 上比较，只同步 bool 到 CPU
  - 降低 GPU 同步频率（每 5 次迭代）
  - 批量 GPU 同步

- **新增测试**：
  - CoxPH Efron reference parity test（vs statsmodels）
  - DBSCAN min_samples=1、高维路径、Cython fallback
  - Quantile SCAD objective parity（vs FISTA-LLA）
  - 三端交叉测试（numpy vs torch）
  - CuPy smoke tests
  - Weighted score test

### 新增 (2026-06-26)

- **无监督 Benchmark**：12 算法 × 3 后端，对比 sklearn
  - 最佳：TruncatedSVD 28.6x、IncrementalPCA 21.9x、DBSCAN 21.0x、NMF 19.9x

- **DBSCAN 优化**：
  - Cython `_dbscan_cy_fast.pyx`：`dbscan_labels_from_pairs` + `dbscan_labels_from_csr` — 全 pipeline 在 C 中运行
  - CPU：p≤12 用 cKDTree query_pairs + Cython（比 sklearn 快 3-4 倍）；p>12 用 sklearn BLAS + Cython CSR（与 sklearn 持平）
  - GPU（PyTorch CUDA）：全在设备上执行 — 距离、稀疏图、label propagation、border 分配，零 GPU→CPU 传输
  - GPU label propagation 用 `scatter_reduce_(amin)`，2-5 次迭代收敛
  - GPU（P100）：p=5 比 sklearn 快 **14-17 倍**，p=50 快 **3-4 倍**

- **UMAP 优化**：
  - 稀疏图 + 负采样（GPU 16.7x 加速）
  - GPU 原生 scatter-add（无 CPU 传输）
  - `nn_method` 参数支持 NNDescent

- **IncrementalPCA**：batch_size 默认改为 n（GPU 0.4x → 21.9x）
- **MiniBatchNMF**：自动 batch、HtH 预计算、同步节流（GPU 0.1x → 3.2x）

- **CuPyBackend**：补全 30+ 缺失方法（qr、svd、bool、zeros_like 等）
- **TorchBackend**：添加 qr、svd、solve
- **Backend Utils**：统一 `scatter_add_1d` 和 `scatter_add_2d`
- **构建系统**：合并 7 个 setup 文件为单一 `setup.py`

### 新增 (2026-06-24)

### 新增 (2026-06-24)

- **完整 Benchmark 套件**：
  - GLM Solver：7 family × 10 penalty × 7 solver × 3 backend（70 组合）
  - 新模块：Panel（8 estimator）、GAM、ANOVA（5 函数）— 3 backend × 3 规模
  - 无监督：12 算法 × 3 backend vs sklearn
  - 外部对比：statgpu vs linearmodels、pygam、scipy、sklearn

- **CuPyBackend**：补全 30+ 缺失方法（qr、svd、bool、zeros_like、solve、norm 等）
  - TruncatedSVD、IncrementalPCA、DBSCAN GPU backend 现可正常工作

- **TorchBackend**：添加 qr、svd、solve 方法

- **无监督优化**：
  - IncrementalPCA：batch_size 默认改为 n（GPU 0.4x → 21.1x）
  - MiniBatchNMF：batch 自动调整 + HtH 预计算 + 同步节流（GPU 0.1x → 3.2x）
  - UMAP：`nn_method` 参数（auto/exact/nndescent）、epoch 减少、float32 优化

- **ANOVA 修复**：
  - f_oneway：向量化 group 统计量（cupy 0.7x → 3.4x）
  - f_twoway：torch dtype 兼容性修复

- **Panel**：BetweenOLS 接受 `time_ids` 参数，API 一致性

- **GAM**：`knot_method`（quantile/uniform）和 `gamma` 参数，用于与 pygam 对齐

### 新增 (2026-06-19)

- **LossBase 架构** (Phase 1):
  - 从 `GLMLoss` 提取 `LossBase` 作为所有损失函数的通用基类
  - `GLMLoss` 现继承自 `LossBase`（向后兼容）
  - 新损失类型自动继承全部 10 种惩罚和 6 种求解器
  - 求解器类型注解从 `GLMLoss` 更新为 duck-typed `LossBase`

- **新损失类型**:
  - `QuantileLoss`: 分位数回归的 pinball 损失（对应 R `quantreg::rq()`）
  - `HuberLoss`: 稳健 M-估计器损失（对应 R `MASS::rlm()`）
  - `CoxPartialLikelihoodLoss`: Cox PH 负对数偏似然（对应 R `survival::coxph()`）
    - 支持 Breslow 和 Efron tie 处理
    - CPU-only (numpy)；GPU 加速请用 `statgpu.survival.CoxPH`

- **损失注册表** (`statgpu.losses._registry`):
  - `register_loss(name)`: 注册自定义损失类的装饰器
  - `get_loss(name, **kwargs)`: 损失实例化工厂函数
  - `list_losses()`: 列出所有已注册损失（GLM + 非 GLM）

- **新增文件**: `statgpu/losses/__init__.py`, `_base.py`, `_registry.py`, `_quantile.py`, `_huber.py`, `_cox_ph.py`
- **测试**: `dev/tests/test_losses.py` 64 个测试全部通过

### 新增 (2026-06-17)

- **P2 模块拓展** (PR #72)：
  - 5 个模块升级：ANOVA (15%→60%)、Covariance (30%→60%)、Panel (45%→70%)、Splines (35%→60%)、Kernel Methods (60%→80%)
  - 所有新功能支持 numpy/cupy/torch 三端计算
  - 17 个新源文件，112 个新测试（全部通过）
  - 外部对标验证：scipy、sklearn、statsmodels（精度：coef diff ≤ 1e-14）

- **ANOVA**：
  - `f_twoway`：二因素 ANOVA（支持/不支持交互项，Type I SS 分解）
  - `f_welch`：方差不齐的 Welch ANOVA（Welch 1951，Welch-Satterthwaite df）
  - `tukey_hsd`：Tukey HSD 事后检验（studentized range 分布）
  - `bonferroni`：Bonferroni 校正的两两 t 检验
  - `cohens_f`：Cohen's f 效果量
  - `partial_eta_squared`：偏 eta 平方
  - 文件：`_twoway.py`、`_welch.py`、`_posthoc.py`、`_effect_size.py`

- **Covariance**：
  - `ShrunkCovariance`：通用收缩估计器（匹配 sklearn）
  - `MinCovDet`：稳健 MCD 估计（FAST-MCD，Rousseeuw & Van Driessen 1999）
    - 多阶段算法：30 次随机启动 → top 10 → 完整 C-steps
    - 一致性校正因子（Croux & Haesbroeck 1999）
    - 匹配 sklearn MinCovDet，correlation = 1.000000
  - `GraphicalLasso`：稀疏逆协方差估计（Friedman et al. 2008）
  - `GraphicalLassoCV`：交叉验证的 graphical lasso
  - 文件：`_robust.py`、`_graphical_lasso.py`、`_shrinkage.py`（扩展）

- **Panel**：
  - `PooledOLS`：混合 OLS（支持 nonrobust/robust/clustered/HAC）
  - `BetweenOLS`：实体均值 OLS
  - `FirstDifferenceOLS`：一阶差分 OLS
  - `FamaMacBeth`：两步法回归（截面 OLS → 时间序列均值 + NW SE）
  - `hac_covariance`：Newey-West HAC 估计器（Bartlett 核，自动带宽）
  - 文件：`_pooled.py`、`_between.py`、`_first_diff.py`、`_fama_macbeth.py`、`_covariance.py`（扩展）

- **Splines**：
  - `SplineTransformer`：sklearn 兼容的 fit/transform API
  - `cyclic_cubic_spline_basis`：周期性三次样条（零空间投影法）
  - `thin_plate_spline_basis`：多维平滑样条（φ(r) = r²log(r)）
  - 文件：`_transformer.py`、`_cyclic.py`、`_thin_plate.py`

- **Kernel Methods**：
  - `chi2_kernel`：指数化卡方核（numpy 后端使用 sklearn Cython 加速）
  - `Nystroem`：核近似（SVD 归一化，匹配 sklearn）
  - `KernelPCA`：核主成分分析
  - RBF kernel 优化：float32 分块计算，CPU 上比 sklearn 快 3.5-13x
  - 文件：`_nystroem.py`、`_kpca.py`、`_kernels.py`（扩展 + 优化）

### 优化 (2026-06-17)

- **RBF kernel numpy 性能**：
  - 大矩阵（n>2000）自动使用 float32（内存带宽减半）
  - 分块计算避免 OOM（n=50000 不再崩溃）
  - 所有运算复用同一 buffer（峰值内存 = 1 个 n×m 矩阵）
  - 性能：n=5000 快 3.8x，n=10000 快 3.5x，n=50000 快 13.4x

- **Nystroem GPU 优化**：
  - K_mm 特征分解移至 CPU（避免小矩阵的 GPU kernel launch 开销）
  - 归一化矩阵存储在 CPU，仅在需要时转到 GPU
  - 输出与 sklearn 完全一致（correlation = 1.000000）

- **数据一致性**：
  - GPU 输入 → GPU 输出（不再自动转 numpy）
  - float64 输入小矩阵 → float64 输出
  - float64 输入大矩阵 → float32 输出（避免 OOM）

### 验证 (2026-06-17)

- **三端基准测试**（Tesla P100-16GB，n=5000-100000）：
  - LedoitWolf：torch 比 sklearn 快 44.8x（n=100000）
  - Nystroem：cupy 比 sklearn 快 43.7x（n=100000）
  - RBF Kernel：cupy 快 797x，torch 快 929x（n=10000）
  - ANOVA：torch 比 scipy 快 2.1x（n=100000）
- **精度**：所有模块与外部框架差异 ≤ 1e-14（float64）
- **112 个测试**：5 个测试文件覆盖所有 P2 模块，全部通过
- **Benchmark JSON**：`results/p2_benchmark_final.json`（含 GPU warmup）

### Code Review 第 9-10 轮 (2026-06-15)

**Bug 修复：**
- Newton 求解器收敛条件过严 10000 倍（`_norm2_dev` 返回 L2 范数而非平方范数）
- `_resolve_loss_name` 从错误模块导入——CV 流水线会抛出 `ImportError`
- ElasticNet Lipschitz 对 `"en"` 别名返回 0
- Debiased inference 清除了 `_resid`/`_X_design`/`_y`，导致 `rsquared`/`aic`/`bic` 失效
- `fista_lla_path` 在 XtX 快速路径中忽略 `sample_weight`（GPU 和 numpy 均受影响）
- `_fit_gpu_backend` 缺少 `xp_ones` 导入——大特征 GPU 拟合会 NameError

**性能优化：**
- 删除 `_solver_utils.py`（442 行重复代码）
- IRLS：将 `_to_backend(y)` 提升到闭包外（原来每迭代调用 30 次），复用 `eta_raw` 矩阵乘法
- Fused dispatch 字典提升为模块级常量
- `xp.sum(sw*ps)` → `xp.dot(sw,ps)`——避免 O(n) 临时分配

**重构：**
- 统一 `_fit_gpu`/`_fit_torch` 为单一 `_fit_gpu_backend` 方法（-468 行）
- 提取 `_nesterov_momentum`/`_nesterov_update` 辅助函数（6 个文件 12 处）
- 提取梯度裁剪常量到 `solvers/_constants.py`
- 为所有公共求解器函数添加类型注解
- 添加 `_call_with_weight` 辅助函数替代 8 个 `try/except TypeError` 块
- 修复顶层 `__init__.py` 重复导入
- 将 `SelectivePenalty` 线程局部单例改为每次调用新建实例
- 缓存 `_family_for_loss()` 结果

### 重构 (2026-06-14)

- **顶层模块重组（Phase 0-6）**：
  - 提取 `statgpu/solvers/` 为通用顶级模块，包含 6 个求解器（FISTA、FISTA-BB、FISTA-LLA、Newton、L-BFGS、ADMM）。求解器现在与 loss 无关——适用于任何实现 `GLMLoss` 接口的 loss。
  - 提取 `statgpu/cross_validation/`，包含 `CVEstimatorBase`、`kfold_indices`、`hash_cv_data`、`batch_mse`、`run_cv`。被 `linear_model` 和 `survival` 共用。
  - 将 `PenalizedGeneralizedLinearModel`（3968 行）拆分为 mixin 架构：`_base.py` + `_fit_mixin.py`（2185 行）+ `_inference_mixin.py`（1174 行）+ `_predict_mixin.py`（215 行）。
  - 重组 `linear_model/` 为 `wrappers/`（13 个模型）、`penalized/`（mixin + 9 个子类 + CV）、`cv/`（4 个 CV wrapper）、`legacy/`（6 个文件）。
  - 将 GLM 特有融合函数移至 `glm_core/_fused.py`。
  - 在 `GLMLoss` 基类中添加优化提示属性（`_lipschitz_safety`、`_momentum_beta_cap`、`_has_constant_hessian` 等）——solver 读取这些属性而非硬编码 loss 名称。
  - 清理 `nonparametric/` 中 4 个重复文件。
  - 62 个安全网测试 + 远程 GPU 验证（Tesla P100）：51/51 精度基准测试全部通过。

- **新增 wrapper**：
  - `AdaptiveLasso` — adaptive L1 惩罚（Zou 2006）
  - `SCADRegression` — SCAD 惩罚（Fan & Li 2001）
  - `MCPRegression` — MCP 惩罚（Zhang 2010）

- **修复：adaptive_l1/scad GPU 后端兼容性**：
  - `_irls_ridge_init_cd` 现在使用后端无关的 `xp` 操作，而非仅 numpy 代码。之前在 CuPy/Torch 上会报 `TypeError`。
  - 无 CPU↔GPU 传输——计算保持在原始设备上。

- **文档**：
  - 修复 28 个模型文档的数学公式显示分隔符（`\[ \]` → `$$ $$`）。
  - 更新 AGENTS.md 的模块结构描述。
  - 在 AGENTS.md 中添加 changelog 写作规范。

### 新增 (2026-06-13 ~ 2026-06-14)

> PR #55~#58 由原始 PR #36（GLM+Penalty 完整模块）拆分而来。PR #36 实现了完整的 GLM + 惩罚系统，在完整矩阵基准测试中达到 1043/1043 ALL PASS (100%)。

- **PR #36 — GLM+Penalty 完整模块（原始，拆分为 PR-A~D）**:
  - 7 个 GLM 族：`squared_error`, `logistic`, `poisson`, `gamma`, `inverse_gaussian`, `negative_binomial`, `tweedie`
  - 10 个惩罚：`none`, `l1`, `l2`, `elasticnet`, `scad`, `mcp`, `adaptive_l1`, `group_lasso`, `group_mcp`, `group_scad`
  - 6 个求解器：`exact`, `newton`, `lbfgs`, `irls`, `fista`, `fista_bb` — 按族+惩罚组合调度
  - 3 个后端：CPU (NumPy), CuPy, PyTorch — 自动设备选择
  - 关键技术特性：
    - LLA 路由处理非凸惩罚（SCAD, MCP, group 变体）
    - 对数链接 GLM 增广截距处理（Poisson, gamma 等）
    - 迭代相关 Lipschitz 计算
    - Async FISTA 处理 GLM+非光滑惩罚（n=5000 时 2-5.5x 加速）
    - L-BFGS 融合惩罚梯度修复 — 正确收敛到 `loss_grad + α·coef = 0`
    - CuPy/Torch 后端 GPU sync 批处理优化
    - GLM loss+gradient 核融合
  - 基准测试结果 (v23c)：
    | Section | 描述 | 测试数 | 状态 |
    |---------|------|--------|------|
    | A | 跨后端计时+精度 | 816 | 全通过 |
    | B | vs sklearn | 13 | 全通过 |
    | D | vs statsmodels | 68 | 全通过 |
    | E | 跨求解器一致性 | 146 | 全通过 |
    | **总计** | | **1043** | **全通过** |
  - GPU 加速 (Section A)：
    | 规模 | CPU 平均 | Torch 平均 | 加速 |
    |------|----------|-----------|------|
    | n=500, p=50 | 953ms | 954ms | 1.00x |
    | n=2000, p=200 | 3995ms | 9108ms | 0.44x |
    | n=5000, p=500 | 2875ms | 1313ms | **2.19x** |
  - n=5000 求解器级别：fista-Torch 2.56x, newton-Torch 2.10x, irls-Torch 2.40x
  - 文件：
    - 核心求解器 & GLM：`statgpu/glm_core/_solver.py`, `_negative_binomial.py`, `_irls.py`, `_gamma.py`, `_inverse_gaussian.py`, `_tweedie.py`
    - 惩罚模型：`statgpu/linear_model/_penalized.py`, `_gamma_glm.py`, `_inverse_gaussian_glm.py`, `_negative_binomial_glm.py`, `_tweedie_glm.py`
    - 惩罚：`statgpu/penalties/_adaptive_l1.py`, `_mcp.py`, `_scad.py`, `_group_lasso.py`, `_group_mcp.py`, `_group_scad.py`
    - 后端：`statgpu/backends/_array_ops.py`, `_cupy.py`
    - 文档：changelog (EN+CN), benchmarks (EN+CN), model docs (GLM, Logistic, Poisson, Ridge; EN+CN), `dev/tests/_bench_v23c_report.md`
  - 完整报告：`dev/tests/_bench_v23c_report.md`

- **PR #55 — 核心 GLM 求解器、后端、惩罚、推断 (PR-A, 来自 PR #36)**:
  - 7 个 GLM 族：squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
  - 10 个惩罚：none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
  - 6 个求解器：irls, fista, fista_bb, admm, lbfgs, newton — 按族+惩罚组合调度
  - 3 个后端：NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) 自动设备选择
  - 统一推断：15 个分布、p 值校正、bootstrap、permutation test
  - 关键技术：LLA 路由处理非凸惩罚（SCAD/MCP）、对数链接 GLM 增广截距、迭代相关 Lipschitz 计算、损失+梯度核融合
  - 稳定性修复：
    - 修复 3 个 Critical NameError（CuPy 路径和循环导入）
    - 修复 torch 设备不匹配（HC2/HC3 leverage 计算）
    - 修复 power-iteration 种子（Lipschitz 计算可复现性）
    - 修复 CuPy cumop dtype 核（空输入处理）
    - 修复 KDE logpdf NameError 和 binomial IRLS deviance 计算
    - 恢复 irls_solver 主循环（意外删除后恢复）
  - 后端改进：
    - 添加 GPU sync 批处理（solver 操作，H6 修复）
    - 拆分 solver 为模块化组件（H4 修复）
    - 相对导入转为绝对导入 `statgpu.xx`
    - 添加后端感知梯度计算
  - 惩罚修复：
    - 添加缺失的 group_mcp/group_scad 到 non_smooth 验证集
    - 更新 group auto-fill 后的派生属性
    - 修复 CompositePenalty 后端处理
  - 测试：
    - 为所有修复添加回归测试
    - 标记 LassoCV 测试为 xfail（PR-B 功能）

- **PR #56 — 惩罚模型 + CV 框架 (PR-B, 来自 PR #36)**:
  - 7 个惩罚估计量：PenalizedLinearRegression, PenalizedLogisticRegression, PenalizedPoissonRegression, PenalizedGammaRegression, PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
  - PenalizedGLM_CV：完整 CV（7 族 × 10 惩罚 × 6 求解器）
  - Lasso, Ridge, ElasticNet 完整推断
  - LogisticRegression, LinearRegression GPU 支持
  - 稳定性修复（8 轮代码审查）：
    - 修复 P0/P1 bug：solver 运行时 NameError + TypeError
    - 修复 GPU/CPU 预测容差（先放宽后收紧到 max_iter=2000 + tol=1e-10）
    - 统一 NB 跨设备路径容差
    - 修复 get_params、sample_weight、backend-aware 问题
    - 将硬编码 penalty/loss 集合合并为共享常量
  - 代码质量：
    - 提取 ~500 行死代码到 legacy 文件
    - 移除 magic numbers，添加命名常量
    - 去重 score/summary 方法（跨估计量）
    - 修复 BOM 编码问题和 __all__ 导出
    - 清理导入，移除自导入
  - 性能：
    - 添加 penalty 操作的批量 GPU sync
    - 优化 penalty 类别检测
  - 测试：
    - 放松后收紧 GPU/CPU 预测容差
    - 修复后移除 xfail 标记

- **PR #57 — 新模块 (PR-C, 来自 PR #36)**:
  - ANOVA：`f_oneway` — GPU 加速单因素方差分析，支持 float32/float64
  - 协方差：`EmpiricalCovariance`, `LedoitWolf`, `OAS` — 收缩协方差估计
  - 面板数据：`PanelOLS`（单/双向固定效应）, `RandomEffects`（Swamy-Arora）, `PanelSummary`, 聚类协方差
  - 样条：`bspline_basis`, `natural_cubic_spline_basis`, 惩罚回归 + GCV
  - 半参数：`GAM`（惩罚 B 样条 + GCV 平滑参数选择）
  - 核方法：`KernelRidge`, `KernelRidgeCV`, 6 个核函数（rbf, polynomial, linear, laplacian, sigmoid, cosine）
  - Python 兼容性：
    - 修复 `__future__` 导入顺序（Python 3.9 兼容）
    - 在 4 个文件中移动 `__all__` 到 `__future__` 之后
    - 修复协方差模块导出
  - 运行时修复：
    - 修复 RandomEffects 组均值计算
    - 添加新模块缺失的 NumpyBackend 方法
    - 修复 panel 测试 fit() 参数顺序（y, X → X, y）
  - 代码审查修复：
    - 第 1 轮修复 8 个 Critical + 2 个 High 问题
    - 修复所有新模块的导入约定
    - 后续轮次修复 H2/M5/M6/L2 问题

- **PR #58 — 基础设施、导出、向后兼容 (PR-D, 来自 PR #36)**:
  - 统一 `statgpu/__init__.py` 导出（~60 个公共名称）
  - `BaseEstimator` 设备管理 + sklearn 兼容 `get_params`/`set_params`
  - `Device` 枚举（CPU/CUDA/TORCH/AUTO）自动检测
  - `kernel_methods/` 和 `splines/` 旧路径向后兼容
  - sklearn 兼容性：
    - 修复 `get_params` 只返回自身 `__init__` 参数（不含父类）
    - 保留 `simultaneous_method` 和 `cov_type` 的字符串标识（sklearn clone() 要求）
  - CoxPH 修复：
    - 在 `_compute_partial_likelihood` 中 null model 路径前定义 `n`
    - 为 null model risk set 添加惩罚警告
  - 代码审查：
    - 修复 `__all__` 导出和导入回退
    - 修复 6 个剩余评论问题

- **PR #48 — 模块重组**:
  - 将 kernel_methods/ 和 splines/ 移至 nonparametric/ 子包
  - 创建 kernel_smoothing/ 子包用于 KDE + 核回归
  - 将 GAM 提取到 semiparametric/ 包
  - 旧导入路径向后兼容
  - IRLS 求解器改进：
    - 修复 log-link 截距初始化（之前使用错误的起始值）
    - 添加每次迭代收敛检查（之前只在结束时检查）
    - 将 `_dev_val` 计算移出 IRLS 循环（性能优化）
  - CuPy 修复：
    - 修复 cummin/cummax 空输入异常处理
    - 修复 cumop dtype 核（非连续数组）
    - 在协方差测试中用 `_to_numpy` 包装 CuPy 数组
  - 代码质量：
    - 从 `_irls.py` 中剥离 BOM 编码
    - 为 `_lasso.py` 添加 `from __future__ import annotations`
    - 将裸 `except Exception` 子句收窄为特定异常
    - 修复 splines `__all__` 导出
  - 安全：
    - 从远程配置中移除硬编码 SSH 凭据
  - 测试：
    - 添加 RTX 4090 的 6 阶段真实数据基准测试套件
    - 为所有 PR #47 代码审查修复添加回归测试
  - Python 3.8 兼容性修复

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
  - 包含 PR #1 到 #64 的完整 CHANGELOG

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

### Fixed (2026-06-10 ~ 2026-06-12)

- **PR #49 Code Review: 110+ fixes across 16 files**:
  - 修复 26 个 P1 bug（merge conflict、NameError、数值公式错误、GPU 路径崩溃等）
  - 修复 55 个 P2 bug（缓存线程安全、后端一致性、边界情况、接口兼容性等）
  - 修复 ~30 个 P3 改进项（死代码清理、magic numbers、性能优化等）
  - 新增 428 个测试用例（远程 GPU Tesla P100 全部通过）
  - 三端精度偏差 < 0.02%（同一 random_state 下）
  - 性能无回退（RidgeCV CuPy 6.8x 加速，PenalizedGLM_CV Torch 3.1x 加速）
  - 删除 ~1300 行死代码
  - 统一 `best_score_` 为负 MSE（sklearn 惯例）
  - 合并 PLAN_UNIFIED.md 门禁与 PR #49 编码规范到 TO_DO.md
  - 统一 CV 框架：
    - 创建 `_cv_base.py`：共享 `kfold_indices`、`CVCache`、`batch_mse`
    - 创建 `_cv_engine.py`：通用 CV 循环引擎
    - 实现 `PenalizedGLM_CV`：完整 family × penalty × solver 矩阵
    - 添加 alpha 值间 warm-start（复用模型实例）
    - 为 RidgeCV 添加批量特征分解（避免逐 alpha 求解）
  - CuPy fused kernel 问题：
    - 发现 SCAD/MCP CuPy fused kernel 数值问题
    - 禁用 SCAD/MCP LLA 路径的 fused kernel
    - 添加诊断脚本和文档
  - Panel 修复：
    - 修复非平衡双向固定效应
    - 修复 PanelOLS 文档
  - Ridge 修复：
    - 修复加权截距计算
    - 修复 ElasticNetCV warm-start（`fit_intercept=False` 时）
  - 代码质量：
    - 用共享导入替换重复的 `_kfold_indices`
    - 修复 Lasso 默认值和缓存键
    - 为 PenalizedGLM_CV 评分添加推断保护

### 新增 (2026-06-07 ~ 2026-06-09)

- **PR #50 — GLM 稀疏 CV 路径添加 val_sample_weight**:
  - 稀疏 GLM 交叉验证的验证样本权重支持
  - 支持不平衡数据集的加权 CV 折
  - 移除多余的 cupy 行
  - 使用 loss_fn.value 用于 numpy 路径
  - 传递未增强的 Xv 到 _evaluate_loss_numpy 用于加权评分

- **PR #53 — 修复加权 Ridge 推断**:
  - 修复加权 Ridge 回归的尺度计算
  - 保留 sample weights 下的 bse/pvalues/conf_int

- **PR #54 — 重构 CV 调度表**:
  - 为 _compute_cv_scores 创建调度表
  - 提取 _cv_fold_general 以更清晰分离
  - 添加路径失败警告和 LLA 清理
  - 修复 Tweedie per-sample loss 符号错误
  - 移除不正确的回退权重
  - 移除死代码和自导入
  - 添加回退警告
  - 优化 Ridge CV 评分
  - 提取硬编码常量为模块级命名变量
  - 添加静默回退警告
  - 修复非高斯 MSE 回退
  - 为非均匀权重与非 L2 惩罚引发清晰错误
  - 添加 loss 公式注释并收窄异常捕获
  - 为 PenalizedGLM_CV 添加 cv_splits 参数以支持自定义折生成器
  - 从 loss 对象默认值参数化 NB alpha 和 Tweedie power
  - 创建统一 loss 公式注册表（替换内联 if/elif 链）
  - 修复 LassoCV cache_key 变量名（缓存重构后）
  - 修复 _res_logistic 返回梯度（sigmoid(eta)-y）而非 loss
  - 修复 Poisson 残差返回梯度、NB 分母、InvGauss clipping
  - 修复加权 Lipschitz 使用 sum(w)、cv_splits 规范化生成器

### Optimized (2026-06-05)

- **Strict sparse GLM CV GPU squeeze pass, round 7**:
  - Reused fold-level initial Lipschitz estimates across sparse GLM alpha paths, including `fista_bb_solver` burn-in checks.
  - Batched CuPy validation scoring for sparse GLM CV; solver trajectories and strict final refits are unchanged.
  - Added a Torch fold-batched strict logistic sparse CV path with per-fold Lipschitz constants and equivalent validation scores.
  - Strict CV still preserves the requested `max_iter` and `tol`; the logistic GPU iteration cap is not applied to strict CV.
  - Matpool P100 strict matrix (`cv=3`, `n_alphas=8`, `max_iter=1000`, `tol=1e-4`) kept all CPU/CuPy/Torch alpha selections matching. Torch was faster than CPU in 18/32 mid/high rows after fold-batched logistic CV; logistic Torch runtimes improved to about `0.46x`-`0.53x` of round-6 timings.
  - `device="auto"` selected CPU for 14 rows and Torch for 18 rows on the same matrix; it was faster than explicit CPU in 27/32 rows, with all alpha selections matching CPU.
  - A follow-up auto-routing pass keeps low-dimensional squared-error sparse CV (`p<256`) on CPU, avoiding the Torch cold-start outlier while preserving high-dimensional Torch acceleration.
  - Round 9 adds CuPy fold-batched strict logistic sparse CV. It keeps explicit `device="cuda"` on the CuPy backend and falls back only to the previous CuPy per-fold path if the helper fails.
  - Round 9 Matpool P100 strict matrix (`warmup=1`, `cv=3`, `n_alphas=8`, `max_iter=1000`, `tol=1e-4`) kept all CPU/CuPy/Torch/auto alpha selections matching CPU. Explicit Torch was faster than CPU in 18/32 rows, explicit CuPy in 8/32 rows, and `device="auto"` in 27/32 rows while selecting CPU for 16 rows and Torch for 16 rows.
  - Targeted logistic CuPy validation matched the previous CuPy per-fold scores to numerical precision and made CuPy faster than CPU on larger `10000x100` and `5000x500` logistic rows; `2000x100` and `2000x500` remain explicit-CuPy hotspots.
  - Validation artifacts: `results/cv_poisson_gamma_lipcache_round5.json`, `results/cv_poisson_gamma_cupy_score_batch_round6.json`, `results/cv_mid_high_after_lipcache_scorebatch_round6.json`, `results/cv_auto_after_lipcache_scorebatch_round6.json`, `results/cv_logistic_foldbatch_round7.json`, `results/cv_mid_high_after_logistic_foldbatch_round7.json`, `results/cv_auto_after_logistic_foldbatch_round7.json`, `results/cv_auto_lowp_sqerr_cpu_round8.json`, `results/cv_logistic_cupy_foldbatch_round9.json`, `results/cv_mid_high_after_cupy_foldbatch_round9.json`.

### 优化 (2026-06-04)

- **小规模 sparse CV GPU 传输优化**:
  - squared-error sparse CV 在只需要 validation score 时不再把 coefficient path 传回主机。
  - Matpool P100 小规模 strict CV (`n=500`, `p=20`, `cv=3`, `n_alphas=8`) 中，`squared_error+l1` 从 CuPy `820ms` 降到 `190ms`，Torch 从 `266ms` 降到 `97ms`；alpha 选择不变，GPU vs CPU 系数 L2 约 `6.9e-06`。
  - logistic sparse CV 仍是 strict 模式热点；为保持 strict 的 `max_iter`/`tol` 语义，未把现有 iteration cap 接入 strict CV。
  - 新增 `dev/tests/benchmark_glm_penalty_external_small.py`，用于小规模 sklearn/statsmodels/R 外部精度与运行时间比较，并显式记录等价惩罚参数映射。
  - 验证产物：`results/cv_strict_sparse_sync_opt_v2_500x20.json` 和 `results/external_glm_penalty_small_gpu_sync_opt_v2.json`。

### 新增 (2026-06-04)

- **Strict-first PenalizedGLM_CV 策略控制**:
  - `PenalizedGLM_CV` 默认保持 `cv_strategy="strict"`，并新增显式 opt-in 的 `cv_strategy="two_stage"` alpha screening。
  - two-stage CV 使用放松的 screening 求解、strict 候选复核，以及 strict 最终 refit。
  - 新增 `ApproximateCVWarning`、`acknowledge_approx`、`refine_top_k`，以及 CV 诊断字段 `cv_strategy_`、`cv_selected_device_`、`refined_mask` 和 stage-1 score 数组。
  - benchmark 脚本可通过 `--cv-strategy` 运行 strict 或 two-stage CV。

### 修复 (2026-06-04)

- **Poisson sparse `PenalizedGLM_CV` 跨后端精度**:
  - strict GPU FISTA 不再使用仅供近似筛选的异步 CV 更新路径。
  - Poisson L1/ElasticNet CV 对几乎平坦的 CV 曲线使用稳定 near-tie 规则；当后端分数差异处于数值噪声量级时，确定性选择更强正则化的 alpha。
  - Matpool P100 远程验证中，`poisson+l1/elasticnet`、`n=500`、`p=20`、`cv=3`、`n_alphas=8` 在 CPU、CuPy、Torch 上选出相同 alpha，系数 L2 差异约 `1.6e-05`。

### 优化 (2026-06-04)

- **GPU sparse GLM CV solver policy**:
  - `solver="auto"` 现在按后端选择 strict-CV sparse GLM 求解器：GPU `poisson+l1` 和 `negative_binomial+l1` 使用 `fista_bb`，Torch `gamma+l1/elasticnet` 使用 `fista_bb`；用户显式指定的 solver 不变。
  - sparse GLM CV path 的首个截距初始化改为 `log(mean(y))`，与 positive-family 常规 fit 初始化一致。
  - Matpool P100 strict 矩阵 (`n=500`, `p=20`, `cv=3`, `n_alphas=8`) 保持 CPU、CuPy、Torch 的 90/90 alpha 一致；相对上一版 strict baseline，targeted speedup 包括 `negative_binomial+l1` Torch `0.37x`、CuPy `0.55x`，`poisson+l1` Torch `0.57x`、CuPy `0.83x`。
  - 验证产物：`results/cv_strict_500x20_gpu_policy_opt_v3.json` 和 `results/cv_two_stage_sparse_auto_policy_opt_500x20.json`。


### 优化 (2026-06-01)

- **后端传输 helper 与 benchmark parser**:
  - CuPy <-> Torch CUDA 转换优先使用 DLPack 零拷贝共享，失败时回退到原安全路径。
  - NumPy -> Torch CUDA 传输在可用时尝试 pinned memory 与 `non_blocking=True`。
  - 新增 `dev/tests/_bench_report_parser.py`，可将 full-matrix benchmark 文本日志汇总为 JSON/Markdown。
  - Benchmark summary 现在包含 backend/family/penalty 行数统计，并支持 `--fail-on-alerts` 作为脚本化 gate。
  - CoxPH/CoxPHCV 统一暴露 Torch CUDA 清理钩子，补齐 GPU memory cleanup 约束。

## 2026-05

### 新增 (2026-05-24 ~ 2026-05-29)

- **PR #37 — GLM 惩罚正确性 + 自动 GPU 路由**:
  - 修复惩罚 GLM predict() 返回逆链接均值尺度预测
  - 基于问题规模的惩罚模型自动 GPU 路由
  - 修复 GPU 后端不可用时的 predict 后端回退
  - 强制显式 GPU 预测后端契约
  - 处理 GPU sample_weight 转换

- **PR #38 — Gamma 逆幂 FISTA**:
  - 链接感知 Gamma FISTA 支持（CPU/CuPy/Torch）
  - 修复逆幂链接函数的目标不匹配
  - 修复逆幂 Gamma FISTA 初始化和 torch dtype 对齐
  - 使用后端原生逆幂 FISTA warm start
  - 修复逆幂 gamma FISTA 初始化和 clipping 一致性
  - 修复 torch FISTA 非高斯截距路径的 dtype
  - 修复整数设计 dtype 提升（跨 GLM 截距路径）
  - 修复 CuPy FISTA 初始化 dtype

- **PR #39~#42 — GLM 求解器重构**:
  - 修复 GLM GPU dtype 和审查回归
  - 重构 GLM 求解器后端 helper
  - IRLS 求解器后端别名和兼容性
  - 测试 IRLS 求解器后端别名

- **PR #43, #44 — 线性推断结果修复**:
  - 重构高斯线性推断 helper
  - 修复 CuPy 推断临界值 dtype
  - 添加共享推断结果容器
  - 完成线性推断结果连接
  - 修复加权惩罚推断状态
  - 清除过时线性推断结果
  - 修复推断边界情况清理
  - 清除 z 结果的过时 t 统计量
  - 清除不可用的 GPU 推断预计算缓存
  - 使用 ridge sandwich 协方差处理惩罚

- **PR #47 — CuPy cummin/cummax 修复**:
  - 修复 CuPy cummin/cummax CUDA 核在非连续数组上的问题
  - adjust_pvalues BH/BY/Hochberg 现在返回正确结果（之前与 statsmodels 0% 一致）
  - 根因：CUDA 核读取顺序内存，但 flip() 返回负步长视图
  - 修复 IRLS log-link 截距初始化
  - 添加每次迭代收敛检查
  - 添加 RTX 4090 的 6 阶段真实数据基准测试套件
  - 从 IRLS 中移除硬编码 SSH 凭据 + 使用后端工具
  - 收窄裸 except 子句
  - 为所有代码审查修复添加回归测试

### 修复 (2026-05-20)

- **v23c: L-BFGS fused penalty gradient 修复**:
  - 根因: `lbfgs_solver` fused GLM 路径只计算 loss 梯度, 遗漏 penalty 梯度
  - L-BFGS 收敛到无正则化解 (`loss_grad ≈ 0`) 而非正确的 `loss_grad + α·coef = 0`
  - 修复: 在 `_fused_glm_value_and_gradient` 调用后添加 `_smooth_penalty_gradient`
  - 影响: 所有 GLM family + smooth penalty (L2, ElasticNet)
  - 修复 9 个 MISMATCH (max|diff| 从 1e-01~1e-02 降至 1e-04~1e-08)
  - 完整基准测试: 1043/1043 ALL PASS
  - 修改文件: `statgpu/glm_core/_solver.py`

### 优化 (2026-05-20)

- **v22g: Async FISTA 与 GPU 优化**:
  - Async FISTA: GLM+非光滑惩罚在 n=5000 时 2-5.5x 加速
  - Lipschitz 重算、y-scaling cap、NB momentum cap、gamma 保守 momentum
  - 回溯优化、梯度裁剪统一
  - CuPy/Torch 后端 GPU sync 优化
  - 修改文件: `statgpu/glm_core/_solver.py`、`statgpu/glm_core/_negative_binomial.py`、`statgpu/backends/_array_ops.py`

- **v23c: 完整矩阵基准测试 (1043 tests)**:
  - 7 families x 10 penalties x 3 scales x 多求解器 x 3 backends
  - Section A 时间: CPU 平均 953ms/3995ms/2875ms, Torch n=5000: 2.19x 加速
  - Section B: 13/13 vs sklearn ALL PASS
  - Section D: 68/68 vs statsmodels ALL PASS
  - Section E: 146/146 跨求解器 ALL PASS
  - 报告: `dev/tests/_bench_v23c_report.md`


### 新增 (2026-05-03 ~ 2026-05-11)

- **PR #27~#29 — 无监督学习 Phase 3/3B/3C**:
  - 新增 12 个估计量：PCA, KMeans, DBSCAN, GaussianMixture, NMF, AgglomerativeClustering, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF, IncrementalPCA, TruncatedSVD
  - 凝聚聚类 GPU 精确路径（single/complete/average/ward linkage）
  - 所有估计量的文档和验证基准测试

- **PR #30, #32 — 凝聚聚类 GPU 精确路径**:
  - GPU 加速精确 linkage（所有距离度量）
  - 支持 single, complete, average, ward linkage

- **PR #33 — 非参数模块审查**:
  - KDE GPU 内存修复
  - 带宽选择 GPU 化
  - Log-sum-exp 数值稳定性修复

- **PR #34, #35 — 文档**:
  - 明确运行时设备选择
  - 明确 Torch 后端文档
  - README 安装和要求更新

## 2026-04

### 新增 (2026-04-26)

- **PR #24 — 精度修复、hochberg/stouffer、包重组**:
  - Phase 1: Ordered 模型跨后端精度修复
  - 使用 torch.compile 和 Triton 核进行 GPU 加速
  - 统一跨包导入为绝对形式（PEP 8）
  - 解决 8 个 Codex 审查评论（shared_mem、lazy pandas、fit_intercept）
  - 添加 CuPy/Numpy 后端缺失的转置
  - 修复 cv_results_ 键命名
  - 在 fit 期间保留公式截距语义

- **PR #26 — README 刷新**:
  - 重组功能、添加模型、推荐可编辑安装
  - 导出 combine_pvalues
  - CuPy 收敛容差对齐：`gtol = 1e-6` → `gtol = self.tol`（与 scipy 一致）
  - CuPy 最小迭代次数从 30 降到 5（小样本下不再被迫多跑无用迭代）
  - 移除 CuPy warm-start 分支，始终从零初始化（与 scipy/torch 一致）
  - PyTorch 从 `optimizer.state_dict()` 捕获真实迭代数，不再虚假报告 `max_iter`
  - PyTorch `strong_wolfe` 不可用时抛出 `RuntimeError`（不再静默降级）
  - 回归测试：`dev/tests/test_ordered_cross_backend.py`（10 个跨后端用例，全部通过）
  - 修改文件：`statgpu/linear_model/_glm_base.py`、`dev/tests/test_ordered_cross_backend.py`

- **Phase 2a: 新增 hochberg (adjust_pvalues) + stouffer (combine_pvalues) 三端实现**:
  - `adjust_pvalues` 新增 `method='hochberg'`（step-up FDR），别名 `fdr_hochberg` / `step_up` / `stepup`
  - `combine_pvalues` 新增 `method='stouffer'`（加权 Z 检验），别名 `ztest` / `weighted_z`
  - stouffer 支持权重，与 cauchy 权重接口一致
  - 批量化支持 `axis` 参数（任意形状数组）
  - 依赖：新增 `norm` distribution proxy（已有 `chi2`）
  - 修改文件：`statgpu/inference/_multiple_testing.py`、`statgpu/inference/_distributions_backend.py`

- **Phase 2b: 测试补齐**:
  - 新增 `TestHochberg` (4 测试): 闭式验证、别名、vs BH、axis 批量化
  - 新增 `TestStouffer` (6 测试): vs scipy、权重、别名、axis、边界条件
  - 新增 `TestCauchyNoWeights` (2 测试): 无权重 cauchy、默认权重等效性
  - 新增 `TestTorchBackend` (6 测试): adjust/combine 各方法的 Torch vs NumPy 一致性
  - 修复 `np._core.numeric` 兼容性（NumPy 1.x vs 2.x），新增 `_normalize_axis_index` helper
  - 测试文件扩展：从 339 行增加到 519 行
  - 远程验证：40/40 通过 (Tesla P100)
  - 修改文件：`dev/tests/test_inference_multiple_testing.py`

- **Phase 3: 包结构审计与整理**:
  - 移动 `_gpu_utils.py` → `backends/_gpu_inference_cupy.py`
  - 移动 `_gpu_utils_torch.py` → `backends/_gpu_inference_torch.py`
  - 合并 `evaluation/` → `metrics/`，删除 `evaluation/` 目录
  - 合并 `glm_core/_backend.py` → `backends/_array_ops.py`
  - 移动 `_cv_base.py` → `linear_model/_cv_base.py`
  - 修正 `core/__init__.py` docstring（移除不存在模块的声明）
  - 添加 `survival/__init__.py` 命名约定注释（`_cuda` / `_cupy` / `_triton`）
  - 更新 18 处 import 站点
  - 删除文件：`_gpu_utils.py`, `_gpu_utils_torch.py`, `_cv_base.py`, `glm_core/_backend.py`, `evaluation/` 目录
  - 所有修改后 `import statgpu` 冒烟测试通过

### 新增 (2026-04-21)

- **PR #19 — Cython Efron 优化**:
  - Cython 优化 Efron 梯度和 Hessian 计算
  - CoxPH 精度和运行时综合基准测试
  - 更新 RidgeCV、LogisticRegressionCV 和 CoxPHCV 文档
  - 修复 logistic cv 重复的 batch log-loss helper 名称
  - 修复 cox cv 缓存键类型和 CUDA 核启动错误暴露
  - 跨文档对齐 CoxPHCV 状态
  - 更新 RidgeCV 和 LogisticRegressionCV 状态为完整实现

- **PR #21 — 分布后端统一**:
  - 将 `_distributions_gpu.py`, `_distributions_torch.py` 合并为单一 `_distributions_backend.py`
  - 通过 `SpecialFunctions` 协议和工厂模式覆盖 3 后端 15 个分布
  - 修复分布后端路由和 torch 设备传播
  - 修复代理 resolve args（rvs 和双侧 critical）
  - 精简代理后端自动解析参数
  - 更新分布 API 文档为统一 3 后端架构

- **PR #22 — 后端工具整合**:
  - 整合重复的后端工具函数
  - 更清晰的后端抽象层

- **CoxPHCV 从接口骨架升级为可训练版本**:
  - 已实现 penalty 网格搜索（K-fold）与最佳 penalty 全量重训流程
  - 支持 `ties='breslow'/'efron'` 与现有 `device` 路径（通过 `CoxPH` 后端执行）
  - 当前边界：`entry` 与 `cluster` 在 `CoxPHCV.fit()` 中暂未支持（显式 `NotImplementedError`）
  - 修改文件:
    - `statgpu/survival/_cox_cv.py`
    - `dev/tests/test_coxph_cv.py`

- **RidgeCV 和 LogisticRegressionCV 完整实现**:
  - 从接口骨架升级为完整功能实现，支持 GPU 加速的交叉验证
  - `RidgeCV` 新增功能:
    - K-fold 交叉验证 (支持自定义 folds 或 folds 生成器)
    - Alpha 网格自动生成 (log-spaced grid)
    - 交叉验证结果缓存 (Blake2b hash key, LRU cache maxsize=64)
    - 支持 `sample_weight` 和 `scoring` 参数
    - 后端支持：CPU (NumPy), GPU (CuPy), GPU (PyTorch)
  - `LogisticRegressionCV` 类似增强
  - 修改文件:
    - `statgpu/linear_model/_ridge_cv.py` - 完整实现 (约 1000 行)
    - `statgpu/linear_model/_logistic_cv.py` - 完整实现
  - 核心 API:
    ```python
    from statgpu.linear_model import RidgeCV, LogisticRegressionCV

    # RidgeCV with automatic alpha grid
    ridge_cv = RidgeCV(alphas=100, cv=5, device='cuda')
    ridge_cv.fit(X, y)
    print(f"Best alpha: {ridge_cv.best_alpha_}")
    print(f"CV scores: {ridge_cv.cv_results_['mean_test_score']}")

    # LogisticRegressionCV with custom alphas
    logit_cv = LogisticRegressionCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5, device='cuda')
    logit_cv.fit(X, y)
    ```

### 新增 (2026-04-20)

- **PR #18 — 远程配置 + 后端增强**:
  - 移除硬编码 SSH 凭据（安全修复）
  - 添加支持环境变量的远程配置模块
  - 为 knockoff filter 添加 Torch GPU 后端支持
  - 添加优化 GPU 实现的 Elastic Net
  - 添加 LassoCV 交叉验证 Lasso 实现
  - 修复远程配置、lasso/elasticnet cv 的审查问题
  - 修复基准测试配置错误消息的环境变量名

- **PR #20 — CoxPHCV CuPy 优化**:
  - 优化 CoxPHCV CuPy Hessian 路径和默认值
  - 加固 coxphcv 环境解析默认值缓存键
  - 添加 CoxPHCV 的 cv 测试
  - 明确 coxcv 默认值和环境回退断言
  - 更新 Cox GPU entry+efron 路径并记录安全推出
  - 同步 Cox 模型文档的 entry+efron GPU 状态

- **CoxPH Efron 实现修复与性能优化**:
  - 修复 Cython Efron 梯度/海森矩阵计算中的数值溢出问题，添加 clipping 保护 (`MAX_LINPRED=700`, `MIN_LINPRED=-700`)
  - 发现 Cython 编译版本存在正确性问题，暂时使用 Python fallback 实现（已验证与数值梯度一致）
  - CoxPH 综合性能对比 (vs statsmodels/lifelines/R survival)：
    - statgpu-Torch GPU 在 n=5000, p=20 规模下实现 **15.44x** 加速 (vs statsmodels)
    - 所有 statgpu 后端系数精度与 statsmodels 一致 (Max Diff < 4e-12)
    - C-index 计算已修复，CPU/CuPy/Torch 现在使用相同的精确分块向量化算法
  - 修改文件:
    - `statgpu/survival/_cox_efron_cy.pyx` - 添加 exp() clipping 保护
    - `statgpu/survival/_cox.py` - 使用 Python fallback 用于 Efron 梯度计算
  - 基准测试结果:
    - n=1000, p=10: statgpu-Torch 2.05x, lifelines 3.33x, R survival 21.6x (vs statsmodels)
    - n=5000, p=20: statgpu-Torch **15.44x**, lifelines 3.42x (vs statsmodels)
  - 测试脚本:
    - `dev/scripts/test_coxph_fit.py` - CoxPH 拟合与 lifelines 对比
    - `dev/scripts/final_verification.py` - 综合验证脚本
  - 报告:
    - `results/coxph_benchmark_report_2026-04-20.md` - 综合性能对比报告

### 新增 (2026-04-18)

- **PR #16 — Torch 后端支持**:
  - 增强 Ridge 和 CoxPH 模型的 Torch 支持
  - 添加内存管理改进
  - 修复 torch 后端/设备问题
  - 修复可复现性问题
  - 避免 Cox torch 路径中的循环同步
  - 收紧验证容差

- **PR #17 — Elastic Net 实现**:
  - 添加 Elastic Net（优化 GPU 实现）
  - 将优化代码集成到核心实现
  - 添加 Elastic Net 文档和 changelog 更新
  - 添加基准测试和测试脚本
  - 从大规模基准运行器中移除硬编码 SSH 凭据
  - 收紧基于环境变量的远程基准运行器的 SSH 认证逻辑
  - 允许使用发现的默认 SSH 密钥的密码短语

- **Elastic Net 实现与基准测试**:
  - 新增 `ElasticNet` 类，结合 L1 和 L2 正则化，使用 FISTA 求解器
  - 支持 CPU (NumPy)、GPU (CuPy) 和 GPU (PyTorch) 后端
  - 新增文件:
    - `statgpu/linear_model/_elasticnet.py` - Elastic Net 实现
    - `dev/benchmarks/benchmark_elasticnet_sklearn.py` - sklearn 对比
    - `dev/benchmarks/benchmark_glmnet_full.R` - R glmnet 对比
    - `dev/benchmarks/benchmark_statgpu_full.py` - statgpu vs glmnet
    - `dev/benchmarks/benchmark_large_scale.py` - 大规模性能测试
    - `dev/benchmarks/run_full_benchmark.py` - 统一基准运行器
    - `dev/benchmarks/run_large_scale.py` - 远端运行器
    - `dev/benchmarks/generate_complete_report.py` - 报告生成器
    - `dev/scripts/remote_elasticnet_smoke.py` - 基础验证
    - `dev/scripts/remote_stability_en.py` - 数值稳定性测试
  - 基准测试结果:
    - 所有后端与 sklearn 最大系数差异 < 3e-8
    - statgpu CPU 赢得 4/6 对比 R glmnet
    - statgpu Torch 在 5/6 大规模测试中最快 (83%)
    - 最大加速比：**4.36x** vs sklearn (n=100k, p=500)
  - 文档:
    - `docs/models/elastic-net.md` - 中文文档
    - `docs/en/models/elastic-net.md` - 英文文档
    - `results/benchmark_complete_summary.md` - 综合基准测试总结

- **PyTorch 后端修复** (Torch Backend Fixes):
  - 修复 `_base.py` 中 `_get_backend()` 方法，正确处理 `Device.TORCH`
  - 修复 `_gpu_utils_torch.py` 中的导入路径问题
  - 修复 `compute_aic_bic_torch()` 中的变量名错误
  - 修复 `_linear.py`, `_logistic.py`, `_ridge.py` 中的设备字符串处理（从 `device.value` 改为 `"cuda"`/`"cpu"`）
  - 修复 `_logistic.py` 中 `y_arr.astype()` 对 Torch tensor 的兼容性
  - **修复 `_linear.py` 中 Cholesky 求解器的 `upper` 参数错误** (`L.T` 是上三角，应使用 `upper=True`)
  - 性能结果 (Tesla P100):
    - LinearRegression Torch GPU: 数值精度 ~1e-15 (修复前 ~0.22)
    - LogisticRegression Torch GPU: 数值精度 ~1e-14
    - Lasso Torch GPU: 数值精度 ~1e-5
    - Ridge Torch GPU: 数值精度 ~1e-15
    - CoxPH Torch GPU: 数值精度 ~1e-15

- **PyTorch 后端完整实现** (Torch Backend Complete):
  - ✅ 所有核心模型支持 Torch 后端 (LinearRegression, Ridge, Lasso, LogisticRegression, CoxPH)
  - ✅ 非参数模块支持 (KDE, KernelRegression)
  - ✅ 特征选择模块支持 (Knockoff)
  - ✅ 完整基准测试和文档
  - 新增文件:
    - `statgpu/_gpu_utils_torch.py` - Torch GPU 工具函数
    - `statgpu/inference/_distributions_torch.py` - 分布对象 (norm, t, F)
  - 修改文件:
    - `statgpu/linear_model/_linear.py` - 添加 `_fit_torch()`
    - `statgpu/linear_model/_ridge.py` - 添加 `_fit_torch()`
    - `statgpu/linear_model/_logistic.py` - 添加 `_fit_torch()`
    - `statgpu/linear_model/_lasso.py` - 添加 `_fit_torch()`
    - `statgpu/survival/_cox.py` - 添加 `_fit_torch()`
    - `statgpu/nonparametric/_kernel_common.py` - 添加 Torch 支持
    - `statgpu/feature_selection/_knockoff_utils.py` - 添加 Torch 支持
  - 基准测试结果:
    - 小数据集 (2K×50): Torch 与 CuPy 性能接近 (<20% 差距)
    - 大数据集 (50K×200): CuPy 领先 2-5x (线性代数优化更成熟)
    - 所有模型数值精度 <1e-6 vs CPU
  - 文档更新:
    - `docs/guides/pytorch-backend.md` - PyTorch 后端使用指南
    - `docs/en/guides/pytorch-backend.md` - English version
    - `dev/docs/torch_backend_final_report.md` - 最终报告

- **API 清理** (API Cleanup):
  - 删除 `LinearRegression.bse_`, `LinearRegression.tvalues_`, `LinearRegression.pvalues_` property
  - 删除 `LogisticRegression.bse_`, `LogisticRegression.pvalues_` property
  - **原因**: 这些 property 是为了测试代码临时添加的，正确做法是测试代码使用内部属性 `_bse`, `_pvalues`
  - **影响**: 测试代码需要改用 `model._bse[1:]` 和 `model._pvalues[1:]` (排除截距)

### 新增 (2026-04-17)

- **PyTorch 后端** (Phase 1-5 完成):
  - 新的 GPU 后端替代方案，使用 PyTorch 2.0+
  - **已完成模型**:
    - ✅ Ridge 回归：完整协方差 (HC1/HC2/HC3/HAC) + 推断
    - ✅ LogisticRegression: IRLS 求解器 + 完整推断
    - ✅ Lasso: FISTA 求解器 + Debiased 推断 + Simultaneous 推断
    - ✅ CoxPH: Breslow 近似 + 完整推断 + C-index + Baseline Hazard
  - 新增文件:
    - `statgpu/inference/_distribution_utils_torch.py` - 特殊函数 (betainc, gammainc, erf 等)
    - `statgpu/inference/_distributions_torch.py` - 分布对象 (norm, t, F)
    - `statgpu/backends/_torch.py` - 后端适配器 (50+ NumPy 兼容方法)
  - 修改文件:
    - `statgpu/linear_model/_ridge.py` - 添加 `_fit_torch()`, `_robust_covariance_torch()`
    - `statgpu/linear_model/_logistic.py` - 添加 `_fit_torch()` 带 IRLS
    - `statgpu/linear_model/_lasso.py` - 添加 `_fit_torch()`, `_compute_inference_debiased_torch()`, `_compute_simultaneous_inference_torch()`
    - `statgpu/linear_model/_linear.py` - 添加 `_fit_torch()`, HAC 协方差
    - `statgpu/survival/_cox.py` - 添加 `_fit_torch()`, `_compute_log_likelihood_torch()`, `_compute_gradient_hessian_torch()`, `_compute_cindex_torch()`, `_compute_baseline_hazard_gpu()`, `_compute_baseline_hazard_torch()`
    - `statgpu/_config.py` - 添加 `Device.TORCH` 支持
  - 功能:
    - Ridge、LogisticRegression、Lasso、CoxPH 的完整 GPU 加速
    - Lasso Debiased 推断 (Javanmard-Montanari / Zhang-Zhang 方法)
    - Lasso Simultaneous 推断 (max-|Z| multiplier bootstrap)
    - 稳健协方差支持 (HC1/HC2/HC3/HAC)
    - CoxPH Baseline Hazard 估计 (Breslow 方法)
    - PyTorch 旧版本 (< 2.0) 回退到 SciPy
    - 数值精度：系数与 NumPy 差异在 1e-14 以内
  - **大规模性能** (Tesla P100, 50K×200):
    - Ridge HC3: Torch GPU 0.067s vs CuPy GPU 0.064s (4% 差距)
    - Logistic HC1: Torch GPU 0.099s vs CuPy GPU 0.102s (Torch 胜!)
    - Lasso: Torch GPU 0.081s vs CuPy GPU 0.076s (7% 差距)
    - CoxPH: Torch GPU 1.94s vs CuPy GPU 0.42s (CuPy 更快，因 baseline hazard 优化)
    - GPU 相比 CPU 提供 60x 加速用于稳健协方差
  - 文档:
    - `dev/docs/torch_backend_full_feature_report.md` - 完整基准报告
    - `dev/docs/torch_backend_implementation_summary.md` - 实现总结
    - `docs/guides/pytorch-backend.md` - PyTorch 后端指南（中英文）
    - `dev/docs/torch_benchmark_data.json` - 结构化基准数据（供前端使用）
    - `dev/docs/torch_backend_gap_analysis.md` - 功能完整性对比报告
  - 测试:
    - `dev/scripts/test_lasso_debiased_torch.py` - Lasso Debiased 推断测试
    - `dev/scripts/test_coxph_torch.py` - CoxPH Torch 后端测试
    - `dev/scripts/remote_test_lasso_debiased_torch.py` - 远程 GPU 测试
    - `dev/scripts/remote_test_coxph_torch.py` - 远程 GPU 测试
  - 安装：`pip install statgpu[torch]`

### 新增 (2026-04-15)

### 新增 (2026-04-11 ~ 2026-04-15)

- **PR #10 — HAC 协方差支持**:
  - LinearRegression 和 LogisticRegression 的 HAC 协方差
  - Newey-West 带宽选择
  - 修复 Ridge 推断的 penalized bread
  - 为 CV 骨架添加 NotImplementedError
  - 明确 CV 类的已实现 vs 仅接口范围

- **PR #11 — 新模型文档**:
  - Knockoff 特征选择文档
  - 新模型文档

- **PR #12 — 分布兼容层**:
  - 添加旧版分布函数兼容层
  - 重构推断方法以统一后端访问
  - 修复 Lasso GPU sync 开销（移除不必要的传输）
  - 修复分布代理 resolve args（rvs 和双侧 critical）
  - 预计算 Lasso 排除索引（性能优化）
  - 明确 t-ppf 二分边界文档

- **PR #13 — F 检验 p 值处理**:
  - 完美拟合 F 检验 p 值处理（返回接近零的 p 值）
  - 优化 Lasso p 值计算边界情况

- **PR #14 — 核回归 + Lasso GPU 优化**:
  - 添加核回归实现（NumPy/CuPy 支持）
  - 优化 Lasso GPU 计算逻辑
  - 修复完美拟合情况下的 F 统计量 p 值
  - 减少非参数 API 中的 GPU 索引内存使用
  - 修复非参数 API 命名

- **PR #15 — Lasso 推断 GPU 支持**:
  - 添加去偏 Lasso 同时推断（GPU nodewise 瓶颈）
  - 优化 CN/EN 模型文档结构和引用
  - 修复 API 命名、全设计缓存键
  - 移除冗余数组转换
  - 避免去偏矩阵哈希路径中的不必要复制

### 新增 (2026-04-03 ~ 2026-04-07)

- **PR #1 — CoxPH 聚类稳健协方差**:
  - 新增 `cov_type="cluster"` 用于分组 sandwich 协方差估计
  - Breslow tie 处理改进
  - 新增 CoxPH 基准测试脚本

- **PR #2 — 运行时比较表**:
  - 跨 CPU/GPU 和外部框架的可复现运行时比较表
  - 添加多目标线性回归形状处理
  - 添加多目标 sklearn 和 R 基准测试脚本
  - 修复 Ridge.score CUDA 预测的主机转换
  - 优化诊断和逐步选择
  - 改进 Cox 推断路径
  - 修复跨模型的缓存/收敛处理

- **PR #3 — 基准测试结构重构**:
  - 重构基准测试结构并更新文档

- **PR #4 — 可插拔后端抽象**:
  - 创建 BackendBase ABC + NumPy/CuPy/Torch 实现
  - 移除冗余模型实现（两个 LinearRegression 类、三个 Ridge 变体）
  - 多后端支持的清晰路径
  - 用后端抽象层规范化代码库

- **PR #5 — Ridge 推断支持**:
  - 与 LinearRegression 完整推断对齐
  - `cov_type`: nonrobust/hc0/hc1 (CPU + GPU)
  - `summary()`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic`

- **PR #6 — Logistic Regression 评估指标**:
  - 综合评估指标：ROC, AUC, 混淆矩阵
  - `evaluate_binary_classification` 函数
  - 修复 CuPy 在 logistic 评估方法中的安全性
  - 添加 y_score 的有限性检查
  - 对齐 CuPy/Torch 精度回退与 NumPy
  - 通过委托消除指标重复
  - 缓存训练评估指标以供复用

- **PR #7, #8 — Bug 修复和实验结果**:
  - 各种 bug 修复
  - 更新实验结果

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
- 新增非参数能力与导出：
  - KDE：`fit_kde`、`kde_pdf`、`kde_bootstrap_confidence_interval`
  - KDE 核函数：`gaussian/rectangular/triangular/epanechnikov/biweight/cosine/optcosine/triweight`
  - KDE 带宽规则：`nrd0`、`nrd`
  - Kernel Regression：`fit_kernel_regression`、`kernel_regression_predict`、`KernelRegression`
  - Kernel Regression 新增 `kernel_metric='full'|'diagonal'` 与 `bandwidth_per_feature`
- 新增 kernel regression 对标脚本：
  - `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- 非参数基准能力补充：
  - `dev/benchmarks/benchmark_kde_vs_scipy.py` 统一输出 statgpu CPU/GPU 与 SciPy 对照
  - `dev/benchmarks/benchmark_nonparametric_vs_r.py` 支持 `--statgpu-backend numpy/cupy`
  - `dev/benchmarks/benchmark_nonparametric_vs_r.py` 的 KDE CI 支持 `--ci-method normal/bootstrap`
  - 统一补齐 KDE / KernelReg NW / KernelReg Local Linear / KDE CI 的 CPU、GPU、R、SciPy、statsmodels 对照
- 新增 knockoff 基准脚本：
  - `dev/benchmarks/benchmark_knockoff_fixedx.py`
  - `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
  - `benchmark_knockoff_vs_baselines.py` 新增可选 `knockpy` 基线对比能力（环境可用时）
- 新增多重检验指南：
  - `docs/guides/multiple-testing-combine-pvalues.md`

### 改进

- Lasso `gpu_ols_inference` 路径将更多推断步骤放在 GPU 侧，减少 CPU 传输与 SciPy 依赖。
- `LinearRegression` 的 CPU HAC 路径新增自适应精度选择（mixed/float64 快速探测 + 形状分桶缓存），用于降低大规模场景回摆风险。
- Kernel Regression 的多维 local-linear 路径改为批处理向量化求解；远端运行 `run_id=20260415_120903` 在保持精度对齐下显著提速（dim=3：CPU 约 4.81x、GPU 约 115.5x；dim=5：CPU 约 5.39x、GPU 约 116.4x）。
- KDE 1D Numba 快路径将本地 SciPy 相对耗时从约 1.39x（慢）优化到约 0.58x（快）。
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
- 新增非参数验证覆盖：
  - `dev/tests/test_inference_kde.py`（9 passed, 1 skipped）
  - `dev/tests/test_nonparametric_kernel_regression.py`（13 passed, 1 skipped）
- Kernel Regression 公平核口径远端验证（`run_id=20260415_103036`）确认在对角核设置下与 statsmodels 达到机器精度对齐。
- 新增/刷新统一三方协方差对比产物（同设定、可审计）：
  - `results/remote_covariance_full_compare_2026-04-10.json`
  - 覆盖 `statsmodels` / `statgpu CPU` / `statgpu GPU` 的 `hc2/hc3/hac` 时间与精度对比
