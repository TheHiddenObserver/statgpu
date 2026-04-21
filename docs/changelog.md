# Changelog

> 语言：中文  
> 最后更新：2026-04-18  
> 页面定位：变更记录  
> 切换：[English](en/changelog.md)

语言切换：[English](en/changelog.md)

## 2026-04

### 新增 (2026-04-21)

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
