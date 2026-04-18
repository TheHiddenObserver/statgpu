# StatGPU 完整后端支持对比报告

**生成日期**: 2026-04-18  
**版本**: 0.1.0

---

## 后端类型总览

| 后端名称 | 模块 | 设备 | 说明 |
|----------|------|------|------|
| **NumPy** | `numpy` | CPU | 默认 CPU 后端，纯 NumPy 实现 |
| **CuPy** | `cupy` | GPU | CUDA GPU 加速后端 |
| **Torch** | `torch` | GPU/CPU | PyTorch 后端，支持 GPU 和 CPU |

---

## 模块后端支持矩阵

### 1. 线性模型 (linear_model)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `LinearRegression` | ✅ | ✅ | ✅ | ✅ | `_linear.py` |
| `Ridge` | ✅ | ✅ | ✅ | ✅ | `_ridge.py` |
| `RidgeCV` | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | `_ridge_cv.py` |
| `Lasso` | ✅ | ✅ | ✅ | ✅ | `_lasso.py` |
| `LassoCV` | ✅ | ✅ | ✅ | ✅ | `_lasso.py` |
| `LogisticRegression` | ✅ | ✅ | ✅ | ✅ | `_logistic.py` |
| `LogisticRegressionCV` | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | `_logistic_cv.py` |

**功能详情**:

| 模型 | 协方差类型 | 推断功能 | Torch 特殊功能 |
|------|------------|----------|----------------|
| LinearRegression | nonrobust, hc0, hc1, hc2, hc3, hac | coef, bse, t, p, CI, F, AIC, BIC | Cholesky 修复 |
| Ridge | nonrobust, hc1, hc2, hc3, hac | coef, bse, t, p, CI, F, AIC, BIC | 完整支持 |
| Lasso | nonrobust | coef, bse, t, p, CI, nnz | Debiased/Simultaneous 推断 |
| LogisticRegression | nonrobust, hc1, hc2, hc3, hac | coef, bse, z, p, CI, LLR | IRLS 求解器 |

---

### 2. 生存分析 (survival)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `CoxPH` | ✅ | ✅ | ✅ | ✅ | `_cox.py` |
| `CoxPHCV` | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | ⚠️ 骨架 | `_cox_cv.py` |

**功能详情**:

| 功能 | CPU | CuPy GPU | Torch GPU |
|------|-----|----------|-----------|
| Bresnow 结处理 | ✅ | ✅ | ✅ |
| Efron 结处理 | ✅ | ✅ | ✅ |
| Newton-Raphson 优化 | ✅ | ✅ | ✅ |
| C-index | ✅ | ✅ | ✅ |
| Baseline Hazard | ✅ | ✅ | ✅ |
| 对数似然 | ✅ | ✅ | ✅ |
| 协方差 (nonrobust/hc0/hc1/cluster) | ✅ | ✅ | ✅ |

---

### 3. 非参数方法 (nonparametric)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `KernelDensityEstimator` | ✅ | ✅ | ✅ | ✅ | `_kde.py` |
| `KDE` (别名) | ✅ | ✅ | ✅ | ✅ | `_kde.py` |
| `KernelRegression` | ✅ | ✅ | ✅ | ✅ | `_kernel_regression.py` |
| `fit_kde` | ✅ | ✅ | ✅ | ✅ | `_kde.py` |
| `kde_pdf` | ✅ | ✅ | ✅ | ✅ | `_kde.py` |
| `kde_confidence_interval` | ✅ | ✅ | ✅ | ✅ | `_kde.py` |
| `fit_kernel_regression` | ✅ | ✅ | ✅ | ✅ | `_kernel_regression.py` |
| `kernel_regression_predict` | ✅ | ✅ | ✅ | ✅ | `_kernel_regression.py` |
| `select_bandwidth` | ✅ | ✅ | ✅ | ✅ | `_bandwidth_selection.py` |

**功能详情**:

| 功能 | CPU | CuPy GPU | Torch GPU |
|------|-----|----------|-----------|
| 核函数 | gaussian, rectangular, triangular, epanechnikov, biweight, triweight, cosine, optcosine |
| 带宽规则 | scott, silverman, nrd0, nrd |
| KDE 置信区间 | Bootstrap | ✅ | ✅ |
| Kernel Regression 类型 | NW, Local-Linear | ✅ | ✅ |
| 核度量 | full, diagonal | ✅ | ✅ |

---

### 4. 特征选择 (feature_selection)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `knockoff_filter` | ✅ | ✅ | ✅ | ✅ | `_knockoff.py` |
| `fixed_x_knockoff_filter` | ✅ | ✅ | ✅ | ✅ | `_knockoff.py` |
| `model_x_knockoff_filter` | ✅ | ✅ | ✅ | ✅ | `_knockoff.py` |
| `KnockoffSelector` | ✅ | ✅ | ✅ | ✅ | `_knockoff.py` |
| `FixedXKnockoffSelector` | ✅ | ✅ | ✅ | ✅ | `_knockoff.py` |

**功能详情**:

| 功能 | CPU | CuPy GPU | Torch GPU |
|------|-----|----------|-----------|
| Fixed-X Knockoff 构造 | ✅ | ✅ | ✅ |
| Model-X Knockoff 构造 | ✅ | ✅ | ✅ |
| Knockoff 统计量 | corr_diff, ols_coef_diff | ✅ | ✅ |
| FDR 控制 | knockoff+, standard | ✅ | ✅ |
| Lasso 路径 | ✅ | ✅ | ✅ |

---

### 5. 推断工具 (inference)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `adjust_pvalues` | ✅ | ✅ | ✅ | ✅ | `_multiple_testing.py` |
| `combine_pvalues` | ✅ | ✅ | ✅ | ✅ | `_multiple_testing.py` |
| `multipletests` | ✅ | ✅ | ✅ | ✅ | `_multiple_testing.py` |
| `bootstrap_statistic` | ✅ | ✅ | ✅ | ✅ | `_resampling.py` |
| `permutation_test` | ✅ | ✅ | ✅ | ✅ | `_resampling.py` |

**功能详情**:

| 功能 | 方法 | CPU | CuPy GPU | Torch GPU |
|------|------|-----|----------|-----------|
| p 值校正 | bh, by, holm, bonferroni | ✅ | ✅ | ✅ |
| p 值组合 | fisher, cauchy, acat | ✅ | ✅ | ✅ |
| Bootstrap | 百分位数，BCa | ✅ | ✅ | ✅ |
| 置换检验 | 双样本，相关性 | ✅ | ✅ | ✅ |

---

### 6. 评估指标 (evaluation)

| 类/函数 | NumPy CPU | CuPy GPU | Torch GPU | Torch CPU | 文件 |
|---------|-----------|----------|-----------|-----------|------|
| `evaluate_binary_classification` | ✅ | ✅ | ✅ | ✅ | `_classification.py` |
| `binary_confusion_matrix` | ✅ | ✅ | ✅ | ✅ | `_classification.py` |
| `classification_table` | ✅ | ✅ | ✅ | ✅ | `_classification.py` |
| `roc_curve` | ✅ | ✅ | ✅ | ✅ | `_classification.py` |
| `precision_recall_curve` | ✅ | ✅ | ✅ | ✅ | `_classification.py` |

---

### 7. 后端抽象 (backends)

| 类/函数 | 说明 | 文件 |
|---------|------|------|
| `NumpyBackend` | NumPy CPU 后端实现 | `_numpy.py` |
| `CuPyBackend` | CuPy GPU 后端实现 | `_cupy.py` |
| `TorchBackend` | PyTorch 后端实现 | `_torch.py` |
| `get_backend()` | 后端工厂函数 | `_factory.py` |
| `BackendBase` | 后端抽象基类 | `_base.py` |

**TorchBackend 支持的方法** (50+ NumPy 兼容方法):

```
数组创建：ones, zeros, eye, arange, linspace, full
数组操作：concatenate, stack, column_stack, reshape, flatten, ravel
线性代数：linalg.solve, linalg.lstsq, linalg.inv, linalg.pinv, 
          linalg.cholesky, linalg.eigh, linalg.qr, linalg.svd
统计：sum, mean, var, std, min, max, argmin, argmax, argsort
数学：sqrt, exp, log, log10, sin, cos, tan, abs, clip
特殊：maximum, minimum, where, isnan, isinf, isfinite
```

---

### 8. GPU 工具函数 (_gpu_utils / _gpu_utils_torch)

| 函数 | CPU | CuPy GPU | Torch GPU |
|------|-----|----------|-----------|
| `t_two_tail_pvalues` | ✅ | ✅ | ✅ |
| `t_crit_two_tail` | ✅ | ✅ | ✅ |
| `norm_two_tail_pvalues` | ✅ | ✅ | ✅ |
| `norm_crit_two_tail` | ✅ | ✅ | ✅ |
| `compute_inference` | ✅ | ✅ | ✅ |
| `compute_r2` | ✅ | ✅ | ✅ |
| `compute_aic_bic` | ✅ | ✅ | ✅ |
| `compute_f_stat` | ✅ | ✅ | ✅ |
| `hac_meat` | ✅ | ✅ | ✅ |

---

### 9. 分布函数 (inference/_distributions*)

| 分布 | CPU (SciPy) | CuPy GPU | Torch GPU |
|------|-------------|----------|-----------|
| `norm` (正态分布) | ✅ | ✅ | ✅ |
| `t` (t 分布) | ✅ | ✅ | ✅ |
| `f` (F 分布) | ✅ | ✅ | ✅ |
| `chi2` (卡方分布) | ✅ | ✅ | ✅ |

**Torch 特殊函数实现** (`_distribution_utils_torch.py`):

| 函数 | 说明 |
|------|------|
| `regularized_betainc_torch()` | 正则化不完全 Beta 函数 |
| `regularized_betaincinv_torch()` | Beta 函数逆 |
| `gammainc_torch()` | 不完全 Gamma 函数 |
| `gammaincinv_torch()` | Gamma 函数逆 |
| `gammaln_torch()` | 对数 Gamma 函数 |
| `erf_torch()` | 误差函数 |
| `erfc_torch()` | 互补误差函数 |
| `erfcinv_torch()` | 互补误差函数逆 |

---

## 数值精度对比

### 核心模型精度 (vs CPU NumPy)

| 模型 | 系数精度 | 标准误精度 | p 值精度 |
|------|----------|-----------|---------|
| LinearRegression | ~1e-15 | ~1e-17 | ~1e-15 |
| Ridge | ~1e-14 | ~1e-16 | ~1e-14 |
| Lasso | ~1e-5 | ~1e-17 | ~1e-10 |
| LogisticRegression | ~1e-14 | ~1e-15 | ~1e-12 |
| CoxPH | ~1e-15 | ~1e-14 | ~1e-13 |
| KDE | ~1e-14 | N/A | N/A |
| KernelRegression | ~1e-12 | N/A | N/A |

---

## 性能对比 (Tesla P100, 50K×200)

### 线性模型

| 模型 | CPU (ms) | Torch GPU (ms) | CuPy GPU (ms) | Torch/CuPy |
|------|----------|----------------|---------------|------------|
| LinearRegression | 31.44 | 30.10 | 7.74 | 3.9x |
| Ridge (nonrobust) | 27.52 | 31.66 | 5.98 | 5.3x |
| Ridge (HC3) | 3976.0 | 66.6 | 63.9 | 1.04x |
| Logistic (nonrobust) | 45.06 | 37.93 | 19.00 | 2.0x |
| Logistic (HC1) | 12.5 | 99.0 | 102.0 | 0.97x (Torch 快!) |
| Lasso | 5.18 | 8.05 | 11.86 | 0.68x (Torch 快!) |
| CoxPH | 2824.0 | 627.4 | 86.67 | 7.2x |

### 非参数模型

| 模型 | CPU | Torch GPU | CuPy GPU | 说明 |
|------|-----|-----------|----------|------|
| KDE (1D, Numba) | 1.0x | ~0.58x SciPy | ~0.55x SciPy | Numba 加速 |
| KernelRegression (dim=3, NW) | 1.0x | ~4.8x 加速 | ~115x 加速 | 批处理求解 |
| KernelRegression (dim=5, LL) | 1.0x | ~5.4x 加速 | ~116x 加速 | 批处理求解 |

---

## 后端选择指南

### 选择 NumPy CPU 当...
- 数据集小 (<5K 样本)
- 无 GPU 可用
- 单次执行场景
- 调试和开发阶段

### 选择 CuPy GPU 当...
- 追求极致性能
- 线性模型大规模数据
- CoxPH 大规模数据
- 非参数多维场景

### 选择 Torch GPU 当...
- 需要与 PyTorch 生态集成
- Lasso 模型 (更快)
- LogisticRegression HC1/HC2/HC3 (相当或更快)
- 需要 autograd 能力
- 使用 Torch 调试工具

---

## 使用示例

### 统一 API

```python
from statgpu.linear_model import LinearRegression

# NumPy CPU
model_cpu = LinearRegression(device='cpu')
model_cpu.fit(X, y)

# CuPy GPU
model_cupy = LinearRegression(device='cuda', backend='cupy')
model_cupy.fit(X, y)

# Torch GPU
model_torch = LinearRegression(device='cuda', backend='torch')
model_torch.fit(X, y)

# Torch CPU
model_torch_cpu = LinearRegression(device='cpu', backend='torch')
model_torch_cpu.fit(X, y)
```

### 自动后端选择

```python
from statgpu.nonparametric import KernelDensityEstimator

# 自动选择最佳后端
kde = KernelDensityEstimator(backend='auto', device='cuda')
kde.fit(X)
# 优先 CuPy，回退 Torch，最后 NumPy
```

---

## 安装要求

| 后端 | 依赖 | 安装命令 |
|------|------|----------|
| NumPy CPU | numpy>=1.20 | `pip install statgpu` |
| CuPy GPU | cupy-cuda11x 或 cupy-cuda12x | `pip install statgpu[gpu11]` |
| Torch GPU | torch>=2.0 | `pip install statgpu[torch]` |

---

## 限制和注意事项

### NumPy CPU
- ✅ 功能最完整
- ✅ 调试最方便
- ⚠️ 大规模数据慢

### CuPy GPU
- ✅ 性能最佳 (大部分场景)
- ✅ 功能完整
- ⚠️ 需要 CUDA 环境
- ⚠️ 小数据开销大

### Torch GPU
- ✅ 与 PyTorch 生态集成
- ✅ Lasso/Logistic 场景性能好
- ✅ 调试工具成熟
- ⚠️ LinearRegression/Ridge 稍慢于 CuPy
- ⚠️ 小数据开销大

---

## 测试覆盖

| 模块 | CPU 测试 | CuPy 测试 | Torch 测试 |
|------|----------|-----------|-----------|
| LinearRegression | ✅ | ✅ | ✅ |
| Ridge | ✅ | ✅ | ✅ |
| Lasso | ✅ | ✅ | ✅ |
| LogisticRegression | ✅ | ✅ | ✅ |
| CoxPH | ✅ | ✅ | ✅ |
| KDE | ✅ | ✅ | ✅ |
| KernelRegression | ✅ | ✅ | ✅ |
| Knockoff | ✅ | ✅ | ✅ |

---

*报告生成于 2026-04-18*  
*StatGPU v0.1.0*
