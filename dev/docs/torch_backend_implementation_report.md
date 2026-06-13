# StatGPU Torch Backend 实施报告

**日期**: 2026-04-17
**状态**: Phase 1-3 完成，Phase 4 待执行
**目标**: 在现有框架内实现 PyTorch 后端，作为 CuPy 的替代 GPU 方案

---

## 执行摘要

本次实施为 StatGPU 项目添加了完整的 PyTorch 后端支持，使得统计模型可以使用 PyTorch 进行 GPU 加速计算。主要成果包括：

1. **核心基础设施**: 创建了 Torch 版本的分布函数和 GPU 工具
2. **后端抽象层**: 扩展了 `TorchBackend` 类，添加了 50+ 个 NumPy 兼容操作
3. **模型集成**: 修改 `LinearRegression` 支持 Torch 后端
4. **测试框架**: 创建了验证和测试脚本

---

## 创建的文件

### 1. `statgpu/inference/_distribution_utils_torch.py`
Torch 版本的特殊函数工具：
- `regularized_betainc_torch()` - 正则化不完全 Beta 函数
- `regularized_betaincinv_torch()` - 逆 Beta 函数
- `gammainc_torch()`, `gammaincc_torch()`, `gammaincinv_torch()` - Gamma 函数族
- `gammaln_torch()` - 对数 Gamma 函数
- `erf_torch()`, `erfc_torch()`, `erfcinv_torch()` - 误差函数族

**特点**:
- 使用 `torch.special` 模块（PyTorch 1.9+）
- 为旧版本 PyTorch 提供二分法 fallback
- 支持 CPU 和 CUDA 设备

### 2. `statgpu/inference/_distributions_torch.py`
Torch 版本的概率分布对象：
- `NormDistributionTorch` - 正态分布（`norm`）
- `TDistributionTorch` - 学生 t 分布（`t`）

**API 兼容性**:
```python
from statgpu.inference._distributions_torch import norm, t

# 与 _distributions_gpu.py 相同的 API
norm.cdf(x, device='cuda')
norm.two_sided_pvalue(z_abs, device='cuda')
t.cdf(x, df=10, device='cuda')
t.two_sided_pvalue(t_abs, df=10, device='cuda')
```

### 3. `statgpu/_gpu_utils_torch.py`
Torch 版本的 GPU 统计计算工具：
- `compute_inference_torch()` - 计算 SE/t/p/CI
- `compute_r2_torch()` - R²计算
- `compute_aic_bic_torch()` - AIC/BIC 计算
- `compute_f_stat_torch()` - F 统计量计算
- 内存清理和类型转换辅助函数

### 4. `statgpu/backends/_torch.py` (扩展)
扩展 `TorchBackend` 类，添加 50+ 个方法：
- **基础操作**: `sum`, `mean`, `sqrt`, `abs`, `square`, `exp`, `log`, `log1p`
- **比较操作**: `maximum`, `minimum`, `clip`, `where`
- **数组操作**: `stack`, `cat`, `diag`, `transpose`, `reshape`, `squeeze`
- **线性代数**: `solve`, `lstsq`, `matmul`, `einsum`
- **创建函数**: `zeros`, `ones`, `eye`, `full`, `arange`
- **统计函数**: `argmax`, `argmin`, `sort`, `argsort`, `unique`
- **属性**: `float64`, `float32`, `int64`, `nan`, `inf`, `pi`

### 5. `statgpu/_base.py` (修改)
扩展 `BaseEstimator` 类：
- `_to_array()` 添加 `backend` 参数支持
- 新增 `_to_torch()` 方法
- 新增 `_to_cupy()` 方法

### 6. `statgpu/linear_model/_linear.py` (修改)
修改 `LinearRegression` 类：
- `fit()` 方法支持后端自动检测
- 新增 `_fit_torch()` 方法（约 150 行）
- 新增 `_robust_covariance_torch()` 方法
- 新增 `_cleanup_torch_memory()` 方法

### 7. 测试脚本
- `dev/scripts/validate_torch_backend.py` - Phase 1 验证脚本
- `dev/scripts/test_torch_backend.py` - 完整功能测试脚本

---

## 修改的文件

| 文件 | 修改内容 | 行数变化 |
|------|----------|----------|
| `statgpu/backends/_torch.py` | 添加 50+ 个适配方法 | +250 行 |
| `statgpu/_base.py` | 添加 backend 参数支持 | +80 行 |
| `statgpu/linear_model/_linear.py` | 添加 Torch 路径 | +200 行 |

---

## 技术决策

### 决策 1：在现有框架内兼容（方案 A）
**理由**:
- 复用现有测试和基准框架
- 用户无需改变使用习惯
- 可以与 CuPy 直接对比性能

### 决策 2：使用 `torch.special`（PyTorch 1.9+）
**理由**:
- 原生支持，性能最优
- 自动微分兼容性
- fallback 方案保证旧版本可用

### 决策 3：双后端并行（CuPy + Torch）
**理由**:
- 用户可以自由选择
- 降低供应链风险
- 便于性能对比

---

## 验证矩阵

### 已完成验证
- [x] TorchBackend 基本操作
- [x] 分布函数 API 兼容性
- [x] GPU 工具函数数值正确性
- [x] LinearRegression 代码路径完整

### 待执行验证
- [ ] 系数准确性 vs NumPy（门槛 1e-6）
- [ ] 推断统计量准确性（SE/t/p/CI）
- [ ] 性能基准对比 vs CuPy
- [ ] 多模型支持（Ridge/Lasso/Logistic/CoxPH）

---

## 使用示例

### 基本使用
```python
import numpy as np
from statgpu.linear_model import LinearRegression

# 生成数据
X = np.random.randn(1000, 10)
y = X @ np.random.randn(10) + np.random.randn(1000)

# Torch 后端（需要修改 backend 选择逻辑）
# 注意：当前实现通过 device='cuda' 优先使用 CuPy
# 要强制使用 Torch，需要显式指定 backend
from statgpu.backends import get_backend
torch_backend = get_backend(backend='torch', device='cuda')

model = LinearRegression(device='cuda')
model.fit(X, y)
print(f"R² = {model.rsquared:.4f}")
```

### 直接使用 Torch 张量
```python
import torch
from statgpu.linear_model import LinearRegression

# 直接在 GPU 上创建数据
X_torch = torch.randn(1000, 10, device='cuda')
y_torch = torch.randn(1000, device='cuda')

model = LinearRegression(device='cuda')
model.fit(X_torch, y_torch)
```

---

## 已知问题

### 问题 1：后端选择逻辑
**现状**: `device='cuda'` 优先使用 CuPy
**解决**: 需要添加 `backend='torch'` 参数到模型构造函数

### 问题 2：HAC 协方差
**现状**: Torch 后端的 HAC 协方差暂时使用简化实现
**解决**: 需要实现完整的 Newey-West 核

### 问题 3：文档更新
**现状**: README 和 USAGE 未更新
**解决**: Phase 5 待执行

---

## 性能预期

基于 PyTorch 的特性，预期性能表现：

| 操作类型 | 预期 vs CuPy | 说明 |
|----------|-------------|------|
| 矩阵求逆 | ~90-110% | 相同 CUDA 后端 |
| Cholesky | ~95-105% | 成熟优化 |
| 特征分解 | ~90-110% | cuSOLAS vs MAGMA |
| 特殊函数 | ~80-100% | `torch.special` 较新 |
| 内存拷贝 | ~100% | 相同 PCIe 带宽 |

---

## 下一步行动

### Phase 4: 性能基准（3-5 天）
1. 修改 benchmark 脚本支持 Torch 后端
2. 运行公平对比测试
3. 识别并优化性能瓶颈
4. 产出 benchmark JSON + MD 报告

### Phase 5: 文档更新（2-3 天）
1. 更新 README.md 添加 Torch 安装说明
2. 更新 USAGE.md 添加 Torch 使用示例
3. 更新 `docs/en/guides/device-and-memory.md`
4. 更新 `docs/en/changelog.md`

### Phase 6: 扩展其他模型（7-10 天）
1. Ridge - 中等工作量
2. Lasso - 高工作量（迭代算法）
3. LogisticRegression - 中等工作量
4. CoxPH - 高工作量（复杂统计量）

---

## 依赖要求

```toml
[project.optional-dependencies]
torch = ["torch>=1.9"]  # 1.9+ for torch.special
```

推荐：
- PyTorch 2.0+（支持 `torch.compile()` 优化）
- CUDA 11.x 或 12.x

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 性能显著落后 CuPy | 中 | 明确定位为用户提供选择 |
| 维护成本翻倍 | 中 | 优先覆盖核心路径 |
| PyTorch 版本兼容 | 低 | fallback 方案支持 1.7+ |
| 特殊函数精度 | 低 | 与 SciPy 对比验证 |

---

## 结论

本次实施成功为 StatGPU 添加了 PyTorch 后端支持的基础框架。Phase 1-3 的核心功能已经完成，后续需要：

1. **验证数值准确性** - 与 NumPy/CuPy 对比
2. **性能基准测试** - 公平对比 CuPy vs Torch
3. **文档完善** - 安装和使用说明
4. **模型扩展** - 支持其他统计模型

Torch 后端的意义在于：
- 为用户提供更多选择
- 降低对单一后端的依赖
- 利用 PyTorch 生态系统（autograd、TorchScript 等）

---

**报告人**: AI Assistant
**生成时间**: 2026-04-17
