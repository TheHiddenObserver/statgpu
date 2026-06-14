# CuPy ElementwiseKernel 数值不稳定问题

## 问题描述

CuPy 的 `ElementwiseKernel` 在 SCAD/MCP 的 LLA continuation 路径中会导致系数发散到 NaN/inf。

## 复现步骤

```python
import numpy as np
import cupy as cp
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
from statgpu.backends import _to_numpy

rng = np.random.RandomState(42)
n, p = 500, 20
X_np = rng.randn(n, p)
y_np = X_np @ np.linspace(2, 0.5, p) + rng.randn(n) * 0.5
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

# SCAD on CuPy — 发散
model = PenalizedGeneralizedLinearModel(
    loss="squared_error", penalty="scad", alpha=0.1,
    device="cuda", compute_inference=False, max_iter=200
)
model.fit(X_cu, y_cu)
coef = _to_numpy(model.coef_)
print(f"coef: {coef}")  # 全是 NaN 或 inf

# SCAD on numpy — 正确
model_np = PenalizedGeneralizedLinearModel(
    loss="squared_error", penalty="scad", alpha=0.1,
    device="cpu", compute_inference=False, max_iter=200
)
model_np.fit(X_np, y_np)
print(f"coef: {model_np.coef_[:3]}")  # 正确值
```

## 关键代码位置

- **fused kernel**: `statgpu/glm_core/_solver.py` `_get_sqerr_proximal_cupy()` (line 1395)
- **LLA 路径**: `statgpu/glm_core/_solver.py` `fista_lla_path()` (line 814)
- **当前 workaround**: line 986 禁用 CuPy fused path，使用 unfused 路径

## 已排除的原因

1. ✅ Fused kernel 本身逻辑正确（单独测试 thresh=0 和 thresh=1e-15 结果一致）
2. ✅ LLA 权重计算正确（numpy 和 cupy 结果一致）
3. ✅ 梯度裁剪逻辑正确
4. ✅ 不是 pip 缓存问题（清缓存后仍然复现）

## 关键观察

- L1 penalty 在 CuPy fused path 上正常工作（corr=1.000000）
- SCAD/MCP 在 CuPy fused path 上发散（corr=nan，iter=3680）
- SCAD/MCP 在 unfused path 上正常工作（corr=1.000000，iter=426）
- Torch fused path（torch.compile）对所有 penalty 都正常

## 可能的根因

`fista_lla_path` 的梯度裁剪逻辑（line 1031-1038）使用 `_to_numpy` 强制 GPU 同步，这可能与 CuPy `ElementwiseKernel` 的 JIT 编译交互导致数值不稳定。

## 调试方向

1. **禁用梯度裁剪**：注释掉 line 1031-1038 的梯度裁剪代码，测试 fused path 是否正常
2. **替换 `ElementwiseKernel` 为 `@cp.fuse()`**：用 CuPy 的 `@cp.fuse()` 装饰器替代 `ElementwiseKernel`，看是否解决问题
3. **添加发散检测**：在 FISTA 循环中检测 NaN/inf 系数，提前终止
4. **检查 CuPy 版本**：在不同 CuPy 版本上测试，可能是 CuPy bug

## 验证标准

修复后需要满足：
```python
# SCAD on CuPy 与 numpy 结果一致
assert np.corrcoef(model_np.coef_, _to_numpy(model_cu.coef_))[0, 1] > 0.999
# 迭代次数相近
assert abs(model_np.n_iter_ - model_cu.n_iter_) < 50
```
