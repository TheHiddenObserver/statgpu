# 修复 CuPy ElementwiseKernel SCAD/MCP 数值不稳定

## 背景

statgpu 项目中，CuPy 的 `ElementwiseKernel` 在 SCAD/MCP penalty 的 LLA continuation 路径中会导致系数发散到 NaN/inf。L1 penalty 正常，只有 SCAD/MCP 出问题。

## 任务

修复 `statgpu/glm_core/_solver.py` 中 `fista_lla_path` 函数的 CuPy fused path，使其对 SCAD/MCP penalty 也能正常工作。

## 关键文件

- `statgpu/glm_core/_solver.py` — 主要修改文件
  - `_get_sqerr_proximal_cupy()` (line 1395): CuPy fused kernel 定义
  - `fista_lla_path()` (line 814): LLA continuation 路径
  - line 986: 当前禁用了 CuPy fused path
- `statgpu/penalties/_scad.py` — SCAD penalty 的 `lla_weights()` 和 `proximal()` 方法
- `statgpu/penalties/_mcp.py` — MCP penalty 的 `lla_weights()` 和 `proximal()` 方法

## 当前 workaround

line 986 将 `if _is_quadratic and backend in ("torch", "cupy"):` 改为 `if _is_quadratic and backend == "torch":`，禁用 CuPy fused path，使用 unfused 路径。

## 调试方向

1. **禁用梯度裁剪**：`fista_lla_path` 的梯度裁剪（line 1031-1038）使用 `_to_numpy` 强制 GPU 同步，可能与 fused kernel 交互导致问题
2. **替换为 `@cp.fuse()`**：用 `@cp.fuse()` 装饰器替代 `ElementwiseKernel`，看是否解决问题
3. **添加发散检测**：在 FISTA 循环中检测 NaN/inf 系数，提前终止并回退到 unfused 路径
4. **检查 CuPy 版本**：在不同 CuPy 版本上测试

## 验证方法

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

for penalty in ['l1', 'scad', 'mcp']:
    model_np = PenalizedGeneralizedLinearModel(loss='squared_error', penalty=penalty, alpha=0.1, device='cpu', compute_inference=False, max_iter=200)
    model_np.fit(X_np, y_np)
    model_cu = PenalizedGeneralizedLinearModel(loss='squared_error', penalty=penalty, alpha=0.1, device='cuda', compute_inference=False, max_iter=200)
    model_cu.fit(X_cu, y_cu)
    coef_cu = _to_numpy(model_cu.coef_)
    corr = np.corrcoef(model_np.coef_, coef_cu)[0, 1]
    print(f'{penalty}: corr={corr:.6f} np_iter={model_np.n_iter_} cu_iter={model_cu.n_iter_}')
    assert corr > 0.999, f'{penalty} failed: corr={corr}'
    assert abs(model_np.n_iter_ - model_cu.n_iter_) < 50, f'{penalty} iter mismatch'
```

## 约束

- 不能破坏 L1 penalty 的现有行为（corr=1.000000）
- 不能破坏 Torch fused path
- 性能不能显著下降（unfused 路径 ~12ms per fit，fused ~3ms）
