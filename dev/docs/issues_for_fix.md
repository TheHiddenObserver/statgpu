# statgpu 精度与 GPU 效率问题清单

> **目标读者**：接手修复的 AI 模型。本文档包含可直接定位问题的代码路径、复现命令、和预期修复方向。
> **测试环境**：Tesla P100-SXM2-16GB, CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2
> **基准日期**：2026-05-31
> **当前状态**：Section A 816/816 PASS, pytest 51/51 PASS

---

## 一、精度问题

### 问题 1：logistic+l2 CuPy CV 精度差（最高优先级）

**现象**：
```
logistic + l2  corr_cu=0.999284  L2_cu=2.67e+0  alpha=DIFF
logistic + l2  corr_to=1.000000  L2_to=1.35e-14 alpha=DIFF
```

CuPy CV 选出的 coef 与 CPU 差异 2.67（L2 norm），但 Torch CV 完美匹配。说明问题出在 CuPy 的 IRLS/Newton 求解器，不是 CV 框架本身。

**对比**：同一组合的单次 fit（Section A）结果：
```
logistic+l2 n=500  cupy irls: corr OK, diff=6.22e-15
logistic+l2 n=5000 cupy irls: corr OK, diff=2.50e-15
```
单次 fit 没问题，说明 CV 循环中 fold 数据切片后 CuPy 求解器行为不同。

**定位路径**：
- `statgpu/linear_model/_penalized_cv.py` → warm-start 路径 line 280+
- `statgpu/linear_model/_penalized.py` → `_fit_cpu()` line 1023, `_fit_gpu()` line 1280+
- `statgpu/glm_core/_irls.py` → CuPy IRLS 求解

**复现命令**：
```bash
cd /root/statgpu && PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py
```

**修复方向**：
1. 对比 CV fold 中 CuPy IRLS 和 CPU IRLS 的逐迭代行为（coef、梯度、步长）
2. 检查 `_cv_cache` 中预计算的 XtX 是否在 CuPy 上精度一致
3. 检查 warm-start `_init_coef` 从 numpy 传入 CuPy 时是否有 dtype 转换问题

---

### 问题 2：logistic+l1/elasticnet CV 精度差

**现象**：
```
logistic + l1         corr_cu=0.999665  L2_cu=1.46e+0  alpha=DIFF
logistic + l1         corr_to=0.999665  L2_to=1.46e+0  alpha=DIFF
logistic + elasticnet corr_cu=1.000000  L2_cu=1.87e-05 alpha=DIFF
logistic + elasticnet corr_to=0.999837  L2_to=1.33e+0  alpha=DIFF
```

CuPy 和 Torch 选了不同的 alpha，导致 coef 不同。Torch elasticnet 的 L2 diff=1.33 说明选了完全不同的 alpha。

**根因**：logistic 的 FISTA 迭代在不同 backend 上收敛轨迹不同 → validation loss 不同 → alpha 选择不同。

**修复方向**：
1. 统一 CV 中各 backend 的求解器参数（max_iter、tol、warm-start 策略）
2. 或者在 CV 评估时使用更宽松的 alpha 匹配阈值

---

### 问题 3：logistic+none FISTA GPU 收敛到错误解（Section A）

**现象**（Section A）：
```
logistic+none n=2000 cupy  fista:  vs_CPU=2.02  coef=88.37 vs CPU 94.56
logistic+none n=2000 torch fista:  vs_CPU=2.02  coef=88.37 vs CPU 94.56
logistic+none n=5000 cupy  fista:  vs_CPU=1.42  coef=75.80 vs CPU 70.11
logistic+none n=5000 torch fista:  vs_CPU=1.42  coef=75.80 vs CPU 70.11
```

GPU FISTA 对 logistic+none 收敛到与 CPU 完全不同的解。注意 newton/lbfgs 求解器在同组合上没有此问题。

**定位**：`statgpu/glm_core/_solver.py` → `fista_solver()` 的 FISTA 内循环。
- 检查 `_fused_glm_value_and_gradient` 在 logistic 上的实现
- 检查 Armijo backtracking 在 GPU 上的数值行为
- 检查梯度裁剪逻辑（line 1028-1036）

**复现**：
```bash
PYTHONPATH=/root/statgpu python dev/tests/_bench_full_matrix.py --section A
# 搜索 "logistic+none" 相关行
```

---

### 问题 4：logistic+l2 FISTA 未收敛（Section A）

**现象**：
```
logistic+l2 n=500 cupy fista: vs_CPU=3.95e-3, iter=2000 (hit max_iter)
logistic+l2 n=500 cupy fista_bb: vs_CPU=3.95e-3, iter=2000 (hit max_iter)
```

CuPy FISTA 在 logistic+l2 上迭代 2000 次仍未收敛，而 CPU 只需约 333 次。

**修复方向**：检查 CuPy FISTA 的 Lipschitz 常数计算和步长是否与 CPU 一致。

---

### 问题 5：CV alpha 选择不一致

**现象**：20 个 CV 测试组合中，12 个 alpha=DIFF。

| alpha=OK | alpha=DIFF |
|----------|------------|
| squared_error+mcp | squared_error+l2, l1, elasticnet, scad |
| poisson+l1, l2, elasticnet | poisson+scad, mcp |
| gamma 全部 5 个 | logistic 全部 5 个 |

**根因**：不同 backend 的求解器数值路径不同 → validation loss 不同 → 选了不同 alpha。

**修复方向**：
1. 统一各 backend 的求解器行为（理想方案）
2. 或者在 CV 中使用 backend 一致的求解器（如始终用 CPU 求解 validation loss）

---

## 二、GPU 效率问题

### 问题 6：GPU FISTA 小/中规模比 CPU 慢 10-50 倍

**现象**（Section A，n=500 p=50）：

| Family | Penalty | CuPy speedup | Torch speedup |
|--------|---------|--------------|---------------|
| squared_error | none | 0.04x | 0.08x |
| squared_error | group_lasso | 0.02x | 0.07x |
| squared_error | group_mcp | 0.02x | 0.03x |
| logistic | none | 0.08x | 0.15x |
| tweedie | l2 (irls) | 0.01x | — |

**根因**：
1. 每次 FISTA 迭代有 3-5 个 GPU kernel launch，launch 开销 > 计算收益
2. CuPy `ElementwiseKernel` 首次 JIT 编译开销
3. 梯度裁剪中的 `_to_numpy()` 强制 GPU→CPU 同步

**关键代码**：`statgpu/glm_core/_solver.py` → FISTA 内循环 gradient clipping 部分

**修复方向**：
1. 梯度裁剪改成 device-side 判断，去掉 `_to_numpy` 同步
2. 对小问题（n*p < 阈值）自动 fallback 到 CPU
3. 考虑 `@cp.fuse()` 替代 `ElementwiseKernel` 减少 JIT 开销

---

### 问题 7：GPU FISTA 大规模 squared_error 仍慢

**现象**（n=5000 p=500）：

| Family | Penalty | CuPy speedup | Torch speedup |
|--------|---------|--------------|---------------|
| squared_error | l2 (fista) | 0.08x | 0.17x |
| squared_error | l1 (fista) | 0.69x | 0.60x |
| squared_error | scad | 1.48x | 1.61x |
| squared_error | mcp | 1.43x | 1.55x |

L1 的 FISTA 仍比 CPU 慢，但 SCAD/MCP 已经有加速（因为迭代次数多，摊薄了 kernel 开销）。

**修复方向**：对 squared_error 的 FISTA，预计算 XtX 并用 `XtX @ y_k - Xty` 替代 `X.T @ (X @ y_k - y)`。

---

### 问题 8：GPU CV 比 CPU 慢 2-14 倍

**现象**：

| Loss | Penalty | CPU(ms) | CuPy(ms) | Torch(ms) | CuPy 慢 | Torch 慢 |
|------|---------|---------|----------|-----------|---------|----------|
| squared_error | scad | 3234 | 18392 | 14290 | 5.7x | 4.4x |
| squared_error | mcp | 3494 | 18977 | 15245 | 5.4x | 4.4x |
| logistic | scad | 8691 | 125258 | 80142 | 14.4x | 9.2x |
| poisson | mcp | 4693 | 70884 | 50500 | 15.1x | 10.8x |

**根因**：
1. CV 对每个 alpha × 每个 fold 调用 `model.fit()`，每次有 kernel launch 开销
2. 虽然已添加 `_cv_cache` 预计算 XtX，但 warm-start 路径仍需多次 `model.fit()` 调用
3. 每次 `model.fit()` 的设备初始化、数据传输等固定开销

**修复方向**：
1. 将 FISTA 内循环直接实现在 CV 代码中，避免反复调用 `model.fit()`
2. 在 device 上保持数据，避免 fold 间的数据传输
3. 考虑 batch CV（多个 alpha 同时求解）

---

### 问题 9：GPU 仅在特定求解器上有加速

**GPU 有加速的情况**（n=5000 p=500，speedup > 2x）：

| Family | Penalty | Backend | Solver | Speedup |
|--------|---------|---------|--------|---------|
| squared_error | l2 | cupy | exact | 3.35x |
| squared_error | l2 | cupy | irls | 2.84x |
| logistic | none | torch | newton | 3.00x |
| poisson | none | cupy | newton | 3.16x |
| tweedie | none | torch | newton | 3.24x |

**规律**：`newton`/`irls`/`exact` 求解器 GPU 加速好（每次迭代做矩阵分解，GPU 并行优势大）。`fista` 仅在大规模 SCAD/MCP 和 torch backend 上有加速。

---

## 三、已修复的问题（不需要再处理）

| 问题 | 修复方式 |
|------|---------|
| CuPy fused kernel SCAD/MCP 发散 | `_solver.py:987` 启用 CuPy fused path |
| logistic 大 eta 数值不稳定 | `_logistic.py` 改用 stable softplus 公式 |
| CV Ridge 精度不足 | `_penalized_cv.py` 统一 eigendecomposition |
| CV XtX 重复计算 | `_penalized.py` 添加 `_cv_cache` |
| CV cache 被 fit() 删除 | `_penalized.py` 修改 cache 生命周期 |
| CuPy NB IRLS 发散 | `_irls.py` 添加对称化、jitter、residual fallback |
| group MCP/SCAD proximal 精度 | `_group_mcp.py` / `_group_scad.py` vectorized proximal 对齐 |
| 梯度裁剪 GPU→CPU 同步 | `_solver.py` 改成 device-side 裁剪 |

---

## 四、测试命令

```bash
# 远程服务器
ssh -p 28838 root@hz-4.matpool.com
source /root/miniconda3/envs/myconda/bin/conda.sh && conda activate myconda
cd /root/statgpu

# 1. pytest（51 个测试，含回归测试）
PYTHONPATH=/root/statgpu python -m pytest dev/tests/test_glm_penalty_review_fixes.py -v

# 2. Section A 全量 fit 测试（816 个测试，7 families × 10 penalties × 多 solver × 3 backends × 3 scales）
PYTHONPATH=/root/statgpu python dev/tests/_bench_full_matrix.py --section A

# 3. CV benchmark（4 losses × 5 penalties × 3 backends = 20 组合）
PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py

# 4. 单独测试某个组合
PYTHONPATH=/root/statgpu python -c "
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel as PGLM
import numpy as np
X = np.random.randn(5000, 500)
y = (X @ np.ones(500) + np.random.randn(5000) > 0).astype(float)
m = PGLM(loss='logistic', penalty='l2', alpha=0.01, device='cuda', max_iter=500)
m.fit(X, y)
print(f'iter={m.n_iter_} coef_norm={np.linalg.norm(m.coef_):.4f}')
"
```

---

## 五、关键文件索引

| 文件 | 内容 | 关注行号 |
|------|------|---------|
| `statgpu/glm_core/_solver.py` | FISTA/ADMM/IRLS 求解器，fused kernel | 985-1060 (fused path), 1100-1200 (unfused FISTA) |
| `statgpu/glm_core/_logistic.py` | Logistic loss value/gradient | stable softplus 实现 |
| `statgpu/glm_core/_irls.py` | IRLS 求解器 | CuPy NB 修复 |
| `statgpu/linear_model/_penalized.py` | PGLM 主类，GPU/CPU 调度 | 1023 (_fit_cpu), 1280 (_fit_gpu), 1680 (_fit_torch) |
| `statgpu/linear_model/_penalized_cv.py` | CV 循环 | 280+ (warm-start path), 94 (Ridge batch solve) |
| `statgpu/penalties/_group_mcp.py` | Group MCP proximal | vectorized 实现 |
| `statgpu/penalties/_group_scad.py` | Group SCAD proximal | vectorized 实现 |
| `dev/tests/test_glm_penalty_review_fixes.py` | 回归测试 | 51 个测试 |
| `dev/tests/_bench_full_matrix.py` | 全量 benchmark | Section A-H |
| `dev/tests/benchmark_cv_full.py` | CV benchmark | 4 losses × 5 penalties |
