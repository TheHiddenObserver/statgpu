# statgpu 问题总结：精度与性能

> 本文档汇总当前代码中已知的精度和性能问题，供新模型快速了解情况并定位修复。
> 测试环境：Tesla P100-SXM2-16GB，CuPy 13.6.0，Torch 2.0.0+cu117

---

## 一、精度问题

### 1.1 最严重：logistic+none FISTA GPU 收敛到错误解

**现象**：logistic 不加惩罚时，GPU fista/fista_bb 收敛到与 CPU 完全不同的解。

| Scale | Backend | vs_CPU | ||coef|| CPU | ||coef|| GPU |
|-------|---------|--------|-------------|-------------|
| n=2000 p=200 | cupy fista | **2.02** | 94.56 | 88.37 |
| n=2000 p=200 | torch fista | **2.02** | 94.56 | 88.37 |
| n=5000 p=500 | cupy fista | **1.42** | 70.11 | 75.80 |
| n=5000 p=500 | torch fista | **1.42** | 70.11 | 75.80 |

**定位**：`statgpu/glm_core/_solver.py` 中 `fista_lla_path()` 和 `fista_solver()` 的 FISTA 内循环。
- GPU 的 `_fused_glm_value_and_gradient` 或 Armijo backtracking 与 CPU 的数值路径不同
- 检查梯度裁剪逻辑（`_solver.py:1028-1036`）在 GPU 上的行为

**关键代码**：
- `statgpu/glm_core/_solver.py` — `fista_lla_path()` line 985+, `fista_solver()` line 1100+
- `statgpu/glm_core/_logistic.py` — `LogisticLoss.value()` / `.gradient()`

---

### 1.2 CV 精度：logistic 系列 CV 系数差异大

**现象**：logistic 的 CV 结果在不同 backend 间差异显著。

| Loss | Penalty | CuPy corr | Torch corr | CuPy L2 diff | Torch L2 diff |
|------|---------|-----------|------------|--------------|---------------|
| logistic | l2 | 0.999117 | 0.999798 | 2.56e+0 | 1.32e+0 |
| logistic | l1 | 0.999864 | 0.999847 | 3.01e-1 | 6.99e-1 |
| logistic | elasticnet | 0.999497 | 0.999721 | 1.63e-1 | 1.27e+0 |

**根因**：
1. CV 中每个 fold 的 `model.fit()` 调用不同的求解器路径（CPU 用 numpy，GPU 用 cupy/torch）
2. 不同求解器对同一问题收敛到不同的局部最优（logistic 损失非凸）
3. CV 的 alpha 选择依赖 validation loss，不同 coef 导致不同 alpha 被选中

**定位**：`statgpu/linear_model/_penalized_cv.py` 的 warm-start 路径（line 280+）。
- `_evaluate_single()` 在 numpy 上计算 validation loss，但 coef 来自 GPU 求解器
- 需要确保 CV 和 refit 使用一致的求解器

---

### 1.3 CV alpha 选择不一致（60% DIFF）

**现象**：20 个 CV 测试条件中，12 个显示 alpha=DIFF（GPU 选了不同的最优 alpha）。

**受影响组合**：
- squared_error: l2, elasticnet, scad, mcp
- logistic: 全部 5 个 penalty
- poisson: elasticnet, scad, mcp

**不受影响**：
- squared_error: l1
- poisson: l1, l2
- gamma: 全部 5 个 penalty

**根因**：同 1.2，不同 backend 的求解器数值路径不同 → coef 不同 → validation loss 不同 → 选了不同 alpha。

---

### 1.4 negative_binomial+l2 IRLS CuPy 发散

**现象**：`n=2000, p=200` 时 CuPy IRLS 迭代 270 次（CPU 只需 35 次），coef 差异 8.04e-5。

**定位**：`statgpu/glm_core/_irls.py` 中 CuPy 的 IRLS 实现。
- 可能是 CuPy 的线性求解（`_solve`）精度不如 numpy
- 或者 Armijo backtracking 的 tolerance 在 CuPy 上不同

---

### 1.5 group_mcp/group_scad 小规模精度

**现象**：`n=500, p=50` 时 CuPy fista 的 group_mcp/group_scad vs_CPU 差异 2.6e-3 ~ 2.9e-3。

**定位**：`statgpu/penalties/_group_mcp.py` / `_group_scad.py` 的 proximal 算子。
- 可能是 group proximal 的 CuPy 实现与 numpy 有细微差异

---

## 二、性能问题

### 2.1 GPU FISTA 比 CPU 慢（小/中规模）

**现象**：`n=500, p=50` 和 `n=2000, p=200` 时，GPU FISTA 比 CPU 慢 10-50 倍。

| Family | Penalty | Scale | CuPy speedup | Torch speedup |
|--------|---------|-------|--------------|---------------|
| squared_error | none | 500x50 | 0.04x | 0.08x |
| squared_error | group_lasso | 500x50 | 0.02x | 0.07x |
| squared_error | group_lasso | 2000x200 | 0.02x | 0.05x |
| logistic | none | 500x50 | 0.08x | 0.15x |

**根因**：
1. GPU kernel launch 开销在小问题上占主导（每次 FISTA 迭代 3-5 个 kernel）
2. CuPy `ElementwiseKernel` 的 JIT 编译首次调用开销
3. 梯度裁剪中的 `_to_numpy()` 强制 GPU→CPU 同步（`_solver.py:1030-1036`）

**关键代码**：`statgpu/glm_core/_solver.py` 的 FISTA 内循环 gradient clipping 部分。

---

### 2.2 GPU FISTA 在大规模 squared_error 上仍慢

**现象**：即使 `n=5000, p=500`，squared_error + l2 的 CuPy fista 仍比 CPU 慢（0.08x）。

**根因**：squared_error 的 FISTA 使用 `XtX @ y_k - Xty` 计算梯度，这在 GPU 上是矩阵-向量乘法，kernel launch 开销仍大于计算收益（p=500 太小）。

**对比**：`exact` 求解器在 GPU 上有 2.7-3.4x 加速（单次线性求解，GPU 优势明显）。

---

### 2.3 GPU CV 比 CPU 慢 2-14 倍

**现象**：所有 GPU CV 都比 CPU 慢。

| Loss | Penalty | CPU(ms) | CuPy(ms) | Torch(ms) | CuPy 慢 | Torch 慢 |
|------|---------|---------|----------|-----------|---------|----------|
| squared_error | scad | 3415 | 46424 | 12083 | 13.6x | 3.5x |
| squared_error | mcp | 3758 | 52151 | 13136 | 13.9x | 3.5x |
| logistic | scad | 10288 | 115479 | 65190 | 11.2x | 6.3x |
| poisson | mcp | 4865 | 61889 | 42484 | 12.7x | 8.7x |

**根因**：
1. CV 对每个 alpha × 每个 fold 调用 `model.fit()`，每次重新计算 XtX（已部分修复：添加了 `_cv_cache`）
2. 每次 `model.fit()` 的 kernel launch 开销累积
3. `_evaluate_single()` 每次 alpha 都做 GPU→CPU 转移（已部分修复：缓存了 numpy 验证数据）

**关键代码**：`statgpu/linear_model/_penalized_cv.py` 的 warm-start 路径。

---

### 2.4 GPU 加速有效的情况

以下情况 GPU 有 2-3x 加速（`n=5000, p=500`）：

| Family | Penalty | Backend | Solver | Speedup |
|--------|---------|---------|--------|---------|
| squared_error | l2 | cupy/torch | exact/irls | 2.7-3.4x |
| logistic | none/l1/l2 | cupy/torch | newton/fista | 2.6-3.2x |
| poisson | none/l1/l2 | cupy/torch | newton/irls/fista | 2.4-3.3x |
| gamma | l1/l2 | cupy/torch | fista/irls | 2.2-2.7x |
| tweedie | none/l2 | cupy/torch | newton/irls | 2.4-3.2x |
| negative_binomial | l2 | torch | fista/fista_bb | 2.2x |

**规律**：
- `newton`/`irls`/`exact` 求解器 GPU 加速好（每次迭代做矩阵分解，GPU 优势大）
- `fista` 求解器仅在 `torch` backend 大规模时有加速（`torch.compile` 优化）
- `fista` + `cupy` 在大多数情况下比 CPU 慢

---

## 三、已修复的问题

### 3.1 CuPy fused kernel SCAD/MCP 发散 ✅ 已修复

**修复**：`_solver.py:985` 条件从 `backend == "torch"` 改为 `backend in ("torch", "cupy")`。
- CuPy ElementwiseKernel 现在对 SCAD/MCP 正常工作
- corr=1.000000，MaxDiff < 4e-6

### 3.2 CV Ridge 精度 ✅ 已修复

**修复**：`_penalized_cv.py` 中 Ridge CV 和 refit 统一使用 eigendecomposition。
- squared_error+l2 corr: 0.999986 → 1.000000
- L2 diff: 5.63e-2 → 1.08e-2

### 3.3 CV XtX 重复计算 ✅ 已修复

**修复**：`_penalized.py` 添加 `_cv_cache` 支持，CV 预计算 XtX 后传入 model.fit()。
- 避免每个 alpha 重复计算 XtX

---

## 四、修复优先级建议

| 优先级 | 问题 | 影响范围 | 难度 |
|--------|------|----------|------|
| P0 | logistic+none FISTA GPU 收敛错误 | 单次 fit 精度 | 中 |
| P1 | CV logistic 系列精度差 | CV 模型选择 | 高 |
| P2 | GPU FISTA 小规模比 CPU 慢 | 用户体验 | 中 |
| P3 | negative_binomial IRLS CuPy 发散 | 单次 fit 精度 | 低 |
| P4 | group_mcp/group_scad 小规模精度 | 边缘场景 | 低 |

---

## 五、关键文件索引

| 文件 | 内容 |
|------|------|
| `statgpu/glm_core/_solver.py` | FISTA/ADMM/IRLS 求解器，fused kernel，梯度裁剪 |
| `statgpu/glm_core/_logistic.py` | Logistic loss 的 value/gradient |
| `statgpu/glm_core/_irls.py` | IRLS 求解器实现 |
| `statgpu/linear_model/_penalized.py` | PenalizedGeneralizedLinearModel，GPU/CPU 调度 |
| `statgpu/linear_model/_penalized_cv.py` | PenalizedGLM_CV，CV 循环 |
| `statgpu/penalties/_scad.py` | SCAD penalty + LLA weights |
| `statgpu/penalties/_mcp.py` | MCP penalty + LLA weights |
| `statgpu/penalties/_group_mcp.py` | Group MCP proximal |
| `statgpu/penalties/_group_scad.py` | Group SCAD proximal |
| `dev/tests/_bench_full_matrix_output_section_A.txt` | 816 个 fit 测试完整结果 |
| `dev/tests/_bench_cv_full_output.txt` | CV 测试结果（修复前） |
| `dev/docs/cupy_fused_kernel_scad_issue.md` | SCAD/MCP CuPy fused kernel 问题记录 |

---

## 六、测试命令

```bash
# 远程服务器
ssh -p 28838 root@hz-4.matpool.com
source /root/miniconda3/envs/myconda/bin/conda.sh && conda activate myconda

# pytest
cd /root/statgpu && PYTHONPATH=/root/statgpu python -m pytest dev/tests/test_glm_penalty_review_fixes.py -v

# Section A benchmark (全量 fit 测试)
PYTHONPATH=/root/statgpu python dev/tests/_bench_full_matrix.py --section A

# CV benchmark
PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py
```
