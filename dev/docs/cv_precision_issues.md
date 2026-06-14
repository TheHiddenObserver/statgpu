# CV 精度问题：诊断与修复指南

> **目标读者**：接手修复的 AI 模型
> **测试环境**：Tesla P100-SXM2-16GB, CuPy 13.6.0, Torch 2.0.0+cu117
> **日期**：2026-05-31
> **当前状态**：pytest 54/54 PASS, Section A 816/816 PASS

---

## 一、问题概述

CV benchmark（7 families × 5 penalties × 3 backends = 35 组合）中，部分组合的 CV 系数与 CPU 基线差异较大。

**当前最严重的 5 个问题**：

| # | 组合 | CuPy corr | Torch corr | 严重程度 |
|---|------|-----------|------------|---------|
| 1 | negative_binomial + scad | **0.928** | 1.000 | 🔴 |
| 2 | negative_binomial + mcp | **0.938** | **0.948** | 🔴 |
| 3 | inverse_gaussian + scad | 1.000 | 0.9994 | 🟡 |
| 4 | inverse_gaussian + mcp | 0.9996 | 1.000 | 🟡 |
| 5 | tweedie + scad | 1.000 | 0.9986 | 🟡 |

---

## 二、根因分析

### 核心问题：`fista_lla_path` 的 warm-start 与内部 continuation path 冲突

**代码路径**：

```
PenalizedGLM_CV.fit() → model.fit(X, y)
  → _fit_loss_backend() (line 3545+)
    → _use_lla_fista = True (for SCAD/MCP + GLM losses)
    → fista_lla_path() (line 3746)
```

**问题机制**：

1. CV 循环按 alpha 从大到小排序，每个 alpha 调用 `model.fit()`
2. `model.fit()` 对 SCAD/MCP 调用 `fista_lla_path()`
3. `fista_lla_path()` 内部有自己的 continuation path：从 `lambda_max` 到 target alpha，分 20 步
4. 如果传入 warm-start（上一个 CV alpha 的 coef），它会作为 continuation path 第一步的初始值
5. 这**改变了整个 continuation 轨迹**，因为第一步的 LLA weights 基于 warm-start coef 而非 zeros

**为什么对某些组合有益，对某些有害**：

- **有益**（如 IG+SCAD, Logistic+l2）：warm-start coef 接近目标解，continuation 收敛更快更准
- **有害**（如 NB+SCAD, NB+MCP）：warm-start coef 的 sparsity pattern 与目标 alpha 不同，导致 LLA weights 错误引导 continuation 走向不同的局部最优

### 关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `glm_core/_solver.py` | 837 | `fista_lla_path` 函数签名 |
| `glm_core/_solver.py` | 1000-1030 | warm-start 初始化代码 |
| `glm_core/_solver.py` | 1054-1068 | squared_error fused path 的 continuation loop |
| `glm_core/_solver.py` | 1116-1134 | GLM unfused path 的 continuation loop + LLA weights |
| `linear_model/_penalized.py` | 3703-3758 | `_use_lla_fista` 路径调用 `fista_lla_path` |
| `linear_model/_penalized.py` | 3735-3744 | warm-start 提取逻辑 |
| `linear_model/_penalized_cv.py` | 407-412 | CV 循环中设置 `model._init_coef` |

---

## 三、已尝试的修复方案

### 方案 A：全 warm-start（v6）

```python
# _penalized.py line 3735
if init is not None:
    _init_np = _to_numpy(init).ravel()
    _warm_coef = _init_np[:p]
    _warm_intercept = float(_init_np[p])
```

**结果**：
- ✅ NB+SCAD CuPy: 0.920 → 1.000
- ✅ NB+MCP: 0.94/0.94 → 1.000/1.000
- ❌ NB+SCAD Torch: 1.000 → 0.917
- ❌ NB+L1 Torch: 1.000 → 0.967
- ❌ Tweedie+SCAD: 0.9999 → 0.9988

### 方案 B：仅 CuPy warm-start（v7）

```python
# _penalized.py line 3738
if init is not None and backend_name == "cupy":
    _warm_coef = _init_np[:p]
```

**结果**：
- ✅ Torch NB+SCAD 恢复: 0.917 → 1.000
- ✅ Torch NB+L1 恢复: 0.967 → 1.000
- ❌ CuPy NB+SCAD 退化: 1.000 → 0.928
- ❌ CuPy NB+MCP 退化: 1.000 → 0.938

### 方案 C：无 warm-start（v5，当前回退基线）

```python
# 不传 init_coef/init_intercept 给 fista_lla_path
```

**结果**：
- ❌ NB+SCAD CuPy: 0.920
- ❌ NB+MCP: 0.94/0.94
- ✅ 其他组合稳定

---

## 四、建议的修复方向

### 方向 1：只在 continuation path 最后一步使用 warm-start（推荐）

`fista_lla_path` 的内部 continuation 有 20 步（从 `lambda_max` 到 target alpha）。warm-start 应该只用于最后一步（target alpha），而不是第一步（lambda_max）。

**实现思路**：
```python
# 在 fista_lla_path 的 continuation loop 中
for _cont_i, cont_alpha in enumerate(alpha_path):
    is_last = (_cont_i == len(alpha_path) - 1)
    
    if is_last and init_coef is not None:
        # 最后一步：用 warm-start
        coef = init_coef_device
    else:
        # 非最后一步：从 zeros 开始（或用上一步的结果）
        pass
    
    # LLA + FISTA 内循环
    for _lla_i in range(max_lla_per_step):
        lla_w = scad_penalty.lla_weights(coef)
        ...
```

### 方向 2：Warm-start 只影响 LLA 内循环，不影响 continuation

在 continuation path 的每一步中，用 warm-start 作为 LLA 内循环的初始值，但 continuation 本身仍从 zeros 开始。

### 方向 3：接受当前精度，标记为已知限制

NB+SCAD/MCP 的 CV 精度（corr=0.92-0.94）可能已经是 non-convex penalty + GLM loss 的合理精度。可以：
- 在文档中标记为已知限制
- 对 NB+SCAD/MCP 使用更大的 `n_alphas` 或更细的 alpha grid
- 或者对 NB 使用不同的 CV 策略（如 fixed alpha 而非 grid search）

---

## 五、测试命令

```bash
# 远程服务器
ssh -p 28838 root@hz-4.matpool.com
source /root/miniconda3/envs/myconda/bin/conda.sh && conda activate myconda
cd /root/statgpu

# pytest（54 个测试）
PYTHONPATH=/root/statgpu python -m pytest dev/tests/test_glm_penalty_review_fixes.py -v

# CV benchmark（7 families × 5 penalties × 3 backends）
PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py

# Section A 全量 fit 测试（816 个测试）
PYTHONPATH=/root/statgpu python dev/tests/_bench_full_matrix.py --section A
```

---

## 六、当前代码状态

**已修复的问题**（不需要再处理）：
- CuPy fused kernel SCAD/MCP 发散 → 已启用 CuPy fused path
- logistic 大 eta 数值不稳定 → stable softplus
- CV Ridge 精度不足 → eigendecomposition 统一
- CV XtX 重复计算 → `_cv_cache`
- CV cache 被 fit() 删除 → 生命周期修复
- CuPy NB IRLS 发散 → 对称化 + jitter + residual fallback
- group MCP/SCAD proximal 精度 → vectorized 对齐
- 梯度裁剪 GPU→CPU 同步 → device-side 裁剪

**未解决的问题**：
- NB+SCAD/MCP CV 精度（corr=0.92-0.95）
- IG+SCAD/MCP CV 精度（corr=0.999-1.000，边缘）
- Tweedie+SCAD CV 精度（corr=0.999，边缘）
- GPU CV 性能（比 CPU 慢 2-15x，P100 硬件限制）

---

## 七、关键文件

| 文件 | 内容 |
|------|------|
| `statgpu/glm_core/_solver.py` | `fista_lla_path` 实现 |
| `statgpu/linear_model/_penalized.py` | PGLM 主类，`_fit_loss_backend` |
| `statgpu/linear_model/_penalized_cv.py` | CV 循环 |
| `dev/tests/benchmark_cv_full.py` | CV benchmark |
| `dev/tests/_bench_full_matrix.py` | Section A 全量 benchmark |
| `dev/tests/test_glm_penalty_review_fixes.py` | pytest 回归测试 |
| `dev/docs/issues_for_fix.md` | 之前的问题汇总文档 |
