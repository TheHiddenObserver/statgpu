# 修复提示词

请修复 statgpu 的 CV 精度问题。完整分析文档在 `dev/docs/cv_precision_issues.md`。

## 问题

CV benchmark 中 negative_binomial + scad/mcp 的系数与 CPU 基线差异大（corr=0.92-0.95）。

## 根因

`fista_lla_path()`（`statgpu/glm_core/_solver.py` line 837）内部有 continuation path（20 步，从 lambda_max 到 target alpha）。当 CV 传入 warm-start coef 时，它会改变整个 continuation 轨迹，导致某些组合（NB+SCAD/MCP）收敛到不同的局部最优。

## 已尝试的方案

1. **全 warm-start**：修复了 CuPy NB，但退化了 Torch NB
2. **仅 CuPy warm-start**：修复了 Torch NB，但退化了 CuPy NB
3. **无 warm-start**：NB+SCAD CuPy corr=0.920（当前基线）

## 建议的修复方向

**在 `fista_lla_path` 的 continuation loop 中，只在最后一步使用 warm-start**：

```python
# statgpu/glm_core/_solver.py, fista_lla_path 函数
# line ~1054 (squared_error fused path) 和 line ~1116 (GLM unfused path)

for _cont_i, cont_alpha in enumerate(alpha_path):
    is_last = (_cont_i == len(alpha_path) - 1)
    
    # 只在最后一步使用 warm-start
    if is_last and init_coef is not None:
        coef = <init_coef converted to device array>
    elif not is_last:
        # 非最后一步：用 zeros 或上一步的 coef
        pass
    
    # LLA + FISTA 内循环不变
```

这样 continuation path 仍从 zeros 开始正常运行，但最后一步（target alpha）用 warm-start 加速收敛。

## 验证

```bash
ssh -p 28838 root@hz-4.matpool.com
cd /root/statgpu && PYTHONPATH=/root/statgpu

# 1. pytest
python -m pytest dev/tests/test_glm_penalty_review_fixes.py -v

# 2. CV benchmark（重点看 NB+scad, NB+mcp）
python dev/tests/benchmark_cv_full.py

# 3. Section A
python dev/tests/_bench_full_matrix.py --section A
```

**预期结果**：
- NB+SCAD CuPy: corr > 0.999
- NB+MCP: corr > 0.999
- 其他组合不退化
- Section A 816/816 保持 PASS

## 关键文件

- `statgpu/glm_core/_solver.py` — `fista_lla_path` 实现（line 837+）
- `statgpu/linear_model/_penalized.py` — `_fit_loss_backend` 调用 `fista_lla_path`（line 3746）
- `statgpu/linear_model/_penalized_cv.py` — CV 循环设置 warm-start（line 407-412）
- `dev/tests/benchmark_cv_full.py` — CV benchmark
