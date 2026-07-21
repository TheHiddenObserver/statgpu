# PR #79 最终真实 GPU 验证

> 日期：2026-07-21  
> 硬件：Tesla P100-SXM2-16GB  
> 后端：NumPy、CuPy CUDA、Torch CUDA

PR #79 已完成仓库审查计划要求的真实 GPU 验证。所有强制 gate 均通过，
没有发现 PR 引入的回归，也没有遗留 CRITICAL 或 HIGH 级正确性问题。

## 验证汇总

| Gate | 结果 |
|---|---|
| GPU smoke | 160 passed，0 failed，2 个预期 skip |
| 三后端正确性 | 1100 passed，0 failed，124 skipped，1 个 strict XFAIL |
| Metamorphic | 10/10 通过；记录 1 个已知有限输入问题 |
| 设备纯度 | 审计 3 个模型族，完整设计矩阵传回 CPU 次数为 0 |
| 显存 | CuPy 与 Torch 各重复 15 次，未发现泄漏 |
| 性能 | 两个 GPU 后端完成 3 个规模的同步计时 |
| 外部验证 | Ridge 对齐 scikit-learn；线性回归对齐 statsmodels |
| 完整测试 | CPU 1100 passed；GPU 1100 passed |

Gate B 从 1036 passed、40 failed 改进到 1100 passed、0 failed。唯一的
strict XFAIL 仅适用于 scikit-learn <=1.2，并且可在 base SHA 上复现，因此不是
PR #79 回归。

## 正确性修复

真实 GPU 执行暴露并修复了面板推断、秩亏 PooledOLS、后端数组构造、Nystroem、
线性模型 wrapper、debiased inference 状态保存、带权 GLM fused dispatch 以及
StepwiseSelector clone 等问题。

其中最严重的根因包括：

- CPU 分布临界值标量直接与 GPU 数组运算；
- 将 Torch 专用的 `device=` 参数传给 NumPy/CuPy 数组构造函数；
- 通过 `np.asarray` 隐式转换 CuPy 数组；
- 带权 GLM fused loss/gradient 发生无限递归；
- diagnostics 使用前清除了拟合后的 inference 状态。

## 性能基线

| 数据形状 | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

这些数据仅作为所记录 Tesla P100 环境下的回归基线，不构成跨硬件性能保证。

## 已知后续工作

- Issue #81：统一的后端原生 NaN/Inf 输入验证；
- Issue #82：为兼容 scikit-learn <=1.2 clone 协议而进行的公开构造器统一重构。

这两项均不阻塞 PR #79 已验证的有限输入路径。

## 复现与证据

- `dev/reviews/pr79_physical_gpu_validation.md`
- `dev/plans/pr79_gpu_review_fix_test_plan.md`
- `dev/tests/test_pr79_physical_gpu.py`
- `dev/validation/pr79_gpu_orchestrator.py`
- `dev/validation/pr79_results.py`
- 结果目录约定：`results/pr79/<UTC-run-id>/`
