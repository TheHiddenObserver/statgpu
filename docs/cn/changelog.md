# Changelog

> 语言：中文  
> 最后更新：2026-07-21  
> 页面定位：变更记录  
> 切换：[English](../en/changelog.md)

## 2026-07

### 修复（2026-07-21）— PR #79 真实 GPU 完整验证

Tesla P100 完整验证已在代码 head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad` 上通过。

- Gate A：160 passed，0 failed，2 个预期 skip。
- Gate B：1100 passed，0 failed，124 skipped，1 个 strict XFAIL。
- Gate C：10/10 个 metamorphic 检查通过。
- Gate D：审计路径未发生完整设计矩阵 GPU-to-CPU 传输。
- Gate E：CuPy 与 Torch 各重复 15 次，未发现显存泄漏。
- Gate F：记录三个规模下的同步 Tesla P100 性能基线。
- Gate G：Ridge/scikit-learn 与线性回归/statsmodels 对齐通过。
- 最终完整测试：CPU 1100 passed；GPU 1100 passed。

Gate B 从 **1036 passed / 40 failed / 159 skipped** 改进至
**1100 passed / 0 failed / 124 skipped / 1 strict XFAIL**。该版本限定的 clone
XFAIL 可在 base SHA `a4879fb` 上复现，并由 issue #82 跟踪。

该轮真实 GPU 验证修复了面板 device mismatch、字符串 cluster factorization、
秩亏 PooledOLS、Torch 专用 `device=` 泄漏、CuPy 13.x 与 Nystroem 构造、
Debiased Lasso 拟合状态丢失、带权 GLM fused 递归以及 StepwiseSelector 旧版 clone
契约等问题。

### 修复（2026-07-21）— 验证后的 review-fix 循环

完整 GPU 验证之后又执行了一轮 review → fix → test → re-review。清理后的代码
head 为 `ff72424071ec7ca52399146dbd8a556534c9e6c3`。

新增修复包括：

- `LinearRegression.fit` 与 `predict` 在后端解析前保留 CuPy/Torch 原生输入，
  不再提前执行 NumPy 转换；
- PooledOLS HAC 通过经过验证的 `time_index` 稳定排序，显式消除输入行顺序依赖；
- PooledOLS 使用有效设计秩计算 residual degrees of freedom；
- 远程验证器加入 shell `pipefail`、必须显式提供的精确 SHA、不可变 base worktree
  以及 reset/clean 检查；
- 将公式控制的截距语义与公开、clone 可见的 `fit_intercept` 构造参数分离；
- 修正 CPU、CuPy、Torch 三条带权 `LinearRegression` 路径：截距列同步乘
  `sqrt(weight)`，修复 multi-output 广播，统一权重验证，分别保留原始与带权残差，
  并在奇异设计下使用稳定 least-squares fallback；
- Patsy 删除缺失行后，按照保留的原始行位置对齐原始长度的 formula sample weights。

永久回归测试位于 `dev/tests/test_pr79_final_review_fixes.py`，覆盖
scikit-learn/statsmodels 对齐、秩亏与 HAC 不变量、公式截距及缺失行语义、非法权重、
multi-output WLS、orchestrator 精确 SHA、pipeline 失败传播，以及可选的真实
CuPy/Torch parity。

### 最新 head 的验证边界

GitHub Actions Tests run #477 已在清理后的代码 head `ff72424` 上通过：

- Python 3.9、3.10、3.11、3.12 regression matrix；
- static contracts、编译与完整测试收集；
- 完整 CPU suite。

验证后的修改涉及 CuPy/Torch 带权 `LinearRegression` 路径。因此，在 PR #79 从
Draft 改为 Ready for review 之前，仍必须针对精确的最新代码 head 执行一次聚焦的
真实 GPU 复验。先前 P100 完整验证仍是 `2f18e5d` 的有效证据，但不会被表述为后续
代码的 exact-head 验证。所需命令与验收标准见
`dev/reviews/pr79_physical_gpu_validation.md`。

### 性能基线 — Tesla P100

以下结果来自已完成真实 GPU 验证的 head，只作为特定硬件与环境下的回归基线，
不构成可跨环境推广的性能保证。

| 数据形状 | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

环境：Tesla P100-SXM2-16GB、Python 3.9、CuPy 13.6.0、
PyTorch 2.0.0+cu117。

### 已知非阻塞后续工作

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81)：补齐共享的后端原生
  NaN/Inf 输入验证契约。
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82)：统一重构公开 estimator
  构造函数，以满足 scikit-learn <=1.2 clone identity contract。
- Torch Cox Hessian 的 `O(n*p*p)` 中间量仍作为独立性能优化任务保留。

## 历史变更记录

截至 2026-07-14 的详细记录保留在
[归档 changelog](changelog-history-through-2026-07-14.md)。
