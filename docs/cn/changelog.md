# Changelog

> 语言：中文  
> 最后更新：2026-07-21  
> 页面定位：变更记录  
> 切换：[English](../en/changelog.md)

## 2026-07

### 修复（2026-07-21）— PR #79 最终真实 GPU 正确性审查

- **面板推断与秩亏 PooledOLS**：
  - 根因：CPU 分布临界值直接参与 CuPy/Torch 数组运算，字符串 cluster 标签被送入
    数值 GPU 构造函数，秩亏设计仍依赖不稳定的直接求解。
  - 影响：clustered inference 可能因 device 或 object dtype 报错，奇异设计的系数与
    协方差也可能不稳定。
  - 修复：使用后端感知 helper 转换临界值；在 CPU 将标签 factorize 为元数据后，仅将
    整数编码复制到设备；秩亏时使用稳定的 least-squares/pseudoinverse 路径。
  - 文件：`statgpu/panel/_utils.py`、`statgpu/panel/_pooled.py`。

- **跨后端数组构造与 CuPy 13.x 兼容**：
  - 根因：将 Torch 专用的 `device=` 参数传给 NumPy/CuPy `asarray`，线性模型 wrapper
    还尝试隐式执行 `np.asarray(cupy_array)`。
  - 影响：合法的显式 CUDA 输入在模型计算前即失败。
  - 修复：只在 Torch 路径传递 `device=`；按后端保护 Nystroem 数组构造；仅在公开
    输出边界执行显式 backend-to-NumPy 转换。
  - 文件：`statgpu/backends/_utils.py`、
    `statgpu/nonparametric/kernel_methods/_nystroem.py`、
    `statgpu/linear_model/wrappers/_linear.py`。

- **Debiased Lasso 拟合后 diagnostics**：
  - 根因：inference 清理逻辑删除 `_resid`、`_X_design` 与 `_y`，但 `rsquared`、AIC、
    BIC 等 diagnostics 仍依赖这些状态。
  - 影响：成功完成 inference fit 后，估计器可能无法提供已公开的 diagnostics。
  - 修复：在 NumPy、CuPy、Torch 路径保留拟合后的 inference 状态。
  - 文件：`statgpu/linear_model/penalized/_inference_mixin.py`。

- **带权 GLM fused loss/gradient 递归**：
  - 根因：`_weighted_loss_and_grad()` 携带权重再次调用
    `loss.fused_value_and_gradient()`，后者又调度回 `_weighted_loss_and_grad()`。
  - 影响：FISTA-BB 正确重定向至 FISTA 后，带权 smooth-penalty logistic fit 可能触发
    `RecursionError`。
  - 修复：直接从逐样本 loss 与 score 计算带权目标和梯度，并保持归约在所选后端。
  - 文件：`statgpu/glm_core/_fused.py`。

- **StepwiseSelector 旧版 sklearn clone 行为**：
  - 根因：构造函数使用规范化或复制后的对象替换公开参数，违反 scikit-learn <=1.2
    使用的 constructor identity 检查。
  - 影响：`sklearn.base.clone()` 无法 clone StepwiseSelector。
  - 修复：原样保留公开构造参数，并将规范化运行状态放入私有属性。
  - 文件：`statgpu/feature_selection/_stepwise.py`。

### 优化（2026-07-21）— Tesla P100 同步性能基线

正确性 gate 通过后，使用 warmup 与后端同步进行真实 GPU 计时。以下结果只作为该环境
下的回归基线，不构成可跨硬件推广的性能保证。

| 数据形状 | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

环境：Tesla P100-SXM2-16GB、Python 3.9、CuPy 13.6.0、
PyTorch 2.0.0+cu117。审计报告：
`dev/reviews/pr79_physical_gpu_validation.md`。

### 改进（2026-07-21）— 验证与发布证据

- 新增可复现的真实 GPU 验证计划、远程 orchestrator、共享 GPU fixture、结果聚合、
  device-transfer 审计、显存检查、性能计时和外部参考对齐。
- 新增 `dev/tests/test_pr79_physical_gpu.py` 以及 `dev/validation/` 下的配套脚本。
- 新增最终审查产物 `dev/reviews/pr79_physical_gpu_validation.md`，并提供中英文用户摘要：
  `docs/en/releases/pr79-final-validation.md` 与
  `docs/cn/releases/pr79-final-validation.md`。

### 验证（2026-07-21）— 全部 gate 通过

| Gate | 内容 | 结果 |
|---|---|---|
| A | GPU smoke | 160 passed，0 failed，2 个预期 skip |
| B | NumPy/CuPy/Torch 正确性 | 1100 passed，0 failed，124 skipped，1 个 strict XFAIL |
| C | Metamorphic 性质 | 10/10 通过；记录 1 个已知有限输入问题 |
| D | 设备纯度 | 完整设计矩阵传回 CPU 次数为 0；审计 3 个模型族 |
| E | 显存 | CuPy 与 Torch 各重复 15 次，未发现泄漏 |
| F | 性能 | 两个 GPU 后端均记录 3 个同步规模 |
| G | 外部参考 | Ridge 对齐 scikit-learn；线性回归对齐 statsmodels |
| Final | 完整测试 | CPU 1100 passed；GPU 1100 passed |

Gate B 从 **1036 passed / 40 failed / 159 skipped** 改进至
**1100 passed / 0 failed / 124 skipped / 1 strict XFAIL**。scikit-learn <=1.2
下的 clone XFAIL 可在 base SHA `a4879fb` 上对相同 26 个 estimator 复现，因此不是
PR #79 引入的回归。

### 已知非阻塞后续工作

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81)：补齐共享的后端原生
  NaN/Inf 输入验证契约。目前 Ridge 有一条路径未在 CUDA kernel 前拒绝非有限输入。
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82)：统一重构公开 estimator
  构造函数，以满足 scikit-learn <=1.2 clone identity contract。
- Torch Cox Hessian 仍会物化 `O(n*p*p)` 中间量，作为独立性能优化任务保留。

这些发现均不阻塞 PR #79 已验证的有限输入路径。

## 历史变更记录

截至 2026-07-14 的详细记录保留在
[归档 changelog](changelog-history-through-2026-07-14.md)。
