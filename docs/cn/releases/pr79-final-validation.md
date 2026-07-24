# PR #79 最终验证

> 最终 review head：`c85750d63d4e6dbc9d988847566c20f5fa862e91`  
> 日期：2026-07-24  
> 硬件：Tesla P100-SXM2-16GB  
> 后端：NumPy、CuPy CUDA、Torch CUDA

PR #79 已完成全仓库正确性 review、exact-head CI 验证与维护中的真实 GPU 验收。目前没有已知未闭合的 CRITICAL 或 HIGH 级生产缺陷。

## 最终状态

| Gate | 结果 |
|---|---|
| GitHub Actions | PASS — exact-head Tests run #545 |
| Python matrix | PASS — 3.9、3.10、3.11、3.12 |
| 完整 CPU suite | PASS — 1074 passed、275 skipped、0 failed |
| canonical clean-head smoke | PASS — `canonical_eligible=True` |
| 维护中的 P100 suite | PASS — 33 passed、2 个预期 skip、0 failed |
| CoxPH 完整维护路径 parity | PASS |
| Linear 与 Panel 维护路径 | PASS |

## 最终 review 闭合的用户可见合同

- CoxPH 在三后端统一 line search、收敛、终止原因、最终 KKT、Hessian、协方差与拟合状态。
- delayed-entry robust/cluster 推断在 `compute_inference=True` 时显式报错；`compute_inference=False` 时允许仅估计，推断字段保持未设置。
- Cox 预测和评分保留 estimator 后端。
- `PooledOLS.predict()` 不再对 CuPy 或 Torch 输入进行 eager NumPy 转换。
- PooledOLS HAC 使用经过验证的稳定 `time_index` 排序。
- 秩亏 PooledOLS 使用有效秩计算 residual degrees of freedom；拟合空间结果仍有效，系数级推断标记为 `NOT_COMPARABLE`。
- PR79 canonical report 只能由经过验证的 clean exact-head artifact 渲染。missing、non-finite、duplicate、failed、dirty 或 wrong-SHA 证据全部 fail closed。

## 证据口径

维护中的真实 GPU 验收计数为 **33/33 passed**。另外执行的旧诊断脚本未纳入维护 pytest Gate，由 Issue #83 跟踪。

旧的硬编码 `results/pr79/final/final_accuracy_report.*` 文件不符合当前 renderer schema，不能作为权威结果。只有在 exact target SHA 上重新执行完整 raw matrix，并通过 `aggregate_results.py` 与 `emit_final_report.py` 后，才可以重新提交 full canonical report。

## 后续工作

- Issue #81：统一后端原生 NaN/Inf 验证；
- Issue #82：为 scikit-learn <=1.2 clone compatibility 重构公开构造器；
- Issue #83：转换或移除未纳入维护测试树的旧 GPU 诊断脚本。

这些事项均不阻塞 PR #79 已验证的有限输入与维护路径。

## 复现与证据

- `dev/reviews/pr79_physical_gpu_validation.md`；
- `dev/tests/test_pr79_physical_gpu.py`；
- `dev/benchmarks/pr79/`；
- `dev/validation/pr79_checks/`；
- 结果目录约定：`results/pr79/<UTC-run-id>/`。
