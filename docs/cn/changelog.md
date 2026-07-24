# Changelog

> 语言：中文  
> 最后更新：2026-07-24  
> 页面定位：变更记录  
> 切换：[English](../en/changelog.md)

## 2026-07

### 验证（2026-07-24）— PR #79 exact-head 最终闭环

最终 review 的生产代码 head 为
`c85750d63d4e6dbc9d988847566c20f5fa862e91`。

- exact-head GitHub Actions Tests run #545 通过；
- Python 3.9、3.10、3.11、3.12 regression job 全部通过；
- 完整 CPU suite 为 **1074 passed、275 skipped、0 failed**；
- clean-head canonical smoke pipeline 通过，`canonical_eligible=True`，verdict 为 `PASS`；
- 维护中的 Tesla P100 suite 执行 **33 个检查全部通过**，另有两个预期 skip；
- CoxPH、Linear 与 Panel 的维护路径均满足 PR79 验收合同。

另外执行的六个旧 GPU 诊断脚本没有纳入维护 pytest Gate。其转换、替换或移除由
[Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83) 跟踪。

### 修复（2026-07-24）— 最终公开合同与文档同步

- 修正 CoxPH delayed-entry 支持矩阵：robust/cluster covariance 在
  `compute_inference=True` 时显式报错；同一拟合在 `compute_inference=False` 时允许仅估计，
  推断字段保持未设置。
- 明确 `CoxPHCV` 在最终 refit 时执行相同 inference guard。
- 文档化 PooledOLS 后端保持预测、稳定 HAC `time_index` 排序和有效秩 residual degrees of freedom。
- 明确秩亏 PooledOLS：fitted value、prediction、RSS、rank 与拟合空间检查仍有效；
  系数级推断由于不唯一识别而标记为 `NOT_COMPARABLE`。
- 同步 README、中英文 CoxPH/Panel 模型页、双语 release summary 与 PR79 审计报告。
- 删除陈旧的硬编码 final accuracy artifact。只有在 exact target SHA 上重新执行完整 raw campaign，
  并通过当前 aggregator 与 renderer 后，才可以重新提交 full canonical report。

### 修复（2026-07-23）— PR #79 完整 review 闭环

- 统一 CPU/CuPy/Torch CoxPH 的最终 KKT、line search、终止状态和公共结果字段；
- 新增默认 strict、显式 opt-in 的 approx 稳健推断与 provenance 字段；
- Cox 预测与评分保持后端原生，baseline hazard 使用向量化风险集，移除受影响的 Torch Hessian materialization，
  并避免 nonrobust GPU 推断无条件复制完整训练数据；
- 强化 PR79 diagnostics 与 canonical report：missing、failed、duplicate、non-finite、dirty、wrong-SHA 证据全部 fail closed；
- 新增行为回归并同步中英文 Cox 支持矩阵。

### 验证历史（2026-07-21）

较早的 Tesla P100 完整 campaign 在代码 head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad` 上通过：

- Gate A：160 passed、0 failed、2 个预期 skip；
- Gate B：1100 passed、0 failed、124 skipped、1 个 strict XFAIL；
- Gate C：10/10 metamorphic 检查通过；
- Gate D：审计路径未发生完整设计矩阵 GPU-to-CPU 传输；
- Gate E：CuPy 与 Torch 各重复 15 次，未发现显存泄漏；
- Gate F：记录三个规模下的同步 Tesla P100 性能基线；
- Gate G：Ridge/scikit-learn 与线性回归/statsmodels 对齐通过。

后续在 `786af9e2eb4742a56e5203b4380b03aec63a3ac8` 上进行的 exact-head campaign
又通过了 17/17 个 focused physical-GPU 检查。这些历史 SHA 仍是可审计证据，
但上方 2026-07-24 条目才是最终 PR head 闭环。

### 性能基线 — Tesla P100

以下为特定硬件下的回归基线，不构成跨硬件性能保证。

| 数据形状 | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

环境：Tesla P100-SXM2-16GB、Python 3.9、CuPy 13.6.0、PyTorch 2.0.0+cu117。

### 已知非阻塞后续工作

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81)：共享的后端原生 NaN/Inf 验证；
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82)：为 scikit-learn <=1.2 clone identity 重构公开构造器；
- [Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83)：转换或移除未纳入维护测试树的旧 GPU 诊断脚本。

## 历史变更记录

截至 2026-07-14 的详细记录保留在
[归档 changelog](changelog-history-through-2026-07-14.md)。
