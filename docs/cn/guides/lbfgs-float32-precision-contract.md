# L-BFGS Float32 精度契约

> 语言：中文<br>
> 最后更新：2026-09-15<br>
> 页面定位：数值精度契约<br>
> 切换：[English](../../en/guides/lbfgs-float32-precision-contract.md)

## 适用范围

本页描述普通光滑 L-BFGS 路径维护中的 **float32 数值比较契约**。它不改变 L-BFGS 算法、声明的统计目标、solver 选择、解析权重语义，也不改变显式 backend/device 的权威性。

算法本身见 [Solver Algorithms](solver-algorithms.md)。

## 为什么 float32 不能把某一个 backend 的系数当作逐坐标 oracle

L-BFGS 在有限精度下具有路径依赖。目标函数/梯度 reduction 中很小的舍入差异，会在迭代后期改变曲率对历史、Armijo 回溯，以及最终是通过梯度判据还是通过“参数步长已小到无法表示”而终止。

因此，两个都合法的原生 float32 执行可以出现“系数差明显大于目标函数差”的情况。NumPy float32 的结果不能被当作 CuPy 或 Torch float32 的精确系数 oracle。

这一点只针对 float32 的数值契约。float64 仍然是严格的跨 backend 参考精度。

## 维护中的验收维度

一个 float32 L-BFGS 结果需要联合满足以下维度，而不是只比较系数：

1. **目标函数一致性**：完整声明目标必须在适合 float32 的数值尺度内与参考结果一致；
2. **驻点性**：最终梯度范数必须足够小，包括某个 backend 因参数分辨率而不是梯度判据终止的情况；
3. **同 backend 的 float64 参考**：float32 结果要与同一个 backend、同一份数值数据的 float64 求解结果比较；
4. **跨 backend 诊断**：记录 float32 参数差，但单凭该差异不能判定某个 backend 错误；
5. **解析权重整体缩放**：所有正解析权重乘同一个正常数时，统计目标保持不变；有限精度造成的优化路径差异仍按同一个 float32 objective/stationarity 契约验收；
6. **执行一致性**：diagnostic trace 必须复现 production solver，公开 estimator 路径必须保持调用者请求的 backend/device 与实际 L-BFGS 身份。

Issue #160 经审阅的物理证据矩阵采用以下保守边界：

| 数值量 | Float32 验收边界 |
|---|---:|
| 与同 backend float64 的 objective 绝对差 | `2e-6` |
| 与同 backend float64 的参数最大绝对差 | `2e-3` |
| 最终梯度范数 | `2e-4` |
| 与 NumPy float32 的 objective 绝对差 | `2e-6` |
| 与 NumPy float32 的参数最大绝对差 | `2e-3` |
| 解析权重整体缩放后的 objective 差 | `2e-6` |
| 解析权重整体缩放后的参数最大绝对差 | `1e-3` |
| 解析权重整体缩放后的梯度范数差 | `2e-4` |

其中“与 NumPy float32 的参数差”只是**诊断边界**，并不表示 NumPy float32 的系数向量在数学上具有更高权威性。

## Float64 契约保持严格

Issue #160 不会放宽 float64 L-BFGS 契约。保留的物理矩阵中，float64 的 NumPy/CuPy/Torch 解以及全局权重缩放 control 都在或接近 float64 舍入误差尺度上对齐。当应用需要严格的跨 backend 参数可复现性时，float64 仍是合适的精度层级。

## 实际使用建议

普通使用无需特殊处理：当 float32 数据进入受维护的 L-BFGS 路径时，statgpu 保留原生 float32 执行。

如果应用要求的是严格的跨 backend 系数可复现性，而不仅是 float32 级别的目标函数与驻点一致性，应使用 float64 输入。只要目标函数、驻点性、同 backend float64 对照与执行 provenance 均在契约内，就不应仅凭较小的 float32 系数差异判断某个 backend 失败。

物理证据与经审阅的 Option-A 决策分别保存在 `dev/reviews/issue160_float32_lbfgs_matrix.json` 与 `dev/reviews/issue160_float32_lbfgs_contract_decision.md`。
