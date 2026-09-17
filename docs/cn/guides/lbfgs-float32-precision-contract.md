# L-BFGS Float32 数值行为

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：面向用户的数值精度说明  
> 切换：[English](../../en/guides/lbfgs-float32-precision-contract.md)

## 适用范围

本页说明如何理解 NumPy、CuPy 与 Torch 上的原生 float32 L-BFGS 结果。它不改变 L-BFGS 算法、统计目标、analytic-weight 语义、solver 选择或显式 device 行为。

算法本身见 [求解器算法](solver-algorithms.md)。

## 为什么不同 backend 的 float32 系数可能不同

L-BFGS 在有限精度下具有路径依赖。objective 与 gradient reduction 中很小的舍入差异，可能改变：

- 哪些 curvature pair 进入 limited-memory history；
- Armijo backtracking 进行多少步；
- 后续 accepted iterate 的具体序列；
- 最终是通过 gradient criterion，还是因为进一步参数步长已经低于 float32 分辨率而停止。

因此，两个都有效的 float32 执行可能出现“系数差明显大于 objective value 差”的现象。

所以 NumPy float32 coefficient vector **不应被当作** CuPy 或 Torch float32 的逐坐标精确 oracle。

## 更合理的比较方式

判断两个 float32 L-BFGS fit 是否数值一致时，应联合查看整个数值状态，而不是只比较最大 coefficient difference：

1. **Objective value**：不同执行应在适合 float32 的数值尺度上优化同一个声明目标。
2. **Stationarity**：最终 gradient 或等价 stopping diagnostic 应表明结果已经接近 stationary point。
3. **同 backend 的 float64 结果**：需要更严格诊断时，可把 float32 与同一个 backend、同一问题的 float64 求解比较。
4. **统计不变量**：例如 normalized-weight convention 下，把全部正 analytic weight 同乘一个常数不应改变统计目标。
5. **请求的执行路径**：显式 backend/device 请求应继续在该 backend 上执行，而不是静默替换成 CPU 求解。

因此，只要 objective 与 stationarity 在预期的 float32 尺度上相符，单独一个不大的系数差并不足以说明 backend 出错。

## Analytic weights

对于支持 analytic `sample_weight` 的 L-BFGS route，归一化加权目标为

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

所有正权重同乘一个常数不会改变这个目标。有限精度下优化轨迹仍可能有轻微差别，但统计 target 不变。

受支持 penalized GLM 路径上 analytic weight 的推断含义见 [Penalized GLM 推断](penalized-glm-inference.md)。

## 什么时候使用 float64

如果应用要求更严格的跨 backend coefficient reproducibility，或者很小的 coefficient difference 本身就具有实质意义，应使用 float64。

如果目标是保留原生低精度执行，并且 float32 尺度的 objective/stationarity 精度已经足够，则可以使用 float32。调查可疑数值差异时，float64 也是更合适的诊断参考。

## 实际解释方式

如果 NumPy、CuPy 与 Torch 的 float32 L-BFGS 得到略有不同的 coefficient vector：

- 先比较 objective value；
- 检查 convergence/stationarity diagnostic；
- 确认各次运行使用相同数据、objective normalization、weights、penalty 与 stopping controls；
- 需要更严格参考时，把每个 backend 与其 float64 运行比较；
- 如果应用真正要求系数高度一致，而不仅是优化到等价 objective，应直接使用 float64。

不要为了强迫不同 backend 的 float32 coefficient 近似逐位一致而修改统计目标或改变 solver 语义。

## 相关文档

- [求解器算法](solver-algorithms.md) — L-BFGS update 与 line-search 算法
- [设备与 GPU 内存](device-and-memory.md) — 显式 backend/device 行为
- [Penalized GLM 推断](penalized-glm-inference.md) — 加权目标与推断语义
