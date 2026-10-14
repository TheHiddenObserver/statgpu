# 基准面板与结果解读

> 语言：中文<br>
> 最后更新：2026-10-09<br>
> 切换：[English](../../en/guides/benchmarks.md)

## 查看结果

- [打开交互式基准面板](../../assets/benchmarks/index.html)
- [了解筛选、图表、指标与数据来源](../../en/guides/statgpu_benchmark_dashboard.md)
- [对齐工作负载并复现测量](../benchmarks.md)

先选择环境和模型类别，再按方法、求解器、规模及后端缩小范围。**Metric scope** 用于区分拟合（Fit）、交叉验证（CV）、推断（Inference）、预测（Prediction）和特征选择（Selection）。明细面板只显示所选记录中实际存在的指标组；缺少面板不表示结果为零，也不表示库不支持该功能。

面板包含交叉验证记录。**Cross-validation** 面板显示可用的交叉验证评估、最终重拟合与总用时、选中的参数、评分、收敛信息，以及运行失败或不可用的原因。比较数值前，请先阅读状态和评分方向。

## 覆盖范围与数据来源

面板使用日期不早于 **2026-06-01** 的基准数据源。当前覆盖范围及数量以部署的[数据源清单](../../assets/benchmarks/data/source_inventory.json)、[解析报告](../../assets/benchmarks/data/parse_report.json)和[标准化结果](../../assets/benchmarks/data/benchmark_data.json)为准。数据源数、运行记录数和模型条目数是不同的统计量；筛选后显示的内容可能少于整个数据包。

已有记录涉及 GLM 与惩罚 GLM、线性模型、稳健与分位数回归、生存分析、无监督学习、有序模型、非参数方法、面板模型、协方差估计及 ANOVA。不同方法、指标、规模和后端的覆盖程度不同。当前数据包的 Feature Selection 类别尚无符合要求的结构化数据源。2026 年 4 月的 ElasticNet、LassoCV、综合验证、Cox 软件包比较和 knockoff 结果不在面板日期范围内。缺少原始计时及精度记录、只保留舍入汇总的分布报告也未纳入。

## 比较时注意

- 对齐环境、工作负载、目标函数、求解器、数值精度和计时范围。拟合加推断、完整交叉验证的用时不能直接当作仅拟合用时。
- 加速比大于一表示快于指定参考对象，小于一表示更慢。实验脚本报告的比值和根据数据计算的比值具有不同来源。
- 仅验证正确性的记录不提供时间或加速比，缺失值也不是零。
- 历史测量只描述记录中的源码和环境，不能自动代表当前版本或自己的硬件。

例如，[选择后 OLS 推断基准](../../../dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py)测量 NumPy CPU 与 CuPy CUDA 上的完整拟合加推断过程，并比较系数及推断结果。它不是仅推断阶段的加速测量；`device` 选择硬件，`inference_method="post_selection_ols"` 选择统计方法。

面向贡献者的数据接入、构建和测试流程见[面板维护指南](../../../frontend/docs/benchmark-dashboard-maintenance.md)。
