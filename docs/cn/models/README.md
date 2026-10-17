<a id="模型总览"></a>

# 模型与统计方法

> 语言：中文  
> 最后更新：2026-10-17  
> 切换：[English](../../en/models/README.md)

先根据要回答的问题选择方法族，再通过族内概览选择教程。首次拟合可先走完
[入门路径](../usage.md)。当前求解器、惩罚项、后端与推断支持范围，以
[已实现方法](../guides/implemented-methods.md)、
[求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)和对应方法页为准。

<a id="回归与-glm"></a>
<a id="专业统计模块"></a>
<a id="无监督学习"></a>

## 按问题选择方法族

| 想回答的问题 | 从这里开始 | 进一步选择 |
|---|---|---|
| 响应如何随预测变量变化？ | [回归](regression/README.md) | 响应类型、收缩、分位数或稳健损失 |
| 组均值是否有差异，或怎样处理多个检验？ | [统计检验](testing/README.md) | ANOVA 设计、事后比较或 P 值校正 |
| 多个变量怎样共同变化？ | [协方差估计](covariance/README.md) | 经验、收缩、稳健或稀疏精度估计 |
| 需要密度、平滑响应曲线或非线性特征？ | [非参数方法](nonparametric/README.md) | KDE/局部回归、核方法、样条或 GAM |
| 应该保留哪些预测变量？ | [特征选择](feature-selection/README.md) | AIC/BIC 逐步搜索，或 knockoff 选择及其假设 |
| <a id="生存分析"></a><a id="如何选择-cox-估计器"></a>响应是事件发生前的时间？ | [生存分析](survival/README.md) | 比较 Cox 拟合、交叉验证和更广泛的惩罚项选择 |
| 观测跨个体、跨时间重复出现？ | [面板数据](../panel/README.md) | 比较不同的面板估计器 |
| 没有响应变量，希望发现结构或学习表示？ | [无监督学习](../unsupervised/README.md) | 分解、聚类、混合模型或流形方法 |

各方法族入口说明如何选择方法，并链接到族内全部教程。需要核对构造参数、方法或拟合属性时，
查阅[完整 API 参考](../README.md#完整-api-参考)。

## 核心参考

| 页面 | 内容 |
|---|---|
| [损失函数](../reference/losses.md) | 损失函数定义与数值性质 |
| [求解器算法](../guides/solver-algorithms.md) | 求解器更新公式与适用前提 |
| [损失函数 × 惩罚项 × 求解器框架](../guides/loss-penalty-solver-framework.md) | 组件分工与组合方式 |
| [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) | 显式求解器与自动分发范围 |
| [推断 API](../guides/inference-api.md) | 分布、多重检验、重抽样与推断工具入口 |

## 支持范围的阅读方式

- NumPy、CuPy 与 Torch 是不同的数组后端。需要加速器计算时，请先核对[设备指南](../guides/device-and-memory.md)与对应模型的设备限制。
- 后端支持可能因求解器、惩罚项、推断方法和可选依赖而不同，应查看详细兼容性文档，而不要从某个模型族的概览自行推断。
- 具体模型的统计定义、输入限制、预测、推断和 CV 行为，以对应模型页为准。
