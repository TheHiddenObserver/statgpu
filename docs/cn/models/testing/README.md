# 统计检验

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../../en/models/testing/README.md)

[模型与统计方法](../README.md) / 统计检验

先明确假设与数据设计。比较组均值和处理一组 P 值彼此相关，但需要作出不同的方法选择。

## 比较组均值

先阅读 [ANOVA 概览](anova.md)，理解假设、实验单位、适用前提与支持的设计，再选择教程：

| 设计或问题 | 教程 | 选择依据 |
|---|---|---|
| 一个因素、相互独立的组，并采用共同方差模型 | [经典单因素 ANOVA](anova-oneway.md) | 检验各组均值是否全部相等 |
| 一个因素、相互独立的组，且方差不齐 | [Welch ANOVA](anova-welch.md) | 考虑不同组的方差差异 |
| 两个交叉因素，每个单元具有相同重复数 | [平衡双因素 ANOVA](anova-twoway.md) | 分别考察主效应与交互作用 |
| 哪些组对存在差异？ | [事后比较](anova-posthoc.md) | 选择比较及其多重性校正 |

各教程说明输入要求与不支持的设计；不能仅凭方法族名称推断支持重复测量或不平衡设计。

## 处理一组检验

[多重检验](multiple-testing.md)解释错误率标准、校正方法与完整示例。
如果已有多个 P 值，需要选择控制哪种错误率，或理解合并证据的含义，从这里开始。

若希望在 knockoff 假设下选择变量，请转到独立的[特征选择方法族](../feature-selection/README.md)。

## 查阅完整接口

- [ANOVA API](../../reference/anova-api.md)：函数、结果字段、输入要求与设备限制
- [P 值校正与合并 API](../../guides/multiple-testing-combine-pvalues.md)：函数签名、轴、权重与后端行为
- [推断 API](../../guides/inference-api.md)：相关结果对象、排列检验与自助法工具
- [分布 API](../../guides/distribution-api.md)：统计计算使用的分布函数
