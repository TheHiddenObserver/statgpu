# 回归

> 语言：中文  
> 最后更新：2026-10-17  
> 切换：[English](../../../en/models/regression/README.md)

[模型与统计方法](../README.md) / 回归

先确定响应类型与希望估计的量，再决定是否需要系数收缩或不同的损失函数。
下面的教程提供完整示例；推断、求解器和设备限制仍需以具体方法页为准。

## 按响应选择模型

| 响应或问题 | 教程 | 选择依据 |
|---|---|---|
| 连续响应，条件均值近似线性 | [LinearRegression](linear-regression.md) | 普通最小二乘与系数推断 |
| 二元响应 | [LogisticRegression](logistic-regression.md) | 事件概率与分类 |
| 非负整数计数 | [PoissonRegression](poisson-regression.md) | Poisson 响应模型下的期望计数 |
| 有序类别 | [有序 logit/probit](ordered.md) | 利用类别顺序，但不假设相邻类别等距 |
| 其他受支持的响应族，或统一的 GLM 接口 | [广义线性模型](generalized-linear-model.md) | 先选分布族与链接函数，再核对普通与惩罚模型接口 |

## 为线性响应模型选择收缩方式

| 需求 | 教程 | 选择依据 |
|---|---|---|
| 收缩系数，不使用 L1 选择惩罚 | [Ridge](ridge.md) | L2 正则化与 RidgeCV |
| 稀疏线性拟合 | [Lasso](lasso.md) | L1 正则化、LassoCV 与推断选项 |
| 结合 L1 选择与 L2 收缩 | [Elastic Net](elastic-net.md) | 混合惩罚与 ElasticNetCV |
| 根据初始拟合调整 L1 惩罚 | [Adaptive Lasso](adaptive-lasso.md) | 初始估计与特征尺度会影响惩罚权重 |
| 用非凸惩罚减轻大系数的收缩 | [SCAD](scad.md) 与 [MCP](mcp.md) | 比较惩罚形状与优化限制 |

## 选择不同的目标或残差损失

- [分位数回归](quantile.md)：估计条件分位数，而非条件均值。
- [稳健回归](robust.md)：当较大响应残差需要采用不同于最小二乘的处理方式时，使用 Huber、Bisquare 或 Fair 损失。

## 继续分析

- [Lasso 推断原理](lasso-inference.md)：去偏修正、逐变量精度估计与同时置信区间
- [交叉验证](../../guides/cross-validation.md)：调参与最终重拟合
- [推断模式](../../guides/inference-modes.md)与[回归诊断](../../guides/regression-diagnostics.md)：解释和检查拟合
- [线性模型 API](../../reference/linear-model-api.md)与[估计器通用 API](../../reference/estimator-api.md)：完整接口
- [损失函数](../../reference/losses.md)：底层损失定义与数值性质
- [求解器 × 惩罚项兼容性矩阵](../../guides/solver-penalty-matrix.md)与[设备指南](../../guides/device-and-memory.md)：支持的配置

事件时间响应见 [生存分析](../survival/README.md)；带有个体/时间结构的观测见[面板方法](../../panel/README.md)。
