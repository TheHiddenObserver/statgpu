# 非参数方法

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../../en/models/nonparametric/README.md)

[模型与统计方法](../README.md) / 非参数方法

先确定需要学习的是密度、响应曲线，还是新特征。非参数方法仍有适用假设；
各教程分别说明调参、输入、预测和后端限制。

## 估计密度或局部响应曲线

[KDE 与核回归](nonparametric.md)从两个不同的问题出发：没有响应变量时，观测集中在哪里；
有连续响应时，响应如何随预测变量变化。本页介绍密度估计、局部加权平均、局部线性回归与带宽选择。

## 用相似度预测或学习表示

先读[核方法](kernel-methods.md)，理解共有的核概念与方法选择，再按需要的输出选择教程：

| 需求 | 教程 | 选择依据 |
|---|---|---|
| 用核预测数值响应 | [KernelRidge](kernel-ridge.md) | 使用给定的正则化强度拟合 |
| 在训练折中选择该正则化强度 | [KernelRidgeCV](kernel-ridge-cv.md) | 交叉验证后重拟合 |
| 没有响应变量时学习非线性坐标 | [KernelPCA](kernel-pca.md) | 输出嵌入，而非响应预测 |
| 较小且可重复使用的核特征映射 | [Nystroem](kernel-nystroem.md) | 利用训练地标点近似核，供后续模型使用 |

## 构造平滑项或可加模型

- [样条基函数](splines.md)：构造可重复使用的基特征，包括 `SplineTransformer`；基函数本身不拟合响应。
- [GAM / 半参数模型](semiparametric.md)：为连续响应拟合平滑可加效应。当前实现使用惩罚最小二乘，不提供 GLM 的 `family` 或 `link` 接口。

响应分布族模型见[回归](../regression/README.md)；其他不依赖响应的表示与聚类方法见
[无监督学习](../../unsupervised/README.md)。

## 完整参考

- [核方法 API](../../reference/kernel-methods-api.md)：四类估计器、成对核函数、形状与数值/后端限制
- [生存分析与平滑 API](../../reference/survival-smoothing-api.md)：KDE、核回归、带宽选择、样条与 GAM 接口
- [交叉验证](../../guides/cross-validation.md)与[设备指南](../../guides/device-and-memory.md)：共有分析流程与后端说明
