# 无监督学习

> 语言：中文
> 最后更新：2026-05-02
> 本页：无监督模型总览
> English: [English](../en/models/unsupervised.md)

## 概览

`statgpu.unsupervised` 提供降维、聚类、密度聚类、混合模型和非负矩阵分解相关 estimator。接口尽量沿用常见的 `fit`、`transform`、`predict`、`fit_predict` 和 `score` 风格；不适合某个模型语义的方法不会强行暴露。

更详细的逐模型说明位于 [docs/unsupervised/](../unsupervised/README.md)，其中分别说明 objective function、估计过程、设备路径、输出字段、限制和验证方式。

## 模型总览

| Estimator | 主要用途 | 核心准则 |
|---|---|---|
| [PCA](../unsupervised/pca.md) | 线性降维 | 最大化投影方差 / 最小化 rank-k 重构误差 |
| [KMeans](../unsupervised/kmeans.md) | 原型聚类 | 最小化 squared Euclidean inertia |
| [DBSCAN](../unsupervised/dbscan.md) | 带噪声的密度聚类 | Density reachability 与 connected components |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | 概率软聚类 | 使用 EM 最大化 diagonal Gaussian mixture log likelihood |
| [NMF](../unsupervised/nmf.md) | 非负 parts-based 分解 | 在非负约束下最小化 Frobenius 重构误差 |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | 层次聚类 | 贪心 single-linkage 合并 |

## 设备行为

大多数无监督 estimator 提供 `device="auto"`、`"cpu"`、`"cuda"` 和 `"torch"`，并遵循项目统一设备规则。显式 GPU device 要么在对应后端运行，要么给出清晰错误；不应静默回退到 CPU。部分算法的设备支持范围更窄，使用前请查看逐模型页面。

## 说明

无监督 estimator 通常不提供 standard errors、p-values、confidence intervals、AIC 或 BIC 等统计推断字段，除非模型本身自然定义这些量。因此这部分文档重点说明算法目标、exact 与 iterative 行为、设备支持和输出语义。

关于具体 API 行为和逐模型限制，请继续查看上面链接的模型页面。
