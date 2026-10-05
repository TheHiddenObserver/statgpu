# 无监督学习

> 语言：中文
> 最后更新：2026-09-29
> 本页：无监督模型总览
> 切换：[English](../../en/models/unsupervised.md)

## 概览

`statgpu.unsupervised` 提供降维、聚类、密度聚类、混合模型、非负矩阵分解、流形嵌入（manifold embedding）和近似最近邻搜索等估计器。API 沿用 `fit`、`transform`、`predict`、`fit_predict` 和 `score` 风格。

## 模型总览

| 估计器 | 主要用途 | 核心准则 |
|---|---|---|
| [PCA](../unsupervised/pca.md) | 线性降维 | 最大化投影方差 / 最小化秩 k 重构误差 |
| [KMeans](../unsupervised/kmeans.md) | 原型聚类 | 最小化平方欧氏惯性 |
| [DBSCAN](../unsupervised/dbscan.md) | 带噪声的密度聚类 | 密度可达性与连通分量 |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | 概率软聚类 | 用 EM 最大化 Gaussian mixture 的对数似然 |
| [NMF](../unsupervised/nmf.md) | 基于部件的非负分解 | 在非负约束下最小化 Frobenius 重构误差 |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | 层次聚类 | 贪心连接合并 |
| [TruncatedSVD](../unsupervised/truncated-svd.md) | 不中心化低秩投影 | 最小化秩 k 稠密重构误差 |
| [MiniBatchKMeans](../unsupervised/minibatch-kmeans.md) | 较大规模原型聚类 | 用小批量更新近似最小化惯性 |
| [IncrementalPCA](../unsupervised/incremental-pca.md) | 分批线性降维 | 近似中心化的秩 k 重构 |
| [MiniBatchNMF](../unsupervised/minibatch-nmf.md) | 较大规模非负矩阵分解 | 小批量 Frobenius 重构损失 |
| [UMAP](../unsupervised/umap.md) | 流形嵌入 | 近似近邻布局力更新 |
| [TSNE](../unsupervised/tsne.md) | 流形可视化 | 亲和度分布之间的 KL 散度 |

## 设备行为

多数无监督估计器都提供 `device="auto"`、`"cpu"`、`"cuda"` 和 `"torch"`，并遵循项目统一的设备规则。显式请求 GPU 时，要么在对应后端执行，要么给出明确错误，不会静默回退到 CPU；部分算法的设备支持范围更窄，使用前请查看对应模型页。

## 输入验证

稠密无监督估计器共享同一套按后端执行的有限性检查：NaN/Inf 会在 SVD、特征分解、距离计算或迭代更新之前被拒绝，从而给出稳定的公开错误，而不是各算法不同的底层异常。

## 说明

无监督估计器通常不提供标准误、p 值、置信区间、AIC 或 BIC 等统计推断字段，除非模型本身自然定义了这些量。因此本文档重点说明算法目标、精确与迭代行为的区别、设备支持和输出语义。

十二个公开估计器及完整构造和方法约定见[无监督学习索引](../unsupervised/README.md)与 [API 参考](../unsupervised/api-reference.md)。NNDescent 是 UMAP 使用的内部近邻搜索实现，并不是 `statgpu.unsupervised` 额外导出的估计器。
