# 无监督学习

> 语言：中文
> 最后更新：2026-07-14
> 本页：无监督模型总览
> English: [English](../../en/models/unsupervised.md)

## 概览

`statgpu.unsupervised` 提供降维、聚类、密度聚类、混合模型、非负矩阵分解、manifold embedding 和近似最近邻搜索相关 estimator。API 沿用 `fit`、`transform`、`predict`、`fit_predict` 和 `score` 风格。

## 模型总览

| Estimator | 主要用途 | 核心准则 |
|---|---|---|
| [PCA](../unsupervised/pca.md) | 线性降维 | 最大化投影方差 / 最小化 rank-k 重构误差 |
| [KMeans](../unsupervised/kmeans.md) | 原型聚类 | 最小化 squared Euclidean inertia |
| [DBSCAN](../unsupervised/dbscan.md) | 带噪声的密度聚类 | Density reachability 和 connected components |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | 概率软聚类 | 使用 EM 最大化 Gaussian mixture log likelihood |
| [NMF](../unsupervised/nmf.md) | 非负 parts-based 分解 | 在非负约束下最小化 Frobenius 重构误差 |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | 层次聚类 | 贪心 linkage 合并 |
| [TruncatedSVD](../unsupervised/truncated-svd.md) | 不中心化低秩投影 | 最小化 rank-k dense 重构误差 |
| [MiniBatchKMeans](../unsupervised/minibatch-kmeans.md) | 较大规模原型聚类 | 使用 mini-batch 更新近似最小化 inertia |
| [IncrementalPCA](../unsupervised/incremental-pca.md) | 分批线性降维 | 近似 centered rank-k 重构 |
| [MiniBatchNMF](../unsupervised/minibatch-nmf.md) | 较大规模非负矩阵分解 | Mini-batch Frobenius 重构损失 |
| [UMAP](../unsupervised/umap.md) | Manifold embedding | Fuzzy graph cross-entropy |
| [NNDescent](../unsupervised/umap.md) | 近似最近邻搜索 | 迭代邻居候选优化 |
| [TSNE](../unsupervised/tsne.md) | Manifold visualization | Affinity 分布之间的 KL divergence |

## 设备行为

多数无监督 estimator 提供 `device="auto"`、`"cpu"`、`"cuda"` 和 `"torch"`，并遵循项目统一设备规则。显式 GPU device 要么在对应后端运行，要么给出清晰错误；不应静默回退到 CPU。部分算法的设备支持范围更窄，使用前请查看逐模型页面。

## 输入验证

稠密无监督估计器共享后端感知的 finite-input 检查。NaN/Inf 会在 SVD、特征分解、
距离计算或迭代更新前被拒绝，从而返回稳定的公共错误，而不是各算法不同的底层异常。

## 说明

无监督 estimator 通常不提供 standard errors、p-values、confidence intervals、AIC 或 BIC 等统计推断字段，除非模型本身自然定义这些量。因此这部分文档重点说明算法目标、exact 与 iterative 行为、设备支持和输出语义。
