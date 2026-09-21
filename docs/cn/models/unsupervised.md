# 无监督学习

> 语言：中文
> 最后更新：2026-09-21
> 本页：无监督模型总览
> 切换：[English](../../en/models/unsupervised.md)

## 概览

`statgpu.unsupervised` 提供降维、聚类、密度聚类、混合模型、非负矩阵分解、流形嵌入和近似最近邻搜索等估计器。接口沿用 `fit`、`transform`、`predict`、`fit_predict` 和 `score` 风格。

## 模型总览

| 估计器 | 主要用途 | 核心准则 |
|---|---|---|
| [PCA](../unsupervised/pca.md) | 线性降维 | 最大化投影方差 / 最小化秩 k 重构误差 |
| [KMeans](../unsupervised/kmeans.md) | 原型聚类 | 最小化欧氏平方惯性（inertia） |
| [DBSCAN](../unsupervised/dbscan.md) | 带噪声的密度聚类 | 密度可达性与连通分量 |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | 概率软聚类 | 使用 EM 最大化高斯混合对数似然 |
| [NMF](../unsupervised/nmf.md) | 非负的基于部件分解 | 在非负约束下最小化 Frobenius 重构误差 |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | 层次聚类 | 按 linkage 准则贪心合并 |
| [TruncatedSVD](../unsupervised/truncated-svd.md) | 不中心化低秩投影 | 最小化稠密数据上的秩 k 重构误差 |
| [MiniBatchKMeans](../unsupervised/minibatch-kmeans.md) | 较大规模原型聚类 | 使用小批量更新近似最小化 inertia |
| [IncrementalPCA](../unsupervised/incremental-pca.md) | 分批线性降维 | 近似中心化的秩 k 重构 |
| [MiniBatchNMF](../unsupervised/minibatch-nmf.md) | 较大规模非负矩阵分解 | 小批量 Frobenius 重构损失 |
| [UMAP](../unsupervised/umap.md) | 流形嵌入 | 模糊图交叉熵 |
| [NNDescent](../unsupervised/umap.md) | 近似最近邻搜索 | 迭代邻居候选优化 |
| [TSNE](../unsupervised/tsne.md) | Manifold visualization | 高低维相似度分布之间的 KL 散度 |

## 设备行为

多数无监督估计器提供 `device="auto"`、`"cpu"`、`"cuda"` 和 `"torch"`，并遵循项目统一的设备规则。显式指定 GPU 设备时，要么在对应后端执行，要么明确报错；不会静默回退到 CPU。部分算法的设备支持范围更窄，使用前请查看逐模型页面。

## 输入验证

稠密无监督估计器会先执行与后端一致的有限值检查。NaN/Inf 会在 SVD、特征分解、
距离计算或迭代更新前被拒绝，从而返回稳定的公共错误，而不是各算法不同的底层异常。

## 说明

无监督估计器通常不提供标准误、p 值、置信区间、AIC 或 BIC 等统计推断字段，除非模型本身自然定义这些量。因此，这部分文档重点说明算法目标、精确或迭代求解方式、设备支持和输出语义。
