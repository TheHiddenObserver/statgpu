# 无监督学习

> 切换：[English](../../en/unsupervised/)

支持 GPU 的降维、聚类、流形学习和矩阵分解接口。

## 求解器速查

| 模型 | 公开控制项 | 支持值 |
|---|---|---|
| [PCA](pca.md#求解器支持) | `svd_solver` | `auto`、`full`、`covariance`、`randomized` |
| [NMF](nmf.md#求解器支持) | `solver` | 仅 `mu` |

其他页面即使没有公开 `solver` 参数，也会说明各自固定的估计算法。

## 降维

- [PCA](pca.md)
- [增量 PCA](incremental-pca.md)
- [截断 SVD](truncated-svd.md)

## 矩阵分解

- [非负矩阵分解](nmf.md)
- [小批量 NMF](minibatch-nmf.md)

## 聚类

- [K-means](kmeans.md)
- [小批量 K-means](minibatch-kmeans.md)
- [层次聚类](agglomerative-clustering.md)
- [DBSCAN](dbscan.md)
- [高斯混合模型](gaussian-mixture.md)

## 流形学习

- [t-SNE](tsne.md)
- [UMAP](umap.md)

## 概览

- [完整无监督学习目录](README.md)
