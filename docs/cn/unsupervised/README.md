# 无监督学习

> 语言：中文
> 最后更新：2026-09-21
> 本页：无监督学习索引
> 切换：[English](../../en/unsupervised/README.md)

## 概览

`statgpu.unsupervised` 提供 scikit-learn 风格的无监督学习估计器，并遵循明确的 CPU、CuPy/CUDA 和 Torch CUDA 设备语义。这个目录按模型分别介绍目标函数、估计过程、后端行为、输出字段、适用限制以及与外部实现的对照。

## 模型列表

- [PCA](pca.md)：精确或随机化主成分分析。
- [KMeans](kmeans.md)：Lloyd 聚类，支持随机初始化和 greedy k-means++ 初始化。
- [DBSCAN](dbscan.md)：稠密欧氏空间中的密度聚类，并提供可选的 statgpu Cython CPU 加速路径。
- [GaussianMixture](gaussian-mixture.md)：高斯混合模型，支持 diagonal、spherical、tied 和 full 四种协方差结构，并使用 EM 拟合。
- [NMF](nmf.md)：以 Frobenius 重构损失为目标，使用乘法更新的非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：稠密数据上的精确层次聚类，支持 single、complete、average 和 ward linkage。
- [TruncatedSVD](truncated-svd.md)：面向稠密、不中心化输入的截断 SVD。
- [MiniBatchKMeans](minibatch-kmeans.md)：面向较大规模稠密数据的小批量 KMeans。
- [IncrementalPCA](incremental-pca.md)：面向稠密数据的分批主成分分析。
- [MiniBatchNMF](minibatch-nmf.md)：面向稠密数据的小批量非负矩阵分解。
- [UMAP](umap.md)：稠密欧氏数据上的精确 UMAP v1。
- [TSNE](tsne.md)：稠密欧氏数据上的精确 t-SNE v1。

## 支持矩阵

| 估计器 | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / 秩 k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | 欧氏平方惯性（inertia） |
| `DBSCAN` | 支持 | 支持 | 支持 | 密度可达性与连通分量 |
| `GaussianMixture` | 支持 | 支持 | 支持 | 高斯混合对数似然 |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 支持 | 支持 | 层次 linkage 合并准则 |
| `TruncatedSVD` | 支持 | 支持 | 支持 | 不中心化低秩重构 |
| `MiniBatchKMeans` | 支持 | 支持 | 支持 | 小批量欧氏平方惯性 |
| `IncrementalPCA` | 支持 | 支持 | 支持 | 分批中心化低秩重构 |
| `MiniBatchNMF` | 支持 | 支持 | 支持 | 小批量 Frobenius 重构损失 |
| `UMAP` | 支持 | 支持，含主机端 SciPy 图构建 | 支持，含主机端 SciPy 图构建 | 模糊图交叉熵 |
| `TSNE` | 支持 | 支持 | 支持 | 高低维相似度分布之间的 KL 散度 |

显式指定 `device="cuda"` 或 `device="torch"` 时不会静默回退到 CPU；依赖不可用或模型不支持时会明确报错。

## 验证说明

这些无监督估计器均有对应的单元测试，并在适用时与 scikit-learn、cuML、umap-learn 或 openTSNE 等外部实现进行数值对照。GPU 路径还会检查设备语义和 CPU/GPU 结果一致性。性能会明显依赖数据规模、维数、硬件和数据驻留位置，因此外部基准只应作为参考，实际使用时建议针对自己的工作负载重新测量。
