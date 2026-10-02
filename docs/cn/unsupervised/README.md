# 无监督学习

> 语言：中文
> 最后更新：2026-10-02
> 本页：无监督学习索引
> 切换：[English](../../en/unsupervised/README.md)

## 概览

`statgpu.unsupervised` 提供 sklearn 风格的无监督学习估计器，并遵循显式 CPU、CuPy/CUDA、Torch CUDA 设备语义。本目录按模型拆分，说明各自的目标函数、估计过程、后端行为、输出字段、限制与外部验证方式。

## 模型列表

- [PCA](pca.md)：精确或随机化（randomized）主成分分析。
- [KMeans](kmeans.md)：Lloyd 迭代聚类，支持 random 与贪心 k-means++ 初始化。
- [DBSCAN](dbscan.md)：稠密欧氏距离的密度聚类，支持可选的 statgpu 自有 Cython CPU 快速路径。
- [GaussianMixture](gaussian-mixture.md)：支持 diagonal、spherical、tied、full 四种协方差结构的 Gaussian mixture，用 EM 拟合。
- [NMF](nmf.md)：在 Frobenius 损失下用乘性更新求解的非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：稠密数据的精确层次聚类，支持 single、complete、average、ward 连接。
- [TruncatedSVD](truncated-svd.md)：不对输入做中心化的稠密截断 SVD。
- [MiniBatchKMeans](minibatch-kmeans.md)：面向较大规模稠密数据的 mini-batch KMeans。
- [IncrementalPCA](incremental-pca.md)：按批次处理的稠密主成分分析。
- [MiniBatchNMF](minibatch-nmf.md)：按小批量处理的稠密非负矩阵分解。
- [UMAP](umap.md)：稠密欧氏距离的精确 UMAP 第一版。
- [TSNE](tsne.md)：稠密欧氏距离的精确 t-SNE 第一版。

## 支持矩阵

| 估计器 | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / 秩 k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | 平方欧氏惯性（inertia） |
| `DBSCAN` | 支持 | 支持 | 支持 | 密度可达性与连通分量 |
| `GaussianMixture` | 支持 | 支持 | 支持 | Gaussian mixture 对数似然 |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 支持 | 支持 | 层次聚类的连接合并准则 |
| `TruncatedSVD` | 支持 | 支持 | 支持 | 不中心化低秩重构 |
| `MiniBatchKMeans` | 支持 | 支持 | 支持 | 小批量平方欧氏惯性 |
| `IncrementalPCA` | 支持 | 支持 | 支持 | 分批中心化低秩重构 |
| `MiniBatchNMF` | 支持 | 支持 | 支持 | 小批量 Frobenius 重构损失 |
| `UMAP` | 支持 | 支持（图的组装在主机侧用 SciPy 完成） | 支持（图的组装在主机侧用 SciPy 完成） | 模糊图交叉熵 |
| `TSNE` | 支持 | 支持 | 支持 | 高低维亲和度之间的 KL 散度 |

显式 `device="cuda"` 和 `device="torch"` 不会静默回退到 CPU；依赖不可用或模型不支持时会明确报错。

## 验证说明

这些无监督估计器均有对应的单元测试，并在适用时与 scikit-learn、cuML、umap-learn 或 openTSNE 等外部实现进行数值对照。GPU 路径还会检查设备语义和 CPU/GPU 结果一致性。性能会明显依赖数据规模、维数、硬件和数据驻留位置，因此外部基准只应作为参考，实际使用时建议针对自己的工作负载重新测量。

这些外部软件包仅作为验证或基准测试的对照；生产代码保持 statgpu 自有实现。
