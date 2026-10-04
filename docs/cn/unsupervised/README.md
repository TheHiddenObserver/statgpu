# 无监督学习

> 语言：中文
> 最后更新：2026-10-04
> 本页：无监督学习索引
> 切换：[English](../../en/unsupervised/README.md)

## 概览

`statgpu.unsupervised` 提供 sklearn 风格的无监督学习估计器，并遵循显式 CPU、CuPy/CUDA、Torch CUDA 设备语义。本目录按模型拆分，说明各自的目标函数、估计过程、后端行为、输出字段、限制与外部验证方式。

## 从你的问题选择模型

| 你想回答什么问题？ | 可以先看 | 重点检查什么？ |
|---|---|---|
| 能否用更少的数值特征概括数据变化？ | [PCA](pca.md)；不希望中心化时看 [TruncatedSVD](truncated-svd.md) | 成分载荷、解释的变异与重构效果，不要只看前两个坐标 |
| 能否把非负数据分解成可相加的组成部分？ | [NMF](nmf.md) | 非负因子与重构误差；均值中心化可能破坏非负输入要求 |
| 能否把观测分成指定数量的紧凑组？ | [KMeans](kmeans.md) | 聚类中心、标签与簇内平方和（inertia）；标签只是任意编号，不代表大小或等级 |
| 是否需要为重叠的组给出成员概率？ | [GaussianMixture](gaussian-mixture.md) | 成员概率与拟合的协方差结构；比较成分数时保持数据和评分方式一致 |
| 是否需要按局部密度形成组，并标出可能的噪声点？ | [DBSCAN](dbscan.md) | 邻域尺度、最低密度与噪声标记；特征单位会影响欧氏距离 |
| 是否需要层次结构，而不只是一次固定分组？ | [AgglomerativeClustering](agglomerative-clustering.md) | 合并结构和连接方式，以及不同切分层级如何改变分组 |
| 是否主要想把邻域关系画在低维空间？ | [UMAP](umap.md) 或 [TSNE](tsne.md) | 不同随机种子和参数下的稳定性；图上分离本身不能证明总体存在不同类别 |
| 是否需要分批处理数据？ | [IncrementalPCA](incremental-pca.md)、[MiniBatchKMeans](minibatch-kmeans.md) 或 [MiniBatchNMF](minibatch-nmf.md) | 对应模型的批量大小、初始化和 `partial_fit` 要求 |

这些实现面向稠密输入。特别是，当前 TruncatedSVD 不是稀疏文本处理流程。UMAP 提供 `nn_method="exact"` 和 `"nndescent"`（近似）近邻搜索，`"auto"` 会选择路径，但图组装仍在主机端使用 SciPy。TSNE 使用精确的稠密距离计算。这两种可视化估计器均不支持对新数据调用 `transform`。用于大数据集前，先阅读对应模型的限制。

## 第一次使用的顺序

1. 明确每一行代表什么，以及哪些数值特征和单位有意义。拟合前处理缺失值和非有限值。基于距离的结果可能对特征尺度很敏感，应根据问题决定如何缩放，而不是机械地统一标准化。
2. 准备有代表性的小规模数值样本，按对应模型页面的说明使用 `device="cpu"` 完成第一次拟合。核对输入形状、拟合属性和支持的预测/转换方法；采用 sklearn 风格命名，不代表每个估计器都实现了所有方法。
3. 按上表第三列检查结果，并比较合理的参数设置。使用留出观测评价时，预处理和模型选择只能使用训练部分。不同模型的 `score()` 含义可能不同，不应直接按数值高低比较无关模型。
4. 确认小规模流程正确后，再考虑分批处理或受支持的 GPU 后端。检查模型的内存限制，并测量包含必要数据传输的完整工作负载；小规模拟合不保证在 GPU 上更快。

下面的模型页面提供各自的示例、参数、输出约定和失败模式。本索引用于帮助选择下一步阅读内容。

## 先跑一个小型 CPU 流程

这个 PCA 示例把三个数值特征压缩到两个。第三列几乎是第一列的副本；模拟特征的单位可比较，因此这里不额外缩放。先用训练行拟合，再用同一个模型转换新观测。

<!-- learner-example: unsupervised-pca -->
```python
import numpy as np
from statgpu.unsupervised import PCA

rng = np.random.default_rng(12)
X = rng.normal(size=(120, 3))
X[:, 2] = X[:, 0] + 0.05 * rng.normal(size=120)
X_train, X_test = X[:90], X[90:]
pca = PCA(n_components=2, svd_solver="full", device="cpu")
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)
X_reconstructed = pca.inverse_transform(Z_test)
print("reduced shapes:", Z_train.shape, Z_test.shape)
print("explained fraction:", round(float(pca.explained_variance_ratio_.sum()), 4))
print("held-out reconstruction MSE:", round(float(np.mean((X_test - X_reconstructed) ** 2)), 4))
```

两个降维结果的形状是 `(90, 2)` 和 `(30, 2)`。保留的方差比例接近 1，留出观测的重构误差很小，因为这组模拟数据接近二维。这个结果说明压缩效果，不是分类结果，也不是科学重要性的检验。成分解释、符号不唯一、缩放和求解器选择见 [PCA](pca.md)。

## 模型列表

- [PCA](pca.md)：精确或随机化（randomized）主成分分析。
- [KMeans](kmeans.md)：Lloyd 迭代聚类，支持 random 与贪心 k-means++ 初始化。
- [DBSCAN](dbscan.md)：稠密欧氏距离的密度聚类，支持可选的 statgpu 自有 Cython CPU 快速路径。
- [GaussianMixture](gaussian-mixture.md)：支持 `diagonal`、`spherical`、`tied`、`full` 四种协方差结构的高斯混合模型，用 EM 算法拟合。
- [NMF](nmf.md)：在 Frobenius 损失下用乘性更新求解的非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：稠密数据的精确层次聚类，支持 single、complete、average、ward 连接。
- [TruncatedSVD](truncated-svd.md)：不对输入做中心化的稠密截断 SVD。
- [MiniBatchKMeans](minibatch-kmeans.md)：面向较大规模稠密数据的小批量 KMeans。
- [IncrementalPCA](incremental-pca.md)：按批次处理的稠密主成分分析。
- [MiniBatchNMF](minibatch-nmf.md)：按小批量处理的稠密非负矩阵分解。
- [UMAP](umap.md)：基于稠密欧氏数据的 UMAP，支持精确或近似近邻搜索，图组装在主机端使用 SciPy。
- [TSNE](tsne.md)：稠密欧氏距离的精确 t-SNE 第一版。

## 支持矩阵

| 估计器 | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / 秩 k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | 簇内平方和：各样本到所属聚类中心的平方欧氏距离之和 |
| `DBSCAN` | 支持 | 支持 | 支持 | 密度可达性与连通分量 |
| `GaussianMixture` | 支持 | 支持 | 支持 | 高斯混合模型的对数似然 |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 支持 | 支持 | 层次聚类的连接合并准则 |
| `TruncatedSVD` | 支持 | 支持 | 支持 | 不中心化低秩重构 |
| `MiniBatchKMeans` | 支持 | 支持 | 支持 | 小批量聚类的簇内平方和 |
| `IncrementalPCA` | 支持 | 支持 | 支持 | 分批中心化低秩重构 |
| `MiniBatchNMF` | 支持 | 支持 | 支持 | 小批量 Frobenius 重构损失 |
| `UMAP` | 支持 | 支持（图的组装在主机侧用 SciPy 完成） | 支持（图的组装在主机侧用 SciPy 完成） | 模糊图交叉熵 |
| `TSNE` | 支持 | 支持 | 支持 | 高低维亲和度之间的 KL 散度 |

显式 `device="cuda"` 和 `device="torch"` 不会静默回退到 CPU；依赖不可用或模型不支持时会明确报错。

## 验证说明

这些无监督估计器均有对应的单元测试，并在适用时与 scikit-learn、cuML、umap-learn 或 openTSNE 等外部实现进行数值对照。GPU 路径还会检查设备语义和 CPU/GPU 结果一致性。性能会明显依赖数据规模、维数、硬件和数据驻留位置，因此外部基准只应作为参考，实际使用时建议针对自己的工作负载重新测量。

这些外部软件包仅作为验证或基准测试的对照；生产代码保持 statgpu 自有实现。
