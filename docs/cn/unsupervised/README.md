# 无监督学习

> 语言：中文
> 最后更新：2026-10-13
> 本页：无监督学习索引
> 切换：[English](../../en/unsupervised/README.md)

## 概览

没有标签的数据表仍然可能包含有用的结构：多个测量值可能记录了几乎相同的变化，观测之间可能形成分组，相近观测也可能沿着单列数据中看不出的规律分布。列均值和单变量图有助于初步检查，却不能概括这些特征之间或观测之间的关系。无监督学习首先要明确想概括哪一种关系，因为这里没有已知目标标签来告诉模型什么才是正确答案。

三类问题对应不同的输出：

- **表示数据：** 能否用更少的坐标或可相加的组成部分重构测量中关心的信息？重构误差小，说明所选表示能够近似数据，并不证明每个成分都有科学含义。
- **组织观测：** 哪些观测应视为相似，是否每个观测都必须归入某一组？围绕中心划分、建立概率模型、连接高密度区域和构建层次结构，回答的是这个问题的不同版本。
- **探索图形：** 在测量特征所定义的空间中，哪些观测互为近邻？图形可以提示值得研究的模式，但图上的孤岛并不自动成为发现的类别。

先确定问题，再选择算法。例如，压缩相关的传感器读数与划分传感器的运行状态，即使用同一张表也是不同任务。先查看原始特征、简单汇总或线性投影，能说明这些方法遗漏了什么，再考虑更专门的方法。没有现成标签时，应根据目的检查稳定性、重构效果或邻域关系，并用拟合之外的信息核实解释。

`statgpu.unsupervised` 为这些任务提供 sklearn 风格的估计器。下方模型指南把学习动机与目标函数、拟合输出和限制联系起来；各页的设备部分说明显式 CPU、CuPy/CUDA 和 Torch CUDA 行为。

UMAP 可使用内部的 NNDescent 实现进行近似近邻搜索，但该搜索不是单独导出的估计器。各模型按其适用操作提供 `fit`、`transform`、`predict`、`fit_predict` 和 `score` 风格。

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
| 是否需要分批处理数据？ | [IncrementalPCA](incremental-pca.md)、[MiniBatchKMeans](minibatch-kmeans.md) 或 [MiniBatchNMF](minibatch-nmf.md) | 对应模型的批量大小、初始化和 `partial_fit` 要求，包括 MiniBatchNMF 首批全零特征的限制 |

这些实现面向稠密输入。特别是，当前 TruncatedSVD 不是稀疏文本处理流程。UMAP 提供 `nn_method="exact"` 和 `"nndescent"`（近似）近邻搜索，`"auto"` 始终选择精确搜索，但图组装仍在主机端使用 SciPy。TSNE 使用精确的稠密距离计算。这两种可视化估计器均不支持对新数据调用 `transform`。用于大数据集前，先阅读对应模型的限制。

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

[完整无监督 API 参考](api-reference.md)列出十二个估计器的默认值、方法签名、输出形状与增量批次限制。

- [PCA](pca.md)：精确或随机化（randomized）主成分分析。
- [KMeans](kmeans.md)：Lloyd 迭代聚类，支持 random 与贪心 k-means++ 初始化。
- [DBSCAN](dbscan.md)：稠密欧氏距离的密度聚类，可使用编译扩展加速 CPU 计算。
- [GaussianMixture](gaussian-mixture.md)：支持 `diag`（对角协方差）、`spherical`、`tied`、`full` 四种协方差结构的高斯混合模型，用 EM 算法拟合。
- [NMF](nmf.md)：在 Frobenius 损失下用乘性更新求解的非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：稠密数据的精确层次聚类，支持 single、complete、average、ward 连接。
- [TruncatedSVD](truncated-svd.md)：不对输入做中心化的稠密截断 SVD。
- [MiniBatchKMeans](minibatch-kmeans.md)：面向较大规模稠密数据的小批量 KMeans。
- [IncrementalPCA](incremental-pca.md)：按批次处理的稠密主成分分析。
- [MiniBatchNMF](minibatch-nmf.md)：按小批量处理的稠密非负矩阵分解。
- [UMAP](umap.md)：基于稠密欧氏数据的 UMAP，支持精确或近似近邻搜索，图组装在主机端使用 SciPy。
- [TSNE](tsne.md)：稠密欧氏距离的精确 t-SNE。

<a id="模型总览"></a>

## 支持矩阵

| 估计器 | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / 秩 k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | 簇内平方欧氏距离之和：各样本到所属聚类中心的平方距离相加 |
| `DBSCAN` | 支持 | 支持 | 支持 | 密度可达性与连通分量 |
| `GaussianMixture` | 支持 | 支持 | 支持 | 高斯混合模型的对数似然 |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 支持 | 支持 | 层次聚类的连接合并准则 |
| `TruncatedSVD` | 支持 | 支持 | 支持 | 不中心化低秩重构 |
| `MiniBatchKMeans` | 支持 | 支持 | 支持 | 小批量聚类的簇内平方和 |
| `IncrementalPCA` | 支持 | 支持 | 支持 | 分批中心化低秩重构 |
| `MiniBatchNMF` | 支持 | 支持 | 支持 | 小批量 Frobenius 重构损失 |
| `UMAP` | 精确搜索，至少二维 | 支持（图的组装在主机侧用 SciPy 完成） | 支持（图的组装在主机侧用 SciPy 完成） | 近似近邻布局力更新 |
| `TSNE` | 支持 | 支持 | 支持 | 高低维亲和度之间的 KL 散度 |

显式 `device="cuda"` 和 `device="torch"` 不会静默回退到 CPU；依赖不可用或模型不支持时会明确报错。

## 验证说明

比较不同实现时，应对齐目标函数、初始化、容差和预处理。具体限制及数值注意事项见各模型指南；应检查输出有限性，以及模型提供的收敛状态。性能应针对自己的完整工作负载重新测量。

在 CPU 上用 `DBSCAN` 处理超过 12 个特征时，需安装 scikit-learn，因为该配置使用其 `NearestNeighbors` 搜索邻居。依赖、内存要求和已知限制见 [DBSCAN 指南](dbscan.md)。

当前具体限制见 [UMAP](umap.md) 和 [DBSCAN](dbscan.md)；支持 CPU 不代表每种参数与数据组合都可靠。

## 设备行为

多数无监督估计器都提供 `device="auto"`、`"cpu"`、`"cuda"` 和 `"torch"`。显式请求的 GPU 后端不可用时，拟合会报错。部分算法仍包含 CPU 计算或返回 NumPy 数组；依赖 GPU 路径前，请查看对应模型页中的操作支持、输出位置和限制。

## 输入验证

观测矩阵会在 SVD、特征分解、距离计算或迭代更新之前接受 NaN/Inf 检查；发现非有限值时会抛出公开的验证错误。这不代表所有构造参数都已检查：数值控制参数及 MiniBatchKMeans 的显式初始中心也必须有限，但当前并非全部得到可靠验证。详见[公共输入规则](api-reference.md#公共输入设备与状态)和[初始中心注意事项](minibatch-kmeans.md#数值与使用注意事项)。

## 说明

无监督估计器通常不提供标准误、p 值、置信区间、AIC 或 BIC 等统计推断字段，除非模型本身自然定义了这些量。

完整构造和方法约定见 [API 参考](api-reference.md)。
