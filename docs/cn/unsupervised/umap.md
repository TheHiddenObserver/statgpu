# UMAP

> 语言：中文
> 最后更新：2026-10-06
> 切换：[English](../../en/unsupervised/umap.md)
> 路径：`statgpu.unsupervised.UMAP`

## 概览

`UMAP` 在输入空间构造模糊近邻图（fuzzy neighbor graph），并优化低维嵌入（embedding）。它既支持稠密、精确的欧氏近邻，也支持内置的 NNDescent 近似近邻搜索。

## 何时使用

UMAP 可用于观察局部近邻结构。应比较多个随机种子、邻居数与 `min_dist` 设置。图中不同岛状区域的距离、面积或大小，不能直接解释为人群间差异或占比。

吸引力使相连的近邻靠近，排斥力使点分散。本实现的图权重和力更新与 umap-learn 不同，因此相同参数值不代表等价的嵌入。应把结果视为近似近邻布局；具体差异见[进阶公式](#进阶近邻图与布局公式)。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import UMAP
```

## 一个可独立运行的 CPU 示例

<!-- learner-example: umap -->
```python
import numpy as np
from statgpu.unsupervised import UMAP

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.4, (20, 3)), rng.normal(2, 0.4, (20, 3))])
model = UMAP(n_neighbors=5, n_epochs=20, init="random", nn_method="exact", random_state=0, device="cpu")
embedding = model.fit_transform(X)
print(embedding.shape, model.n_epochs_)
```

嵌入形状为 `(40, 2)`，只对应参与拟合的行。短迭代示例用于说明 API，不代表已获得理想可视化质量。不支持新数据 `transform`，应保留行标识以便追溯样本。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#umap)。显式请求的 GPU 不可用时会报错。

## 参数

- `n_neighbors` 控制近邻数量；调节时应比较局部细节与更广范围的结构。
- `min_dist` 和 `spread` 控制低维吸引曲线，不能修复不可靠的输入近邻图。
- `n_components` 控制输出维数；CPU 上至少使用二维。`metric` 当前只支持欧氏距离。
- `init="random"` 配合 `random_state` 可按种子初始化；谱初始化存在下文所述限制。
- `n_epochs` 与 `learning_rate` 控制优化；`negative_sample_rate` 与 `repulsion_strength` 控制抽样排斥。数值参数应全部有限。
- `nn_method="auto"` 使用精确搜索。`nn_method="nndescent"` 请求近似近邻，但 NumPy 2 的 CPU 路径当前会失败。`device` 选择数值后端；GPU 拟合仍需要主机内存。

全部构造默认值和近似搜索控制参数见 [API 参考](api-reference.md#umap)。

## 后端与主机边界

距离计算、图权重和嵌入数组使用所选的 NumPy、CuPy 或 Torch 后端。在 NumPy 2 上，数组分派存在限制：安装了 Torch 时，CPU 负采样还可能经由 Torch CPU 执行。模糊并集图的组装会把 O(n*k) 的边索引和边权重复制到主机内存，由 SciPy 的稀疏 COO/CSR 结构完成组装，再复制回所选后端。谱初始化和吸引曲线拟合也使用主机端 SciPy，因此 GPU 拟合仍需要 CPU 运算和主机内存。精确近邻还需要 O(n²) 的稠密距离矩阵内存；近似路径可用时，`nn_method='nndescent'` 能避开这个矩阵，但在 NumPy 2 的 CPU 路径上当前会失败。

## 近似与解释边界

`nn_method='exact'` 以 float32 距离运算穷举稠密欧氏近邻，舍入可能影响距离接近的情况；`nn_method='nndescent'` 是近似搜索，并按所选后端执行。两种模式都要经过上述基于 SciPy 的主机侧模糊并集组装；因此图组装仍需要主机内存。

## 输出

`embedding_`、`graph_`、`n_epochs_`、`n_features_in_`。

## FAQ

不支持稀疏输入、非欧氏 `metric`，也不支持对新样本调用 `transform`。近似近邻可以通过 `nn_method='nndescent'` 启用；图的组装仍然需要 SciPy 和主机内存。


## 当前限制

- `nn_method="auto"` 始终选择精确搜索。NumPy 2 上的 CPU `nn_method="nndescent"` 当前会失败，应改用精确搜索。
- CPU `n_components=1` 当前会在力累积时失败，请至少使用两个维度。
- 稀疏 `init="spectral"` 路径可能保留常量图特征向量，而漏掉一个所需的有效方向。特征求解器的起始向量不受 `random_state` 控制；需要按种子初始化时，应使用 `init="random"`。
- 很大的共同特征偏移可能在转为 float32 或计算展开距离时丢失细小间距。应先以 float64 中心化，再拟合；输出有限不代表近邻图可靠。
- 样本间距离极大时，精确搜索可能把样本自身选为邻居，删除自环后便无法保留足够的不同邻居。应先以 float64 中心化，再把所有特征除以由训练数据确定的同一个正尺度，使坐标处于适中范围；需要比较的数据应复用同一偏移与尺度。统一缩放保持欧氏近邻顺序，逐特征分别缩放则会改变距离度量。仅中心化不能解决这一限制。
- 数值参数应全部有限；非有限学习率不一定被拒绝，却可能产生无效嵌入。

## 进阶：近邻图与布局公式

### 目标函数

标准 UMAP 的参考目标是高维图权重 `w_ij` 与低维亲和度 `q_ij` 之间的模糊集交叉熵：

$$
\sum_{i<j}\left[
w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}
\right].
$$

求和只包含不同观测构成的无序点对，不包含自环。分子权重为零的项按连续延拓记为零。其中 $q_{ij}=(1+a\|y_i-y_j\|^{2b})^{-1}$，正的曲线参数 $a,b$ 由 `min_dist` 与 `spread` 确定。这是参考目标，不表示当前的力更新会精确最小化它。

### 估计方程

默认用稠密、精确的搜索选出 `n_neighbors` 个近邻（`nn_method='auto'` 解析为 `exact`）；也可以显式请求内置的 NNDescent。随后构造对称的模糊隶属度图，并执行吸引力及抽样排斥力更新。当前更新不是上式标准交叉熵的精确梯度，因此不能把本实现视为与 umap-learn 数值等价。

#### 图权重

设选中近邻的有序距离为 $d_{i1},\ldots,d_{ik}$，当前实现采用

$$
\rho_i=d_{i1},\qquad
\sigma_i=\max\left(\frac{1}{k}\sum_{j=1}^{k}\max(d_{ij}-\rho_i,0),10^{-12}\right),
\qquad
v_{ij}=\exp\left(-\frac{\max(d_{ij}-\rho_i,0)}{\sigma_i}\right).
$$

未选中的有向边权重为零。对称图采用模糊并集 $w_{ij}=v_{ij}+v_{ji}-v_{ij}v_{ji}$，并删除自环。如果近邻中存在重复观测，$\rho_i$ 可以为零。

这里按近邻距离超出 $\rho_i$ 的平均量确定带宽，并未像 umap-learn 的 `smooth_knn_dist` 那样求解局部隶属度总和的校准条件。因此，即使近邻相同，优化开始前的图权重也可能不同。`min_dist` 和 `spread` 控制低维吸引曲线，不参与这个高维图计算；调节它们不能修复不可靠的近邻图。

### 布局的力更新

每轮抽取 `n_samples * negative_sample_rate` 对均匀随机源点和目标点用于排斥，而不是对每条吸引边分别抽样。

亲和度曲线为 $q_{ij}=(1+a\,r_{ij}^{b})^{-1}$，其中 $r_{ij}=\|y_i-y_j\|^2$。当前吸引项正比于 $w_{ij}q_{ij}(y_i-y_j)$，抽样排斥项正比于 $q_{ij}^2(y_i-y_j)$。标准交叉熵梯度还包含额外的距离相关因子。因此应把输出视为近似近邻布局，并直接检验其实际用途。

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [UMAP API 参考](api-reference.md#umap)。

<a id="references"></a>

## 参考文献

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

- umap-learn 开发者。[`smooth_knn_dist` 源码参考](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#smooth_knn_dist)。
