# KMeans

> 语言：中文
> 最后更新：2026-10-13
> 切换：[English](../../en/unsupervised/kmeans.md)

## 概览

假设你有设备的温度、压力和振动记录，却没有标注每条记录属于哪种运行状态。只看总体均值，可能把几种反复出现的模式混在一起；逐个指标设置阈值，又可能遗漏真正区分这些模式的指标组合。你希望用少量典型画像概括数据，并把每条观测分配给其中一个画像。

`KMeans` 把这些画像表示为聚类中心：先把每条记录分给最近的中心，再把中心移到所属记录的均值，如此交替更新。它用预先指定数量的代表性中心概括数据。中心是平均画像，不一定对应实际出现过的某条记录。

## 何时使用

当“离某个代表性中心有多远”能够表达你关心的相似性，且希望得到固定数量、较紧凑的组时，可使用 KMeans。应先考虑特征单位：未经缩放时，数值范围很大的指标可能主导平方距离。标准化是一种选择，但它也会改变“相似”的含义。[K-means 几何解释](https://scikit-learn.org/stable/modules/clustering.html#k-means)说明了为什么细长或弯曲的组可能不适合这个目标。

如果分组需求有所不同，可以据此选择邻近方法：

- [MiniBatchKMeans](minibatch-kmeans.md) 保留以中心概括数据的目标，改用较小批次更新。
- [GaussianMixture](gaussian-mixture.md) 适合组间有重叠、希望用模型表达软归属及协方差形状的情况。
- [DBSCAN](dbscan.md) 寻找密度连通区域，允许部分观测不属于任何簇，不必给每条记录强行分组。

即使数据中没有截然分开的真实群体，KMeans 也会给出一个划分。为簇赋予实际含义前，应查看中心画像是否有用，以及结果在不同随机种子和合理预处理选择下是否稳定。

## 导入路径

```python
from statgpu.unsupervised import KMeans
```

## 目标函数 / 损失函数

用所属中心代替一条记录，会产生重构误差。KMeans 将所有记录的平方误差相加，寻找让这个总量尽可能小的中心和分配方式；这个总量称为惯性（inertia）：

$$
\min_{C, z} \sum_{i=1}^{n} \left\|x_i - c_{z_i}\right\|_2^2 .
$$

其中 $x_i$ 是第 $i$ 条记录，$n$ 是记录数，$C$ 是所有中心的集合，$c_{z_i}$ 是分配给该记录的中心。距离取平方后，较大的偏离会获得更大权重。固定分组时，组内均值使平方误差和最小；固定中心时，选择最近中心使每条记录的误差最小。这两点就引出了下面的交替更新。

<a id="估计方程"></a>
## 拟合步骤与更新公式

实现采用 Lloyd 迭代：

- 用 `random` 或贪心的 `k-means++` 初始化聚类中心。
- 用下式计算平方距离，把每个样本分配到最近中心：
  $$
  d_{ij}^2 = \left\|x_i\right\|_2^2 + \left\|c_j\right\|_2^2 - 2 x_i^\top c_j .
  $$
- 把每个聚类中心更新为该簇样本的均值。
  $$
  c_j = \frac{1}{|\{i: z_i = j\}|}\sum_{i:z_i=j} x_i .
  $$
- 空簇用距离当前所属中心最远的样本重置。
- 当中心位移的平方不超过 `tol` 或达到 `max_iter` 时停止。
- 运行 `n_init` 次初始化，保留惯性最低的结果。

## 参数

- `n_clusters`：簇数量。
- `init`：`"k-means++"` 或 `"random"`；不支持可调用（callable）的 `init`。
- `n_init`：`"auto"` 对 k-means++ 使用 `1`，对 random 使用 `10`。
- `max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: kmeans -->
```python
import numpy as np
from statgpu.unsupervised import KMeans

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
model = KMeans(n_clusters=2, n_init=5, random_state=0, device="cpu")
labels = model.fit_predict(X)
print(labels.shape, model.cluster_centers_.shape, model.inertia_)
```

标签形状为 `(60,)`，中心形状为 `(2, 2)`；标签编号没有大小顺序。惯性是平方距离之和，只适合在相同行和量纲下比较，不能仅凭其下降决定簇数。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#kmeans)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`KMeans` 是非凸迭代优化器，不提供标准误或显著性检验；不同初始化可能得到不同的局部最优，可复现性取决于 `random_state`、`init`、`n_init`、`max_iter` 和 `tol`。

## 输出字段

- `cluster_centers_`
- `labels_`
- `inertia_`
- `n_iter_`
- `n_features_in_`

## FAQ

**为什么标签 ID 和 sklearn 不同但聚类看起来一样？**
簇编号本身是任意的；验证应使用惯性、中心匹配或对标签置换不变的指标。

**支持稀疏输入或 `sample_weight` 吗？**
不支持。稠密 KMeans 会对稀疏输入和 `sample_weight` 明确报错。


## 数值与使用注意事项

平方距离通过展开范数计算。很大的共同偏移可能引起消减误差，影响距离、惯性乃至标签。可先减去由训练数据确定的特征偏移，再拟合，并在预测时减去相同偏移；平移不会改变欧氏几何关系。报告中心时可加回偏移，恢复原始单位。

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [KMeans API 参考](api-reference.md#kmeans)。

<a id="references"></a>

## 参考文献

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)* (pp. 1027-1035). Society for Industrial and Applied Mathematics.
