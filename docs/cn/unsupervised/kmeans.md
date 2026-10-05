# KMeans

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/kmeans.md)

## 概览

`KMeans` 通过最小化平方欧氏簇内误差，把稠密观测分成 `n_clusters` 个簇。支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 何时使用

如果希望把样本分成预先指定数量、在欧氏距离下较紧凑的组，可使用 KMeans。应合理缩放特征，并比较不同随机种子下的中心与惯性。它会给所有样本分组，包括离群点；需要噪声标签时可考虑 DBSCAN。

## 导入路径

```python
from statgpu.unsupervised import KMeans
```

## 目标函数 / 损失函数

`KMeans` 最小化惯性（inertia）：

$$
\min_{C, z} \sum_{i=1}^{n} \left\|x_i - c_{z_i}\right\|_2^2 .
$$

其中 `C` 是聚类中心，`z_i` 是样本 `i` 的簇标签。

## 估计方程

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

`KMeans` 是非凸迭代优化器，不提供严格的统计推断；不同初始化可能得到不同的局部最优，可复现性取决于 `random_state`、`init`、`n_init`、`max_iter` 和 `tol`。

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

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)* (pp. 1027-1035). Society for Industrial and Applied Mathematics.
