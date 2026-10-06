# PCA

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/pca.md)

## 概览

`PCA` 对已中心化的稠密数据估计一组正交低维基，使其捕捉方差最大的方向。支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 何时使用

当多个数值特征高度相关，且希望用少量线性方向概括它们时，可使用 PCA。模型会自动中心化，但不会自动统一量纲；应先选择合理的单位，并只用训练数据拟合预处理。可结合累计解释方差和留出数据的重构误差选择成分数，高方差本身不代表科学上的重要性。

## 导入路径

```python
from statgpu.unsupervised import PCA
```

## 目标函数 / 损失函数

对中心化数据 `X_c = X - mean(X)`，PCA 求解：

$$
\begin{aligned}
\max_{W \in \mathbb{R}^{p \times k}} \quad
& \operatorname{tr}\left(W^\top X_c^\top X_c W\right) \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

保留 `k` 个主成分时，也等价于在正交投影约束下最小化秩为 `k` 的平方重构误差：

$$
\begin{aligned}
\min_{W \in \mathbb{R}^{p \times k}} \quad
& \left\|X_c - X_c W W^\top\right\|_F^2 \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

二者等价，因为中心化后的总方差是固定的。

## 估计方程

- `svd_solver="covariance"` 先计算
  $$
  \Sigma = \frac{X_c^\top X_c}{n - 1}
  $$
  再用 `eigh` 求解
  $$
  \Sigma v_j = \lambda_j v_j .
  $$
- `svd_solver="full"` 计算
  $$
  X_c = U S V^\top
  $$
  并把 `V.T` 的各行作为主成分。
- `svd_solver="auto"` 在 `n_samples >= n_features` 时使用协方差矩阵配合 `eigh`，否则使用完整 SVD。
- `svd_solver="randomized"` 通过随机投影、幂迭代（power iteration）和一个小矩阵的 SVD 近似前若干个右奇异向量。
- 解释方差（explained variance）计算为
  $$
  \operatorname{explained\_variance}_j = \frac{s_j^2}{n - 1}.
  $$
- `explained_variance_ratio_` 是保留方差除以中心化后的总方差。

## 参数

- `n_components`：保留的主成分个数；为 `None` 时保留所有可行的主成分。
- `svd_solver`：`"auto"`、`"full"`、`"covariance"` 或 `"randomized"`。
- `whiten`：为 `True` 时，`transform` 后的得分会除以 `sqrt(explained_variance_)`。
- `random_state`、`n_oversamples`、`iterated_power`：控制随机化 SVD 的参数。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: pca -->
```python
import numpy as np
from statgpu.unsupervised import PCA

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 4))
X[:, 3] = X[:, 0] + 0.05 * rng.normal(size=80)
model = PCA(n_components=3, svd_solver="full", device="cpu")
Z = model.fit_transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, np.mean((X - X_hat) ** 2))
```

坐标形状为 `(80, 3)`，重构形状为 `(80, 4)`。`components_` 的每一行表示特征空间中的一个方向，不是类别；方向整体变号不改变含义。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#pca)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`PCA` 没有统计推断意义上的严格推断模式；这里的“精确/近似”指的是分解算法：

- `full` 和 `covariance` 在稠密输入上是精确求解器，误差仅来自浮点运算。
- `randomized` 是近似的截断 SVD，由 `random_state`、`n_oversamples` 和 `iterated_power` 控制。
- 主成分的符号不可识别：`v` 与 `-v` 表示同一个主成分。

## 输出字段

- `components_`
- `mean_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`

## FAQ

**为什么主成分和 sklearn 差一个符号？**
特征向量与奇异向量的符号并不唯一；验证时应使用对齐符号后的比较，或直接比较子空间。

**白化（whitening）做了什么？**
白化将投影坐标除以 `sqrt(explained_variance_)`。在正方差方向得到准确的精确分解时，训练数据的主成分得分具有单位样本方差；随机化求解只能近似达到这一效果，过采样与幂迭代会影响近似质量。白化不保证新观测的协方差为单位矩阵。


## 数值与使用注意事项

协方差求解器先计算未中心化的二阶矩，再减去均值乘积。当共同偏移远大于实际变化时，可能发生严重消减误差，甚至错误地返回全零解释方差比例。此时应使用 `svd_solver="full"`，或在拟合前减去由训练数据确定的偏移，并对后续数据做相同处理。对方差为零的成分进行白化可能返回非有限坐标，应减小秩或关闭白化。公开的 `inverse_transform` 方法会对含 NaN 或无穷值的输入坐标抛出 `ValueError`，因此应传入有限的主成分得分。

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [PCA API 参考](api-reference.md#pca)。

<a id="references"></a>

## 参考文献

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559-572. https://doi.org/10.1080/14786440109462720
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer Series in Statistics. Springer. https://doi.org/10.1007/b98835
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
