# NMF

> 语言：中文
> 最后更新：2026-10-14
> 切换：[English](../../en/unsupervised/nmf.md)

## 概览

NMF 用指定数量的共享基本模式，近似一张非负数据表。输入的每一行是一条样本，每一列是一个特征。它学习两张非负表：`H` 的每一行是一个基本模式，记录该模式在所有特征上的数值；`W` 的每一行对应一个样本，记录该样本给各个模式乘多大的系数。把各模式乘以样本自己的系数再相加，就得到该样本的近似重构。写成矩阵形式，就是用 `W @ H` 近似 `X`。

若有 `n` 个样本、`p` 个特征和 `k` 个模式，`X` 的形状为 `(n, p)`，`H` 为 `(k, p)`，`W` 为 `(n, k)`。两张学到的表都只能包含零或正数，因此各项贡献相加，不会正负抵消；系数不必加起来等于 1。模式不一定对应真实的物理组成部分，分解也不一定唯一。

### 示例：两个基本光谱

假设测了 60 份样品，每份都记录相同 5 个波长的光强。这里 `X` 的形状为 `(60, 5)`，基本模式就是光谱。我们想用两个基本光谱 A 和 B 来近似这些强度，而不是把每份样品硬分成某一种类型。`H` 有 2 行、5 列，两行分别记录 A 和 B；`W` 有 60 行、2 列，每行记录一份样品的两个系数。

例如，设 A 为 `[1, 0, 2, 0, 1]`，B 为 `[0, 2, 0, 2, 1]`。某份样品的系数为 `[2, 0.5]`，近似强度就是 `2*A + 0.5*B = [2, 1, 4, 1, 2.5]`。将这 5 个重构强度与该样品的实测值比较，就能检查近似得好不好。这组数只用于演示乘法，不是对实际拟合结果的保证。

<!-- worked-example: nonnegative-spectra -->
```python
import numpy as np

H = np.array([[1., 0., 2., 0., 1.], [0., 2., 0., 2., 1.]])
w = np.array([2., 0.5])
x_hat = w @ H
print(x_hat)  # [2.  1.  4.  1.  2.5]
```

系数是乘数，不是基本光谱的个数，也不是概率。这里 `n_components=2` 才表示有两个基本光谱，样品对它们的系数分别是 2 和 0.5，不必加起来等于 1。学到的基本光谱不一定就是实际存在的纯物质。一般来说，基本模式组成的表也称为字典，样本系数也称为激活系数。

## 何时使用

当非负模式相加是有用的解释时，可对稠密非负测量使用 NMF。PCA 虽能压缩数据，但带符号的方向允许正负贡献相互抵消，不容易对应“若干部分相加”的解释。不要对输入做均值中心化；如果关注的是相对基线的正负偏离，[PCA](pca.md) 可能更合适。[TruncatedSVD](truncated-svd.md) 也近似未经中心化的矩阵，但允许带符号因子相互抵消。需要分批学习共享字典时，可考虑 [MiniBatchNMF](minibatch-nmf.md)。

`n_components` 决定提取多少个基本模式，不一定等于矩阵的实际秩。可以先留出有代表性的样本，用其余样本拟合，再调用 `transform` 和 `inverse_transform`，比较留出样本的重构与实测值。换几个随机种子重复拟合，看看是否仍得到相似的模式。比较时要先配对相似模式，并考虑尺度：把一个模式放大 2 倍、对应系数缩小一半，重构结果不会变。

拟合目标把各样本、各特征的误差平方后相加，再乘以二分之一，这就是下文的平方 Frobenius 重构损失。某一列若换成数值大得多的单位，就会更影响总误差。输入非负只是算法要求，并不说明这个误差指标一定适合你的问题。

## 导入路径

```python
from statgpu.unsupervised import NMF
```

当前估计器支持 CPU、CuPy/CUDA 与 Torch CUDA 上的稠密输入。

## 目标函数 / 损失函数

拟合因子需要求解一个带约束的非凸问题：

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

`components_` 存储 `H`；`fit_transform` 返回 `W`。

<a id="估计方程"></a>
## 拟合步骤与更新公式

实现采用乘性更新：

$$
W \leftarrow W \odot
\frac{XH^\top}{WHH^\top + \varepsilon}
$$

$$
H \leftarrow H \odot
\frac{W^\top X}{W^\top W H + \varepsilon}
$$

`init="random"` 时，如果行数不少于成分数，就按随机种子抽取数据行作为初始字典；否则使用按数据均值缩放的正随机元素。初始激活由数据与字典计算。重构误差会定期检查，并在最后一次迭代检查；检查间隔取决于后端。`transform(X)` 会固定已拟合的 `H`，为新数据求解新的 `W`，并执行 `max_iter` 次乘性更新；`tol` 只控制 `fit` 的停止条件，不会让转换求解提前结束。

## 参数

- `n_components`：基本模式的个数，重构矩阵的秩不超过这个数；为 `None` 时使用 `min(n_samples, n_features)`。
- `init`：仅支持 `"random"`。
- `solver`：仅支持 `"mu"`。
- `beta_loss`：仅支持 `"frobenius"`。
- `max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: nmf -->
```python
import numpy as np
from statgpu.unsupervised import NMF

rng = np.random.default_rng(0)
X = rng.uniform(0.1, 1.0, (60, 2)) @ rng.uniform(0.1, 1.0, (2, 5))
model = NMF(n_components=2, max_iter=100, random_state=0, device="cpu")
W = model.fit_transform(X)
X_hat = model.inverse_transform(W)
print(W.shape, model.components_.shape, np.linalg.norm(X - X_hat))
```

沿用前面的光谱示例，`components_[0]` 和 `components_[1]` 分别是学到的基本光谱 A 和 B，各含 5 个强度；`W[i]` 是第 `i` 份样品的两个系数。因此 `W` 为 `(60, 2)`，`components_` 为 `(2, 5)`，`W @ model.components_` 得到全部 60 份样品的 `(60, 5)` 重构强度表。`reconstruction_err_` 是 Frobenius 范数，不是其平方，也不是平均每行误差。后续 `transform` 会固定已学到的字典，只求解新的激活系数。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#nmf)。显式请求的 GPU 不可用时会报错。

## 极小数值单位

乘性更新中的固定绝对稳定项可能淹没数值极小的正观测。例如，严格正且精确为秩一的矩阵，按 `1e-12` 的尺度表示时，可能得到全零重构，即使 `tol=0` 也如此。仅增加迭代次数不能修复这种塌缩状态。观测本身极小时，绝对 `reconstruction_err_` 很小也可能误导判断；还应检查相对误差和逐特征重构误差。

从有代表性的训练数据中选择一个有限正尺度，拟合前把所有特征除以同一个尺度。后续调用 `transform` 时继续使用它，重构结果再乘回该尺度以恢复原始单位。这保持非负性，只给 Frobenius 目标乘上统一的正常数；逐特征分别缩放则会改变各特征的权重。不要通过均值中心化或添加正偏移来处理这个限制。

<!-- learner-example: nmf-units -->
```python
import numpy as np
from statgpu.unsupervised import NMF

X = 1e-12 * np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
scale = float(X.max())
if not np.isfinite(scale) or scale <= 0:
    raise ValueError("Choose a finite positive training scale")
model = NMF(n_components=1, random_state=5, device="cpu")
W = model.fit_transform(X / scale)
X_hat = model.inverse_transform(W) * scale
new_rows = 1e-12 * np.array([[5., 10.]])
W_new = model.transform(new_rows / scale)
new_hat = model.inverse_transform(W_new) * scale
print("相对重构误差：", np.linalg.norm(X - X_hat) / np.linalg.norm(X))
print("新观测按原始单位重构：", new_hat)
```

这个简单秩一示例的相对误差接近零，`new_hat` 接近 `[[5e-12, 1e-11]]`。缩放是数值防护措施，不保证任意数据都能收敛。拟合字典采用缩放后的单位；`W @ (model.components_ * scale)` 可在不修改估计器的情况下得到相同的原始单位重构。后续还需调用 `transform` 时，应保持已拟合字典不变。

## 近似与解释边界

`NMF` 不提供标准误或显著性检验；目标函数非凸。乘性更新尝试寻找局部解；结果质量取决于初始化和停止准则，耗尽迭代次数不代表已经收敛。

## 输出字段

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**输入可以有负数吗？**
`fit`、`fit_transform`、`transform` 和 `predict` 会拒绝负的观测值。`inverse_transform` 只将传入坐标乘以 `components_`，允许负坐标，也可能返回负值；需要非负重构时，应传入非负因子。

**支持坐标下降（coordinate descent）吗？**
不支持。仅支持 Frobenius 损失下的乘性更新（MU）。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [NMF API 参考](api-reference.md#nmf)。

<a id="references"></a>

## 参考文献

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13* (pp. 556-562). MIT Press.
- scikit-learn Developers. [NMF：非负因子与重构目标](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)。
