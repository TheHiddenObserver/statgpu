# NMF

> 语言：中文
> 最后更新：2026-10-06
> 切换：[English](../../en/unsupervised/nmf.md)

## 概览

`NMF` 把非负的稠密数据分解为两个非负因子 `W` 与 `H`。支持 Frobenius 损失下的乘性更新（multiplicative updates），并支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 何时使用

对于强度、计数等可解释为非负加和的特征，可使用 NMF。它不会中心化输入。应结合因子的实际含义和重构误差选择秩；因子存在缩放和排列的不唯一性，不应直接视为唯一的科学机制。

## 导入路径

```python
from statgpu.unsupervised import NMF
```

## 目标函数 / 损失函数

拟合因子需要求解一个带约束的非凸问题：

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

`components_` 存储 `H`；`fit_transform` 返回 `W`。

## 估计方程

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

- `n_components`：隐变量维度；为 `None` 时使用 `min(n_samples, n_features)`。
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

两个因子分别为 `(60, 2)` 的 `W` 和 `(2, 5)` 的 `components_`。`reconstruction_err_` 是 Frobenius 范数，不是其平方，也不是平均每行误差。后续 `transform` 会固定已学到的字典，只求解新的激活系数。

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

`NMF` 没有严格推断模式；目标函数非凸。乘性更新尝试寻找局部解；结果质量取决于初始化和停止准则，耗尽迭代次数不代表已经收敛。

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
