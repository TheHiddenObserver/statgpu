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
