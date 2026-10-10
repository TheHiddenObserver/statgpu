# MiniBatchNMF

> 语言：中文
> 最后更新：2026-10-06
> 切换：[English](../../en/unsupervised/minibatch-nmf.md)

## 概览

`MiniBatchNMF` 在稠密的小批量（mini-batch）数据上拟合非负低秩分解。支持 Frobenius 损失和乘性更新（MU）的小批量算法，覆盖 CPU、CuPy/CUDA 与 Torch CUDA。

## 何时使用

需要分批学习非负加和结构时，可使用 MiniBatchNMF。`fit` 接收全部数据，外部数据流可用 `partial_fit`。应固定秩与特征含义，并用 `transform` 返回的因子检查重构，不要假设该求解过程与拟合时的更新完全相同。

## 导入路径

```python
from statgpu.unsupervised import MiniBatchNMF
```

## 目标函数 / 损失函数

在非负约束下，模型最小化 Frobenius 重构损失的逐批近似：

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

## 估计方程

每一批中，`MiniBatchNMF` 先固定 `H`，近似求解该批的激活 `W_batch`。`fit` 会在一整轮数据内固定 `H`，累加 `A = sum(W_batch.T @ W_batch)` 和 `B = sum(W_batch.T @ X_batch)`，再更新 `H`；`partial_fit` 则跨调用累加这些统计量。基本乘性更新形式如下：

$$
W \leftarrow W \odot \frac{XH^\top}{WHH^\top + \epsilon},
\qquad
H \leftarrow H \odot \frac{W^\top X}{W^\top WH + \epsilon}.
$$

## 参数

- `n_components`：分解的秩；为 `None` 时使用 `min(n_samples, n_features)`。
- `init`：支持 `"random"`。
- `batch_size`、`max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: minibatch-nmf -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

rng = np.random.default_rng(0)
X = rng.uniform(0.1, 1.0, (60, 2)) @ rng.uniform(0.1, 1.0, (2, 5))
model = MiniBatchNMF(n_components=2, random_state=0, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
W = model.transform(X)
X_hat = model.inverse_transform(W)
print(W.shape, model.components_.shape, np.linalg.norm(X - X_hat))
```

因子形状分别为 `(60, 2)` 和 `(2, 5)`。`partial_fit` 后的 `reconstruction_err_` 只描述该批次及拟合时的因子；上例另外计算的全数据重构误差含义不同。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#minibatchnmf)。显式请求的 GPU 不可用时会报错。

## 首批中全零特征的限制

如果某个特征在首次 `partial_fit` 的整个批次中都为零，字典中对应的整列可能被更新为零。当前乘性更新无法用后续正值重新激活这一列。首批全部为零时，整个字典都可能锁定在零；增大 `max_iter` 或调整 `tol` 不能修复这种状态。

初始化前应缓冲一批有代表性的数据，使预计会出现正值的每个特征都得到覆盖。如果后续批次出现了字典全零列无法表示的新特征，应新建估计器，用包含该特征的代表性保留数据重新拟合。不要为了绕过限制而随意给数据加正偏移，这会改变分解问题本身。

下面先合并两个批次再初始化，可避开全零字典列的问题，但不保证已收敛或得到最优分解。

<!-- learner-example: minibatch-nmf-warmup -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

first = np.array([[1., 0.], [2., 0.], [3., 0.]])
second = np.array([[1., 1.], [2., 2.], [3., 3.]])
warmup = np.vstack([first, second])
if np.any(np.all(warmup == 0, axis=0)):
    raise ValueError("Buffer more representative rows before initialization")
model = MiniBatchNMF(n_components=1, random_state=0, device="cpu")
model.partial_fit(warmup)
reconstructed = model.inverse_transform(model.transform(second))
print("active dictionary columns:", np.any(model.components_ > 0, axis=0))
print("reconstructed second feature:", reconstructed[:, 1])
```

此时字典两列均有正值，第二个特征的重构也为正。若不缓冲，而是依次对 `first`、`second` 调用 `partial_fit`，第二个特征的重构会始终为零。数据流发生变化时，应逐特征检查重构效果。

## 极小数值单位

固定绝对稳定项也可能在 `fit` 和 `partial_fit` 中淹没数值极小的正观测。按 `1e-12` 尺度表示的精确秩一矩阵，即使每批的所有特征均为正，也可能重构成几乎全零。这与上面的全零特征初始化限制不同；仅缓冲正值行还不够。增加 `max_iter` 或设置 `tol=0` 也不能解决。

应从有代表性的缓冲训练行中选定一个有限正尺度，所有批次及后续 `transform` 都复用它。把每个特征除以同一个尺度，再将 `inverse_transform` 输出乘回该尺度以恢复原始单位。统一缩放保持非负性，只给 Frobenius 目标乘上同一个常数；逐特征缩放会改变目标中各特征的权重。不要添加偏移。字典的单位解释见 [NMF 的输入单位说明](nmf.md#极小数值单位)。

<!-- learner-example: minibatch-nmf-units -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchNMF

first = 1e-12 * np.array([[1., 2.], [2., 4.]])
second = 1e-12 * np.array([[3., 6.], [4., 8.]])
scale = float(first.max())
if not np.isfinite(scale) or scale <= 0:
    raise ValueError("Choose a finite positive training scale")
model = MiniBatchNMF(n_components=1, random_state=5, device="cpu")
for batch in (first, second):
    model.partial_fit(batch / scale)
X = np.vstack([first, second])
W = model.transform(X / scale)
X_hat = model.inverse_transform(W) * scale
print("相对重构误差：", np.linalg.norm(X - X_hat) / np.linalg.norm(X))
```

这两个正值成比例的批次可得到接近零的相对误差。此处固定尺度有效，是因为两批数值量级相近；它不能保证任意数据流或动态范围都可靠。应检查相对误差和逐特征误差，不要只依赖绝对 `reconstruction_err_` 很小。

## 近似与解释边界

`MiniBatchNMF` 是非凸的近似分解方法；`partial_fit` 的结果依赖批次顺序，而普通 `fit` 会汇总一整轮数据的统计量后再更新成分。它面向可扩展的矩阵分解，不提供严格的统计推断。

## 输出字段

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**支持负数或稀疏输入吗？**
不支持稀疏输入。`fit`、`partial_fit`、`fit_transform`、`transform` 和 `predict` 要求观测值非负。`inverse_transform` 只将传入坐标乘以 `components_`，允许负坐标，也可能返回负值；需要非负重构时，应传入非负因子。

**支持坐标下降（CD）求解器或其他 beta 损失吗？**
不支持。仅支持乘性更新（MU）和 Frobenius 损失。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [MiniBatchNMF API 参考](api-reference.md#minibatchnmf)。

<a id="references"></a>

## 参考文献

- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. *Advances in Neural Information Processing Systems*, 13.
- Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S.-I. (2009). *Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation*. Wiley.
- scikit-learn Developers. `sklearn.decomposition.MiniBatchNMF`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
