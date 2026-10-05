# TruncatedSVD

> 语言：中文
> 最后更新：2026-10-05
> 路径：`statgpu.unsupervised.TruncatedSVD`

## 概览

`TruncatedSVD` 在不对输入矩阵做中心化的前提下计算低秩投影；这一点与 `PCA` 不同，也更适合稠密的 LSA 类流程。

## 何时使用

如果数据原点具有意义，中心化会改变问题，可使用 TruncatedSVD。与 PCA 不同，较大的特征均值可能主导首个方向。当前实现要求稠密输入，不能直接用于稀疏文本矩阵。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TruncatedSVD
```

## 目标函数

给定秩 `k`，Truncated SVD 求解：

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

## 估计方程

精确路径计算：

$$
X = U \Sigma V^\top.
$$

随机化路径先把 `X` 投影到随机的低维子空间，每轮幂迭代后重新正交化，并使用确定性的主成分符号约定，最后对一个小矩阵做 SVD。

## 参数

`n_components`、`algorithm`、`n_iter`、`n_oversamples`、`random_state`、`device`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: truncated-svd -->
```python
import numpy as np
from statgpu.unsupervised import TruncatedSVD

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5)) + 2.0
model = TruncatedSVD(n_components=2, algorithm="full", device="cpu")
Z = model.fit_transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, X_hat.shape, model.explained_variance_ratio_.sum())
```

坐标形状为 `(60, 2)`，重构时不会加回均值。解释方差是投影坐标中心化后的方差（除以 `n`），尽管拟合本身不中心化；它不等同于保留奇异值平方和占原矩阵平方范数的比例。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#truncatedsvd)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`algorithm="full"` 是稠密数据上的精确 SVD；`algorithm="randomized"` 是近似算法，比较时应使用对符号和子空间不变的指标。

## 输出

`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`n_components_`、`n_features_in_`。

## FAQ

不支持稀疏输入和 ARPACK。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [TruncatedSVD API 参考](api-reference.md#truncatedsvd)。

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.
