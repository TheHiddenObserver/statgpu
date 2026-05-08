# MiniBatchNMF

> 语言：中文
> 最后更新：2026-05-07
> English: [English](../en/unsupervised/minibatch-nmf.md)

## 概览

`MiniBatchNMF` 从 dense mini-batches 中拟合非负低秩分解。Phase 3C 支持 Frobenius loss 和 MU-style mini-batch update，覆盖 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import MiniBatchNMF
```

## 目标函数 / 损失函数

在非负约束下，模型最小化 Frobenius reconstruction loss 的 mini-batch 近似：

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

## 估计方程

每个 batch 中，`MiniBatchNMF` 先固定当前 `H` 更新 batch activations `W_batch`，再用乘法更新公式更新 `H`：

$$
W \leftarrow W \odot \frac{XH^\top}{WHH^\top + \epsilon},
\qquad
H \leftarrow H \odot \frac{W^\top X}{W^\top WH + \epsilon}.
$$

## 参数

- `n_components`：分解秩；`None` 使用 `min(n_samples, n_features)`。
- `init`：v1 支持 `"random"`。
- `batch_size`、`max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import MiniBatchNMF

nmf = MiniBatchNMF(n_components=8, batch_size=1024, max_iter=50, random_state=0, device="torch")
W = nmf.fit_transform(X)
X_hat = nmf.inverse_transform(W)
```

## strict/approx 差异

MiniBatchNMF 是非凸且依赖 batch order 的近似分解方法，目标是可扩展 factorization，不提供 strict statistical inference。

## 输出字段

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**v1 支持负数或 sparse input 吗？**
不支持。输入必须是 dense 且非负。

**v1 支持 CD solver 或其他 beta loss 吗？**
不支持。Phase 3C 仅支持 MU-style update 和 Frobenius loss。

## 外部验证

- 测试：`dev/tests/test_unsupervised_minibatch_nmf.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3c.py`。
- 最新远程 artifact：`results/unsupervised_phase3c_opt7_20260507_185500.json`。
- Baseline：sklearn `MiniBatchNMF`，对齐 rank、batch size、初始化和迭代次数。

## References

- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. *Advances in Neural Information Processing Systems*, 13.
- Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S.-I. (2009). *Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation*. Wiley.
- scikit-learn Developers. `sklearn.decomposition.MiniBatchNMF`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
