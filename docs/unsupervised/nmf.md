# NMF

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/nmf.md)

## 概览

`NMF` 把非负 dense data 分解为非负 factors `W` 和 `H`。Phase 2 支持 Frobenius loss 下的 multiplicative updates，并支持 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import NMF
```

## 目标函数 / 损失函数

拟合 factors 需要求解非凸约束问题：

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

`components_` 存储 `H`；`fit_transform` 返回 `W`。

## 估计方程

实现使用 multiplicative updates：

$$
W \leftarrow W \odot
\frac{XH^\top}{WHH^\top + \varepsilon}
$$

$$
H \leftarrow H \odot
\frac{W^\top X}{W^\top W H + \varepsilon}
$$

Factors 使用按 `X` 均值缩放的正随机值初始化。重构误差每 10 次迭代和最后一次迭代检查。`transform(X)` 会固定已经拟合的 `H`，为新数据更新新的 `W`。

## 参数

- `n_components`：latent dimension；`None` 使用 `min(n_samples, n_features)`。
- `init`：仅支持 `"random"`。
- `solver`：仅支持 `"mu"`。
- `beta_loss`：仅支持 `"frobenius"`。
- `max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import NMF

X = np.abs(np.random.default_rng(0).normal(size=(1000, 32)))

nmf = NMF(n_components=8, random_state=0, device="cuda")
W = nmf.fit_transform(X)
X_hat = nmf.inverse_transform(W)
```

## strict/approx 差异

NMF 没有 strict inference 模式。目标函数是非凸的，multiplicative updates 收敛到依赖初始化和停止准则的局部解。

## 输出字段

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**输入可以有负数吗？**
不可以。`X` 包含负数时 NMF 会报错。

**支持 coordinate descent 吗？**
不支持。Phase 2 仅支持 Frobenius loss 下的 MU。

## 外部验证

- 测试：`dev/tests/test_unsupervised_nmf.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase2.py`。
- Baseline：sklearn `NMF(solver="mu", beta_loss="frobenius")`。
- 最新远程矩阵：CPU/CuPy/Torch reconstruction differences 处于浮点噪声量级；sklearn reconstruction error 与 statgpu CPU 同尺度。

## References

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13* (pp. 556-562). MIT Press.
