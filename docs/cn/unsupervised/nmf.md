# NMF

> 语言：中文
> 最后更新：2026-09-29
> 切换：[English](../../en/unsupervised/nmf.md)

## 概览

`NMF` 把非负的稠密数据分解为两个非负因子 `W` 与 `H`。Phase 2 支持 Frobenius 损失下的乘性更新（multiplicative updates），并支持 CPU、CuPy/CUDA 与 Torch CUDA。

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

两个因子用按 `X` 均值缩放的正随机值初始化；重构误差每 10 次迭代以及最后一次迭代检查。`transform(X)` 会固定已拟合的 `H`，为新数据求解新的 `W`。

## 参数

- `n_components`：隐变量维度；为 `None` 时使用 `min(n_samples, n_features)`。
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

## 严格与近似模式的差别

`NMF` 没有严格推断模式；目标函数非凸，乘性更新会收敛到依赖初始化与停止准则的局部解。

## 输出字段

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**输入可以有负数吗？**
不可以。`X` 包含负数时 NMF 会报错。

**支持坐标下降（coordinate descent）吗？**
不支持。Phase 2 仅支持 Frobenius 损失下的乘性更新（MU）。

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_nmf.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised_phase2.py`。
- 对齐基线：sklearn 的 `NMF(solver="mu", beta_loss="frobenius")`。
- 最新远程矩阵：CPU/CuPy/Torch 之间的重构差异处于浮点噪声量级；sklearn 的重构误差与 statgpu CPU 处于同一尺度。

## References

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13* (pp. 556-562). MIT Press.
