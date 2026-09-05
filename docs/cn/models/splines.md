# 样条基函数

> 语言: 中文
> 最后更新: 2026-07-14
> 页面定位: 模型文档
> 切换: [English](../../en/models/splines.md)

语言切换：[English](../../en/models/splines.md)

## 概览（Overview）

样条模块提供一组用于构建设计矩阵的基函数工具。`bspline_basis` 使用 De Boor 递推计算 B 样条基；`natural_cubic_spline_basis` 通过边界二阶导数为零的约束构造自然三次样条；`cyclic_cubic_spline_basis` 在两端强制函数值、一阶导数和二阶导数连续，适合周期变量；`thin_plate_spline_basis` 使用薄板样条径向核构造多维基。`SplineTransformer` 则把 B 样条封装为 sklearn 兼容的 `fit`/`transform` 接口，便于加入流水线。所有接口均支持 NumPy、CuPy 和 Torch 后端。

使用这些基函数的广义可加模型（GAM）请参见 [GAM](semiparametric.md)。

## 路径（Path）

```
statgpu.nonparametric.splines.bspline_basis
statgpu.nonparametric.splines.natural_cubic_spline_basis
statgpu.nonparametric.splines.cyclic_cubic_spline_basis
statgpu.nonparametric.splines.thin_plate_spline_basis
statgpu.nonparametric.splines.SplineTransformer
```

## 目标函数（Objective Function）

**B 样条基**通过 De Boor 递归计算。零次基函数为：

$$
B_{i,0}(x) = \begin{cases} 1 & \text{若 } t_i \le x < t_{i+1} \\ 0 & \text{其他} \end{cases}
$$

对于次数 $k \ge 1$：

$$
B_{i,k}(x) = w_1 \, B_{i,k-1}(x) + (1 - w_2) \, B_{i+1,k-1}(x)
$$

其中

$$
w_1 = \frac{x - t_i}{t_{i+k} - t_i}, \qquad w_2 = \frac{x - t_{i+1}}{t_{i+k+1} - t_{i+1}}
$$

约定 $0/0 = 0$。

**自然三次样条**基：将三次 B 样条基投影到边界二阶导数约束（$f'' = 0$，在两个边界节点处）的零空间上。与对应的普通 B 样条基相比，基的维度减少 2。

**周期三次样条**基：把三次 B 样条投影到三个周期边界约束的零空间。设 $a=\min(\text{knots})$、$b=\max(\text{knots})$，则要求

$$
f(a)=f(b),\qquad f'(a)=f'(b),\qquad f''(a)=f''(b).
$$

因此相对普通 B 样条减少 3 个自由维度，并保证跨周期边界的平滑性。

**薄板样条**基：输入维数为 $d$、惩罚阶数为 $m$ 时，径向基函数为

$$
\phi(r)=
\begin{cases}
r^{2m-d}\log(r), & d \text{ 为偶数},\\
r^{2m-d}, & d \text{ 为奇数},
\end{cases}
$$

其中 $r=\lVert x-\xi_j\rVert_2$ 是样本到节点 $\xi_j$ 的欧氏距离。一维且 $m=2$ 时，$\phi(r)=r^3$；二维且 $m=2$ 时，$\phi(r)=r^2\log(r)$。基矩阵还包含 $[1,x_1,\ldots,x_d]$ 的低阶多项式列，以保证表示完整。

**SplineTransformer**：为每个输入特征生成 B 样条特征，并使用 `'uniform'`、`'quantile'` 或自定义数组放置节点。每个特征的输出维度在 `include_bias=True` 时为 `n_knots + degree - 1`，否则为 `n_knots + degree - 2`。外推模式包括：

- `error`：超出边界时报错；
- `constant`：钳制到边界；
- `linear`：沿边界切线延拓；
- `continue`：继续边界处的多项式片段。

## 估计方程（Estimating Equation）

评估采用直接递推，无需求解回归系统。`cyclic_cubic_spline_basis` 通过 SVD 计算周期约束矩阵的零空间；`thin_plate_spline_basis` 通过向量化广播计算成对距离。`SplineTransformer` 为每个特征使用后端原生 Cox–de Boor 递推，并显式处理四种外推语义，不再把完整数组交给 SciPy。

## 协方差 / 推断（Covariance / Inference）

样条基函数是确定性计算工具，不产生推断输出（无标准误、p 值或置信区间）。如需使用样条进行统计推断，请参见 [GAM](semiparametric.md) 模型，该模型将惩罚样条与 GCV 平滑参数选择相结合。

## 后端执行与验证边界

SplineTransformer 的节点学习和四种外推均使用 NumPy/CuPy/Torch 共享递推。
在已拟合对象切换输入后端时，仅转移节点元数据，不转移完整训练设计。
已验证 NumPy/Torch-CPU 外推一致性；真实 CUDA 显存与性能验证仍待完成。

`thin_plate_spline_basis` 同样使用 device-aware 分配和标量安全的径向运算，并在
构造基函数前验证 x、knots 与 penalty order。自然样条的 QR fallback 会在约束矩阵
所在设备创建单位矩阵。

## strict / approx 区别

样条基计算没有 strict/approx 模式。NumPy、CuPy 与 Torch 使用同一递推；已验证 NumPy/Torch-CPU 紧容差一致性，但真实 CUDA parity 与性能仍待验证。

## 参数（Parameters）

**bspline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` |
| `knots` | 必需 | 内部节点位置（严格递增） |
| `degree` | `3` | 样条次数 |
| `xp` | `None` | 数组模块（`numpy`、`cupy` 或 `torch`）；若为 `None` 则从 `x` 推断 |

**natural_cubic_spline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` |
| `knots` | 必需 | 内部节点位置（严格递增） |
| `xp` | `None` | 数组模块；若为 `None` 则从 `x` 推断 |

**cyclic_cubic_spline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` |
| `knots` | 必需 | 内部节点位置（严格递增） |
| `xp` | `None` | 数组模块；若为 `None` 则从 `x` 推断 |

**thin_plate_spline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` 或 `(n, d)` |
| `knots` | 必需 | 节点位置，形状 `(m,)` 或 `(m, d)`，维数必须与 `x` 一致 |
| `penalty_order` | `2` | 惩罚阶数 $m$，控制平滑程度 |
| `xp` | `None` | 数组模块；若为 `None` 则从 `x` 推断 |

**SplineTransformer**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `n_knots` | `5` | 节点数，包含边界节点 |
| `degree` | `3` | 样条次数，3 表示三次样条 |
| `knots` | `'uniform'` | `'uniform'`、`'quantile'`，或形状为 `(n_knots, n_features)` 的数组 |
| `include_bias` | `True` | 是否保留分割单位性质带来的全部基函数 |
| `extrapolation` | `'constant'` | `'error'`、`'constant'`、`'linear'` 或 `'continue'` |
| `device` | `'auto'` | 计算设备 |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.nonparametric.splines import (
    bspline_basis,
    natural_cubic_spline_basis,
    cyclic_cubic_spline_basis,
    thin_plate_spline_basis,
    SplineTransformer,
)
import numpy as np

x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)

# CPU：B 样条基
B = bspline_basis(x, knots, degree=3, xp=np)
print(f"基矩阵形状: {B.shape}")  # (500, 14)

# CPU：自然三次样条基
B_nat = natural_cubic_spline_basis(x, knots, xp=np)
print(f"自然样条基形状: {B_nat.shape}")  # (500, 12)

# CPU：周期三次样条基
B_cyc = cyclic_cubic_spline_basis(x, knots, xp=np)
print(f"周期样条基形状: {B_cyc.shape}")  # (500, 11)

# CPU：一维薄板样条基
B_tp = thin_plate_spline_basis(x, knots, penalty_order=2, xp=np)
print(f"薄板样条基形状: {B_tp.shape}")  # (500, 12)

# CPU：二维薄板样条基
xy = np.column_stack([np.linspace(0, 1, 200), np.linspace(0, 1, 200)])
knots_2d = np.column_stack([
    np.linspace(0.1, 0.9, 5),
    np.linspace(0.1, 0.9, 5),
])
B_tp2 = thin_plate_spline_basis(xy, knots_2d, penalty_order=2, xp=np)
print(f"二维薄板样条基形状: {B_tp2.shape}")  # (200, 8)

# CPU：sklearn 兼容的转换器
X = np.random.randn(500, 3)
transformer = SplineTransformer(n_knots=10, degree=3, knots="quantile")
X_spline = transformer.fit_transform(X)
print(f"转换后形状: {X_spline.shape}")  # (500, 30)
```

**CuPy（GPU）**：

```python
import cupy as cp

x_gpu = cp.asarray(x)
knots_gpu = cp.asarray(knots)

B_gpu = bspline_basis(x_gpu, knots_gpu, degree=3, xp=cp)
print(f"GPU 基矩阵形状: {B_gpu.shape}")  # (500, 14)

B_nat_gpu = natural_cubic_spline_basis(x_gpu, knots_gpu, xp=cp)
print(f"GPU 自然样条基形状: {B_nat_gpu.shape}")  # (500, 12)

B_cyc_gpu = cyclic_cubic_spline_basis(x_gpu, knots_gpu, xp=cp)
print(f"GPU 周期样条基形状: {B_cyc_gpu.shape}")  # (500, 11)

B_tp_gpu = thin_plate_spline_basis(
    x_gpu, knots_gpu, penalty_order=2, xp=cp
)
print(f"GPU 薄板样条基形状: {B_tp_gpu.shape}")  # (500, 12)
```

**PyTorch（GPU）**：

```python
import torch

x_t = torch.tensor(x, device='cuda')
knots_t = torch.tensor(knots, device='cuda')

B_t = bspline_basis(x_t, knots_t, degree=3, xp=torch)
print(f"Torch 基矩阵形状: {B_t.shape}")  # (500, 14)

B_cyc_t = cyclic_cubic_spline_basis(x_t, knots_t, xp=torch)
print(f"Torch 周期样条基形状: {B_cyc_t.shape}")  # (500, 11)

B_tp_t = thin_plate_spline_basis(
    x_t, knots_t, penalty_order=2, xp=torch
)
print(f"Torch 薄板样条基形状: {B_tp_t.shape}")  # (500, 12)
```

## 输出（Outputs）

**bspline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + degree + 1)`。

**natural_cubic_spline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + 1)`。

**cyclic_cubic_spline_basis**：返回形状为 `(n, n_knots + degree + 1 - 3)` 的三次样条基；减少的 3 维对应三个周期边界约束。

**thin_plate_spline_basis**：返回形状为 `(n, m + d + 1)` 的矩阵，其中 $m$ 为节点数、$d$ 为输入维数。前 $m$ 列是径向基，其余 $d+1$ 列为截距和一次多项式项。

**SplineTransformer 拟合属性**：

| 属性 | 形状 | 说明 |
|---|---|---|
| `knots_` | 数组列表 | 每个特征的节点位置 |
| `boundary_lo_` | `(n_features,)` | 每个特征的下边界 |
| `boundary_hi_` | `(n_features,)` | 每个特征的上边界 |
| `n_features_in_` | int | 输入特征数 |
| `n_features_out_` | int | 输出特征数 |

**SplineTransformer 方法**：

| 方法 | 说明 |
|---|---|
| `fit(X, y=None)` | 从训练数据学习节点并返回 `self` |
| `transform(X)` | 把数据变换为 B 样条特征 |
| `fit_transform(X, y=None)` | 在一步内拟合并变换 |
| `get_feature_names_out(input_features=None)` | 返回输出特征名称 |

## 常见问题（FAQ）

**自然样条与普通 B 样条有何区别？** 自然样条在边界处强制线性，减少数据范围边缘的过拟合。当边界行为很重要时，使用自然样条。

**什么时候使用周期三次样条？** 当特征具有明确周期结构（例如一年中的日期或角度）时使用。该基保证拟合函数及其前两阶导数在周期边界处衔接。

**什么时候使用薄板样条？** 薄板样条天然支持多维平滑；与本质上一维的 B 样条相比，它通过径向基函数直接处理 $d$ 维输入。

**SplineTransformer 与直接调用 bspline_basis 有何区别？** `SplineTransformer` 提供多特征、自动节点放置、`fit`/`transform` 语义和流水线兼容；只需计算单个已知节点序列时，可直接调用底层基函数。

**样条的 GPU 加速效果如何？** 递推已向量化并保留在设备端，但加速取决于样本量、次数、节点数和后端；完成当前 CUDA benchmark 前不作统一倍数承诺。

## 外部验证（External Validation）

- B 样条基值与 `scipy.interpolate.BSpline` 验证；相对误差 < 1e-15。
- 自然三次样条精度：$n \le 500$ 时优秀（< 1e-10）；$n = 5000$ 时一般（约 1.5e-6），原因是边界约束投影中的 SVD 条件数。
- `SplineTransformer` 的 uniform 与 quantile 节点策略已与 `sklearn.preprocessing.SplineTransformer` 对照。
- constant、linear 与 continue 外推已验证 NumPy/Torch-CPU 一致性；可选 CuPy 测试需要真实 CUDA 环境。
- `cyclic_cubic_spline_basis` 已验证 $f(a)\approx f(b)$、$f'(a)\approx f'(b)$、$f''(a)\approx f''(b)$ 在 SVD 容差内成立。
- `thin_plate_spline_basis` 已与二维 $\phi(r)=r^2\log(r)$ 的手工计算结果对照。

## 参考文献（References）

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces. In *Constructive Theory of Functions of Several Variables*. Springer.
