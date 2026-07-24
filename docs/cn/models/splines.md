# 样条基函数

> 语言: 中文
> 最后更新: 2026-07-14
> 页面定位: 模型文档
> 切换: [English](../../en/models/splines.md)

语言切换：[English](../../en/models/splines.md)

## 概览（Overview）

样条模块提供 `bspline_basis`、`natural_cubic_spline_basis`、`cyclic_cubic_spline_basis`、`thin_plate_spline_basis` 以及 sklearn 风格的 `SplineTransformer`。这些接口支持 NumPy、CuPy 和 Torch；SplineTransformer 使用后端原生 Cox–de Boor 递推。

使用这些基函数的广义可加模型（GAM）请参见 [GAM](semiparametric.md)。

## 路径（Path）

- `statgpu.nonparametric.splines.bspline_basis`
- `statgpu.nonparametric.splines.natural_cubic_spline_basis`
- `statgpu.nonparametric.splines.cyclic_cubic_spline_basis`
- `statgpu.nonparametric.splines.thin_plate_spline_basis`
- `statgpu.nonparametric.splines.SplineTransformer`

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

**周期三次样条**在两端约束函数值、一阶导数与二阶导数连续。
**薄板样条**使用径向核；二维且惩罚阶数为 2 时为
\(\phi(r)=r^2\log r\)。

`SplineTransformer` 为每个特征学习 uniform、quantile 或自定义节点，并支持：

- `error`：超出边界时报错；
- `constant`：钳制到边界；
- `linear`：沿边界切线延拓；
- `continue`：继续边界处的多项式片段。

## 估计方程（Estimating Equation）

评估采用直接递推，无需求解回归系统。SplineTransformer 不再将完整数组交给 SciPy，而是在所选后端构造完整基矩阵。

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

**SplineTransformer**：`n_knots=5`、`degree=3`、`knots='uniform'`、
`include_bias=True`、`extrapolation='constant'`，并支持 `device='auto'`。

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.nonparametric.splines import bspline_basis, natural_cubic_spline_basis
import numpy as np

x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)

# CPU：B 样条基
B = bspline_basis(x, knots, degree=3, xp=np)
print(f"基矩阵形状: {B.shape}")  # (500, 14)

# CPU：自然三次样条基
B_nat = natural_cubic_spline_basis(x, knots, xp=np)
print(f"自然样条基形状: {B_nat.shape}")  # (500, 12)
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
```

**PyTorch（GPU）**：

```python
import torch

x_t = torch.tensor(x, device='cuda')
knots_t = torch.tensor(knots, device='cuda')

B_t = bspline_basis(x_t, knots_t, degree=3, xp=torch)
print(f"Torch 基矩阵形状: {B_t.shape}")  # (500, 14)
```

## 输出（Outputs）

**bspline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + degree + 1)`。

**natural_cubic_spline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + 1)`。

**cyclic_cubic_spline_basis**：返回满足周期边界约束的三次样条基。

**thin_plate_spline_basis**：返回径向基与低阶多项式列组成的矩阵。

**SplineTransformer**：`fit()` 后提供 `knots_`、`boundary_lo_`、`boundary_hi_`、`n_features_in_` 和 `n_features_out_`；`transform()` 返回与输入/所选后端一致的数组。

## 常见问题（FAQ）

**自然样条与普通 B 样条有何区别？** 自然样条在边界处强制线性，减少数据范围边缘的过拟合。当边界行为很重要时，使用自然样条。

**样条的 GPU 加速效果如何？** 递推已向量化并保留在设备端，但加速取决于样本量、次数、节点数和后端；完成当前 CUDA benchmark 前不作统一倍数承诺。

## 外部验证（External Validation）

- B 样条基值与 `scipy.interpolate.BSpline` 验证；相对误差 < 1e-15。
- 自然三次样条精度：$n \le 500$ 时优秀（< 1e-10）；$n = 5000$ 时一般（约 1.5e-6），原因是边界约束投影中的 SVD 条件数。

## 参考文献（References）

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
