# 样条基函数

> 语言: 中文
> 最后更新: 2026-05-28
> 页面定位: 模型文档
> 切换: [English](../../en/models/nonparametric/splines.md)

语言切换：[English](../../en/models/nonparametric/splines.md)

## 概览（Overview）

样条模块提供样条基函数构造工具。`bspline_basis` 使用 De Boor 递归算法评估 B 样条基矩阵。`natural_cubic_spline_basis` 构造带边界约束（边界节点处二阶导数为零）的自然三次样条基。两者均支持 CPU、CuPy 和 Torch 后端。

使用这些基函数的广义可加模型（GAM）请参见 [GAM](../semiparametric.md)。

## 路径（Path）

`statgpu.nonparametric.splines.bspline_basis`、`statgpu.nonparametric.splines.natural_cubic_spline_basis`

## 目标函数（Objective Function）

**B 样条基**通过 De Boor 递归计算。零次基函数为：

\[
B_{i,0}(x) = \begin{cases} 1 & \text{若 } t_i \le x < t_{i+1} \\ 0 & \text{其他} \end{cases}
\]

对于次数 $k \ge 1$：

\[
B_{i,k}(x) = w_1 \, B_{i,k-1}(x) + (1 - w_2) \, B_{i+1,k-1}(x)
\]

其中

\[
w_1 = \frac{x - t_i}{t_{i+k} - t_i}, \qquad w_2 = \frac{x - t_{i+1}}{t_{i+k+1} - t_{i+1}}
\]

约定 $0/0 = 0$。

**自然三次样条**基：将三次 B 样条基投影到边界二阶导数约束（$f'' = 0$，在两个边界节点处）的零空间上。与对应的普通 B 样条基相比，基的维度减少 2。

## 估计方程（Estimating Equation）

评估是直接的递归计算，无需求解线性系统。

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

## 输出（Outputs）

**bspline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + degree + 1)`。

**natural_cubic_spline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + 1)`。

## 常见问题（FAQ）

**自然样条与普通 B 样条有何区别？** 自然样条在边界处强制线性，减少数据范围边缘的过拟合。当边界行为很重要时，使用自然样条。

**样条的 GPU 加速效果如何？** B 样条基构造在所有样本点上向量化。对于大 $n$（5000+），GPU 上可期望 2-3 倍加速。

## 外部验证（External Validation）

- B 样条基值与 `scipy.interpolate.BSpline` 验证；相对误差 < 1e-15。
- 自然三次样条精度：$n \le 500$ 时优秀（< 1e-10）；$n = 5000$ 时一般（约 1.5e-6），原因是边界约束投影中的 SVD 条件数。

## 参考文献（References）

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
