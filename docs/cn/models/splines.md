# 样条基函数

> 语言: 中文
> 最后更新: 2026-10-05
> 页面定位: 模型文档
> 切换: [English](../../en/models/splines.md)

语言切换：[English](../../en/models/splines.md)

## 概览（Overview）

样条模块提供 `bspline_basis`、`natural_cubic_spline_basis`、`cyclic_cubic_spline_basis`、`thin_plate_spline_basis` 以及 sklearn 风格的 `SplineTransformer`。函数通过显式 `xp` 选择 NumPy、CuPy 或 Torch；`xp=None` 使用 NumPy，不根据输入自动推断。`natural_cubic_spline_basis` 用数值约束投影三次基，近似限制端点曲率为零。SplineTransformer 使用后端原生 Cox–de Boor 递推。

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
B_{i,k}(x) = \frac{x-t_i}{t_{i+k}-t_i}B_{i,k-1}(x)
+\frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(x).
$$

当某项分母为零时，将整个加项定义为零，重复边界节点也遵循此约定。最右端边界使用基函数的极限值，不能直接套用零次基的半开区间约定。

**自然三次样条**基：将三次 B 样条基投影到数值边界二阶导数约束的零空间上，近似满足两个端点处 $f'' = 0$。两个约束独立时，基维度减少 2；一般情况下，减少量等于约束的数值秩。

**周期三次样条**的目标是在评估范围两端 $a=\min(x)$、$b=\max(x)$ 满足

$$
f(a)=f(b),\qquad f'(a)=f'(b),\qquad f''(a)=f''(b).
$$

内部节点须严格位于两端之间。理想情况下，三个独立约束会使基维度减少 3。但当前函数用范围外的零基函数值近似边界导数，因此实际返回的基可能不满足真实的单侧导数约束，列数也可能不同。**需要周期连续性时，请勿依赖此函数。** 若适合分析问题，可手动构造正弦/余弦基作为周期模型的替代。
**薄板样条**使用径向核；二维且惩罚阶数为 2 时为
$\phi(r)=r^2\log r$。径向部分的一般形式为：偶数维使用 $r^{2m-d}\log r$，奇数维使用 $r^{2m-d}$，其中 $m$ 为惩罚阶数，要求 $2m>d$。函数始终附加截距和线性项 $[1,x_1,\ldots,x_d]$，这对应二阶惩罚的多项式部分；其他阶数并未获得完整的阶数专属多项式项。函数只构造特征，不拟合平滑器，也不构造系数约束或惩罚矩阵。

`SplineTransformer` 为每个特征学习 uniform、quantile 或自定义节点，并支持：

- `error`：超出边界时报错；
- `constant`：钳制到边界；
- `linear`：沿边界切线延拓；
- `continue`：继续边界处的多项式片段。

## 估计方程（Estimating Equation）

评估采用直接递推，无需求解回归系统。SplineTransformer 不再将完整数组交给 SciPy，而是在所选后端构造完整基矩阵。

## 协方差 / 推断（Covariance / Inference）

样条基函数是确定性计算工具，不产生推断输出（无标准误、p 值或置信区间）。拟合可加平滑曲线可参见 [GAM](semiparametric.md)，它通过 GCV 选择平滑强度，但同样不提供系数推断或置信带。

## 后端执行与验证边界

SplineTransformer 的节点学习和四种外推均使用 NumPy/CuPy/Torch 共享递推。
在已拟合对象切换输入后端时，仅转移节点元数据，不转移完整训练设计。
选择后端本身不能证明分析所需的精度或速度；请针对实际数据验证基函数和边界行为。

`thin_plate_spline_basis` 使用与设备匹配的数组分配和可处理标量的径向运算，并在构造前验证输入、节点和惩罚阶数。自然样条的 QR 回退会在约束矩阵所在设备创建单位矩阵。

## strict / approx 区别

样条基计算没有严格/近似模式。显式选择后端不会消除自然样条和周期样条边界投影的数值限制。

## 参数（Parameters）

**bspline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` |
| `knots` | 必需 | 内部节点位置（严格递增） |
| `degree` | `3` | 非负整数样条次数 |
| `boundary_lo`、`boundary_hi` | `None` | 可选固定边界；默认覆盖评估点及节点，内部节点须严格位于边界内。新点应复用训练边界，保持同一基定义。 |
| `xp` | `None` | 数组模块（`numpy`、`cupy` 或 `torch`）；若为 `None` 则使用 NumPy |

**natural_cubic_spline_basis**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `x` | 必需 | 评估点，形状 `(n,)` |
| `knots` | 必需 | 内部节点位置（严格递增） |
| `xp` | `None` | 数组模块；若为 `None` 则使用 NumPy |

**cyclic_cubic_spline_basis**：`cyclic_cubic_spline_basis(x, knots, xp=None)`，评估点和内部节点都是一维数组；`xp=None` 使用 NumPy。当前周期约束存在上文所述限制。

**thin_plate_spline_basis**：`thin_plate_spline_basis(x, knots, penalty_order=2, xp=None)`；输入为 `(n,)` 或 `(n,d)`，节点为 `(k,)` 或 `(k,d)`，特征数必须一致。惩罚阶数须为正整数且满足 `2 * penalty_order > d`。

**SplineTransformer**：`n_knots=5`、`degree=3`、`knots='uniform'`、`include_bias=True`、`extrapolation='constant'`、`device='auto'`、`n_jobs=None`。`n_knots` 是包含边界的节点数，须为不小于 3 的整数；分位数节点须各不相同。`degree` 为非负整数，`knots` 也可传入 `(n_knots,n_features)` 数组。`n_jobs` 不用于并行构造。每个特征输出 `n_knots + degree - 1` 列；`include_bias=False` 时少一列。

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

**natural_cubic_spline_basis**：两个边界约束独立时通常返回 `(n, n_knots + 2)`；一般列数为 `n_knots + 4 - 数值秩`。SVD 的基方向任意，首列不是专门的截距列。每次调用都会按评估网格重新计算边界和投影，不能在一组网格拟合系数后，仅凭相同节点就假定另一组网格的基完全相同。该函数不保存训练状态，也不提供线性外推 API；需要复用已拟合变换时，可使用 `SplineTransformer`。

**cyclic_cubic_spline_basis**：返回 `(n, n_knots + 4 - 数值秩)`；本页一维网格上返回 12 列，而非理想周期基的 11 列。返回矩阵不代表周期连续性成立。

**thin_plate_spline_basis**：返回 `(n, n_knots + d + 1)`，依次为径向基、截距和线性列。

**SplineTransformer**：`fit()` 后提供 `knots_`、`boundary_lo_`、`boundary_hi_`、`n_features_in_` 和 `n_features_out_`；`transform()` 返回与输入/所选后端一致的数组。

`fit(X, y=None, sample_weight=None)` 返回 `self`；`fit_transform(X, y=None, sample_weight=None)` 拟合并返回基矩阵。两者都不使用 `y` 或 `sample_weight`，权重不会改变分位数节点。`transform(X)` 与别名 `predict(X)` 返回 `(n_query,n_features_out_)` 基特征，不预测响应。`get_feature_names_out(input_features=None)` 返回字符串列表；可选输入名称数须等于 `n_features_in_`。`get_params(deep=True)` 和 `set_params(**params)` 用于参数管理；修改参数后须重拟合。

## 常见问题（FAQ）

**自然样条与普通 B 样条有何区别？** 自然基近似约束端点曲率为零，可以限制边界波动，但不保证降低过拟合，也不提供可复用的线性外推规则。

**样条的 GPU 加速效果如何？** 递推已向量化并保留在设备端，但加速取决于样本量、次数、节点数和后端；应针对实际工作负载测量，不应假定统一的加速倍数。

## 验证方法

可用相同的扩展节点和次数，对照 `scipy.interpolate.BSpline` 的基值和边界导数。与 scikit-learn 变换器比较时，应对齐节点、`include_bias`、次数和外推方式。自然基的 SVD 列符号或方向可能不同，应比较表示的函数或子空间。周期性需要检查真实单侧边界导数，不能复用实现本身的范围外有限差分作为验证。CPU 检查不等于 GPU 精度或性能证据。

## 参考文献（References）

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
