# 样条基函数

> 语言: 中文
> 最后更新: 2026-10-09
> 页面定位: 模型文档
> 切换: [English](../../en/models/splines.md)

## 把弯曲的特征效应变成可复用输入

样条基把一个数值特征展开成若干局部重叠的曲线，后续回归再组合这些列来学习
非线性关系；构造基函数本身并没有拟合响应。训练和新观测需要使用相同节点及特征
定义时，应选择 `SplineTransformer`。若还需要可加最小二乘模型和自动平滑参数
选择，可使用 [GAM](semiparametric.md)。增加节点能表达更多细节，但本身不会控制过拟合。

只在训练行上拟合变换器，再对验证或测试行调用 `transform`。交叉验证时，应在
每个训练折内学习节点，避免分位数节点提前使用留出观测的信息。

## 拟合一次，再复用同一组特征

下面的 CPU 示例先学习样条特征，再用普通最小二乘拟合响应，并把训练节点复用于
新观测。`include_bias=False` 每个特征块去掉一列，使后续模型可以显式添加截距。
示例曲线没有噪声，只演示两步流程，不能把结果当作泛化性能验证。

<!-- example: spline-transformer-reuse-cpu -->
```python
import numpy as np
from statgpu.nonparametric.splines import SplineTransformer
```

### 准备一列特征与响应

`X_train` 是 `(41, 1)` 数组，每行一个时间点；`y_train` 为 `(41,)`，记录该点的无噪声响应。`[:, None]` 保留“观测 × 特征”的二维结构。按顺序运行本节各段。

```python
X_train = np.linspace(-2.0, 2.0, 41)[:, None]
y_train = np.sin(1.5 * X_train[:, 0])
```

### 学习样条特征

`fit_transform` 只从训练点学习节点，并将这一列特征展开为样条基列；此步骤不使用 `y_train`。

```python
transformer = SplineTransformer(
    n_knots=6, degree=3, include_bias=False,
    extrapolation="constant", device="cpu",
)
B_train = transformer.fit_transform(X_train)
```

### 用基特征拟合响应

在刚生成的 `B_train` 前加一列截距，再用最小二乘估计系数。响应拟合与特征构造是两个不同步骤。

```python
train_design = np.column_stack([np.ones(len(X_train)), B_train])
coefficients = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
```

### 复用训练节点进行预测

`X_query` 是三个新点，形状为 `(3, 1)`。调用已拟合对象的 `transform`，不要在这些点上重新 `fit`。

```python
X_query = np.array([[-1.5], [0.0], [1.5]])
B_query = transformer.transform(X_query)
query_design = np.column_stack([np.ones(len(X_query)), B_query])
prediction = query_design @ coefficients
print(B_train.shape, B_query.shape)
print(prediction.round(3))
```

输出形状为 `(41, 7)` 和 `(3, 7)`，随后约为 `[-0.778, 0.000, 0.778]`。
七列来自 `6 + 3 - 2`，表示基函数特征，并非七个独立观测的原始变量。
`transformer.predict` 返回的仍是这些特征；只有将后续回归的设计矩阵乘以拟合系数，
才得到响应预测。

### 查看特征名称与边界行为

继续使用本节的 `transformer`、`X_query` 和 `B_query`。名称用于识别展开的基列；`predict` 在这个变换器中只是 `transform` 的别名。

```python
names = transformer.get_feature_names_out(["time"])
assert len(names) == B_query.shape[1]
assert np.allclose(transformer.predict(X_query), B_query)
assert np.allclose(transformer.transform([[2.5]]), transformer.transform([[2.0]]))
```
<!-- example-end: spline-transformer-reuse-cpu -->

最后一个断言说明常数外推的含义：2.5 处采用训练上边界 2.0 处的特征值。
这只是边界规则，不能证明真实响应在数据范围外保持不变。`linear` 和 `continue`
采用不同外推假设，结果可能迅速增大。对于含噪声数据，应在训练折内选择节点数和
后续模型的正则化参数，再用独立测试集评估预测。

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

评估采用直接递推，无需求解回归系统。SplineTransformer 使用后端原生递推构造完整基矩阵；实际设备放置受下文所述限制。

## 协方差 / 推断（Covariance / Inference）

样条基函数是确定性计算工具，不产生推断输出（无标准误、p 值或置信区间）。拟合可加平滑曲线可参见 [GAM](semiparametric.md)，它通过 GCV 选择平滑强度，但同样不提供系数推断或置信带。

## 后端执行与验证边界

SplineTransformer 的节点学习和四种外推均使用 NumPy/CuPy/Torch 递推。
输入切换后端时，会按需转移已拟合的节点元数据，不转移完整训练设计。

当前设备选择存在例外：传入的 Torch 张量优先于 `device`。Torch CPU 输入即使
搭配显式 `device="torch"` 或 `device="cuda"`，拟合节点与变换特征仍可能留在 CPU 上；
改用 NumPy 输入时，相同的不可用加速器请求则可能报错。应检查 `knots_` 中的各数组
与返回的基矩阵，不能只看 `model.device`：Torch 查看 `.device`、`.is_cuda`，
CuPy 查看 `.device`，NumPy 数组位于 CPU。需要明确的 CPU 路径时，请传入 NumPy
数组并设置 `device="cpu"`。这一限制没有改变预期的严格[设备约定](../guides/device-and-memory.md)。
选择后端本身不能证明分析所需的精度或速度；请针对实际数据验证基函数和边界行为。

`thin_plate_spline_basis` 在构造前验证评估点、节点和惩罚阶数，并返回所传 `xp` 后端上的数组。

### 自然样条基对计量尺度的敏感性

`natural_cubic_spline_basis` 使用固定绝对步长近似边界导数。改变计量单位可能明显
改变所表示的函数空间：范围长度为 `1e6` 时，返回的基甚至可能无法表示常数；范围
很小时，端点曲率也可能明显不为零。结果有限、列数正确，都不能证明自然边界条件
成立。将评估点及节点一起中心化并缩放到单位区间，可以减小这些已知误差，但不能
让约束变得精确。若分析必须满足自然端点条件，应使用经过独立验证的自然样条构造，
并检查解析边界导数。

### 变换器的输入与输出形状

`SplineTransformer` 接受非空、有限实数的 `(n,p)` 设计矩阵，基函数计算使用
float64。拟合时，一维向量表示单特征；变换时，一维向量可表示单特征的多个点，
或多特征的一个点。但后一种形式当前会在 Torch 上抛出 `TypeError`，因此请始终
显式传入 `(q,p)` 矩阵，单个点也保留 `(1,p)` 形状。输出为 `(q,n_features_out_)`，
须保持特征顺序并沿用拟合节点。单特征的自定义节点也可以是长度为 `n_knots` 的向量。

<a id="strict--approx-区别"></a>

## 数值精度与边界条件

样条基计算没有严格/近似模式。显式选择后端不会消除自然样条和周期样条边界投影的数值限制。

## 完整的模型专属调用

以下名称从 `statgpu.nonparametric.splines` 导入。调用签名与本页的参数和输出说明配合使用。
[共享估计器辅助方法](../reference/estimator-api.md) 不会使这个特征变换器自动具备响应拟合或系数推断能力。

```text
bspline_basis(x, knots, degree=3, xp=None, boundary_lo=None, boundary_hi=None)
natural_cubic_spline_basis(x, knots, xp=None)
cyclic_cubic_spline_basis(x, knots, xp=None)
thin_plate_spline_basis(x, knots, penalty_order=2, xp=None)
SplineTransformer(n_knots=5, degree=3, knots='uniform', include_bias=True, extrapolation='constant', device='auto', n_jobs=None)
SplineTransformer.fit(X, y=None, sample_weight=None)
SplineTransformer.transform(X)
SplineTransformer.fit_transform(X, y=None, sample_weight=None)
SplineTransformer.predict(X)
SplineTransformer.get_feature_names_out(input_features=None)
SplineTransformer.get_params(deep=True)
SplineTransformer.set_params(**params)
```

直接调用 B 样条函数时，至少需要一个内部节点。对新查询批次，仅复用内部节点还
不够，还须传入训练时的 `boundary_lo`、`boundary_hi`，或改用已拟合的变换器。
原始基函数在显式边界外为零；变换器的外推策略是另一套 API。自定义变换器节点
包含两端边界，不必覆盖全部训练行。因此在 `extrapolation="error"` 下，`fit`
可能成功，而 `fit_transform` 会在随后变换越界训练行时抛出错误。

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

**SplineTransformer**：`n_knots=5`、`degree=3`、`knots='uniform'`、`include_bias=True`、`extrapolation='constant'`、`device='auto'`、`n_jobs=None`。`n_knots` 是包含边界的节点数，须为不小于 3 的整数；分位数节点须各不相同。`degree` 为非负整数，`knots` 也可传入 `(n_knots,n_features)` 数组。`n_jobs` 不用于并行构造。`device` 表示请求的设备，但 Torch 张量输入可能覆盖该请求，见上文设备例外。每个特征输出 `n_knots + degree - 1` 列；`include_bias=False` 时少一列。

## 进阶：直接构造基矩阵

只有需要直接控制节点或比较基函数家族时，才使用底层函数。下面是独立于响应拟合示例的计算网格：500 个一维评估点和 10 个内部节点。这里没有响应变量，也不会拟合模型。按顺序运行本节各段。

<!-- example: spline-raw-bases-cpu -->
```python
import numpy as np
from statgpu.nonparametric.splines import bspline_basis
```

<a id="raw-spline-setup"></a>

### 固定网格、节点与边界

```python
x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)
boundary_lo, boundary_hi = 0.0, 1.0
```

三次 B 样条返回 14 列，来自 `10 + 3 + 1`。若以后在新点计算同一组基，必须沿用这里的节点与两个边界。

```python
B = bspline_basis(
    x, knots, degree=3, xp=np,
    boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B.shape)  # (500, 14)
```

### 比较边界投影

复用本节的 `x` 和 `knots`。自然基近似约束两端曲率为零；周期基的真实单侧导数约束并不可靠。下面只展示返回矩阵，不能用列数或有限值来证明边界条件成立。

```python
from statgpu.nonparametric.splines import (
    natural_cubic_spline_basis, cyclic_cubic_spline_basis,
)

B_nat = natural_cubic_spline_basis(x, knots, xp=np)
B_cyc = cyclic_cubic_spline_basis(x, knots, xp=np)
print(B_nat.shape, B_cyc.shape)  # (500, 12), (500, 12) on this grid
```

### 构造径向薄板特征

继续复用本节的一维 `x` 和 `knots`。二阶薄板基由 10 个径向列和截距、线性列组成，共 12 列；它本身不提供平滑惩罚或响应拟合。

```python
from statgpu.nonparametric.splines import thin_plate_spline_basis

B_tp = thin_plate_spline_basis(x, knots, penalty_order=2, xp=np)
print(B_tp.shape)  # (500, 12)
```

二维输入中，每行是一个平面点，节点也必须有两列。这里构造一个小网格和五个节点；输出八列，包括五个径向列与 `[1, x1, x2]`。

```python
u, v = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 10))
xy = np.column_stack([u.ravel(), v.ravel()])
knots_2d = np.array([[0.1, 0.1], [0.1, 0.9], [0.5, 0.5],
                     [0.9, 0.1], [0.9, 0.9]])
B_tp2 = thin_plate_spline_basis(xy, knots_2d, penalty_order=2, xp=np)
print(B_tp2.shape)  # (200, 8)
```
<!-- example-end: spline-raw-bases-cpu -->

### 可选：在 GPU 上评价同一组 B 样条

先完成本节[固定网格、节点与边界](#raw-spline-setup)，复用 `x`、`knots`、`boundary_lo`、`boundary_hi` 和 `bspline_basis`。根据安装情况选择一个代码段；显式传入数组模块 `xp`，并将两个输入都放到相应 GPU 上。结果仍为 `(500, 14)` 基矩阵，不是响应预测。

CuPy/CUDA：

<!-- example-requires: spline-raw-bases-cpu -->
<!-- example: spline-raw-cupy -->
```python
import cupy as cp

B_gpu = bspline_basis(
    cp.asarray(x), cp.asarray(knots), degree=3, xp=cp,
    boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B_gpu.shape)
```
<!-- example-end: spline-raw-cupy -->

Torch CUDA：

<!-- example-requires: spline-raw-bases-cpu -->
<!-- example: spline-raw-torch -->
```python
import torch

B_t = bspline_basis(
    torch.as_tensor(x, device="cuda"), torch.as_tensor(knots, device="cuda"),
    degree=3, xp=torch, boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B_t.shape)
```
<!-- example-end: spline-raw-torch -->

其他底层函数也按相同方式传入后端数组及匹配的 `xp`。后端不会消除上文说明的自然/周期边界限制，GPU 收益仍需针对实际规模测量。多特征的 `SplineTransformer` 沿用开头的拟合/变换流程，每个输入特征独立展开，列数见参数与输出参考。

## 输出（Outputs）

**bspline_basis**：返回基矩阵 $B$，形状为 `(n, n_knots + degree + 1)`。

**natural_cubic_spline_basis**：两个边界约束独立时通常返回 `(n, n_knots + 2)`；一般列数为 `n_knots + 4 - 数值秩`。SVD 的基方向任意，首列不是专门的截距列。每次调用都会按评估网格重新计算边界和投影，不能在一组网格拟合系数后，仅凭相同节点就假定另一组网格的基完全相同。该函数不保存训练状态，也不提供线性外推 API；需要复用已拟合变换时，可使用 `SplineTransformer`。

**cyclic_cubic_spline_basis**：返回 `(n, n_knots + 4 - 数值秩)`；本页一维网格上返回 12 列，而非理想周期基的 11 列。返回矩阵不代表周期连续性成立。

**thin_plate_spline_basis**：返回 `(n, n_knots + d + 1)`，依次为径向基、截距和线性列。

**SplineTransformer**：`fit()` 后提供 `knots_`、`boundary_lo_`、`boundary_hi_`、`n_features_in_` 和 `n_features_out_`；`transform()` 返回实际解析出的后端数组；受上文设备例外影响，须检查实际位置。

`fit(X, y=None, sample_weight=None)` 返回 `self`；`fit_transform(X, y=None, sample_weight=None)` 拟合并返回基矩阵。两者都不使用 `y` 或 `sample_weight`，权重不会改变分位数节点。`transform(X)` 与别名 `predict(X)` 返回 `(n_query,n_features_out_)` 基特征，不预测响应。`get_feature_names_out(input_features=None)` 返回字符串列表；可选输入名称数须等于 `n_features_in_`。`get_params(deep=True)` 和 `set_params(**params)` 用于参数管理；修改参数后须重拟合。

## 常见问题（FAQ）

**自然样条与普通 B 样条有何区别？** 自然基近似约束端点曲率为零，可以限制边界波动，但不保证降低过拟合，也不提供可复用的线性外推规则。

**样条的 GPU 加速效果如何？** 递推已向量化并保留在设备端，但加速取决于样本量、次数、节点数和后端；应针对实际工作负载测量，不应假定统一的加速倍数。

## 验证方法

可用相同的扩展节点和次数，对照 `scipy.interpolate.BSpline` 的基值和边界导数。与 scikit-learn 变换器比较时，应对齐节点、`include_bias`、次数和外推方式。自然基的 SVD 列符号或方向可能不同，应比较表示的函数或子空间。周期性需要检查真实单侧边界导数，不能复用实现本身的范围外有限差分作为验证。CPU 检查不等于 GPU 精度或性能证据。

## 参考文献（References）

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
