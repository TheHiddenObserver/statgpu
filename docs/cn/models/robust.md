# 稳健回归

> 语言：中文  
> 最后更新：2026-10-09<br>
> 页面定位：模型文档  
> 切换：[English](../../en/models/robust.md)

## 什么时候使用稳健回归？

当连续响应中少数很大的残差可能主导最小二乘拟合时，可以考虑稳健回归。Huber 对小残差采用平方损失，对大残差采用线性损失；Bisquare 与 Fair 使用不同的降权方式。得到的是所选损失定义的条件位置，不一定是条件均值。

`PenalizedRobustRegression` 将这些损失与正则化组合：损失限制异常响应残差的影响，惩罚控制系数收缩。这两种选择解决不同问题。稳健损失不能自动修复错误数据、异常特征值造成的高杠杆或遗漏变量；若关心某个条件分位数，可参见[分位数回归](quantile.md)。

<a id="cpu-example"></a>

## 完整 CPU 示例

依次运行以下步骤。先使用 Huber 与较小的 L2 惩罚，之后再介绍其他损失、非凸惩罚和 GPU。

### 1. 导入

<!-- learner-example: robust-basic -->
```python
import numpy as np
from statgpu.linear_model.penalized import PenalizedRobustRegression
```

### 2. 准备含异常响应的数据

`X` 为 `(320, 2)` 数值矩阵，每行一条观测；`y` 为长度 320 的连续响应。前 240 行用于训练，后 80 行留作评价。下面仅在训练数据中加入 12 个偏大的响应值，留出集不加入这些人为异常值。实际使用时先核查异常值来源、处理缺失值，并保持预测列顺序。

```python
rng = np.random.default_rng(27)
X = rng.normal(size=(320, 2))
y = 1.0 + X @ np.array([1.5, -0.7]) + rng.normal(scale=0.5, size=320)
y[:12] += 12.0
```

### 3. 拟合 Huber 模型

`alpha=0.01` 是演示用的 L2 强度，`solver="auto"` 对这个光滑惩罚目标选择 Newton。默认先估计残差尺度，再由 `epsilon × scale` 确定 Huber 阈值；阈值的含义见后文。

```python
model = PenalizedRobustRegression(
    loss="huber", penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240])
print("Slopes:", np.round(model.coef_, 3))
```

本例斜率约为 `[1.522, -0.723]`，接近模拟的 `[1.5, -0.7]`；这不是所有污染模式下的精度保证。斜率是拟合位置随某个特征增加一单位的变化，条件是其他特征固定。

### 4. 预测并评价留出数据

```python
prediction = model.predict(X[240:])
mae = np.mean(np.abs(y[240:] - prediction))
print("Predictions:", np.round(prediction[:3], 3))
print("Held-out MAE:", round(float(mae), 3))
```

预测数组为 `(80,)`，前三项约为 `[2.590, 1.341, 0.479]`。平均绝对误差（MAE）约为 `0.437`，单位与响应相同，同一评价集上越小越好。这里有意直接计算 MAE；该类的 `score` 返回响应尺度 R²，不是 Huber 损失。
<!-- example-end: robust-basic -->

## 参数选择与常见问题

- `epsilon` 或固定阈值控制对残差的降权，`alpha` 控制惩罚强度；不要混淆二者。按应用需要选择留出评价指标。
- 特征单位会影响惩罚，训练折内学习缩放与调参，再应用到留出数据。较小训练损失不等于更好的泛化。
- Bisquare 与 SCAD/MCP 涉及非凸目标，应关注初始化、数值警告及结果稳定性。
- 稳健拟合不会自动得到稳健标准误或选择调整后的 p 值。共享推断参数仍须满足[具体推断方法的支持条件](../guides/penalized-glm-inference.md)。

## 损失函数

### Huber 损失

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & |y - \eta| \le \delta \\
\delta|y - \eta| - \frac{1}{2}\delta^2 & \text{否则}
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- $\delta\to\infty$ 时趋近 OLS；阈值减小时对大残差的影响进一步受限
- 默认 `epsilon=1.35`；在自动尺度模式下有效阈值为 `epsilon × scale`

### Bisquare（Tukey biweight）损失

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- $|u|>c$ 时损失保持常数、梯度为 0
- 默认 `epsilon=4.685`，常用于获得高斯分布下约 95% 的效率

### Fair 损失

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log\left(1 + \frac{|y-\eta|}{c}\right)\right]
$$

- `smooth_gradient=True`、`has_hessian=True`
- 对残差的降权比硬截断型损失更平缓

## 损失参数与估计器控制

下表描述损失对象，不是估计器的完整构造参数。`PenalizedRobustRegression` 提供 `loss`、`penalty`、`alpha`、`epsilon`、`method`、求解器/设备控制及共享惩罚模型选项。完整参数和方法见[公开实现](../../../statgpu/linear_model/penalized/_penalized_robust.py)、[共享惩罚 API](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel)或已安装版本的 `help(PenalizedRobustRegression)`。

损失对象均从 `statgpu.losses` 导入：`HuberLoss`、`BisquareLoss` 与 `FairLoss`。

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦给定，会使用固定阈值模式 |
| `epsilon` | `1.35` | 与估计尺度相乘得到有效 Huber 阈值 |
| `method` | `"MAD"` | `"MAD"`、`"huber_prop2"` 或 `"joint"` |

`method="joint"` 联合优化系数与 `log_sigma`，与固定尺度的系数优化问题不同。

### `BisquareLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值 |
| `epsilon` | `4.685` | 与估计尺度相乘得到有效阈值 |
| `method` | `"MAD"` | `"MAD"` 或 `"huber_prop2"` |

### `FairLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦给定，会使用固定阈值模式 |
| `epsilon` | `1.35` | 与估计尺度相乘得到 Fair 损失的有效阈值 |
| `method` | `"MAD"` | `"MAD"` 或 `"huber_prop2"` |

## 尺度估计

自动尺度处理支持 MAD 与 Huber Proposal 2。`PenalizedRobustRegression` 在系数优化前估计尺度，因此普通 `MAD` / `huber_prop2` 拟合在求解阶段使用已经确定的有效阈值。

- **MAD**：$\hat\sigma=\operatorname{median}(|r_i|)/0.6745$
- **Huber Proposal 2**：通过固定点迭代估计尺度
- Huber 使用 $\delta=\epsilon\hat\sigma$
- Bisquare 与 Fair 都通过内部有效 `delta` 使用 $c=\epsilon\hat\sigma$

显式给定 `delta` 时直接使用固定阈值。`method="joint"` 仅属于 Huber，并定义单独的系数—尺度联合优化问题。

## 更换损失、惩罚或计算设备

### Huber 与 SCAD

先完成 [CPU 示例](#cpu-example)。以下 CPU 小节都复用其中的导入与 `X`、`y`，只使用前 240 行训练。SCAD 是非凸惩罚，自动调度使用 LLA 与 FISTA；下面的 alpha 仅为演示。

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-scad -->
```python
scad_model = PenalizedRobustRegression(
    loss="huber", penalty="scad", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-scad -->

### Bisquare 与 MCP

Bisquare 对超过阈值的残差给出零梯度，MCP 则改变系数的惩罚形式；两者都与上一种配置不同。先完成 [CPU 示例](#cpu-example)，再复用其中的导入和训练数据，在同一留出集上评价这类选择。

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-bisquare -->
```python
bisquare_model = PenalizedRobustRegression(
    loss="bisquare", penalty="mcp", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-bisquare -->

### Fair 与 L2

先完成 [CPU 示例](#cpu-example)，再复用其中的导入和训练数据。Fair 对残差的降权较平缓；本例保留 L2 惩罚，`auto` 使用 Newton。

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-fair -->
```python
fair_model = PenalizedRobustRegression(
    loss="fair", penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-fair -->

### GPU（Torch CUDA）

先完成 [CPU 示例](#cpu-example)，再复用其中的导入与 `X`、`y`，显式指定 Torch CUDA。需要安装 Torch 并有可用 CUDA 设备；显式设备不可用时会报错。尺度估计的 CPU 边界见后文说明。

```python
gpu_model = PenalizedRobustRegression(
    loss="huber", penalty="scad", alpha=0.1, device="torch",
).fit(X[:240], y[:240])
```

### 底层求解器接口

先完成 [CPU 示例](#cpu-example)，再复用其中的训练数组。底层调用需要自行选择损失和惩罚，不会自动添加截距；下面有意拟合无截距、固定 Huber 阈值为 1 的目标，与前面的自动尺度模型不同。普通建模优先使用估计器接口。

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-direct-solver -->
```python
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(delta=1.0)
coef, n_iter = fista_solver(
    loss, SCADPenalty(alpha=0.1), X[:240], y[:240],
)
```
<!-- example-end: robust-direct-solver -->

## 求解器兼容性

下表描述当前公开模型/底层求解路径；`sample_weight` 仍需同时满足具体损失函数和求解器的带权支持约定。

| 求解器 | Huber | Bisquare | Fair | 说明 |
|--------|:---:|:---:|:---:|------|
| Proximal Newton | ✅（光滑目标） | ✅（光滑目标） | ✅（光滑目标） | 当前通用实现对 L2/无惩罚执行 Newton；非光滑请求转到 FISTA |
| FISTA | ✅ | ✅ | ✅ | 稀疏/近端路径 |
| FISTA-BB | ✅（受支持组合） | ✅（受支持组合） | ✅（受支持组合） | 自适应步长 |
| FISTA-LLA | ✅ | ✅ | ✅ | SCAD/MCP 等非凸惩罚的 LLA 路径 |
| IRLS | ❌（当前未开放） | ✅（L2/无惩罚） | ✅（L2/无惩罚） | 当前公共分发不会为 Huber 选择 IRLS |
| Newton | ✅ | ✅ | ✅ | `solver="auto"` 下光滑 L2/无惩罚的主要路径 |
| L-BFGS | ✅（光滑、无权重/均匀权重） | ✅（光滑、无权重/均匀权重） | ✅（光滑、无权重/均匀权重） | 通用非 GLM `LossBase` 当前未声明直接非均匀带权 L-BFGS |
| ADMM | ✅（受支持形式） | ✅（受支持形式） | ✅（受支持形式） | 共享入口当前只接受未传或均匀 `sample_weight` |

### `solver="auto"` 的主要分发

| 惩罚 | 主要路径 | 说明 |
|---------|---------------|-------|
| L2 / 无惩罚 | Newton | 稳健损失提供梯度与 Hessian |
| L1 / ElasticNet | FISTA | 近端稀疏路径 |
| SCAD / MCP | FISTA + LLA | 非凸惩罚通过局部线性近似形成加权凸子问题 |
| 自适应 L1 | FISTA / LLA 路径 | 使用自适应加权近端形式 |
| 分组惩罚 | Group FISTA / Group FISTA-LLA | 使用对应的分组近端算子 |

## 算法说明

### 光滑 Huber / Bisquare / Fair

对于 L2/无惩罚目标，当前自动分发使用 Newton。损失函数提供实际梯度和 Hessian，求解器使用线性系统与 Armijo 回溯完成更新。

### SCAD / MCP

非凸惩罚通过 LLA 转换成局部加权凸问题，再由 FISTA 系列内层求解。对于非光滑惩罚，底层 `proximal_newton_solver` 会发出 `RuntimeWarning` 并使用 FISTA，因为该求解器没有实现 Hessian 度量近端子问题。

### Huber IRLS 的当前状态

`HuberLoss.irls()` 会抛出 `NotImplementedError`。对 Huber 回归，`PenalizedRobustRegression(..., solver="irls")` 会抛出 `ValueError`。固定阈值 Huber 的标准 IRLS 权重

$$
w_i=\frac{\psi_\delta(r_i)}{r_i}
=\min\left(1,\frac{\delta}{|r_i|}\right)
$$

与 Huber 一阶条件具有直接关系。因此，Huber 可以自然地构造 IRLS 更新，但该 API 不支持这一算法。L2/无惩罚 Huber 拟合可用 Newton；受支持的近端惩罚可用 FISTA。

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | `(p,)` float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `loss` | str | 损失名称 |

## 外部验证

- **Huber**：使用经典 Huber 损失形式；与 `MASS::rlm(psi=psi.huber)` 做数值比较时需要对齐调节常数和尺度估计约定。当前 Huber IRLS 不在公开支持矩阵中。
- **Bisquare**：使用 Tukey biweight 损失形式；外部比较需要对齐调节常数和尺度约定，当前非凸惩罚路径使用 LLA/FISTA。
- **Fair**：使用 Fair 损失形式；外部比较需要对齐调节常数和尺度约定。

## 注意事项

- 尺度计算目前会使用 NumPy 主机数组；完成尺度预计算后，数值优化继续使用所选 NumPy/CuPy/Torch 后端。
- `sample_weight` 是否可用取决于损失函数、求解器和具体模型路径，而不是“所有 robust solver 自动支持”。
- 三种损失都提供 Hessian 数值原语；这支持光滑 Newton 路径，但不等价于“任意非光滑惩罚都支持 Proximal Newton”。
- Huber IRLS 当前未作为公开求解路径开放；显式选择该组合会按照当前兼容性约定失败。

## 参考文献

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using IRLS. *Communications in Statistics*, A6(9), 813-827.
