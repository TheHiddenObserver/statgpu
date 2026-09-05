# 分位数回归

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../../en/models/quantile.md)

## 概述

`QuantileLoss` 实现 quantile 回归的 pinball（check）损失。`QuantileRegression` 使用固定 FISTA 拟合；`PenalizedQuantileRegression` 则增加 penalty-aware 分发，并为 SCAD/MCP 提供专门的 proximal-IRLS 路径。

| 组件 | 路径 |
|------|------|
| 损失 | `statgpu.losses.QuantileLoss` |
| 独立模型 | `statgpu.linear_model.QuantileRegression` |
| 惩罚模型 | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| 专用求解器 | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| 低层坐标求解器 | `statgpu.solvers._quantile_cd.quantile_cd_solver` |
| R 等价 | `quantreg::rq()` |

## 目标函数

Pinball 损失，在分位数 τ ∈ (0, 1) 处：

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad \rho_\tau(u) = u \cdot (\tau - \mathbf{1}\{u < 0\})
$$

逐样本梯度（subgradient，在 u=0 处）：

$$
\frac{\partial \ell}{\partial \eta} = -\tau + \mathbf{1}\{y - \eta < 0\}
$$

关键属性：梯度是阶梯函数，不随残差大小变化。因此 `has_hessian = False`、`smooth_gradient = False`。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值范围 (0, 1)。τ=0.5 为中位数回归。 |

无尺度参数；quantile 回归是尺度无关的。

## 求解器支持

两个估计器接口有意采用不同契约：

| 估计器 / `solver` 值 | 支持情况 | 实际路径 |
|---|:---:|---|
| `QuantileRegression` | 固定 | FISTA；bootstrap 推断使用 batched pinball FISTA |
| `PenalizedQuantileRegression(solver="auto")` | 支持 | 按下表根据惩罚分发 |
| `solver="fista"` | 支持 | FISTA；但 L2/none 会在内部用 quantile IRLS 稳定求解 |
| `solver="fista_bb"` | 支持 | 对普通兼容惩罚使用 BB 步长 FISTA |
| `solver="irls"` | 仅 L2/none | Quantile 专属 IRLS |
| `solver="admm"` | 兼容惩罚 | 拆分求解替代路径；只支持均匀样本权重 |
| `newton`、`lbfgs`、`exact` | 不支持 | pinball loss 没有 Hessian，且 exact 仅适用于 Ridge |

`proximal_irls_quantile_solver` 与 `quantile_cd_solver` 是内部或低层函数名，不能作为 `solver=` 的取值。

## 惩罚兼容性

| 惩罚 | 求解器 (auto) | 说明 |
|---------|---------------|-------|
| l2 / none | FISTA 分发，内部使用 quantile IRLS | 光滑惩罚；内部会收紧 IRLS 容差。 |
| l1 / elasticnet | FISTA | Proximal/subgradient 路径。 |
| SCAD / MCP | Proximal IRLS + LLA | 模型专属 continuation 子路径。 |
| adaptive_l1 | 加权 L1 FISTA | 数据驱动 proximal 权重。 |
| group penalties | Group proximal 路径 | 具体分发取决于 group penalty。 |

## 示例

### 独立模型（含统计推断）

```python
from statgpu.linear_model import QuantileRegression

# 中位数回归，含 kernel 标准误
model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="kernel",   # Powell (1991) sandwich
    kernel="epa",                # Epanechnikov 核
    bandwidth="hsheather",       # Hall-Sheather 带宽
)
model.fit(X, y)
print(model.coef_)        # 系数
print(model._bse)         # 标准误
print(model._pvalues)     # p 值
print(model._conf_int)    # 95% 置信区间

# Bootstrap 推断，使用批量 FISTA（GPU 加速）
model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="bootstrap",
    n_bootstrap=200,
    device="cuda",         # 或 "torch" / "cpu"
)
model.fit(X, y)
```

### 带惩罚项的分位数回归

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# 中位数回归 (τ=0.5)
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)
print(model.coef_)

# 上四分位数 + L2 惩罚
model = PenalizedQuantileRegression(quantile=0.75, penalty='l2', alpha=0.01)
model.fit(X, y)

# 下四分位数 + MCP
model = PenalizedQuantileRegression(quantile=0.25, penalty='mcp', alpha=0.1)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### GPU (cupy-CUDA)

```python
import cupy as cp
X_cp = cp.asarray(X)
y_cp = cp.asarray(y)

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_cp, y_cp)
```

### 加权 Quantile

```python
sample_weight = np.ones(n)
sample_weight[:50] = 5.0  # 前 50 个样本权重加倍

model = PenalizedQuantileRegression(quantile=0.5, penalty='l2', alpha=0.01)
model.fit(X, y, sample_weight=sample_weight)
```

## 算法详解

### 分位数近端 IRLS（SCAD/MCP）

**源码：**`statgpu/solvers/_proximal_irls_quantile.py`

这是 quantile loss 配合 SCAD 或 MCP 时使用的模型专属 continuation 路径。令残差 $r_i=y_i-x_i^\top\beta$、目标分位数为 $\tau$，则

$$
\rho_\tau(r)=r\left(\tau-\mathbf 1\{r<0\}\right).
$$

在每个 continuation 与局部线性近似（LLA）步骤中，statgpu 构造

$$
a_i
=
\frac{
\tau\mathbf 1\{r_i\ge0\}
+
(1-\tau)\mathbf 1\{r_i<0\}
}{
\max(|r_i|,\varepsilon)
},
\qquad
\omega_j=P_\lambda'(|\beta_j|).
$$

令 $A=\operatorname{diag}(a_i)$，则

$$
g=\tilde X^\top A(y-\tilde X\beta),
\qquad
h=\operatorname{diag}(\tilde X^\top A\tilde X),
$$

所有坐标并行更新为

$$
\beta_j^{\mathrm{new}}
=
\frac{
\mathcal S_{n\omega_j}(g_j+h_j\beta_j)
}{h_j}.
$$

这是适合 GPU 的 Jacobi 风格对角 majorization 步骤，并不是 cyclic coordinate descent。解析样本权重会先归一化到总和为 $n$，再乘入 $a_i$。收敛比较保留在所选设备上，只按节流间隔同步布尔结果。

**主要参考文献：**

- Koenker, R., & Bassett, G. (1978). [Regression quantiles](https://doi.org/10.2307/1913643). *Econometrica*, 46(1), 33–50.
- Wu, Y., & Liu, Y. (2009). [Variable selection in quantile regression](https://www3.stat.sinica.edu.tw/sstest/j19n2/j19n222/j19n222.html). *Statistica Sinica*, 19(2), 801–817.
- Zou, H., & Li, R. (2008). [One-step sparse estimates in nonconcave penalized likelihood models](https://doi.org/10.1214/07-AOS520). *Annals of Statistics*, 36(4), 1509–1533.

### 分位数坐标下降

**源码：**`statgpu/solvers/_quantile_cd.py`

这个仅 NumPy 的低层求解器在 LLA 权重与 cyclic coordinate update 之间交替，求解

$$
\min_\beta
\sum_{i=1}^n\rho_\tau(y_i-x_i^\top\beta)
+
\sum_{j=1}^p\omega_j|\beta_j|.
$$

令 $r\ge0$ 时 $\psi_\tau(r)=\tau$，否则 $\psi_\tau(r)=-(1-\tau)$，实现中的坐标更新为

$$
\beta_j
\leftarrow
\frac{
\mathcal S_{\omega_j}
\left(
\sum_i x_{ij}\psi_\tau(r_i^{(-j)})
\right)
}{
\sum_i x_{ij}^2
}.
$$

该函数可供进阶用户直接导入，但不会被估计器自动分发选中。其签名虽然包含 `sample_weight`，当前实现并未使用该参数；观测权重会影响结果时，应使用估计器路由的 FISTA 或 quantile-proximal-IRLS 路径。

**主要参考文献：** Wu, Y., & Liu, Y. (2009). [Variable selection in quantile regression](https://www3.stat.sinica.edu.tw/sstest/j19n2/j19n222/j19n222.html). *Statistica Sinica*, 19(2), 801–817.

### IRLS（L2/无惩罚）

**主要参考文献：** Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

使用 Frisch-Newton 算法（匹配 statsmodels `QuantReg`）：
1. IRLS 权重：w_i = (τ + (1−2τ)·1_{r_i<0}) / max(|r_i|, ε)
2. 求解加权最小二乘：(X'WX + n·α·I) β = X'Wy
3. 重复至收敛（~5-15 次迭代）

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | (p,) float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 外部验证

- **R `quantreg::rq()`**: IRLS 路径系数与 Frisch-Newton IRLS 匹配到 1e-6。
- **sklearn `QuantileRegressor`**: HiGHS LP 求解器产生相同的 active set 和系数（tol=1e-8）。
- **FISTA-LLA 对等性**: Proximal IRLS-CD 与 FISTA-LLA 的 active set 一致（rtol=0.15）。

## 注意事项

- Score 使用加权 pinball 损失：`score()` 返回负平均 pinball 损失以兼容 sklearn。
- `sample_weight` 全求解器支持。
- GPU 设备（`cuda`/`torch`）不静默回退 CPU。
- 大规模问题（n=10K, p=500）GPU 比 CPU 快 ~49x。

## 参考文献

- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.
