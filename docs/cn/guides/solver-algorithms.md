# 求解器算法

> 语言：简体中文
> 最后更新：2026-09-04
> 切换：[English](../../en/guides/solver-algorithms.md)

## 范围与统一目标

statgpu 的多数估计器求解复合目标

$$
\min_{\beta\in\mathbb R^p}
F(\beta)
=
f(\beta)+P(\beta),
$$

其中 $f$ 是平均数据拟合损失，$P$ 可以为零、Ridge 等光滑惩罚，或者非光滑/非凸惩罚。截距通常不进入 $P$。

本页按当前源码列出 **12 条核心求解路径**：其中 10 个低层函数由 `statgpu.solvers` 导出，IRLS 以 `statgpu.glm_core.IRLSSolver` 导出，`exact` 则是估计器内部的闭式路径。融合 GPU kernel 与 group-aware LLA 是这些路径的实现方式，不另算公开求解器名称。

::: warning API 边界
该清单比任意单个估计器接受的 `solver=` 取值更广。除非已经核对[求解器 × 惩罚兼容矩阵](solver-penalty-matrix.md)，否则优先使用 `solver="auto"`。直接求解器函数需要 loss 与 penalty 对象，属于进阶接口。
:::

## 完整清单

| # | 路径 | 公开状态 | 主要用途 | 后端 |
|---:|---|---|---|---|
| 1 | FISTA | `fista_solver`；估计器会调度 | 凸非光滑惩罚和一般复合目标 | NumPy、CuPy、Torch |
| 2 | FISTA-BB | `fista_bb_solver`；估计器会调度 | 使用自适应谱步长的稀疏 GLM | NumPy、CuPy、Torch |
| 3 | FISTA-LLA | `fista_lla_path`；continuation 子路径 | SCAD、MCP、Group SCAD、Group MCP | NumPy、CuPy、Torch |
| 4 | Newton-Raphson | `newton_solver`；估计器会调度 | 无惩罚或 L2 的光滑损失 | NumPy、CuPy、Torch |
| 5 | Proximal-Newton facade | `proximal_newton_solver`；低层直接 API | 光滑情形走 Newton；非光滑情形转 FISTA | NumPy、CuPy、Torch |
| 6 | GLM IRLS | `IRLSSolver`；估计器会调度 | 普通 GLM 与受支持的光滑惩罚 GLM | NumPy、CuPy、Torch |
| 7 | Quantile Proximal IRLS | `proximal_irls_quantile_solver`；内部调度子路径 | Quantile loss 配合 SCAD/MCP continuation | NumPy、CuPy、Torch |
| 8 | Quantile coordinate descent | `quantile_cd_solver`；低层直接 API | Quantile loss 的 weighted-L1 LLA 子问题 | NumPy |
| 9 | L-BFGS | `lbfgs_solver`；估计器会调度 | 不希望保存完整 Hessian 的光滑目标 | NumPy、CuPy、Torch |
| 10 | L-BFGS-B | `lbfgs_b_solver`；低层直接 API | 带 box constraint 的光滑目标 | NumPy、CuPy、Torch |
| 11 | ADMM | `admm_solver`；可显式选择 | 拆分光滑 loss 与近端 penalty 更新 | NumPy、CuPy、Torch |
| 12 | Exact Ridge | 估计器 `solver="exact"` | squared error + L2 | NumPy、CuPy、Torch |

旧页面遗漏了 `quantile_cd_solver`；旧中文页还完全缺少 L-BFGS/L-BFGS-B、ADMM 与 exact。

## 公共记号：近端算子

对步长 $\gamma>0$，近端算子定义为

$$
\operatorname{prox}_{\gamma P}(v)
=
\arg\min_u
\left\{
\frac{1}{2}\lVert u-v\rVert_2^2+\gamma P(u)
\right\}.
$$

当 $P(\beta)=\lambda\lVert\beta\rVert_1$ 时，它就是逐元素软阈值：

$$
\mathcal S_{\gamma\lambda}(v_j)
=
\operatorname{sign}(v_j)
\max\left(|v_j|-\gamma\lambda,0\right).
$$

## 1. FISTA

**源码：**`statgpu/solvers/_fista.py`

FISTA 把近端梯度更新与 Nesterov 动量结合。令 $L_k$ 为局部 Lipschitz 估计，则

$$
\beta_{k+1}
=
\operatorname{prox}_{P/L_k}
\left(
z_k-\frac{1}{L_k}\nabla f(z_k)
\right),
$$

$$
t_{k+1}
=
\frac{1+\sqrt{1+4t_k^2}}{2},
\qquad
z_{k+1}
=
\beta_{k+1}
+
\frac{t_k-1}{t_{k+1}}
\left(\beta_{k+1}-\beta_k\right).
$$

实现会在需要时进行 backtracking，保存当前最佳目标以便发散恢复，并可针对陡峭的指数 link loss 限制或关闭动量。GPU 路径会融合近端与动量操作。默认控制为 `max_iter=1000`、`tol=1e-4`。

## 2. FISTA-BB

**源码：**`statgpu/solvers/_fista_bb.py`

FISTA-BB 用相邻参数和光滑梯度估计局部步长：

$$
s_k=\beta_k-\beta_{k-1},
\qquad
y_k=\nabla f(\beta_k)-\nabla f(\beta_{k-1}).
$$

两个 Barzilai-Borwein 候选步长为

$$
\gamma_k^{\mathrm{BB1}}
=
\frac{s_k^\top s_k}{s_k^\top y_k},
\qquad
\gamma_k^{\mathrm{BB2}}
=
\frac{s_k^\top y_k}{y_k^\top y_k}.
$$

burn-in 后会交替使用两种步长，并将其截断到安全范围；动量方向与下降方向冲突时执行 adaptive restart。SCAD/MCP 及其 group 版本会关闭 BB 自适应，因为变化的 LLA 权重会使谱步长不稳定。二次损失可能转回标准 FISTA。

## 3. FISTA-LLA

**源码：**`statgpu/solvers/_fista_lla.py` 与 `_fista_lla_group_contract.py`

局部线性近似在 $\beta^{(m)}$ 附近把非凸惩罚替换为 weighted L1 surrogate：

$$
P_\lambda(|\beta_j|)
\approx
P_\lambda(|\beta_j^{(m)}|)
+
P_\lambda'(|\beta_j^{(m)}|)
\left(|\beta_j|-|\beta_j^{(m)}|\right).
$$

去掉常数后，每个外层迭代求解

$$
\min_\beta
f(\beta)
+
\sum_{j=1}^p\omega_j^{(m)}|\beta_j|,
\qquad
\omega_j^{(m)}
=
P_\lambda'(|\beta_j^{(m)}|),
$$

内层使用 FISTA。递减 continuation path $\lambda_1>\cdots>\lambda_M$ 为困难的 SCAD/MCP 问题提供 warm start。Group SCAD/MCP 对 $\lVert\beta_g\rVert_2$ 应用同样近似，并使用 group-aware FISTA 内层问题。

## 4. Newton-Raphson

**源码：**`statgpu/solvers/_newton.py`

对光滑目标，定义

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

带数值正则的 Newton 更新为

$$
\left(H_k+\delta I\right)d_k=g_k,
\qquad
\beta_{k+1}=\beta_k-a_kd_k,
$$

其中 $\delta=10^{-10}$，$a_k$ 由 Armijo backtracking 选择。常量 Hessian 会被缓存；真正的秩亏可回退到 least squares，其他数值错误不会被静默吞掉。该函数只接受光滑惩罚和均匀 `sample_weight`。

## 5. Proximal-Newton facade

**源码：**`statgpu/solvers/_proximal_newton.py`

完整的非光滑 Proximal Newton 应求解 Hessian metric 子问题

$$
d_k
=
\arg\min_d
\left\{
g_k^\top d
+
\frac{1}{2}d^\top H_kd
+
P(\beta_k+d)
\right\}.
$$

当前尚未实现这个 Hessian-metric proximal 子问题。公开函数在 L2/无惩罚时执行 Newton；收到非光滑惩罚时先发出 `RuntimeWarning`，再转给 `fista_solver`。这条明确边界可避免把错误的 Euclidean-prox shortcut 描述成 Proximal Newton。

## 6. GLM IRLS

**源码：**`statgpu/glm_core/_irls.py`

在当前均值 $\mu_i$ 和线性预测子 $\eta_i=g(\mu_i)$ 处，IRLS 构造

$$
z_i
=
\eta_i
+
(y_i-\mu_i)g'(\mu_i),
\qquad
w_i
=
\frac{\left(d\mu_i/d\eta_i\right)^2}{V(\mu_i)}.
$$

随后求解 weighted Ridge 系统

$$
\left(\tilde X^\top W\tilde X+\lambda D\right)\beta_{\mathrm{new}}
=
\tilde X^\top Wz,
$$

其中 $D$ 通常不惩罚截距。backtracking line search 会检查已注册的 family/link 目标。`IRLSSolver` 支持解析 sample weight 和 NumPy、CuPy、Torch。普通 GLM 中 `solver="auto"` 当前会解析为 IRLS；需要无惩罚 IRLS 时设置 `C=0`。

## 7. Quantile Proximal IRLS

**源码：**`statgpu/solvers/_proximal_irls_quantile.py`

对残差 $r_i=y_i-x_i^\top\beta$ 与分位数 $\tau$，check loss 为

$$
\rho_\tau(r)
=
r\left(\tau-\mathbf 1\{r<0\}\right).
$$

在每个 continuation 与 LLA 步内，实现使用

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

令 $A=\operatorname{diag}(a_i)$，

$$
g=\tilde X^\top A(y-\tilde X\beta),
\qquad
h=\operatorname{diag}(\tilde X^\top A\tilde X),
$$

所有坐标并行更新：

$$
\beta_j^{\mathrm{new}}
=
\frac{
\mathcal S_{n\omega_j}(g_j+h_j\beta_j)
}{h_j}.
$$

这是一种适合 GPU 的 Jacobi 式对角 majorization，并不是 cyclic coordinate descent。解析 sample weight 在归一化到总和为 $n$ 后乘入 $a_i$。

## 8. Quantile coordinate descent

**源码：**`statgpu/solvers/_quantile_cd.py`

这个仅 NumPy 的低层求解器，在 LLA 权重与 cyclic coordinate update 之间交替，目标为

$$
\min_\beta
\sum_{i=1}^n\rho_\tau(y_i-x_i^\top\beta)
+
\sum_{j=1}^p\omega_j|\beta_j|.
$$

令 $r\ge0$ 时 $\psi_\tau(r)=\tau$，否则 $\psi_\tau(r)=-(1-\tau)$，实际坐标更新是

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

该函数可供进阶用户直接导入，但当前自动调度不会选择它。其签名虽然包含 `sample_weight`，当前实现尚未消费该参数；观测权重很重要时应使用已调度的 FISTA/Proximal IRLS 路径。

## 9. L-BFGS

**源码：**`statgpu/solvers/_lbfgs.py`

L-BFGS 根据最近若干组

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k,
\qquad
\rho_k=(y_k^\top s_k)^{-1}
$$

近似 $H_k^{-1}g_k$。标准 two-loop recursion 不需要形成稠密 Hessian 即可得到 $d_k=-B_kg_k$，随后用 Armijo backtracking 检查完整光滑目标。默认 history size 为 10，只接受 L2/无惩罚和均匀 `sample_weight`。

## 10. L-BFGS-B

**源码：**`statgpu/solvers/_lbfgs_b.py`

L-BFGS-B 加入逐坐标 box constraint

$$
\ell_j\le\beta_j\le u_j
$$

并投影候选点：

$$
\beta_{k+1}
=
\Pi_{[\ell,u]}
\left(\beta_k+a_kd_k\right).
$$

收敛判断使用 projected gradient：在活跃边界上继续指向边界外的分量会被置零。上下界必须与系数同形、不得包含 NaN，并满足 $\ell_j\le u_j$。该路径同样只接受光滑惩罚和均匀 sample weight。

## 11. ADMM

**源码：**`statgpu/solvers/_admm.py`

ADMM 引入 $z$ 并求解

$$
\min_{\beta,z}
f(\beta)+P(z)
\quad\text{subject to}\quad
\beta=z.
$$

使用 scaled dual variable $u$ 后，

$$
\beta^{k+1}
=
\arg\min_\beta
\left[
f(\beta)
+
\frac{\rho}{2}
\lVert\beta-z^k+u^k\rVert_2^2
\right],
$$

$$
z^{k+1}
=
\operatorname{prox}_{P/\rho}
\left(\beta^{k+1}+u^k\right),
\qquad
u^{k+1}
=
u^k+\beta^{k+1}-z^{k+1}.
$$

小型常量 Hessian 系统使用缓存的 Cholesky；其他 loss 子问题使用 Nesterov 加速梯度迭代，`cg_max_iter` 是保留的历史参数名。原始与对偶残差为

$$
r_{\mathrm p}=\lVert\beta-z\rVert_2,
\qquad
r_{\mathrm d}=\rho\lVert z^k-z^{k-1}\rVert_2.
$$

开启自适应 $\rho$ 后，当一侧残差超过另一侧十倍时会将 $\rho$ 加倍或减半。当前仅接受均匀 sample weight。

## 12. Exact Ridge

**源码：**`statgpu/linear_model/penalized/_fit_mixin.py` 中的 `_solve_exact_numpy`、`_solve_exact_cupy` 与 `_solve_exact_torch`

对中心化/加权后的 $X_c$ 与 $y_c$，估计器求解

$$
\hat\beta
=
\left(
X_c^\top W X_c+n_{\mathrm{eff}}\alpha I
\right)^{-1}
X_c^\top W y_c.
$$

它对应平均 squared error 加 $\alpha\lVert\beta\rVert_2^2/2$ 的约定。截距由加权均值重建，不被惩罚。NumPy 使用直接求解并在秩亏时回退 pseudoinverse；CuPy 优先 Cholesky；Torch 使用 `torch.linalg.solve`。自动调度在 CPU 上偏好 exact Ridge，而同类 GPU 问题使用 Newton，因为中小规模 GPU 分解的启动成本可能抵消收益。

## 自动调度与直接可用不是一回事

主要自动策略如下：

| 目标 | 典型自动路径 |
|---|---|
| 普通 `GeneralizedLinearModel` | IRLS |
| CPU 上 squared error + L2 | exact |
| GPU 上 squared error + L2 | Newton |
| 标量或 group SCAD/MCP | FISTA-LLA；quantile loss 使用 Proximal IRLS |
| quantile loss + 凸惩罚 | FISTA |
| squared error + L1/Elastic Net | FISTA |
| 稀疏 GLM | 根据 family、后端、CV 模式和规模选择 FISTA 或 FISTA-BB |
| 光滑 GLM/robust/Cox 目标 | Newton；部分 CV 情况使用 L-BFGS |

`proximal_newton_solver`、`quantile_cd_solver` 与 `lbfgs_b_solver` 可以直接导入，但当前不是 `auto` 的目的地。估计器专属的 ordered-model trust-region Newton、融合 GPU kernel 与 group LLA wrapper 会归入父模型/父路径说明，不再虚增 solver keyword。

准确的接受与拒绝组合见[求解器 × 惩罚兼容矩阵](solver-penalty-matrix.md)，架构关系见 [Loss × Penalty × Solver 框架](loss-penalty-solver-framework.md)。

## 常用控制项与失败信号

| 控制项 | 含义 |
|---|---|
| `max_iter` | 外层最大迭代数；continuation 方法还可能有内层上限 |
| `tol` | 依算法表示梯度、参数步长或残差阈值 |
| `history_size` | L-BFGS/L-BFGS-B 保留的 $(s_k,y_k)$ 组数 |
| `lipschitz_L` | FISTA 路径可选的光滑梯度 Lipschitz 常数 |
| `rho`、`adaptive_rho` | ADMM augmented-Lagrangian 尺度与残差平衡 |
| `alpha_path`、`max_lla_per_step`、`lla_tol` | 非凸惩罚的 continuation 与 LLA 控制 |

达到 `max_iter` 时，受支持路径会发出 convergence warning。求解器返回系数并不等于统计模型合理；仍需检查收敛元数据、objective/KKT 诊断和样本外验证。

## 参考文献

- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- Barzilai, J., & Borwein, J. M. (1988). Two-point step size gradient methods. *IMA Journal of Numerical Analysis*, 8(1), 141–148.
- O'Donoghue, B., & Candès, E. (2015). Adaptive restart for accelerated gradient schemes. *Foundations of Computational Mathematics*, 15(3), 715–732.
- Nocedal, J. (1980). Updating quasi-Newton matrices with limited storage. *Mathematics of Computation*, 35(151), 773–782.
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via ADMM. *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood. *Journal of the American Statistical Association*, 96, 1348–1360.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509–1533.
