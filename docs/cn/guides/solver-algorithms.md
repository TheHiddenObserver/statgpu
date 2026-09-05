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

本页按当前源码列出 **10 条通用求解路径**：其中 8 个低层函数由 `statgpu.solvers` 导出，IRLS 以 `statgpu.glm_core.IRLSSolver` 导出，`exact` 则是估计器内部的闭式路径。模型专用算法只放在对应模型页；分位数专用求解器见[分位数回归](../models/quantile.md#算法详解)。

::: warning API 边界
该清单比任意单个估计器接受的 `solver=` 取值更广。除非已经核对[求解器 × 惩罚兼容矩阵](solver-penalty-matrix.md)，否则优先使用 `solver="auto"`。直接求解器函数需要 loss 与 penalty 对象，属于进阶接口。
:::

## 从模型页面选择

求解器应从正在拟合的模型页面选择，而不是从下方完整算法清单中随意挑选。各链接页面会列出公开选择器、默认值、`auto` 的实际分发路径，以及明确不可用的组合。

| 模型接口 | 求解器控制项 | 模型专属说明 |
|---|---|---|
| `GeneralizedLinearModel` 与支持选择器的类型化 GLM | `solver` | [广义线性模型](../models/generalized-linear-model.md#求解器支持)、[Poisson](../models/poisson-regression.md#求解器支持) |
| `LogisticRegression` 类型化 wrapper | 固定 IRLS；无选择器 | [逻辑回归](../models/logistic-regression.md#求解器支持) |
| Ridge | `solver` | [岭回归](../models/ridge.md#求解器支持) |
| Lasso 与 Elastic Net | `solver`；CV/路径辅助接口还提供 `cpu_solver` | [Lasso](../models/lasso.md#求解器支持)、[Elastic Net](../models/elastic-net.md#求解器支持) |
| Adaptive Lasso、SCAD、MCP | 模型专属 continuation/加权 L1 路径 | [Adaptive Lasso](../models/adaptive-lasso.md#求解器支持)、[SCAD](../models/scad.md#求解器支持)、[MCP](../models/mcp.md#求解器支持) |
| 稳健回归 | 模型专属 `solver` 规则 | [稳健回归](../models/robust.md#求解器支持) |
| Cox 与有序响应模型 | 固定 Newton 路径或惩罚模型 `solver` | [CoxPH](../models/coxph.md#求解器支持)、[有序模型](../models/ordered.md#求解器支持) |
| PCA、KernelPCA、NMF | `svd_solver`、`eigen_solver` 或 `solver` | [PCA](../unsupervised/pca.md#求解器支持)、[核方法](../models/kernel-methods.md#求解器支持)、[NMF](../unsupervised/nmf.md#求解器支持) |

`fista_lla_path`、`proximal_newton_solver`、`lbfgs_b_solver` 等名称属于内部路径或低层 API，并不是所有估计器都能传入的 `solver=` 值。

## 完整清单

| # | 路径 | 公开状态 | 主要用途 | 后端 |
|---:|---|---|---|---|
| 1 | FISTA | `fista_solver`；估计器会调度 | 凸非光滑惩罚和一般复合目标 | NumPy、CuPy、Torch |
| 2 | FISTA-BB | `fista_bb_solver`；估计器会调度 | 使用自适应谱步长的稀疏 GLM | NumPy、CuPy、Torch |
| 3 | FISTA-LLA | `fista_lla_path`；continuation 子路径 | SCAD、MCP、Group SCAD、Group MCP | NumPy、CuPy、Torch |
| 4 | Newton-Raphson | `newton_solver`；估计器会调度 | 无惩罚或 L2 的光滑损失 | NumPy、CuPy、Torch |
| 5 | Proximal-Newton facade | `proximal_newton_solver`；低层直接 API | 光滑情形走 Newton；非光滑情形转 FISTA | NumPy、CuPy、Torch |
| 6 | GLM IRLS | `IRLSSolver`；估计器会调度 | 普通 GLM 与受支持的光滑惩罚 GLM | NumPy、CuPy、Torch |
| 7 | L-BFGS | `lbfgs_solver`；估计器会调度 | 不希望保存完整 Hessian 的光滑目标 | NumPy、CuPy、Torch |
| 8 | L-BFGS-B | `lbfgs_b_solver`；低层直接 API | 带 box constraint 的光滑目标 | NumPy、CuPy、Torch |
| 9 | ADMM | `admm_solver`；可显式选择 | 拆分光滑 loss 与近端 penalty 更新 | NumPy、CuPy、Torch |
| 10 | Exact Ridge | 估计器 `solver="exact"` | squared error + L2 | NumPy、CuPy、Torch |

下方每个算法段落会用文字解释方法，并在推导附近保留简洁的“作者—年份”引用；完整书目信息统一列在页面末尾。这些引用对应数学算法家族；statgpu 的后端 kernel、数值保护、停止规则与估计器分发属于具体实现。

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

FISTA 按 Beck 与 Teboulle（2009）的方案把近端梯度更新与 Nesterov 动量结合；当前实现的重启行为与 O'Donoghue 和 Candès（2015）的自适应重启思想相关。令 $L_k$ 为局部 Lipschitz 估计，则

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

FISTA-BB 保留 Beck 与 Teboulle（2009）的加速近端结构，同时采用 Barzilai 与 Borwein（1988）的谱步长构造。它用相邻参数和光滑梯度估计局部步长：

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

SCAD/MCP 的非凸惩罚背景来自 Fan 与 Li（2001），局部线性近似策略采用 Zou 与 Li（2008）的构造。它在 $\beta^{(m)}$ 附近把非凸惩罚替换为 weighted L1 surrogate：

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

Newton 方向与线搜索采用 Nocedal 与 Wright（2006）的标准表述。对光滑目标，定义

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

复合 Proximal Newton 框架可参见 Lee、Sun 与 Saunders（2014）。完整的非光滑 Proximal Newton 应求解 Hessian metric 子问题

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

IRLS 可参见 Green（1984）以及 McCullagh 与 Nelder（1989）的 GLM 论述。在当前均值 $\mu_i$ 和线性预测子 $\eta_i=g(\mu_i)$ 处，它构造

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

## 7. L-BFGS

**源码：**`statgpu/solvers/_lbfgs.py`

有限内存更新采用 Liu 与 Nocedal（1989）的方法。L-BFGS 根据最近若干组

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k,
\qquad
\rho_k=(y_k^\top s_k)^{-1}
$$

近似 $H_k^{-1}g_k$。标准 two-loop recursion 不需要形成稠密 Hessian 即可得到 $d_k=-B_kg_k$，随后用 Armijo backtracking 检查完整光滑目标。默认 history size 为 10，只接受 L2/无惩罚和均匀 `sample_weight`。

## 8. L-BFGS-B

**源码：**`statgpu/solvers/_lbfgs_b.py`

带边界约束的有限内存方法采用 Byrd 等（1995）的构造。L-BFGS-B 加入逐坐标 box constraint

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

## 9. ADMM

**源码：**`statgpu/solvers/_admm.py`

变量拆分和残差诊断采用 Boyd 等（2011）的 ADMM 表述。该方法引入 $z$ 并求解

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

## 10. Exact Ridge

**源码：**`statgpu/linear_model/penalized/_fit_mixin.py` 中的 `_solve_exact_numpy`、`_solve_exact_cupy` 与 `_solve_exact_torch`

该 Ridge 闭式估计采用 Hoerl 与 Kennard（1970）的基本构造。对中心化/加权后的 $X_c$ 与 $y_c$，估计器求解

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
| 标量或 group SCAD/MCP | FISTA-LLA |
| squared error + L1/Elastic Net | FISTA |
| 稀疏 GLM | 根据 family、后端、CV 模式和规模选择 FISTA 或 FISTA-BB |
| 光滑 GLM/robust/Cox 目标 | Newton；部分 CV 情况使用 L-BFGS |

`proximal_newton_solver` 与 `lbfgs_b_solver` 可以直接导入，但当前不是 `auto` 的目的地。模型专属算法、ordered-model trust-region Newton、融合 GPU kernel 与 group LLA wrapper 会归入父模型/父路径说明，不再虚增 solver keyword。

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

- Barzilai, J., & Borwein, J. M. (1988). [Two-point step size gradient methods](https://doi.org/10.1093/imanum/8.1.141). *IMA Journal of Numerical Analysis*, 8(1), 141–148.
- Beck, A., & Teboulle, M. (2009). [A fast iterative shrinkage-thresholding algorithm for linear inverse problems](https://doi.org/10.1137/080716542). *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://doi.org/10.1561/2200000016). *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). [A limited memory algorithm for bound constrained optimization](https://doi.org/10.1137/0916069). *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
- Fan, J., & Li, R. (2001). [Variable selection via nonconcave penalized likelihood and its oracle properties](https://doi.org/10.1198/016214501753382273). *Journal of the American Statistical Association*, 96(456), 1348–1360.
- Green, P. J. (1984). [Iteratively reweighted least squares for maximum likelihood estimation, and some robust and resistant alternatives](https://doi.org/10.1111/j.2517-6161.1984.tb01288.x). *Journal of the Royal Statistical Society: Series B*, 46(2), 149–192.
- Hoerl, A. E., & Kennard, R. W. (1970). [Ridge regression: Biased estimation for nonorthogonal problems](https://doi.org/10.1080/00401706.1970.10488634). *Technometrics*, 12(1), 55–67.
- Lee, J. D., Sun, Y., & Saunders, M. A. (2014). [Proximal Newton-type methods for minimizing composite functions](https://doi.org/10.1137/130921428). *SIAM Journal on Optimization*, 24(3), 1420–1443.
- Liu, D. C., & Nocedal, J. (1989). [On the limited memory BFGS method for large scale optimization](https://doi.org/10.1007/BF01589116). *Mathematical Programming*, 45, 503–528.
- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*（第 2 版）. Chapman & Hall/CRC.
- Nocedal, J., & Wright, S. J. (2006). [*Numerical Optimization*（第 2 版）](https://doi.org/10.1007/978-0-387-40065-5). Springer.
- O'Donoghue, B., & Candès, E. (2015). [Adaptive restart for accelerated gradient schemes](https://doi.org/10.1007/s10208-013-9150-3). *Foundations of Computational Mathematics*, 15(3), 715–732.
- Zou, H., & Li, R. (2008). [One-step sparse estimates in nonconcave penalized likelihood models](https://doi.org/10.1214/07-AOS520). *Annals of Statistics*, 36(4), 1509–1533.
