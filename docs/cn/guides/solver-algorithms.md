# 求解器算法

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：算法参考  
> 切换：[English](../../en/guides/solver-algorithms.md)

## 概览

statgpu 提供一阶、二阶、近端和闭式等多类求解器。对大多数模型用户，建议先使用 `solver="auto"`；本页属于算法参考，因此会保留具体更新公式、收敛条件、计算后端行为和重要的支持边界。

阅读本页时可以先记住三点：

1. 支持某个计算后端，不代表所有损失函数、惩罚项和权重组合都受支持。
2. `sample_weight` 不会改变显式指定的 `solver`。如果所请求的带权组合不受支持，则直接报错。
3. 权重支持取决于完整的模型路径。尤其是直接调用 L-BFGS 时，非均匀权重目前由维护中的 GLM 损失明确支持，并不会自动扩展到所有 `LossBase`。

模型层面的完整分发表见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 求解器总览

| 求解器 | 最适合 | 后端支持 |
|--------|--------|:---:|
| Proximal IRLS-CD | 分位数回归 + SCAD/MCP | NumPy, CuPy, Torch |
| Proximal Newton | 光滑损失 + L2/无惩罚；非光滑请求使用 FISTA | NumPy, CuPy, Torch |
| FISTA | 一般非光滑惩罚 | NumPy, CuPy, Torch |
| FISTA-BB | GLM + 稀疏惩罚 | NumPy, CuPy, Torch |
| FISTA-LLA | 非凸惩罚的延续/LLA 路径 | NumPy, CuPy, Torch |
| IRLS | 具有维护中 IRLS 表示的损失 | NumPy, CuPy, Torch |
| Newton | 有 Hessian 的光滑损失 | NumPy, CuPy, Torch |
| L-BFGS | 光滑损失、中等参数维度 | NumPy, CuPy, Torch |
| L-BFGS-B | 带盒约束的光滑问题 | NumPy, CuPy, Torch |
| ADMM | 可分/近端形式 | NumPy, CuPy, Torch |
| `exact` | 平方误差 + L2 闭式路径 | NumPy, CuPy, Torch |

后端列只描述数值实现能力；具体模型和损失函数还会进一步限制可用组合。

---

## 1. Proximal IRLS-CD

**文件**：`statgpu/solvers/_proximal_irls_quantile.py`

**用途**：分位数回归 + SCAD/MCP。它把检查损失（pinball loss）的 IRLS 二次上界与非凸惩罚的局部线性近似（LLA）结合起来。

### 算法

对延续路径中的每个 `alpha`：

1. **LLA 外循环。** 在当前系数处计算惩罚导数

   $$
   d_j=P'(|\beta_j|),
   $$

   SCAD/MCP 的当前局部阈值为

   $$
   t_j=n\,d_j.
   $$

2. **IRLS-CD 内循环。** 先计算残差

   $$
   r_i=y_i-x_i^\top\beta.
   $$

   定义

   $$
   q_i=\begin{cases}
   \tau, & r_i\ge 0,\\
   1-\tau, & r_i<0,
   \end{cases}
   \qquad
   w_i^{\mathrm{IRLS}}=\frac{q_i}{\max(|r_i|,\varepsilon)}.
   $$

   如果传入解析权重 `sample_weight=s`，先归一化为

   $$
   \tilde s_i=\frac{n s_i}{\sum_j s_j},
   $$

   再使用

   $$
   w_i=\tilde s_i\,w_i^{\mathrm{IRLS}}.
   $$

3. **并行对角上界更新。** 记 $W=\operatorname{diag}(w)$，计算

   $$
   g=X^\top W(y-X\beta),
   \qquad
   h=\operatorname{diag}(X^\top W X),
   $$

   然后令

   $$
   u=g+h\odot\beta,
   $$

   并对所有坐标并行更新

   $$
   \beta_j^{\mathrm{new}}=\frac{S(u_j,t_j)}{h_j},
   $$

   其中软阈值算子为

   $$
   S(u,t)=\operatorname{sign}(u)\max(|u|-t,0).
   $$

   这里实际采用的是 Jacobi 风格的并行对角上界更新，而不是逐坐标循环更新。

4. **收敛判据。** IRLS 内循环检查

   $$
   \|\beta^{\mathrm{new}}-\beta\|_\infty<\texttt{tol},
   $$

   LLA 外循环检查

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_\infty<\texttt{lla\_tol}.
   $$

### 延续路径与默认值

- 延续路径：从 `lambda_max` 到目标 `alpha`；
- `max_lla_per_step=2`；
- `lla_tol=1e-6`；
- `tol=1e-6`；
- GPU 上的收敛比较尽量留在设备端，只同步最终布尔结果。

### 计算后端

- NumPy：NumPy 矩阵运算；
- CuPy：CuPy 矩阵运算和可用的 GPU 核函数；
- Torch：设备原生张量运算。

---

## 2. Proximal Newton

**文件**：`statgpu/solvers/_proximal_newton.py`

**用途**：光滑损失 + L2/无惩罚，并且普通 Newton 系统具有明确数学定义的场景。

对于一般的非光滑复合目标

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

真正的 Proximal Newton 步应在 Hessian 度量下求解近端子问题，例如

$$
\Delta_k
=\arg\min_{\Delta}
\left\{
\nabla\ell(\beta_k)^\top\Delta
+\frac12\Delta^\top H_k\Delta
+P(\beta_k+\Delta)
\right\}.
$$

直接把普通欧氏近端算子套在 Newton 步上会对应另一个复合目标。因此当前实现只在 L2/无惩罚的光滑路径上执行 Newton；非光滑请求会明确告警并交给 FISTA，直到实现正确的 Hessian 度量近端子问题。

### 算法

对当前迭代点 $\beta_k$，记完整光滑目标为

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

其中这里的 $P$ 仅为 L2 或 0。首先计算

$$
g_k=\nabla\ell(\beta_k)+\nabla P(\beta_k),
$$

以及

$$
H_k=\nabla^2\ell(\beta_k)+\nabla^2P(\beta_k).
$$

实现先将 Hessian 对称化，并加入一个很小的 ridge 稳定项：

$$
\widetilde H_k
=\frac12\left(H_k+H_k^\top\right)+10^{-10}I.
$$

若

$$
\|g_k\|_2\le \texttt{tol},
$$

则认为已经收敛。随后求解

$$
\widetilde H_k d_k=g_k,
$$

并以

$$
\beta_k(t)=\beta_k-t d_k
$$

作为试探点。若线性方程求解被识别为奇异或病态，当前实现不会调用最小二乘求解器，而是直接退回

$$
d_k=g_k,
$$

即最速下降。

由于更新采用“减去方向”的记号，下降方向应满足

$$
g_k^\top d_k>0.
$$

如果该内积非有限或不大于 0，同样改用

$$
d_k=g_k,
\qquad
g_k^\top d_k=\|g_k\|_2^2.
$$

### Armijo 回溯线搜索

从 $t_0=1$ 开始，寻找第一个满足

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-10^{-4}t\,g_k^\top d_k
$$

的步长。若条件不满足，则

$$
t\leftarrow \frac{t}{2}.
$$

最多尝试 25 次。第一个满足条件的候选点被接受：

$$
\beta_{k+1}=\beta_k-t d_k.
$$

如果 25 次试探都失败，则保持 $\beta_{k+1}=\beta_k$，发出线搜索失败警告并结束求解。

### 默认值与计算后端

- 默认 `max_iter=50`；
- 默认 `tol=1e-6`；
- 受支持的 NumPy/CuPy/Torch 路径使用对应的原生线性代数实现。

---

## 3. FISTA（快速迭代收缩阈值算法）

**文件**：`statgpu/solvers/_fista.py`

**用途**：复合目标

$$
F(\beta)=f(\beta)+P(\beta),
$$

其中 $f$ 光滑，而 $P$ 具有近端算子。

### 近端梯度更新

初始化

$$
\beta_0=y_0,\qquad t_0=1.
$$

在第 $k$ 次迭代的动量点 $y_k$ 上计算

$$
g_k=\nabla f(y_k).
$$

给定当前 Lipschitz 常数 $L_k$，步长为

$$
\gamma_k=\frac{1}{L_k},
$$

近端更新为

$$
\beta_{k+1}
=\operatorname{prox}_{\gamma_k P}
\left(y_k-\gamma_k g_k\right).
$$

这里

$$
\operatorname{prox}_{\gamma P}(v)
=\arg\min_x\left\{\gamma P(x)+\frac12\|x-v\|_2^2\right\}.
$$

### 二次上界与回溯

在需要回溯的路径上，令

$$
\Delta_k=\beta_{k+1}-y_k.
$$

候选点需要满足光滑部分的二次上界条件

$$
f(\beta_{k+1})
\le
f(y_k)+g_k^\top\Delta_k
+\frac{L_k}{2}\|\Delta_k\|_2^2+\varepsilon_{\rm slack}.
$$

若不满足，当前实现按

$$
L_k\leftarrow1.5L_k,
\qquad
\gamma_k\leftarrow\frac{1}{L_k}
$$

重新计算近端步，最多回溯 20 次。受支持的异步 GPU 非光滑路径为了避免每次回溯都发生设备同步，会使用经过安全放大的固定 $L_k$；因此该路径的数学更新仍是同一个近端步，但不会逐次执行上述 CPU 式回溯。

### Nesterov 动量

接受 $\beta_{k+1}$ 后，更新

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
$$

$$
y_{k+1}=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

典型的系数收敛判据为

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

对部分自适应惩罚路径，实现还会结合目标函数稳定性判据，避免仅因系数在相近目标值附近小幅振荡而误判。

### 加权路径

在受支持的带权路径上，数据拟合项按归一化解析权重计算。例如逐样本得分为 $\psi_i$ 时，梯度写成

$$
g(\beta)
=\frac{X^\top(s\odot\psi)}{\sum_i s_i}.
$$

带权目标函数与 Lipschitz 估计使用相同的权重定义。FISTA 具有带权实现，并不意味着所有模型组合都自动支持权重。

### 默认值

- 默认 `max_iter=500`；
- 默认 `tol=1e-6`。

---

## 4. FISTA-BB（Barzilai-Borwein）

**文件**：`statgpu/solvers/_fista_bb.py`

**用途**：在 FISTA 近端更新上使用 Barzilai-Borwein 曲率估计来选择步长，适合受支持的 GLM 稀疏惩罚路径。

### Lipschitz 预热与 BB 曲率

先由 Lipschitz 常数 $L$ 定义基准步长

$$
\gamma_L=\frac1L.
$$

在预热阶段使用

$$
\gamma_k=\gamma_L.
$$

预热结束后，令

$$
s_{k-1}=\beta_k-\beta_{k-1},
$$

$$
q_{k-1}=\nabla f(\beta_k)-\nabla f(\beta_{k-1}).
$$

当

$$
s_{k-1}^\top q_{k-1}>0
$$

且曲率信息数值有效时，交替使用

$$
\gamma_k^{\mathrm{BB1}}
=\frac{s_{k-1}^\top s_{k-1}}
{s_{k-1}^\top q_{k-1}},
$$

和

$$
\gamma_k^{\mathrm{BB2}}
=\frac{s_{k-1}^\top q_{k-1}}
{q_{k-1}^\top q_{k-1}}.
$$

默认步长边界为

$$
\gamma_{\min}=10^{-3}\gamma_L,
\qquad
\gamma_{\max}=10^3\gamma_L,
$$

因此选出的 BB 步长会再投影到

$$
\gamma_k
\leftarrow
\min\{\gamma_{\max},\max(\gamma_k,\gamma_{\min})\}.
$$

如果当前曲率对不满足有效性条件，就继续使用已有安全步长，而不是强行形成 BB 比值。

### 近端更新与保护性缩步

在动量点 $y_k$ 计算

$$
g_k=\nabla f(y_k),
$$

然后执行

$$
v_k=y_k-\gamma_k g_k,
$$

$$
\beta_{k+1}=\operatorname{prox}_{\gamma_kP}(v_k).
$$

对于非二次 GLM，当前实现还会定期检查候选点的目标函数与系数范数。如果保护性检查认为步长过大，则执行

$$
\gamma_k\leftarrow\frac{\gamma_k}{2}
$$

并重新计算同一个近端更新，最多尝试 15 次。这一检查是数值保护机制，不改变近端目标本身。

### Nesterov 动量与自适应重启

常规动量使用

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
$$

$$
y_{k+1}
=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

实现使用梯度型自适应重启判据。若

$$
\left(y_{k+1}-\beta_{k+1}\right)^\top
\left(\beta_{k+1}-\beta_k\right)>0,
$$

则清除动量：

$$
t_{k+1}=1,
\qquad
y_{k+1}=\beta_{k+1}.
$$

收敛检查使用

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

平方误差等二次损失不从 BB 曲率获得额外收益，因此保持固定 Lipschitz 步长的 FISTA 行为。SCAD、MCP 及其分组版本也禁用 BB 更新，因为非凸重加权会使割线曲率突然变化。

---

## 5. FISTA-LLA

**文件**：`statgpu/solvers/_fista_lla.py`

**用途**：SCAD、MCP 以及可以表示成局部加权凸惩罚的 LLA 路径。

### 延续路径

设延续参数依次为

$$
\alpha^{(0)}>\alpha^{(1)}>\cdots>\alpha^{(M)}=\alpha_{\rm target}.
$$

在每个 $\alpha^{(m)}$ 上运行 LLA 外循环。记当前 LLA 迭代为 $r$，系数为 $\beta^{(r)}$。

### LLA 权重

对标量非凸惩罚，局部线性近似使用

$$
d_j^{(r)}
=P_{\alpha^{(m)}}'\!\left(|\beta_j^{(r)}|\right).
$$

SCAD 的当前实现为

$$
d_j^{(r)}=
\begin{cases}
\alpha, & |\beta_j^{(r)}|\le\alpha,\\[3pt]
\dfrac{a\alpha-|\beta_j^{(r)}|}{a-1},
& \alpha<|\beta_j^{(r)}|\le a\alpha,\\[8pt]
0, & |\beta_j^{(r)}|>a\alpha,
\end{cases}
$$

而 MCP 为

$$
d_j^{(r)}=
\begin{cases}
\alpha-\dfrac{|\beta_j^{(r)}|}{\gamma},
& |\beta_j^{(r)}|\le\gamma\alpha,\\[8pt]
0, & |\beta_j^{(r)}|>\gamma\alpha.
\end{cases}
$$

当 GLM 通过增广列拟合截距时，截距坐标的 LLA 权重固定为 0，因此截距不参与惩罚。

### 凸近似问题

忽略与 $\beta$ 无关的常数项后，第 $r$ 次 LLA 将原非凸问题近似为

$$
Q_r(\beta)
=f(\beta)+\sum_j d_j^{(r)}|\beta_j|.
$$

因此默认内层就是一个加权 L1 问题。若调用方提供分组 LLA 工厂，则对应形式为

$$
Q_r(\beta)
=f(\beta)+\sum_g D_g^{(r)}\|\beta_g\|_2,
$$

内层改由相应的加权 Group Lasso 近端算子处理。

### FISTA 内层

在固定的 LLA 权重 $d^{(r)}$ 下，令

$$
\gamma_k=\frac{1}{L_k},
\qquad
g_k=\nabla f(y_k).
$$

先计算

$$
v_k=y_k-\gamma_k g_k.
$$

对于默认的加权 L1 内层，逐坐标近端更新为

$$
\beta_{k+1,j}
=S\!\left(v_{k,j},\gamma_k d_j^{(r)}\right),
$$

其中

$$
S(v,t)=\operatorname{sign}(v)\max(|v|-t,0).
$$

随后使用 Nesterov 动量

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
$$

$$
y_{k+1}
=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

内层典型收敛判据为

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

对非二次损失，当前实现会周期性重新估计 Lipschitz 常数。若新旧估计相差超过约 1.5 倍，则更新

$$
\gamma_k=\frac1{L_k}
$$

后继续迭代。平方误差、无权重 GPU 快速路径直接使用

$$
g_k=\frac{X^\top Xy_k-X^\top y}{n}
$$

避免重复计算两次矩阵乘法。

### LLA 外循环收敛

一次内层求解结束后得到 $\beta^{(r+1)}$。若

$$
\|\beta^{(r+1)}-\beta^{(r)}\|_1
<\texttt{lla\_tol},
$$

则结束当前 $\alpha^{(m)}$ 上的 LLA；否则重新计算 $d^{(r+1)}$ 并继续。随后以上一延续点的解作为下一 $\alpha$ 的起点。

当前通用复合路径默认使用 FISTA 内层。只有损失函数明确声明拥有正确的 Hessian 度量近端子问题时，才允许走 Proximal Newton 内层；Cox 路径目前仍保持 FISTA-LLA。

---

## 6. IRLS（迭代重加权最小二乘）

**实现**：`statgpu/glm_core/_irls.py`，以及少数损失函数自己的 `irls()` 方法。

### 分位数 IRLS

当前 `QuantileLoss.irls()` 使用代码中实际实现的 Frisch-Newton 风格重加权。若没有显式提供初始值，则从 OLS 初值开始。每次迭代先计算

$$
r_i=y_i-x_i^\top\beta,
$$

再计算

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf 1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

如果传入解析权重 `sample_weight=s`，先将其归一化为总和 $n$：

$$
\tilde s_i=\frac{n s_i}{\sum_j s_j},
$$

有效权重为

$$
w_i=\tilde s_i w_i^{\mathrm{IRLS}}.
$$

记 $W=\operatorname{diag}(w)$。无惩罚时，更新量由下面的加权最小二乘系统给出：

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}=X^\top W y.
$$

如果使用 L2 惩罚，则在左侧再加入对应的对角 ridge 项；当 `fit_intercept=True` 时，截距坐标不参与惩罚。收敛判据为

$$
\|\beta_{\mathrm{new}}-\beta\|_2<\texttt{tol}.
$$

### GLM IRLS

设链接函数为

$$
\eta=g(\mu),
\qquad
\mu=g^{-1}(\eta),
$$

分布族方差函数为 $V(\mu)$。在第 $k$ 次迭代，先计算

$$
\eta_i^{(k)}=x_i^\top\beta_k,
\qquad
\mu_i^{(k)}=g^{-1}(\eta_i^{(k)}).
$$

IRLS Fisher 工作权重为

$$
w_i^{\rm work}
=\frac{1}
{V(\mu_i^{(k)})\,[g'(\mu_i^{(k)})]^2},
$$

工作响应为

$$
z_i^{(k)}
=\eta_i^{(k)}
+\left(y_i-\mu_i^{(k)}\right)g'(\mu_i^{(k)}).
$$

若用户传入解析权重 $s_i$，实际最小二乘权重为

$$
w_i^{(k)}=s_i\,w_i^{\rm work}.
$$

因此解析 `sample_weight` 与 IRLS 工作权重是两个不同概念：前者来自统计目标，后者来自当前 GLM 二次近似。

令

$$
W_k=\operatorname{diag}\left(w_1^{(k)},\ldots,w_n^{(k)}\right).
$$

若 L2 对角惩罚矩阵为 $R$，额外二次惩罚矩阵为 $\Omega$，则 WLS 候选解满足

$$
\left(X^\top W_kX+R+\Omega\right)\widetilde\beta_{k+1}
=X^\top W_k z^{(k)}.
$$

线性方程只有在真正的秩失败时才退回最小二乘求解。

### IRLS 目标函数回溯

令

$$
\Delta_k=\widetilde\beta_{k+1}-\beta_k.
$$

从 $t=1$ 开始考察

$$
\beta_k(t)=\beta_k+t\Delta_k.
$$

当前实现要求注册的 GLM 目标函数不增加超过数值容差：

$$
F(\beta_k(t))
\le
F(\beta_k)+\varepsilon_F,
$$

其中

$$
\varepsilon_F
=\max\left(10^{-10}|F(\beta_k)|,10^{-6}\right).
$$

若不满足，则

$$
t\leftarrow\frac{t}{2},
$$

最多回溯 30 次。若仍找不到可接受候选点，则保留旧参数并报告线搜索失败。

### GLM IRLS 收敛判据

令

$$
u_i
=\frac{\mu_i-y_i}
{V(\mu_i)g'(\mu_i)}.
$$

有解析权重时使用 $s_i u_i$，并令

$$
n_{\rm eff}=\sum_i s_i;
$$

无权重时 $n_{\rm eff}=n$。归一化数据项得分为

$$
g_f=\frac{X^\top u}{n_{\rm eff}},
$$

再加上对应的 L2/二次惩罚梯度。当前实现以

$$
\|g_f\|_2<\texttt{tol}
$$

作为最终收敛判据，而不是仅凭一次被线搜索截短后的参数变化量判断收敛。

---

## 7. Newton-Raphson

**文件**：`statgpu/solvers/_newton.py`

**用途**：有 Hessian 的光滑损失 + L2/无惩罚。

### Newton 系统

记完整光滑目标为

$$
F(\beta)=\ell(\beta)+P(\beta).
$$

第 $k$ 次迭代计算

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

实现使用

$$
\widetilde H_k
=\frac12(H_k+H_k^\top)+10^{-10}I
$$

进行数值稳定化，并求解

$$
\widetilde H_k d_k=g_k.
$$

代码使用

$$
\beta_k(t)=\beta_k-t d_k
$$

作为试探点。如果直接线性求解发生真正的秩失败，普通 Newton 与 Proximal Newton 不同：这里会使用最小二乘解

$$
d_k=\widetilde H_k^{+}g_k,
$$

其中 $\widetilde H_k^{+}$ 表示由 `lstsq` 得到的广义逆意义解。

若

$$
g_k^\top d_k\le0
$$

或该内积非有限，则改用最速下降

$$
d_k=g_k.
$$

梯度范数满足

$$
\|g_k\|_2\le\texttt{tol}
$$

时停止。对于声明 Hessian 为常数的损失，Hessian 会在循环外计算一次并复用。

### Armijo 回溯

从 $t=1$ 开始，接受第一个满足

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-10^{-4}t\,g_k^\top d_k
$$

的候选点。失败时

$$
t\leftarrow\frac t2,
$$

最多尝试 20 次。如果没有候选点通过 Armijo 条件，则不接受任何未验证的小步，而是保留 $\beta_k$ 并报告线搜索失败。

### 解析 `sample_weight`

对明确提供带权曲率的路径，Newton 在目标函数值、梯度、Hessian 和每个 Armijo 试探点中都使用同一个归一化带权目标：

$$
F(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

因此，把所有有效权重同时乘以一个正数不会改变最优解。均匀权重或与均匀权重数值等价的情况，在定义了兼容路径时继续保持历史无权重数值行为。

---

## 8. L-BFGS / L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

**用途**：光滑目标的有限内存拟牛顿法，以及其盒约束投影版本。

### L-BFGS 曲率历史

记

$$
g_k=\nabla F(\beta_k),
$$

并在接受新点后定义

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k.
$$

只有当

$$
y_k^\top s_k>10^{-12}
$$

时才保存该曲率对，并令

$$
\rho_k=\frac1{y_k^\top s_k}.
$$

当前默认历史长度为

$$
m=10.
$$

超过 $m$ 后丢弃最旧的 $(s,y,\rho)$。

### 双循环递推

从

$$
q=g_k
$$

开始，对历史按从新到旧的顺序计算

$$
\alpha_i=\rho_i s_i^\top q,
\qquad
q\leftarrow q-\alpha_i y_i.
$$

若已有曲率历史，初始逆 Hessian 缩放为

$$
\gamma_k
=\frac{s_{k-1}^\top y_{k-1}}
{y_{k-1}^\top y_{k-1}},
$$

否则取 $\gamma_k=1$。令

$$
r=\gamma_k q.
$$

再按从旧到新的顺序计算

$$
\beta_i^{\rm loop}=\rho_i y_i^\top r,
$$

$$
r\leftarrow r+s_i
\left(\alpha_i-\beta_i^{\rm loop}\right).
$$

最终搜索方向为

$$
p_k=-r.
$$

如果

$$
g_k^\top p_k\ge0,
$$

则放弃该拟牛顿方向，改用

$$
p_k=-g_k.
$$

### L-BFGS Armijo 线搜索

从 $t=1$ 开始，接受第一个满足

$$
F(\beta_k+t p_k)
\le
F(\beta_k)+10^{-4}t\,g_k^\top p_k
$$

的候选点。失败时

$$
t\leftarrow\frac t2,
$$

最多回溯 25 次。

接受新点后重新计算 $g_{k+1}$ 并更新曲率历史。求解器在

$$
\|g_k\|_2<\texttt{tol}
$$

或

$$
\|s_k\|_2<\texttt{tol}
$$

时结束。

### L-BFGS-B：盒约束投影版本

当前 `lbfgs_b_solver` 是 projected-gradient 形式的 L-BFGS-B 路径，而不是带 generalized Cauchy point 的完整 Byrd–Lu–Nocedal–Zhu 算法。对盒约束

$$
\ell_j\le\beta_j\le u_j,
$$

定义投影

$$
\Pi_{[\ell,u]}(v)_j
=\min\{u_j,\max(\ell_j,v_j)\}.
$$

在活动边界上，如果梯度指向盒外，则投影梯度置零：

$$
\bar g_j=
\begin{cases}
0,& \beta_j\le\ell_j\ \text{且}\ g_j>0,\\
0,& \beta_j\ge u_j\ \text{且}\ g_j<0,\\
g_j,& \text{其他情况}.
\end{cases}
$$

两循环递推得到的方向也会删除所有会立即离开可行盒的分量。线搜索候选点为

$$
\beta_k(t)
=\Pi_{[\ell,u]}\left(\beta_k+t p_k\right),
$$

并使用同样的 Armijo 条件。收敛检查使用

$$
\|\bar g_k\|_2<\texttt{tol}.
$$

`lbfgs_b_solver` 当前只接受未传权重或均匀 `sample_weight`；不要把普通 `lbfgs_solver` 的非均匀 GLM 权重能力推断到 L-BFGS-B。

### L-BFGS 的解析 `sample_weight`

直接调用 L-BFGS 时，非均匀权重需要由损失函数明确支持：

| 直接 L-BFGS 路径 | 非均匀 `sample_weight` |
|---|---|
| 维护中的 `GLMLoss` | ✅ 支持 |
| 通用稳健 / 分位数 / Cox `LossBase` | ❌ 不能由无权重支持自动推出 |

对维护中的 GLM，初始梯度、当前目标函数、每个线搜索候选点和接受新点后的梯度都使用同一组归一化权重：

$$
F(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

---

## 9. ADMM（交替方向乘子法）

**文件**：`statgpu/solvers/_admm.py`

将

$$
\min_w f(w)+P(w)
$$

改写为一致性约束问题

$$
\min_{w,z} f(w)+P(z)
\quad\text{s.t.}\quad w=z.
$$

使用缩放对偶变量 $u$ 时，对应的增广拉格朗日可写为

$$
\mathcal L_\rho(w,z,u)
=f(w)+P(z)
+\frac{\rho}{2}\|w-z+u\|_2^2
-\frac{\rho}{2}\|u\|_2^2.
$$

### 外层更新

第 $k$ 次 ADMM 迭代为

$$
w^{k+1}
=\arg\min_w
\left\{
f(w)+\frac{\rho}{2}\|w-z^k+u^k\|_2^2
\right\},
$$

$$
z^{k+1}
=\operatorname{prox}_{P/\rho}(w^{k+1}+u^k),
$$

$$
u^{k+1}
=u^k+w^{k+1}-z^{k+1}.
$$

### $w$ 子问题：平方误差闭式路径

当损失具有常数 Hessian、特征数不超过当前 Cholesky 阈值时，预先分解

$$
A=\frac{X^\top X}{n}+\rho I,
$$

每次外层迭代只需求解

$$
Aw^{k+1}
=\frac{X^\top y}{n}+\rho(z^k-u^k).
$$

当前实现对该路径固定 $\rho$，避免预计算的 Cholesky 分解在改变 $\rho$ 后失效。

### $w$ 子问题：Nesterov 加速梯度路径

一般 GLM 使用内层加速梯度。记当前内层动量点为 $v_j$，则

$$
g_j
=\nabla f(v_j)+\rho(v_j-z^k+u^k).
$$

步长为

$$
\gamma
=\frac{1}{L_f+\rho+10^{-8}},
$$

并更新

$$
w_{j+1}=v_j-\gamma g_j.
$$

随后使用

$$
t_{j+1}=\frac{1+\sqrt{1+4t_j^2}}{2},
$$

$$
v_{j+1}
=w_{j+1}
+\frac{t_j-1}{t_{j+1}}(w_{j+1}-w_j).
$$

内层在

$$
\|w_{j+1}-w_j\|_1
<\texttt{cg\_tol}\times p
$$

时提前结束。虽然参数仍名为 `cg_max_iter` / `cg_tol`，当前非 Cholesky fallback 实际执行的是 Nesterov 加速梯度，而不是共轭梯度。

### 原始/对偶残差与自适应 $\rho$

定义

$$
r_{\rm p}^{k+1}
=\|w^{k+1}-z^{k+1}\|_2,
$$

$$
r_{\rm d}^{k+1}
=\rho\|z^{k+1}-z^k\|_2.
$$

若启用 `adaptive_rho=True`，当前规则为

$$
\rho\leftarrow
\begin{cases}
\min(2\rho,10^4),& r_{\rm p}>10r_{\rm d},\\
\max(\rho/2,10^{-4}),& r_{\rm d}>10r_{\rm p},\\
\rho,& \text{其他情况}.
\end{cases}
$$

改变 $\rho$ 后同步更新内层步长 $\gamma=1/(L_f+\rho+10^{-8})$。外层收敛要求

$$
r_{\rm p}<\texttt{tol}
\qquad\text{且}\qquad
r_{\rm d}<\texttt{tol}.
$$

最终返回 $z$，因为 $z$ 始终是应用惩罚近端算子后的变量。共享 `admm_solver` 当前只接受未传权重或均匀 `sample_weight`；真正非均匀解析权重不属于该入口的当前能力。

---

## 10. `exact`（闭式路径）

**实现位置**：`_fit_mixin._solve_exact_*`

**用途**：平方误差 + L2，并且自动分发选择闭式/特征分解路径的情形。对应的基本线性系统形如

$$
\left(\frac{X^\top X}{n}+\alpha I\right)\beta
=\frac{X^\top y}{n},
$$

截距处理由具体模型单独完成。

---

## 求解器调度

对普通直接拟合，`solver="auto"` 按模型层面的维护表分发。简化表示为：

```text
直接拟合，solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + 稀疏惩罚       → FISTA/FISTA-BB
├── 光滑非高斯 GLM + L2            → Newton
├── SCAD/MCP/adaptive 路径          → LLA + FISTA 系列内层求解器
├── 分位数                          → 分位数专用 FISTA/IRLS 路径
└── 组惩罚                          → Group FISTA / FISTA-LLA
```

`PenalizedGLM_CV` 的光滑 L2 分发与直接拟合相关但有意独立。尤其是 Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 交叉验证/最终重拟合路径使用 L-BFGS，而 logistic、Poisson、Tweedie 的 L2 组合使用 Newton。不要从直接拟合的分发树推断交叉验证行为，应以兼容性矩阵为准。

`sample_weight` 不会改变显式指定的 `solver`。不支持的带权组合会直接报错，而不是选择另一个求解器。

## 参考文献

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Liu, D. C. & Nocedal, J. (1989). On the Limited Memory BFGS Method for Large Scale Optimization. *Mathematical Programming*, 45, 503-528.
- Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. (1995). A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM J. Scientific Computing*, 16(5), 1190-1208.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in Machine Learning*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.
