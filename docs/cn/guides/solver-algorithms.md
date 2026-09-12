# 求解器算法

> 语言：中文  
> 最后更新：2026-09-13  
> 页面定位：算法参考  
> 切换：[英文版](../../en/guides/solver-algorithms.md)

## 概览

statgpu 提供一阶、二阶、近端和闭式等多类求解器。对大多数模型用户，建议先使用 `solver="auto"`；本页属于算法参考，因此会保留具体更新公式、收敛条件、计算后端行为和重要的支持边界。

阅读本页时可以先记住三点：

1. 支持某个计算后端，不代表所有损失函数、惩罚项和权重组合都受支持。
2. `sample_weight` 不会改变显式指定的 `solver`。如果所请求的带权组合不受支持，则直接报错。
3. 权重支持取决于完整的模型路径。尤其是直接调用 L-BFGS 时，非均匀权重目前由维护中的 GLM 损失明确支持，并不会自动扩展到所有 `LossBase`。

模型层面的完整调度表见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 求解器总览

| 求解器 | 最适合 | 后端支持 |
|--------|--------|:---:|
| Proximal IRLS-CD | 分位数回归 + SCAD/MCP | NumPy, CuPy, Torch |
| Proximal Newton | 光滑损失 + L2/无惩罚；非光滑请求使用 FISTA | NumPy, CuPy, Torch |
| FISTA | 一般非光滑惩罚 | NumPy, CuPy, Torch |
| FISTA-BB | GLM + 稀疏惩罚 | NumPy, CuPy, Torch |
| FISTA-LLA | 非凸惩罚的延续/LLA 路径 | NumPy, CuPy, Torch |
| IRLS | 具有维护中 IRLS 表示的损失 | NumPy, CuPy, Torch |
| Newton | 有 Hessian 矩阵的光滑损失 | NumPy, CuPy, Torch |
| L-BFGS | 光滑损失、中等参数维度 | NumPy, CuPy, Torch |
| L-BFGS-B | 带盒约束的光滑问题 | NumPy, CuPy, Torch |
| ADMM | 可分/近端形式 | NumPy, CuPy, Torch |
| `exact` | 平方误差 + L2 闭式路径 | NumPy, CuPy, Torch |

后端列只描述数值实现能力；具体模型和损失函数还会进一步限制可用组合。

---

## 1. Proximal IRLS-CD

**文件**：`statgpu/solvers/_proximal_irls_quantile.py`

**用途**：分位数回归 + SCAD/MCP。它把 check（又称 pinball）损失的 IRLS 二次上界与非凸惩罚的局部线性近似（LLA）结合起来。

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

实现先将 Hessian 对称化，并加入一个很小的 Ridge 稳定项：

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

预热阶段固定使用 $\gamma_L$。之后，令

$$
s_k=\beta_k-\beta_{k-1},
\qquad
q_k=\nabla f(\beta_k)-\nabla f(\beta_{k-1}).
$$

仅当

$$
s_k^\top q_k>\texttt{curvature\_tol}
$$

时接受这组割线信息。两种 BB 步长为

$$
\gamma_k^{\mathrm{BB1}}
=\frac{s_k^\top s_k}{s_k^\top q_k},
$$

$$
\gamma_k^{\mathrm{BB2}}
=\frac{s_k^\top q_k}{q_k^\top q_k}.
$$

当前实现交替使用 BB1/BB2，并把候选步长限制在

$$
10^{-3}\gamma_L
\le
\gamma_k
\le
10^3\gamma_L.
$$

### 近端更新与回溯保护

在动量点 $y_k$ 上计算

$$
\beta_{k+1}
=\operatorname{prox}_{\gamma_k P}
\left(y_k-\gamma_k\nabla f(y_k)\right).
$$

对于非二次损失，周期性检查目标值和更新范数。如果当前 BB 步过于激进，则按

$$
\gamma_k\leftarrow\frac{\gamma_k}{2}
$$

回溯，最多 15 次。平方误差路径禁用 BB 更新，保留固定 Lipschitz 步长的 FISTA 行为。

### Nesterov 动量与自适应重启

动量参数仍使用

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

并构造外推点。若当前外推方向与最新更新方向出现不利的内积关系，即

$$
(y_{k+1}-\beta_{k+1})^\top
(\beta_{k+1}-\beta_k)>0,
$$

则触发自适应重启，把动量重置到当前点。

### 默认行为

- 二次损失：固定 Lipschitz FISTA；
- 非二次损失：预热后交替 BB1/BB2；
- SCAD/MCP 与非凸分组惩罚：不启用通用 BB 路径；
- 收敛判据使用系数差的 L1 范数。

---

## 5. FISTA-LLA（非凸惩罚）

**文件**：`statgpu/solvers/_fista_lla.py`

**用途**：SCAD、MCP、自适应 L1 以及对应分组非凸惩罚的延续/局部线性近似路径。

### LLA 子问题

在第 $m$ 次 LLA 外迭代中，对当前系数 $\beta^{(m)}$ 计算

$$
d_j^{(m)}=P'\left(|\beta_j^{(m)}|\right).
$$

然后用加权 L1 近似原非凸惩罚：

$$
P(\beta)
\approx
\sum_j d_j^{(m)}|\beta_j|.
$$

默认的凸近似对象是

```python
AdaptiveL1Penalty(alpha=1.0, weights=lla_w)
```

因此 SCAD/MCP 的导数权重已经包含目标 `alpha` 的尺度；这里的内部 `alpha=1.0` 不会再次乘一遍目标强度。

若 GLM 使用增强设计矩阵来表示截距，截距坐标会追加一个 0 权重，所以截距不参与 LLA 惩罚。

### SCAD 导数

$$
d_j=
\begin{cases}
\alpha, & |\beta_j|\le\alpha,\\
\dfrac{a\alpha-|\beta_j|}{a-1},
& \alpha<|\beta_j|\le a\alpha,\\
0, & |\beta_j|>a\alpha.
\end{cases}
$$

### MCP 导数

$$
d_j=
\begin{cases}
\alpha-\dfrac{|\beta_j|}{\gamma},
& |\beta_j|\le\gamma\alpha,\\
0, & |\beta_j|>\gamma\alpha.
\end{cases}
$$

### 内层 FISTA

对当前加权 L1 子问题，初始化

$$
y_0=\beta_0,
\qquad
t_0=1,
\qquad
\gamma_0=\frac1{L_0}.
$$

每次计算损失梯度，然后应用带权软阈值

$$
\beta_{k+1,j}
=S\left(y_{k,j}-\gamma_k g_{k,j},
\gamma_k d_j^{(m)}\right).
$$

之后使用标准 Nesterov 动量更新。系数收敛检查为

$$
\sum_j|\beta_{k+1,j}-\beta_{k,j}|<\texttt{tol}.
$$

对非二次损失，当前实现每 20 次内迭代重新估计一次 Lipschitz 常数；只有新估计与当前值相差超过约 1.5 倍时才更新。

平方误差且无解析权重的 GPU 快速路径会预计算

$$
X^\top X,
\qquad
X^\top y,
$$

并用

$$
g(y_k)
=\frac{X^\top Xy_k-X^\top y}{n}
$$

避免每次都重新执行完整数据乘法。一般损失和带权路径使用共享损失对象的梯度接口。

### LLA 外层收敛

若

$$
\sum_j|\beta_j^{(m+1)}-\beta_j^{(m)}|
<\texttt{lla\_tol},
$$

则当前延续点的 LLA 收敛。

各 `alpha` 延续点之间沿用上一点的系数状态。不同惩罚族还可能定义专用初值/热启动策略，但不会改变上述 LLA 子问题定义。

### 分组 LLA

分组 SCAD/MCP 会把标量 $|\beta_j|$ 替换为组范数 $\|\beta_g\|_2$，并构造自适应分组 Lasso 的凸近似。分组权重按惩罚关于组范数的导数生成，截距仍保持零惩罚。

---

## 6. IRLS（迭代重加权最小二乘）

**文件**：`statgpu/solvers/_irls.py`

**用途**：支持 IRLS 表示的 GLM 或其他维护中的损失函数。

### 一般 GLM 工作响应

记

$$
\eta_i=x_i^\top\beta,
\qquad
\mu_i=g^{-1}(\eta_i),
$$

其中 $g$ 是链接函数。当前实现对非恒等链接的 $\eta_i$ 做数值裁剪，再计算 $\mu_i$ 并施加分布族定义域裁剪。

工作权重为

$$
W_i^{\mathrm{work}}
=\frac{1}{V(\mu_i)[g'(\mu_i)]^2},
$$

工作响应为

$$
z_i
=\eta_i+(y_i-\mu_i)g'(\mu_i).
$$

若传入解析权重 $s_i$，则实际 WLS 权重为

$$
W_i=s_iW_i^{\mathrm{work}}.
$$

### 加权最小二乘子问题

每轮求解

$$
\left(
X^\top W X
+\lambda_{\rm ridge}D
+M
\right)\beta_{\mathrm{WLS}}
=X^\top Wz,
$$

其中：

- $D$ 是 Ridge 对角矩阵，截距位置可设为 0；
- $M$ 是可选附加惩罚矩阵；
- 线性系统优先直接求解；只有真正的秩失败才退回最小二乘求解。

公共目标使用平均损失，而底层 WLS 正规方程本身是未归一化求和。因此 `_fit_irls()` 会按目标函数尺度修正 Ridge 强度：

$$
\lambda_{\rm WLS}
=\lambda_{\rm public}
\times
\begin{cases}
n, & \text{无解析权重},\\
\sum_i s_i, & \text{有解析权重}.
\end{cases}
$$

### 线搜索

记

$$
\Delta_k
=\beta_{\mathrm{WLS}}-\beta_k.
$$

从 $t=1$ 开始尝试

$$
\beta_{\mathrm{try}}
=\beta_k+t\Delta_k.
$$

候选点必须满足

$$
F(\beta_{\mathrm{try}})
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
t\leftarrow\frac t2,
$$

最多回溯 30 次。若所有试探点都失败，恢复旧参数并发出警告。

### 收敛判据

实现使用归一化的带惩罚得分，而不是简单的系数差：

$$
\frac{\|g(\beta_k)\|_\infty}
{1+\|\beta_k\|_\infty}
<\texttt{tol}.
$$

默认值：

- `max_iter=100`；
- `tol=1e-8`。

---

## 7. Newton

**文件**：`statgpu/solvers/_newton.py`

**用途**：光滑损失 + 光滑惩罚。

### 完整目标

$$
F(\beta)=\ell(\beta)+P(\beta).
$$

每轮计算

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

然后构造

$$
\widetilde H_k
=\frac12(H_k+H_k^\top)+10^{-10}I.
$$

若

$$
\|g_k\|_2\le\texttt{tol},
$$

则停止。

### Newton 系统

求解

$$
\widetilde H_kd_k=g_k.
$$

与 Proximal Newton 不同，这里的普通 Newton 在直接线性求解发生真正秩失败时，会使用最小二乘求解

$$
d_k
=\arg\min_d
\|\widetilde H_kd-g_k\|_2.
$$

如果得到的方向非有限，或不是下降方向，即

$$
g_k^\top d_k\le0,
$$

则改用

$$
d_k=g_k.
$$

### Armijo 线搜索

更新写成

$$
\beta_{k+1}=\beta_k-t_kd_k.
$$

从 $t_k=1$ 开始，寻找满足

$$
F(\beta_k-t_kd_k)
\le
F(\beta_k)-10^{-4}t_kg_k^\top d_k
$$

的步长；不满足时每次减半，最多 20 次。若线搜索失败，则不接受未经验证的更新。

对 Hessian 恒定的目标，实现会复用 Hessian，避免重复构造。

### 带权目标

在支持解析权重的路径上，Newton 的函数值、梯度、Hessian 和 Armijo 试探点使用同一个归一化带权目标。例如 GLM 路径使用

$$
F_w(\beta)
=\frac{\sum_i s_i\ell_i(\beta)}{\sum_i s_i}
+P(\beta).
$$

因此 `sample_weight` 不会只影响梯度而遗漏函数值或 Hessian。

---

## 8. L-BFGS

**文件**：`statgpu/solvers/_lbfgs.py`

**用途**：光滑目标；只保存有限数量的割线对，不显式形成完整 Hessian。

### 割线对

记

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k.
$$

当

$$
y_k^\top s_k>10^{-12}
$$

时保存该割线对，并定义

$$
\rho_k=\frac{1}{y_k^\top s_k}.
$$

默认历史长度为 10。

### 两循环递推

令

$$
q=g_k.
$$

按从新到旧的顺序计算

$$
\alpha_i
=\rho_i s_i^\top q,
\qquad
q\leftarrow q-\alpha_i y_i.
$$

然后使用初始逆 Hessian 尺度

$$
\gamma_k
=\frac{s_{m}^\top y_m}{y_m^\top y_m},
$$

如果分母过小则取 $\gamma_k=1$。令

$$
r=\gamma_kq,
$$

再按从旧到新的顺序计算

$$
\beta_i
=\rho_i y_i^\top r,
$$

$$
r\leftarrow r+s_i(\alpha_i-\beta_i).
$$

最终搜索方向为

$$
p_k=-r.
$$

若

$$
g_k^\top p_k\ge0,
$$

则改用最速下降方向

$$
p_k=-g_k.
$$

### Armijo 线搜索

从 $t=1$ 开始，检查

$$
F(\beta_k+tp_k)
\le
F(\beta_k)+10^{-4}t g_k^\top p_k.
$$

若不满足，则步长减半，最多 25 次。成功后再计算新梯度并更新割线对。

### 收敛判据

任一条件满足即可停止：

$$
\|g_k\|_2<\texttt{tol},
$$

或

$$
\|\beta_{k+1}-\beta_k\|_2<\texttt{tol}.
$$

### 非均匀解析权重

直接 L-BFGS 的非均匀 `sample_weight` 支持由损失函数显式声明。当前 `LossBase` 默认拒绝这一能力，`GLMLoss` 明确开启。

对受支持的 GLM，初始梯度、当前函数值、每个 Armijo 试探点和接受新点后的梯度都使用同一个归一化带权目标。因此不会出现“梯度带权但线搜索函数值无权”的不一致。

---

## 9. L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs_b.py`

**用途**：光滑盒约束问题

$$
l_j\le\beta_j\le u_j.
$$

当前实现是投影梯度 + 有限内存 BFGS 的变体，不是包含广义 Cauchy 点和子空间最小化的完整经典 L-BFGS-B 实现。

### 投影梯度

定义

$$
g_j^{\mathrm{proj}}
=\begin{cases}
0,
& \beta_j=l_j\ \text{且}\ g_j>0,\\
0,
& \beta_j=u_j\ \text{且}\ g_j<0,\\
g_j,
& \text{其他}.
\end{cases}
$$

即位于边界并继续指向不可行方向的梯度分量被置零。

### 方向与候选点

用投影梯度进入两循环递推，得到方向 $p_k$ 后，候选点为

$$
\beta_{\mathrm{try}}
=\Pi_{[l,u]}(\beta_k+t p_k),
$$

其中 $\Pi$ 表示逐坐标盒投影。

Armijo 线搜索最多 25 次。收敛判据使用投影梯度范数。

### 权重边界

当前共享 `lbfgs_b_solver` 只接受未传或均匀的 `sample_weight`；真正非均匀权重会在拟合前报错。

---

## 10. ADMM

**文件**：`statgpu/solvers/_admm.py`

**用途**：把光滑损失与可近端惩罚拆成共识问题

$$
\min_{w,z}
f(w)+P(z)
\quad\text{s.t.}\quad
w=z.
$$

缩放形式的增广拉格朗日函数为

$$
\mathcal L_\rho(w,z,u)
=f(w)+P(z)
+\frac\rho2\|w-z+u\|_2^2
-\frac\rho2\|u\|_2^2.
$$

### 外循环更新

1. **$w$ 子问题**

   $$
   w^{k+1}
   =\arg\min_w
   f(w)+\frac\rho2\|w-z^k+u^k\|_2^2.
   $$

2. **$z$ 子问题**

   $$
   z^{k+1}
   =\operatorname{prox}_{P/\rho}
   (w^{k+1}+u^k).
   $$

3. **对偶更新**

   $$
   u^{k+1}
   =u^k+w^{k+1}-z^{k+1}.
   $$

### 平方误差快速路径

当损失为无权重平方误差且维度 $p\le2000$ 时，固定 $\rho$，预计算

$$
\frac{X^\top X}{n}+\rho I
$$

的 Cholesky 分解。此路径默认关闭自适应 $\rho$，从而反复复用同一分解。

### 一般 $w$ 子问题

非 Cholesky 路径当前使用 Nesterov 加速梯度，而不是共轭梯度。代码中的部分 `cg_*` 参数名是历史接口，不应据此把当前实现理解成 CG 求解器。

### 残差

原始残差

$$
r_p^k=\|w^k-z^k\|_2,
$$

对偶残差

$$
r_d^k=\rho\|z^k-z^{k-1}\|_2.
$$

两者都小于 `tol` 时收敛。

### 自适应 $\rho$

在允许自适应的路径上：

- 若 $r_p>10r_d$，增大 $\rho$；
- 若 $r_d>10r_p$，减小 $\rho$；
- 调整倍率为 2，并限制在维护中的上下界内；
- 修改 $\rho$ 时同步缩放对偶变量，使缩放形式的增广拉格朗日语义保持一致。

### 返回值与权重

求解结束后返回 $z$ 作为系数估计。

当前共享 `admm_solver` 只接受未传或均匀的 `sample_weight`。真正非均匀权重会在进入数值迭代前报错；不能从 `LossBase` 的带权函数值/梯度能力推断 ADMM 已支持同样权重。

---

## 11. `exact` 路径

`solver="exact"` 不是一个通用迭代优化器；它是平方误差 + L2 的专用闭式/线性代数路径。

设增强设计矩阵为 $X$，惩罚矩阵为 $D$，则求解

$$
\left(X^\top X+\lambda D\right)\beta
=X^\top y.
$$

带解析权重时，对应

$$
\left(X^\top W X+\lambda D\right)\beta
=X^\top Wy,
$$

并使用与公共目标函数一致的权重尺度。

CPU 路径可以使用闭式/特征分解实现；GPU 的 `solver="auto"` 对平方误差 + L2 当前通常转到 Newton，而不是强制使用同一个闭式实现。

---

## 12. 计算后端与数值边界

### NumPy

- CPU 线性代数和数组计算；
- 可使用 NumPy/SciPy 辅助操作，但求解器支持范围仍由模型路径决定。

### CuPy

- 数值迭代保持在 CUDA 数组上；
- 不应仅为收敛判断或函数值评估而复制完整数组到 CPU；
- 允许同步最终标量、布尔结果和小型元数据。

### Torch

- 使用 Torch 原生张量和 `torch.linalg`；
- 显式 CUDA 设备应保持具体设备序号；
- 异构输入容器必须先按公开 `device` 契约转换，再进入数值求解。

## 13. 求解器选择总结

| 场景 | 推荐求解器 |
|------|-----------|
| 平方误差 + L2（CPU） | `exact` / `auto` |
| 平方误差 + L2（GPU） | Newton / `auto` |
| 光滑 GLM + L2/无惩罚 | Newton；部分交叉验证路径使用 L-BFGS |
| 光滑目标且不希望显式形成 Hessian | L-BFGS |
| L1 / ElasticNet | FISTA / FISTA-BB |
| SCAD / MCP / 自适应 L1 | FISTA-LLA / 专用延续路径 |
| 分位数 + SCAD/MCP | Proximal IRLS-CD |
| 分组稀疏 / 分组非凸 | 分组 FISTA / 分组 FISTA-LLA |
| 盒约束光滑问题 | L-BFGS-B |
| 适合变量分裂的近端形式 | ADMM |

若只是在模型层选择算法，优先从 `solver="auto"` 开始；本页主要用于理解实现、比较显式求解器和检查支持边界。