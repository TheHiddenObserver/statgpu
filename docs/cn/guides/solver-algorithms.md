# 求解器算法

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：算法参考  
> 切换：[English](../../en/guides/solver-algorithms.md)

## 概览

statgpu 提供一阶、二阶、近端和闭式等多类求解器。对大多数模型用户，建议先使用 `solver="auto"`；本页属于算法参考，因此会保留具体更新公式、收敛条件、计算后端行为和重要的支持边界。

阅读本页时可以先记住三点：

1. 后端支持并不意味着所有损失函数、惩罚项和权重组合都受支持。
2. `sample_weight` 不会改写显式指定的求解器；如果请求的带权组合不受支持，拟合应明确报错。
3. 权重支持属于具体路径能力。例如，非均匀 direct L-BFGS 权重由维护中的 GLM 路径支持，并不能自动推广到所有 `LossBase`。

模型层的实际分派请查看 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 求解器摘要

| 求解器 | 最适合 | 后端支持 |
|--------|--------|:---:|
| Proximal IRLS-CD | Quantile + SCAD/MCP | NumPy、CuPy、Torch |
| Proximal Newton | 光滑损失 + L2/无惩罚；非光滑请求转交 FISTA | NumPy、CuPy、Torch |
| FISTA | 凸近端/维护中的一阶路径，包括显式 Quantile L2/none | NumPy、CuPy、Torch |
| FISTA-BB | 具有光滑梯度差分结构的 GLM + 稀疏惩罚 | NumPy、CuPy、Torch |
| FISTA-LLA | continuation/LLA 非凸惩罚路径 | NumPy、CuPy、Torch |
| IRLS | 具有维护中 IRLS 表示的损失 | NumPy、CuPy、Torch |
| Newton | 有 Hessian 的光滑损失 | NumPy、CuPy、Torch |
| L-BFGS | 光滑损失、中等维度 | NumPy、CuPy、Torch |
| L-BFGS-B | 带盒约束的光滑问题 | NumPy、CuPy、Torch |
| ADMM | 可分/近端形式 | NumPy、CuPy、Torch |
| `exact` | 平方误差 + L2 闭式路径 | NumPy、CuPy、Torch |

后端列只说明数值实现能力，模型与损失函数的公开契约还会进一步缩窄合法组合。Quantile/check loss 就是一个典型例子：普通 FISTA 既维护于凸稀疏 Quantile 路径，也可显式用于 L2/无惩罚 Quantile；但 L2/none 的 `solver="auto"` 仍优先 Quantile IRLS。FISTA-BB 和共享 ADMM 继续排除 Quantile。底层 direct L-BFGS 保留未传/均匀权重的历史 Quantile 兼容面，但 estimator/CV 层 `solver="lbfgs"` 仍不支持，真正非均匀 Quantile L-BFGS 权重也会 fail closed。

---

## 1. Proximal IRLS-CD

**文件**：`statgpu/solvers/_proximal_irls_quantile.py`

**适用场景**：带 SCAD/MCP 惩罚的分位数回归。算法把 pinball loss 的 IRLS 二次上界与非凸惩罚的局部线性近似（LLA）结合起来。

### 算法

对 continuation 中每个 `alpha`：

1. **LLA 外循环。** 计算局部惩罚导数

   $$
   d_j=P'(|\beta_j|),
   $$

   阈值为

   $$
   t_j=n\,d_j.
   $$

2. **IRLS-CD 内循环。** 对残差

   $$
   r_i=y_i-x_i^\top\beta,
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

   若传入解析权重 `sample_weight=s`，先归一化为

   $$
   \tilde s_i=\frac{n s_i}{\sum_j s_j},
   $$

   再使用

   $$
   w_i=\tilde s_i\,w_i^{\mathrm{IRLS}}.
   $$

3. **并行对角上界。** 记 $W=\operatorname{diag}(w)$。实现计算

   $$
   g=X^\top W(y-X\beta),
   \qquad
   h=\operatorname{diag}(X^\top W X),
   $$

   然后

   $$
   u=g+h\odot\beta,
   $$

   并行更新全部坐标：

   $$
   \beta_j^{\mathrm{new}}=\frac{S(u_j,t_j)}{h_j},
   $$

   其中

   $$
   S(u,t)=\operatorname{sign}(u)\max(|u|-t,0).
   $$

   因而这里是 Jacobi 风格的并行对角上界更新，而不是逐坐标循环扫过。

4. **截距与平坦 LLA surrogate。** Quantile loss 不是二次损失，因此不能通过对 $X$、$y$ 做均值中心化消去截距。`fit_intercept=True` 时，数值设计矩阵增加一列 1，截距作为 pinball 目标的普通坐标直接优化，且其局部惩罚阈值固定为 0。

   若所有 feature-side LLA 导数都恰好为 0，

   $$
   d_1=\cdots=d_p=0,
   $$

   当前 SCAD/MCP surrogate 已无激活惩罚，退化为普通加权分位数回归。这时求解器使用维护中的完整 `QuantileLoss.irls()` WLS 更新来闭合 surrogate，而不是继续对角 Jacobi 近似。平坦 surrogate 的 IRLS tolerance 为

   $$
   \min(\texttt{tol},10^{-8}),
   $$

   与维护中的 Quantile IRLS 精度约定一致。只要存在 $d_j>0$，仍执行普通 Proximal IRLS-CD 内循环。

5. **收敛。** 真正带惩罚的内循环检查

   $$
   \|\beta^{\mathrm{new}}-\beta\|_\infty<\texttt{tol},
   $$

   完全平坦的 surrogate 使用 Quantile IRLS 的 $\ell_2$ 参数变化准则。LLA 外循环检查

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_\infty<\texttt{lla\_tol}.
   $$

### Continuation 与默认值

- continuation path：`lambda_max` 到目标 `alpha`；
- `max_lla_per_step=2`；
- `lla_tol=1e-6`；
- `tol=1e-6`；
- GPU 收敛比较留在设备端，只同步最终布尔值。

---

## 2. Proximal Newton

**文件**：`statgpu/solvers/_proximal_newton.py`

**适用场景**：光滑损失 + L2/无惩罚，此时普通 Newton 系统定义良好。

对真正的非光滑复合目标

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

真正的 Proximal Newton 步应该求解 Hessian 度量下的近端子问题，例如

$$
\Delta_k
=\arg\min_{\Delta}
\left\{
\nabla\ell(\beta_k)^\top\Delta
+\frac12\Delta^\top H_k\Delta
+P(\beta_k+\Delta)
\right\}.
$$

如果把普通欧氏 prox 直接套在 Newton 步上，会优化另一个复合目标。因此当前维护实现只在 L2/无惩罚光滑路径上执行 Newton；非光滑请求会 warning 后转交 FISTA，直到真正的 Hessian-metric proximal 子问题实现为止。

### 完整光滑目标

维护路径使用

$$
F(\beta)=L(\beta)+P(\beta).
$$

损失支持解析 `sample_weight=w` 时，数据拟合项始终使用同一个归一化加权目标：

$$
L(\beta)
=\frac{1}{s}\sum_{i=1}^n w_i\,\ell_i(\eta_i),
\qquad
\eta_i=x_i^\top\beta,
\qquad
s=\sum_i w_i.
$$

无权重时 $w_i=1$、$s=n$。对于可以写成逐样本 score/curvature 的损失，记

$$
\psi_i(\beta)=\frac{\partial\ell_i}{\partial\eta_i},
\qquad
h_i(\beta)=\frac{\partial^2\ell_i}{\partial\eta_i^2},
$$

则

$$
\nabla L(\beta)
=\frac{X^\top\!\left(w\odot\psi(\beta)\right)}{s},
$$

$$
\nabla^2L(\beta)
=\frac{X^\top\operatorname{diag}\!\left(w\odot h(\beta)\right)X}{s}.
$$

对不天然使用逐样本曲率分解的结构化损失，求解器直接调用损失对象的 `hessian()` 或 `fused_gradient_and_hessian()`。

当前 Newton 路径只接受 L2 或无惩罚。L2 约定为

$$
P(\beta)=\frac{\alpha}{2}\|\beta\|_2^2,
\qquad
\nabla P(\beta)=\alpha\beta,
\qquad
\nabla^2P(\beta)=\alpha I.
$$

因此

$$
g_k=\nabla F(\beta_k)=\nabla L(\beta_k)+\alpha\beta_k,
$$

$$
H_k=\nabla^2F(\beta_k)=\nabla^2L(\beta_k)+\alpha I.
$$

无惩罚时取 $\alpha=0$。

### 稳定化 Newton 系统

实现先对 Hessian 对称化，

$$
\bar H_k=\frac12(H_k+H_k^\top),
$$

再加入固定数值稳定项

$$
\widetilde H_k=\bar H_k+10^{-10}I.
$$

若

$$
\|g_k\|_2\le\texttt{tol},
$$

则停止；否则解

$$
\widetilde H_kd_k=g_k.
$$

实现使用“减去方向”的记号，候选点为

$$
\beta_k(t)=\beta_k-td_k.
$$

若线性系统被识别为真正奇异/病态，Proximal Newton 不调用最小二乘，而回退到最速下降：

$$
d_k=g_k.
$$

定义下降量

$$
q_k=g_k^\top d_k.
$$

由于候选点是 $\beta_k-td_k$，下降方向要求

$$
q_k>0.
$$

若 $q_k$ 非有限或不大于 0，则再次使用

$$
d_k=g_k,
\qquad
q_k=\|g_k\|_2^2.
$$

### Armijo 回溯

从

$$
t_0=1
$$

开始，每次失败步长减半：

$$
t_m=2^{-m},
\qquad m=0,1,\ldots,24.
$$

接受第一个满足完整复合目标 Armijo 条件的候选：

$$
F(\beta_k-t_md_k)
\le F(\beta_k)-10^{-4}t_mq_k.
$$

更新为

$$
\beta_{k+1}=\beta_k-t_md_k.
$$

这里 $F=L+P$ 在当前点和每个 trial point 都包含 L2 惩罚值。因为 L2 梯度和曲率已经包含在 $g_k$、$H_k$ 中，trial point 不会额外再做一次欧氏 prox，否则会重复计算 L2 惩罚。

若 25 个候选步长全部失败，求解器恢复

$$
\beta_{k+1}=\beta_k,
$$

发出线搜索 warning 并停止。trial point 上被识别为数值定义域失败的情况只会拒绝当前候选并继续回溯；设备、输入契约等非数值错误仍直接暴露给调用者。

### 初始化、默认值与后端

未提供 `init_coef` 时，

$$
\beta_0=0.
$$

默认：

- `max_iter=50`；
- `tol=1e-6`；
- NumPy/CuPy/Torch 的受支持路径使用各自原生线性代数。

因此当前名为 `proximal_newton_solver` 的 L2/无惩罚路径，在数值上是**带 Hessian 稳定化和 Armijo 回溯的 damped Newton**。真正的非光滑 Hessian-metric Proximal-Newton 子问题尚未实现；非光滑惩罚会在这些 Newton 迭代开始前转交 FISTA。

---

## 3. FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）

**文件**：`statgpu/solvers/_fista.py`

**适用场景**：经典复合优化设定为

$$
F(\beta)=f(\beta)+P(\beta),
$$

其中 $f$ 光滑且 $P$ 有近端算子。statgpu 另外维护显式 Quantile L2/无惩罚以及凸稀疏的一阶路径：该路径把注册的 Quantile 次梯度送入同一 FISTA engine。由于 check loss 非光滑，这属于实现层维护契约，并不表示经典 smooth-gradient FISTA 收敛定理的假设成立；因此 L2/none 下 `solver="auto"` 仍继续优先 Quantile IRLS。

### 近端梯度更新

初始化

$$
\beta_0=y_0,\qquad t_0=1.
$$

在动量点 $y_k$ 计算注册的梯度或维护中的次梯度

$$
g_k=\nabla f(y_k).
$$

当前步长尺度为 $L_k$ 时，

$$
\gamma_k=\frac1{L_k},
$$

执行近端步

$$
\beta_{k+1}
=\operatorname{prox}_{\gamma_kP}
\left(y_k-\gamma_kg_k\right),
$$

其中

$$
\operatorname{prox}_{\gamma P}(v)
=\arg\min_x\left\{\gamma P(x)+\frac12\|x-v\|_2^2\right\}.
$$

### 二次上界回溯

在使用回溯的光滑路径上，记

$$
\Delta_k=\beta_{k+1}-y_k.
$$

trial step 需要满足 smooth part 的二次上界

$$
f(\beta_{k+1})
\le f(y_k)+g_k^\top\Delta_k
+\frac{L_k}{2}\|\Delta_k\|_2^2+\varepsilon_{\rm slack}.
$$

若不满足，则

$$
L_k\leftarrow1.5L_k,
\qquad
\gamma_k\leftarrow\frac1{L_k},
$$

重新计算 prox，最多 20 次。受支持的异步 GPU 非光滑路径使用保守固定 $L_k$，避免每个 backtracking trial 都同步；prox 更新本身不变。Quantile/check loss 同样使用维护中的注册步长/Lipschitz 策略，而不会声称上面的光滑二次上界定理在折点处成立。

### Nesterov 动量

接受 $\beta_{k+1}$ 后，

$$
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}2,
$$

$$
y_{k+1}=\beta_{k+1}
+\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k).
$$

典型系数变化准则为

$$
\|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
$$

部分 adaptive-penalty 路径还结合目标函数稳定性，以避免把“系数小幅振荡但目标几乎不变”误判为未收敛。

### 带权路径

在维护中的带权路径上，若逐样本 score 为 $\psi_i$，则

$$
g(\beta)
=\frac{X^\top(w\odot\psi)}{\sum_iw_i}.
$$

目标函数值使用相同归一化解析权重。因此显式 Quantile L2/无惩罚 FISTA 与其他维护中的 Quantile 路径保持同一个统计加权约定；显式选择 FISTA 不会丢弃 `sample_weight`。

### 惩罚 prox 示例

L1：

$$
\operatorname{prox}_{\gamma\alpha\|\cdot\|_1}(v)
=S(v,\gamma\alpha).
$$

L2：

$$
\operatorname{prox}_{\gamma(\alpha/2)\|\cdot\|_2^2}(v)
=\frac{v}{1+\gamma\alpha}.
$$

无惩罚时 prox 为恒等映射。截距仍不进入 feature penalty。

---

## 4. FISTA-BB

**文件**：`statgpu/solvers/_fista_bb.py`

FISTA-BB 保留 FISTA 的 prox/动量结构，但根据梯度差更新局部步长。记

$$
s_k=\beta_k-\beta_{k-1},
\qquad
y_k=g_k-g_{k-1},
$$

两种 Barzilai-Borwein 候选为

$$
\alpha_k^{\rm BB1}=\frac{s_k^\top s_k}{s_k^\top y_k},
\qquad
\alpha_k^{\rm BB2}=\frac{s_k^\top y_k}{y_k^\top y_k}.
$$

分母数值有效时实现交替使用两种形式，并把步长裁剪到维护中的范围。由于这里把梯度差当作局部曲率信息，公开 Quantile/check-loss FISTA-BB 请求会 fail closed：阶梯次梯度不提供所需的 smooth-gradient 结构。普通 Quantile FISTA 受支持并不意味着 Quantile FISTA-BB 也受支持。

---

## 5. FISTA-LLA

**文件**：`statgpu/solvers/_fista_lla.py`

对非凸惩罚 $P$，在当前系数 $\beta^{(m)}$ 处做局部线性近似，权重为

$$
d_j^{(m)}=P'(|\beta_j^{(m)}|),
$$

并求解凸 surrogate

$$
\min_\beta L(\beta)+\sum_jd_j^{(m)}|\beta_j|.
$$

通用实现从较大的正则化水平 continuation 到请求的目标，并以加权 L1 FISTA 作为内层。分组版本使用对应的组导数和 group-aware 凸 surrogate。Quantile SCAD/MCP 不走这条通用路径，而使用第 1 节的专用 Proximal IRLS-CD。

---

## 6. IRLS（迭代重加权最小二乘）

IRLS 由具体 family/loss 定义。对 Quantile L2/无惩罚，它是 `solver="auto"` 路径，也可以显式选择。

Quantile 残差

$$
r_i=y_i-\eta_i,
$$

对应

$$
q_i=\tau+(1-2\tau)\mathbf1\{r_i<0\},
\qquad
w_i^{\rm IRLS}=\frac{q_i}{\max(|r_i|,\varepsilon)}.
$$

若有解析权重 $s_i$，则按维护中的归一化规则乘到 working weights 上，因此 WLS 步与归一化加权 pinball 目标保持一致。L2 会加入相应 ridge 对角项，同时排除截距坐标。

对普通 GLM，均值为 $\mu_i$、方差函数为 $V(\mu_i)$、link 导数为 $g'(\mu_i)$ 时，Fisher working weight 为

$$
w_i^{\rm work}
=\frac1{V(\mu_i)[g'(\mu_i)]^2},
$$

working response 为

$$
z_i=\eta_i+(y_i-\mu_i)g'(\mu_i).
$$

若传入解析权重 $s_i$，WLS 权重变为 $s_iw_i^{\rm work}$。因此解析 `sample_weight` 与 IRLS working weights 是两个不同对象：前者来自统计目标，后者来自局部二次近似。

---

## 7. Newton-Raphson

**文件**：`statgpu/solvers/_newton.py`

对完整光滑目标

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

计算

$$
g_k=\nabla F(\beta_k),
\qquad
H_k=\nabla^2F(\beta_k).
$$

维护实现先对称化并稳定 Hessian：

$$
\widetilde H_k=\frac12(H_k+H_k^\top)+10^{-10}I,
$$

求解

$$
\widetilde H_kd_k=g_k,
$$

并尝试 $\beta_k-td_k$。真正秩失败时使用最小二乘 Newton 系统；非下降方向回退到最速下降。Armijo 接受第一个满足

$$
F(\beta_k-td_k)
\le F(\beta_k)-10^{-4}t\,g_k^\top d_k
$$

的候选。Quantile 没有 Hessian，因此不进入 Newton。

---

## 8. L-BFGS / L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

L-BFGS 面向光滑目标。对被接受步，记

$$
s_k=\beta_{k+1}-\beta_k,
\qquad
y_k=g_{k+1}-g_k,
$$

仅当

$$
y_k^\tops_k>10^{-12}
$$

时保存曲率对。有限内存 two-loop recursion 产生搜索方向，Armijo 线搜索只在完整目标充分下降时接受候选点。盒约束版本把 trial point 投影到 $[\ell,u]$，并使用 projected-gradient 停止准则。

历史上 direct 底层 `lbfgs_solver(QuantileLoss, ...)` 的未传/均匀权重兼容面继续由 regression 锁定。这并不意味着 estimator/CV 层 Quantile `solver="lbfgs"` 受支持；真正非均匀 direct Quantile L-BFGS 权重仍会 fail closed。

---

## 9. ADMM（交替方向乘子法）

**文件**：`statgpu/solvers/_admm.py`

把

$$
\min_wf(w)+P(w)
$$

改写为

$$
\min_{w,z}f(w)+P(z)
\quad\text{s.t.}\quad w=z.
$$

使用 scaled dual $u$：

$$
w^{k+1}=\arg\min_w\left\{f(w)+\frac\rho2\|w-z^k+u^k\|_2^2\right\},
$$

$$
z^{k+1}=\operatorname{prox}_{P/\rho}(w^{k+1}+u^k),
\qquad
u^{k+1}=u^k+w^{k+1}-z^{k+1}.
$$

平方误差可以使用 Cholesky w-update。共享通用路径则使用内层 Nesterov accelerated-gradient，因此要求光滑损失梯度结构。Quantile/check loss 因而继续被排除在共享 ADMM 之外，即使普通 Quantile FISTA 已受支持。

---

## 10. `exact`（闭式路径）

平方误差 + L2 的维护闭式系统基于

$$
\left(\frac{X^\top X}{n}+\alpha I\right)\beta
=\frac{X^\top y}{n},
$$

并采用 estimator-specific 截距处理。这里的 `exact` 与 Cox 的 exact-ties partial likelihood 无关。

---

## 求解器分派

直接模型拟合时，`solver="auto"` 按维护中的模型层表执行。公开 `none` / `null` 会在 dispatch 前规范化为 `L2(alpha=0)`，因此无惩罚光滑路径沿用 L2 分支。简化视图为：

```text
direct fit with solver="auto"
├── squared_error + L2/none              → CPU exact / GPU Newton
├── quantile + L2/none                   → IRLS
├── quantile + L1/ElasticNet             → 普通 FISTA
├── quantile + SCAD/MCP                  → Proximal IRLS-CD
├── smooth non-Gaussian GLM + L2/none    → Newton
├── squared_error + convex sparse        → FISTA
├── gamma / inverse-Gaussian + sparse    → FISTA
├── logistic / poisson / NB + sparse     → FISTA-BB
├── tweedie + sparse                     → CPU FISTA-BB / GPU FISTA
├── SCAD/MCP（其他标量路径）             → FISTA-LLA
├── adaptive L1                          → 初始化 adaptive weights，再进入凸稀疏 FISTA/FISTA-BB policy
└── group penalties                      → group-aware FISTA / FISTA-LLA
```

对 Quantile L2/无惩罚，显式 `solver="irls"` 与 `auto` 使用同一维护算法；显式普通 `solver="fista"` 也受维护，并且真正进入通用 FISTA engine，不会静默替换成 IRLS。模型/CV 边界上的 Quantile FISTA-BB、L-BFGS 和 ADMM 会在数值 dispatch 前失败；公开底层 Quantile FISTA-BB 与 ADMM 也 fail closed。底层 direct L-BFGS 继续保留未传/均匀权重兼容面，非均匀权重仍被拒绝。稀疏 Quantile 普通 FISTA 与 SCAD/MCP Proximal IRLS-CD 仍是彼此独立的维护算法。

上面的树只是摘要。更精细的 family/backend/problem-size 规则——尤其 Poisson 与 Negative-Binomial 的 CV 稀疏分派——以 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) 为准。

`PenalizedGLM_CV` 的 smooth-L2 policy 与 direct fit 相关但有意分开。Quantile L2/无惩罚 `auto` 候选和最终 refit 使用 IRLS；显式 `solver="fista"` 则在 CV 子模型与最终 refit 中都使用 FISTA。Gamma、逆高斯、负二项的 L2 CV/final-refit 使用 L-BFGS；logistic、Poisson、Tweedie 的 L2 使用 Newton。不要从 direct-fit tree 推断 CV 行为，应查兼容性矩阵。

`sample_weight` 不会改变显式请求的 solver。不支持的带权组合应报错，而不是另选求解器。

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
