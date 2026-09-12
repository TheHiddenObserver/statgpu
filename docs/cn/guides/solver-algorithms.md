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
g_k
=\nabla\ell(\beta_k)+\nabla P(\beta_k),
$$

以及

$$
H_k
=\nabla^2\ell(\beta_k)+\nabla^2P(\beta_k).
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

则认为已经收敛。

随后求解 Newton 线性系统

$$
\widetilde H_k d_k=g_k.
$$

代码采用“减去方向”的记号，因此试探点写成

$$
\beta_k(t)=\beta_k-t d_k.
$$

这与通常写成 $p_k=-\widetilde H_k^{-1}g_k$、再令 $\beta_k+t p_k$ 完全等价。若线性方程求解被识别为奇异或病态，当前实现**不会调用最小二乘求解器**，而是直接退回

$$
d_k=g_k,
$$

也就是在上述“减去方向”的记号下采用最速下降。

在进入线搜索前还会检查下降性。由于更新为 $\beta_k-t d_k$，下降方向应满足

$$
g_k^\top d_k>0.
$$

如果 $g_k^\top d_k$ 非有限或不大于 0，同样改用

$$
d_k=g_k,
\qquad
 g_k^\top d_k=\|g_k\|_2^2.
$$

### Armijo 回溯线搜索

从

$$
t_0=1
$$

开始，寻找第一个满足

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-c\,t\,g_k^\top d_k,
$$

的步长，其中当前实现使用

$$
c=10^{-4}.
$$

若条件不满足，则按

$$
t\leftarrow \frac{t}{2}
$$

继续回溯，最多尝试 25 次。第一个满足 Armijo 条件的候选点被接受：

$$
\beta_{k+1}=\beta_k-t d_k.
$$

如果 25 次试探都失败，则恢复

$$
\beta_{k+1}=\beta_k,
$$

发出线搜索失败警告并结束当前求解过程。

### 默认值与计算后端

- 默认 `max_iter=50`；
- 默认 `tol=1e-6`；
- 受支持的 NumPy/CuPy/Torch 路径使用对应的原生线性代数实现。

---

## 3. FISTA（快速迭代收缩阈值算法）

**文件**：`statgpu/solvers/_fista.py`

**用途**：光滑的数据拟合项 + 具有近端算子的惩罚项。

### 算法

初始化

$$
\beta_0=y_0,\qquad t_0=1.
$$

第 $k$ 次迭代：

1. 在动量点计算梯度

   $$
   g_k=\nabla\ell(y_k);
   $$

2. 做近端梯度更新

   $$
   \beta_{k+1}=\operatorname{prox}_{\alpha/L}
   \left(y_k-\frac{1}{L}g_k\right);
   $$

3. 更新 Nesterov 动量

   $$
   t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
   $$

   $$
   y_{k+1}=\beta_{k+1}
   +\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k);
   $$

4. 在对应路径上使用维护中的收敛判据，例如

   $$
   \|\beta_{k+1}-\beta_k\|_1<\texttt{tol}.
   $$

### 加权路径

在受支持的带权路径上，`sample_weight` 会在入口处转换到选定的计算后端，数据拟合项梯度按归一化权重计算，例如

$$
g=\frac{X^\top(s\odot\psi)}{\sum_i s_i}.
$$

目标函数的带权跟踪使用相同归一化。FISTA 具有带权实现，并不意味着所有模型组合都自动支持权重。

### GPU 路径

受支持的 GPU 路径会尽量把梯度、近端更新、动量更新以及大部分收敛/发散检查留在设备端，并批量减少设备到主机的同步。

### 默认值

- 默认 `max_iter=500`；
- 默认 `tol=1e-6`。

---

## 4. FISTA-BB（Barzilai-Borwein）

**文件**：`statgpu/solvers/_fista_bb.py`

**用途**：带自适应 Barzilai-Borwein 步长的 FISTA，适合受支持的 GLM 稀疏惩罚路径。

### 算法

令

$$
s_{k-1}=\beta_k-\beta_{k-1},
\qquad
y_{k-1}=\nabla\ell(\beta_k)-\nabla\ell(\beta_{k-1}).
$$

两种标准 BB 步长为

$$
\alpha_k^{\mathrm{BB1}}
=\frac{\langle s_{k-1},s_{k-1}\rangle}
{\langle s_{k-1},y_{k-1}\rangle},
$$

以及

$$
\alpha_k^{\mathrm{BB2}}
=\frac{\langle s_{k-1},y_{k-1}\rangle}
{\langle y_{k-1},y_{k-1}\rangle}.
$$

维护中的实现按既定周期交替 BB1/BB2，并对步长做上下界限制；当动量方向与下降方向冲突时执行自适应重启。

SCAD/MCP 及其分组版本禁用 BB 更新，因为 LLA 重加权会突然改变有效次梯度，使基于割线信息的 BB 步长在这些延续路径上不稳定。

---

## 5. FISTA-LLA

**文件**：`statgpu/solvers/_fista_lla.py`

**用途**：SCAD、MCP、adaptive L1 等非凸或迭代重加权惩罚。

### 算法

1. 从 `lambda_max` 到目标 `alpha` 构造延续路径（维护中的默认设置通常为 5 步，非光滑路径为 3 步）。
2. 延续路径中的每一步运行 LLA 外循环。
3. 在当前系数处计算局部惩罚权重。
4. 用对应后端的 FISTA 解当前凸近似问题。
5. 当

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_1<\texttt{lla\_tol}
   $$

   时结束 LLA 外循环。

受支持的 GPU 路径会使用融合的近端/动量核函数，并批量执行标量检查，以减少设备到主机的同步。

---

## 6. IRLS（迭代重加权最小二乘）

**实现方式**：由具体损失函数或分布族提供 `irls()` 方法。

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

GLM IRLS 具有同样的“构造工作响应与工作权重（working response/weights），再解加权最小二乘”的高层结构，但具体工作响应和工作权重由分布族与链接函数决定。这里的 IRLS 工作权重与用户传入的解析 `sample_weight` 不是同一概念。

---

## 7. Newton-Raphson

**文件**：`statgpu/solvers/_newton.py`

**用途**：有 Hessian 的光滑损失 + L2/无惩罚。

### 算法

设完整目标函数的梯度和 Hessian 分别为 $g$ 与 $H$，Newton 方向为

$$
d=-H^{-1}g.
$$

随后执行 Armijo 回溯线搜索，并在需要时加入小的 ridge 稳定项改善数值条件。

### 解析 `sample_weight`

对明确提供带权曲率的路径，Newton 在目标函数值、梯度、Hessian 和每个 Armijo 试探点中都使用同一个归一化带权目标：

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

因此，把所有有效权重同时乘以一个正数不会改变最优解。均匀权重或与均匀权重数值等价的情况，在定义了兼容路径时继续保持历史无权重数值行为。

---

## 8. L-BFGS / L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

**用途**：光滑损失 + 光滑/无惩罚，并且希望避免显式形成完整 Hessian 的场景。

### L-BFGS 算法

L-BFGS 使用标准的有限内存双循环递推和 Armijo 线搜索；维护中的历史长度为 `m=10`。当前目标函数、每个线搜索候选点，以及接受新点后的梯度，都必须使用同一个目标函数定义。

### 解析 `sample_weight`

直接调用 L-BFGS 时，非均匀权重需要由损失函数明确支持：

| 直接 L-BFGS 路径 | 非均匀 `sample_weight` |
|---|---|
| 维护中的 `GLMLoss` | ✅ 支持 |
| 通用稳健 / 分位数 / Cox `LossBase` | ❌ 不能由无权重支持自动推出 |

对维护中的 GLM，初始梯度、当前目标函数、每个线搜索候选点和接受新点后的梯度都使用同一组归一化权重；NumPy/CuPy/Torch 数值计算保持在选定后端。

`L-BFGS-B` 是独立的盒约束实现，不应默认继承 `lbfgs_solver` 的全部权重能力。

---

## 9. ADMM（交替方向乘子法）

**文件**：`statgpu/solvers/_admm.py`

对变量分裂约束 $\beta=z$，维护中的基本结构为：

1. 系数更新

   $$
   \beta^{k+1}=\arg\min_\beta
   L(\beta)+\frac{\rho}{2}\|\beta-z^k+u^k\|_2^2;
   $$

2. 分裂变量的近端更新

   $$
   z^{k+1}=\operatorname{prox}_{P/\rho}(\beta^{k+1}+u^k);
   $$

3. 缩放对偶变量更新

   $$
   u^{k+1}=u^k+\beta^{k+1}-z^{k+1}.
   $$

实现会按照维护中的残差规则调整 `rho`。

---

## 10. `exact`（闭式路径）

**实现位置**：`_fit_mixin._solve_exact_*`

**用途**：平方误差 + L2，并且自动分发选择闭式/特征分解路径的情形。对应的基本线性系统形如

$$
\left(\frac{X^\top X}{n}+\alpha I\right)\beta=\frac{X^\top y}{n},
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
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.