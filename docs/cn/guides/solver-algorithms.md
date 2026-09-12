# 求解器算法

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：算法参考  
> 切换：[English](../../en/guides/solver-algorithms.md)

## 概览

statgpu 提供一组一阶、二阶、proximal 和闭式 solver。对大多数模型用户来说，优先使用 `solver="auto"` 即可；本页主要用于解释各 solver 的算法思想，以及为什么某些 loss / penalty / weight 组合会被支持或拒绝。

阅读下面的表格时，建议先记住三条规则：

1. **支持某个 backend，不等于支持所有模型组合。** solver 可以在 NumPy/CuPy/Torch 上实现，但某个具体的 loss × penalty × weight 路径仍可能不支持。
2. **显式 solver request 不会被静默改写。** 加入 `sample_weight` 不会悄悄把 Newton/L-BFGS 换成其他 solver。
3. **weight support 取决于完整路径。** 尤其是 direct L-BFGS 的非均匀权重，只对明确声明该能力的 loss（目前包括维护中的 GLM loss）开放。

模型层面的完整分发表见 [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)。

## 求解器总览

| 求解器 | 最适合 | 后端支持 |
|--------|--------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | numpy, cupy, torch |
| Proximal Newton | smooth loss + L2/无惩罚；非光滑请求使用 FISTA | numpy, cupy, torch |
| FISTA | 一般非光滑 penalty | numpy, cupy, torch |
| FISTA-BB | GLM + sparse penalty | numpy, cupy, torch |
| FISTA-LLA | 非凸 penalty 的 continuation/LLA | numpy, cupy, torch |
| IRLS | 具有维护中 IRLS 表示的 loss | numpy, cupy, torch |
| Newton | 有 Hessian 的 smooth loss | numpy, cupy, torch |
| L-BFGS | smooth loss、中等参数维度 | numpy, cupy, torch |
| L-BFGS-B | box-constrained smooth problem | numpy, cupy, torch |
| ADMM | 可分/proximal formulation | numpy, cupy, torch |
| exact | squared error + L2 闭式路径 | numpy, cupy, torch |

后端列只描述数值实现能力；estimator 与 loss contract 可以进一步缩小实际可用范围。

---

## 1. Proximal IRLS-CD

**文件**：`statgpu/solvers/_proximal_irls_quantile.py`

**用途**：Quantile regression + SCAD/MCP。它把 pinball loss 的 IRLS 二次 majorization 与非凸 penalty 的 local linear approximation（LLA）结合起来。

### 算法

1. **Continuation path**：从 $\lambda_{\max}$ 沿短的等比路径走到目标 $\alpha$。
2. **LLA 外循环**：
   - 根据当前 coefficient 计算 SCAD/MCP 的局部权重；
   - 用 IRLS + coordinate descent 解对应的 weighted L1-like surrogate；
   - coefficient 稳定后结束当前 LLA step。
3. **IRLS-CD 内循环**：
   - 为 pinball loss 构造 quadratic majorizer；
   - 计算 weighted gradient 与对角 curvature；
   - 做逐坐标 soft-thresholding；
   - coefficient change 小于 tolerance 后停止。

### 收敛

- IRLS 内层：最大 coefficient change < `tol`；
- LLA 外层：最大 coefficient change < `lla_tol`；
- GPU 上的收敛比较尽量留在 device，仅同步最终 boolean。

### 后端

- NumPy：NumPy linear algebra / array ops；
- CuPy：CuPy matrix ops 与可用的 GPU kernel；
- Torch：device-native tensor ops。

---

## 2. Proximal Newton

**文件**：`statgpu/solvers/_proximal_newton.py`

**用途**：smooth loss + L2/无惩罚，并且普通 Newton system 有明确数学定义的场景。

一般的非光滑 proximal-Newton 需要在 Hessian metric 下求解 proximal 子问题。直接套 Euclidean prox 会优化另一个 composite objective，因此 statgpu 不再静默使用这种近似：direct 非光滑请求会明确告警并改走 FISTA；FISTA-LLA 也保持 backend-native FISTA inner solve，直到实现并显式声明正确的 Hessian-metric proximal capability。

### 算法

1. 计算声明 objective 的 gradient 与 Hessian。
2. 求解 Newton system；只有真正的 rank failure 才使用 least-squares fallback。
3. 对完整 objective 做 Armijo backtracking。
4. 如果 Newton direction 不是 descent direction，则改用 steepest descent。

line-search failure 会被暴露出来，而不会被当作成功迭代。

---

## 3. FISTA（快速迭代收缩阈值算法）

**文件**：`statgpu/solvers/_fista.py`

**用途**：smooth data-fit term + 有 proximal operator 的 penalty。

### 算法

1. 初始化 coefficient、momentum point 与 Nesterov scalar。
2. 每次迭代：
   - 在 momentum point 计算 smooth gradient；
   - 做 proximal-gradient step；
   - 更新 Nesterov momentum；
   - 检查维护中的 convergence rule。

### GPU 路径

受支持的 GPU route 会尽量把 gradient、proximal update、momentum update，以及大部分 convergence/divergence check 留在 device，并批量减少 device-to-host synchronization。

### 加权路径

在维护中的 weighted route 上：

- `sample_weight` 在 solver 入口转换到选定 backend；
- data-fit gradient 使用归一化 weighted convention；
- weighted objective tracking 使用相同 normalization。

但 weight 的统计语义仍由 estimator/loss route 定义；“FISTA 有 weighted implementation”不代表所有模型组合都自动获得 weighted support。

---

## 4. FISTA-BB（Barzilai-Borwein）

**文件**：`statgpu/solvers/_fista_bb.py`

**用途**：带自适应 Barzilai-Borwein step size 的 FISTA，适合维护中的 GLM sparse-penalty route。

### 算法

FISTA-BB 保留 FISTA 的 Nesterov/proximal 结构，但用连续两次 coefficient 与 gradient 的 secant information 构造 BB1/BB2 step size，并在 momentum 与 descent 冲突时做 adaptive restart。

### 非凸 penalty

SCAD/MCP 及其 group 版本禁用 BB update。LLA reweighting 会突然改变有效 subgradient，使基于 secant 的 BB step 在这些 continuation path 上不稳定。

---

## 5. FISTA-LLA

**文件**：`statgpu/solvers/_fista_lla.py`

**用途**：SCAD、MCP、adaptive L1 等非凸或迭代重加权 penalty。

### 算法

1. 从较大的 regularization value 构造到目标 `alpha` 的短 continuation path。
2. 每个 continuation step 运行 LLA outer loop。
3. 每次 LLA 把当前 non-convex penalty 替换成对应 convex surrogate，并用 backend-native FISTA 解这个 surrogate。
4. coefficient 稳定后结束当前 LLA step。

未来如果要加入 proximal-Newton inner path，必须先有显式且数学上正确的 Hessian-metric proximal implementation；当前不会用近似路径冒充这一能力。

---

## 6. IRLS（迭代重加权最小二乘）

**实现方式**：由具体 loss/family 提供 IRLS 方法。

**用途**：适用于 statgpu 明确维护 IRLS 表示的 loss，通常配合 L2/无惩罚。

### 通用结构

1. 根据当前 coefficient 构造 working response 与 working weights。
2. 解对应的 weighted least-squares surrogate。
3. 更新 coefficient，直到满足维护中的 convergence rule。

GLM 的 working response/weights 由 family/link 决定。Quantile-specific IRLS 是另一种 majorization，不要把它与 GLM 的 analytic `sample_weight` 混为一谈。

---

## 7. Newton-Raphson

**文件**：`statgpu/solvers/_newton.py`

**用途**：smooth loss + L2/无惩罚，并且提供 Hessian。second-order curvature 稳定、参数维度适中时通常很有效。

### 算法

1. 计算声明 objective 的 gradient 与 Hessian。
2. 求解 Newton system。
3. 用 Armijo backtracking 选择可接受 step。
4. 在需要时加入维护中的小 ridge stabilization 改善数值条件。

### Analytic `sample_weight`

对明确提供 weighted curvature 的 loss/estimator route，Newton 使用同一个归一化 weighted objective：

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

objective、gradient、Hessian 与每个 Armijo trial 都使用同一组权重。因此，把所有 active weights 同时乘以一个正数不会改变最优解。

在定义了兼容 route 的情况下，uniform/等效 uniform 权重继续保持历史 unweighted 数值路径。

---

## 8. L-BFGS / L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

**用途**：smooth loss + smooth/无惩罚，并且希望避免显式形成完整 Hessian 的场景。

### L-BFGS 算法

L-BFGS 使用标准 limited-memory two-loop recursion 与 Armijo line search。当前 objective、每一个 line-search candidate，以及接受新点后的 gradient，都必须基于同一个声明 objective。

### Analytic `sample_weight`

非均匀 weighted direct L-BFGS 在 loss contract 层采用显式 opt-in：

| direct L-BFGS route | 非均匀 `sample_weight` |
|---|---|
| 维护中的 `GLMLoss` | ✅ 支持 |
| generic robust / quantile / Cox `LossBase` | ❌ 不能由无权重支持自动推出 |

对维护中的 GLM，initial gradient、当前 objective、每个 line-search candidate 和 accepted-point gradient 都使用同一组归一化权重；NumPy/CuPy/Torch 数值计算也保持在选定 backend。

uniform weights 继续兼容历史 unweighted L-BFGS route。一个 loss 支持“无权重 L-BFGS”，**不代表**它自动支持真正非均匀的 `sample_weight`。

`L-BFGS-B` 是独立的 box-constrained implementation，不应默认继承 `lbfgs_solver` 的全部 weighting capability。

---

## 9. ADMM（Alternating Direction Method of Multipliers）

**文件**：`statgpu/solvers/_admm.py`

**用途**：适合 variable splitting 的受支持 separable/proximal formulation。

### 通用结构

1. 在 smooth objective + augmented quadratic term 下更新主 coefficient variable。
2. 通过声明的 proximal operator 更新 split variable。
3. 更新 scaled dual variable。
4. 根据维护中的 residual rule 调整 penalty parameter。

---

## 10. `exact`（闭式路径）

**实现位置**：`_fit_mixin._solve_exact_*`

**用途**：squared-error + L2 且维护中的 dispatch 选择 closed-form/eigendecomposition path 的场景。

---

## 求解器调度

对普通 direct fit，`solver="auto"` 遵循模型层面的维护表。可以粗略理解为：

```
direct fit with solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + sparse penalty → FISTA/FISTA-BB
├── smooth non-Gaussian GLM + L2   → Newton
├── SCAD/MCP/adaptive path          → LLA + FISTA-family inner solve
├── quantile                        → quantile-specific FISTA/IRLS path
└── group penalty                   → group-aware FISTA / FISTA-LLA
```

`PenalizedGLM_CV` 有一套相关但有意独立的 smooth-L2 policy。尤其是 Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 CV/final-refit route 使用 L-BFGS，而 logistic、Poisson、Tweedie 的 L2 row 使用 Newton。不要从 direct-fit tree 推断 CV 行为，请以 compatibility matrix 为准。

`sample_weight` 不会静默重写显式 solver request。如果 requested weighted route 不受支持，statgpu 会直接报错，而不是替换另一个 solver。

## 参考文献

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.
