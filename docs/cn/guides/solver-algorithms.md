# 求解器算法

> 语言：中文  
> 最后更新：2026-09-12

## 概述

statgpu 提供 11 种求解器用于惩罚损失最小化。本文档记录每种求解器的算法、收敛条件、后端支持和重要 capability 边界。

## 求解器总览

| 求解器 | 最佳用途 | 后端支持 |
|--------|----------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | numpy, cupy, torch |
| Proximal Newton | 光滑损失 + 光滑惩罚；非光滑情形显式使用 FISTA | numpy, cupy, torch |
| FISTA | 一般非光滑惩罚 | numpy, cupy, torch |
| FISTA-BB | GLM + 稀疏惩罚 | numpy, cupy, torch |
| FISTA-LLA | 非凸惩罚（continuation path） | numpy, cupy, torch |
| IRLS | 光滑损失 + L2 | numpy, cupy, torch |
| Newton | 光滑损失 + L2 | numpy, cupy, torch |
| L-BFGS | 光滑损失，中低维度 | numpy, cupy, torch |
| L-BFGS-B | box-constrained 问题 | numpy, cupy, torch |
| ADMM | 可分惩罚 | numpy, cupy, torch |
| exact | squared_error + L2（闭式解） | numpy, cupy, torch |

后端支持并不表示每一种 loss / penalty / weight 组合都合法；estimator 与 loss contract 可以进一步收窄 generic solver surface。

---

## 1. Proximal IRLS-CD

**文件**: `statgpu/solvers/_proximal_irls_quantile.py`

**用途**: Quantile 回归 + SCAD/MCP 惩罚。将 IRLS 二次上界与非凸惩罚的 LLA 结合。

### 算法

1. **Continuation path**: λ_max → 目标 α（等比序列，3 步）
2. **LLA 外循环**（每步 2-5 次）：
   a. 计算 LLA 权重 w_j = P'(|β_j|)（来自 SCAD/MCP）
   b. **IRLS-CD 内循环**：
      - 计算 IRLS 权重 w_i = τ_i / max(|r_i|, ε)
      - 二次上界 Q(β) = ½ Σ w_i(y_i − X_iβ)²
      - 并行对角化（Jacobi 步）：
        g = X' @ W @ (y − Xβ)
        h = diag(X' @ W @ X)
        β = S(g + h·β, n·α·w) / h
      - 收敛检查: max(|β_new − β_old|) < tol

### 收敛

- IRLS 内层: 系数最大变化 < tol（默认 1e-6）
- LLA 外层: 系数最大变化 < lla_tol
- GPU: 收敛保持在 device 上比较，仅同步 bool

---

## 2. Proximal Newton

**文件**: `statgpu/solvers/_proximal_newton.py`

**用途**: 对光滑损失与 L2/无惩罚目标执行 Newton 更新。

一般非光滑 proximal-Newton 需要求解 Hessian metric 下的 proximal 子问题；旧的 Euclidean-prox 快捷路径会优化错误目标。现在 direct 非光滑调用会明确告警并使用 FISTA；FISTA-LLA 也保持 backend-native FISTA，直到实现并显式声明正确的 metric proximal 能力。

### 算法

1. 对损失和光滑惩罚各计入一次梯度与 Hessian。
2. 仅在真正的秩失败时使用 least-squares 降级。
3. 对完整声明目标执行 Armijo 回溯。
4. Newton 方向不是下降方向时使用最速下降。

---

## 3. FISTA（快速迭代收缩阈值算法）

**文件**: `statgpu/solvers/_fista.py`

**用途**: 有 proximal 算子的任意损失+任意惩罚的通用求解器。

### 算法

1. 初始化 β₀, y₀ = β₀, t₀ = 1
2. 对 k = 1, 2, ...:
   a. 计算梯度 g_k = ∇ℓ(y_k)
   b. Proximal 步: β_{k+1} = prox(β_k − (1/L)·g_k, α/L)
   c. Nesterov 动量: t_{k+1} = (1 + √(1+4t_k²))/2
      y_{k+1} = β_{k+1} + ((t_k−1)/t_{k+1})(β_{k+1} − β_k)

### GPU 异步路径

满足条件时（GPU 后端 + 非光滑惩罚 + CV/二次损失）：
- 梯度计算在 device 上
- 融合 proximal + momentum kernel
- 批量收敛/发散/Lipschitz 检查

### 加权路径

- 入口处将 sample_weight 转为后端原生数组
- 加权梯度 g = X' @ (sw * ψ) / Σsw
- GPU 路径加权 objective 跟踪

---

## 4. FISTA-BB（Barzilai-Borwein）

**文件**: `statgpu/solvers/_fista_bb.py`

**用途**: 自适应 BB 步长。适合 GPU 上 GLM + 稀疏惩罚。

### 算法

1. 使用 Nesterov 动量的 FISTA 主体
2. 替代固定 L⁻¹ 步长，使用 BB1 或 BB2：
   - BB1（长步）: α_k = ⟨s_{k-1}, s_{k-1}⟩ / ⟨s_{k-1}, y_{k-1}⟩
   - BB2（短步）: α_k = ⟨s_{k-1}, y_{k-1}⟩ / ⟨y_{k-1}, y_{k-1}⟩
   其中 s = β_k − β_{k-1}, y = ∇ℓ(β_k) − ∇ℓ(β_{k-1})
3. 每 2 次迭代交替 BB1/BB2
4. 自适应重启（O'Donoghue & Candes 2015）：动量与下降方向相反时重置

### 非凸惩罚禁用 BB

SCAD/MCP/group MCP/group SCAD 禁用 BB 步长。LLA 重加权引起的 subgradient 突变会放大噪声导致发散。

---

## 5. FISTA-LLA

**文件**: `statgpu/solvers/_fista_lla.py`

**用途**: 非凸惩罚（SCAD/MCP/adaptive L1）通过 LLA。一个融合函数中运行 continuation path + LLA + FISTA/proximal Newton。

### 算法

1. **Continuation path**: λ_max → 目标 α（5 步，非光滑损失 3 步）
2. **LLA 外层**（每步 2-5 次）：
   a. 在当前 β 处计算 LLA 权重
   b. **内层求解器**：
      - 复合 LLA 子问题统一使用 backend-native FISTA
      - 未来的 proximal-Newton 路径必须显式提供正确的 Hessian-metric proximal 能力
   c. LLA 收敛 ||β − β_before_lla||₁ < lla_tol

### 融合 Kernel（GPU）

- squared error + GPU: 融合 proximal + momentum kernel（预计算 X'X）
- 通用路径: 融合梯度裁剪 + proximal + momentum
- 批量 GPU 同步: convergence + divergence + Lipschitz 一次 D2H 传输

---

## 6. IRLS（迭代重加权最小二乘）

**实现方式**: 每个损失类有独立的 `irls()` 方法。

**用途**: 光滑惩罚（L2、none）配合 GLM 或 quantile 损失。

### 算法（Quantile IRLS）

1. 初始化 β₀ = OLS 估计
2. 每次迭代：
   a. 计算残差 r = y − Xβ
   b. 计算对应 IRLS weights
   c. 求解 weighted least-squares surrogate
   d. 满足维护中的收敛规则后停止

---

## 7. Newton-Raphson

**文件**: `statgpu/solvers/_newton.py`

**用途**: 光滑损失 + L2/无惩罚。Hessian 条件良好时收敛快。

### 算法

1. 计算梯度 g = ∇ℓ(β) + λ·β 和 Hessian H = ∇²ℓ(β) + λ·I
2. 求解 Newton system
3. Armijo 线搜索与回退
4. 使用 1e-10 小 ridge 做数值稳定化

### Analytic sample weights

对于明确提供 weighted curvature 的 loss，Newton 支持真正的 non-uniform analytic weights。objective value、gradient、Hessian 与每个 Armijo trial 使用同一个归一化 weighted objective：

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

uniform weights 保持历史 unweighted 数值路径；将所有 active weights 同时乘以一个正数不会改变 optimum。

---

## 8. L-BFGS / L-BFGS-B

**文件**: `statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

**用途**: 光滑损失 + 光滑/无惩罚，中低维度，以及适合 quasi-Newton 更新的 GLM 行。

### 算法

标准 L-BFGS two-loop recursion + Armijo line search，history size 默认 `m=10`。当前点、每个 line-search candidate 与 accepted-point gradient 都必须使用同一个声明目标。

### Analytic sample weights

`lbfgs_solver` 对所有既有 consumer 保留 uniform-weight compatibility；真正的 **non-uniform** weighted L-BFGS 由 loss contract 显式 opt-in：

- 维护中的 `GLMLoss` 会 opt-in，并使用与 Newton 相同的归一化 analytic-weight objective；
- active weight vector 进入 initial gradient、当前 line-search objective、每个 candidate objective 和 accepted-point gradient；
- NumPy/CuPy/Torch 数值迭代停留在输入执行后端；
- generic non-GLM `LossBase` consumer 继续对 genuine non-uniform weighted L-BFGS fail closed，除非该 loss 之后独立声明相同 capability。

因此，direct solver 用户应区分“Huber/Quantile/Cox 可以无权重调用 L-BFGS”和“这些 loss 支持 non-uniform `sample_weight`”这两个不同命题；后者当前不能由前者推出。

`L-BFGS-B` 是独立的 box-constrained 实现，不应默认继承 `lbfgs_solver` 的全部 weight capability。

---

## 9. ADMM（Alternating Direction Method of Multipliers）

**文件**: `statgpu/solvers/_admm.py`

**用途**: 可分 objective 的 alternative formulation。

---

## 10. exact（闭式路径）

**实现位置**: `_fit_mixin._solve_exact_*`

**用途**: maintained dispatch 选择 squared_error + L2 闭式路径时使用。

---

## 求解器调度逻辑

```
fit() with solver="auto"
├── squared_error + L2 + numpy → exact
├── squared_error + L2 + GPU  → newton
├── SCAD/MCP/adaptive → fista (LLA 封装)
├── quantile → fista / quantile-specific path
├── squared_error + sparse → fista
├── GLM + GPU + sparse → 按维护表选择 fista_bb / fista
├── CV + L2 → lbfgs / newton
├── 光滑惩罚 + 光滑损失 → newton / irls
└── 默认 sparse → fista_bb
```

`sample_weight` 不会静默重写显式 solver request；estimator-level `solver="auto"` 继续使用对应 weighted/unweighted fit 的维护 dispatch table。

## 参考文献

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods. *SIAM J. Optimization*, 24(3), 1420-1443.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
