# 求解器算法

> 语言：中文  
> 最后更新：2026-07-01

## 概述

statgpu 提供 10 种求解器用于惩罚损失最小化。本文档记录每种求解器的算法、收敛条件、后端支持和超参数。

## 求解器总览

| 求解器 | 最佳用途 | 后端支持 |
|--------|----------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | numpy, cupy, torch |
| Proximal Newton | Huber/Bisquare/Cox + SCAD/MCP | numpy, cupy, torch |
| FISTA | 一般非光滑惩罚 | numpy, cupy, torch |
| FISTA-BB | GLM + 稀疏惩罚 | numpy, cupy, torch |
| FISTA-LLA | 非凸惩罚（continuation path） | numpy, cupy, torch |
| IRLS | 光滑损失 + L2 | numpy, cupy, torch |
| Newton | 光滑损失 + L2 | numpy, cupy, torch |
| L-BFGS | 光滑损失，中低维度 | numpy, cupy, torch |
| exact | squared_error + L2（闭式解） | numpy, cupy, torch |

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

**用途**: 有 Hessian 的光滑损失（Huber、Bisquare、Cox PH）+ 非光滑惩罚（SCAD/MCP 通过 LLA）。5-10 次迭代收敛。

### 算法

1. 计算 Hessian H = X'WX 和梯度 g = X'ψ / n
2. Newton 方向 d = -H⁻¹·g
3. Armijo 线搜索（最多 25 次回退）：
   a. 尝试点: β_try = proximal(β − step·d, step)
   b. 检查复合 Armijo: f(β_try) + g(β_try) ≤ f(β) + g(β) + c·step·g'd
   c. 不满足则步长减半
4. Hessian 奇异或 g'd ≤ 0 → 回退到梯度下降

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
      - 有 Hessian → Proximal Newton（5-10 次迭代）
      - 无 Hessian → FISTA（300+ 次迭代）
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
   b. IRLS 权重 w_i = (τ + (1−2τ)·1_{r_i<0}) / max(|r_i|, ε)
   c. 求解加权 LS: (X'WX + n·α·I)β = X'Wy
   d. ||β_new − β|| < tol → 停止

---

## 7. Newton-Raphson

**文件**: `statgpu/solvers/_newton.py`

**用途**: 光滑损失 + L2 惩罚。Hessian 正定时收敛快。

### 算法

1. 计算梯度 g = ∇ℓ(β) + λ·β 和 Hessian H = ∇²ℓ(β) + λ·I
2. Newton 方向 d = -H⁻¹·g
3. Armijo 线搜索与回退（最多 25 次）
4. Ridge 正则化 1e-10·I 确保稳定性

---

## 求解器调度逻辑

```
fit() with solver="auto"
├── squared_error + L2 + numpy → exact
├── squared_error + L2 + GPU  → newton
├── SCAD/MCP/adaptive → fista (LLA 封装)
│   ├── squared_error → fista_lla（融合）
│   ├── quantile      → proximal_irls_cd
│   ├── has_hessian   → fista_lla → proximal_newton
│   └── no_hessian    → fista_lla → fista
├── quantile（任意惩罚） → fista
├── squared_error + sparse → fista
├── GLM + GPU + sparse → fista_bb
├── CV + L2 → lbfgs / newton
├── 光滑惩罚 + 光滑损失 → newton / irls
└── 默认 sparse → fista_bb
```

## 参考文献

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods. *SIAM J. Optimization*, 24(3), 1420-1443.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
