# 求解器算法

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：算法参考  
> 切换：[English](../../en/guides/solver-algorithms.md)

## 概览

statgpu 提供一阶、二阶、近端和闭式求解器。对大多数模型用户，建议从 `solver="auto"` 开始；本页主要用于解释各求解器的算法思想与适用范围。

阅读本页时，先记住三条规则：

1. **支持某个计算后端，不等于支持所有模型组合。** 求解器可以在 NumPy、CuPy、Torch 上实现，但具体的损失函数、惩罚项和权重组合仍可能不受支持。
2. **`sample_weight` 不会改变显式指定的 `solver`。** 如果所请求的组合不受支持，则直接报错。
3. **权重支持由完整模型路径决定。** 例如，直接调用 L-BFGS 时的非均匀权重目前由 GLM loss 明确支持，而不是所有 `LossBase` 都自动支持。

模型层面的完整分发表见 [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)。

## 求解器总览

| 求解器 | 最适合 | 后端支持 |
|--------|--------|:---:|
| Proximal IRLS-CD | 分位数回归 + SCAD/MCP | NumPy、CuPy、Torch |
| Proximal Newton | 光滑损失 + L2/无惩罚 | NumPy、CuPy、Torch |
| FISTA | 一般非光滑惩罚 | NumPy、CuPy、Torch |
| FISTA-BB | GLM + 稀疏惩罚 | NumPy、CuPy、Torch |
| FISTA-LLA | 非凸惩罚的延续/LLA 路径 | NumPy、CuPy、Torch |
| IRLS | 明确定义 IRLS 表示的损失函数 | NumPy、CuPy、Torch |
| Newton | 有 Hessian 的光滑损失 | NumPy、CuPy、Torch |
| L-BFGS | 光滑损失、中等参数维度 | NumPy、CuPy、Torch |
| L-BFGS-B | 带盒约束的光滑问题 | NumPy、CuPy、Torch |
| ADMM | 可分或可近端化的问题 | NumPy、CuPy、Torch |
| `exact` | squared error + L2 闭式路径 | NumPy、CuPy、Torch |

后端列只描述数值实现能力；具体模型仍可能进一步限制可用组合。

---

## 1. Proximal IRLS-CD

**文件**：`statgpu/solvers/_proximal_irls_quantile.py`

**用途**：分位数回归 + SCAD/MCP。它把 pinball loss 的 IRLS 二次上界与非凸惩罚的局部线性近似（LLA）结合起来。

### 算法

1. 从 $\lambda_{\max}$ 沿短的等比延续路径走到目标 $\alpha$。
2. 每个 LLA 外循环根据当前系数计算 SCAD/MCP 的局部权重。
3. 用 IRLS + coordinate descent 解对应的加权 L1 型近似问题。
4. 系数稳定后结束当前 LLA 步骤。

### 收敛

- IRLS 内层：最大系数变化小于 `tol`；
- LLA 外层：最大系数变化小于 `lla_tol`；
- GPU 上的收敛判断尽量留在设备端，只同步最终布尔结果。

---

## 2. Proximal Newton

**文件**：`statgpu/solvers/_proximal_newton.py`

**用途**：光滑损失 + L2/无惩罚，并且普通 Newton 系统有明确数学定义的场景。

一般的非光滑 Proximal Newton 需要在 Hessian 度量下求解近端子问题。直接套用 Euclidean prox 会优化另一个复合目标，因此当前实现不采用这种近似：非光滑请求会给出提示并改用 FISTA；FISTA-LLA 也继续使用原生后端的 FISTA 内层，直到实现数学上正确的 Hessian 度量近端算法。

### 算法

1. 计算目标函数的梯度与 Hessian。
2. 求解 Newton 系统；只有真正的秩失败才使用最小二乘降级。
3. 对完整目标函数做 Armijo 回溯线搜索。
4. 如果 Newton 方向不是下降方向，则改用最速下降方向。

线搜索失败会作为失败状态暴露出来，不会被当作成功迭代。

---

## 3. FISTA（快速迭代收缩阈值算法）

**文件**：`statgpu/solvers/_fista.py`

**用途**：光滑数据拟合项 + 具有近端算子的惩罚项。

### 算法

1. 初始化系数、动量点和 Nesterov 标量。
2. 每次迭代：
   - 在动量点计算光滑部分的梯度；
   - 做近端梯度更新；
   - 更新 Nesterov 动量；
   - 检查收敛条件。

### GPU 路径

受支持的 GPU 路径会尽量把梯度、近端更新、动量更新以及大部分收敛/发散判断保留在设备端，并减少设备到主机的同步次数。

### 加权路径

在受支持的加权路径中：

- `sample_weight` 在求解器入口转换为所选后端的数组；
- 数据拟合梯度使用归一化加权定义；
- 目标函数跟踪使用相同的归一化方式。

但权重的统计含义仍由具体模型和损失函数定义；底层 FISTA 具备加权实现，并不表示所有模型组合都自动支持权重。

---

## 4. FISTA-BB（Barzilai-Borwein）

**文件**：`statgpu/solvers/_fista_bb.py`

**用途**：带自适应 Barzilai-Borwein 步长的 FISTA，适合受支持的 GLM 稀疏惩罚路径。

FISTA-BB 保留 FISTA 的 Nesterov/近端结构，但利用相邻两次系数和梯度的割线信息构造 BB1/BB2 步长；当动量方向与下降方向冲突时，会进行自适应重启。

### 非凸惩罚

SCAD/MCP 及其 group 版本禁用 BB 更新。LLA 重加权会突然改变有效次梯度，使基于割线信息的 BB 步长在这些延续路径上不稳定。

---

## 5. FISTA-LLA

**文件**：`statgpu/solvers/_fista_lla.py`

**用途**：SCAD、MCP、adaptive L1 等非凸或迭代重加权惩罚。

### 算法

1. 从较大的正则化参数构造到目标 `alpha` 的短延续路径。
2. 每个延续点运行 LLA 外循环。
3. 每次 LLA 用当前的凸近似替代非凸惩罚，并使用所选后端上的 FISTA 求解。
4. 系数稳定后结束当前 LLA 步骤。

未来若加入 Proximal Newton 内层，必须先实现明确且数学上正确的 Hessian 度量近端问题；当前实现不会使用近似路径代替这一能力。

---

## 6. IRLS（迭代重加权最小二乘）

**实现方式**：由具体损失函数或分布族提供 IRLS 方法。

**用途**：适用于 statgpu 明确维护 IRLS 表示的损失函数，通常配合 L2 或无惩罚。

### 通用结构

1. 根据当前系数构造工作响应与工作权重。
2. 求解对应的加权最小二乘近似问题。
3. 更新系数，直到满足收敛条件。

GLM 的工作响应与工作权重由分布族和链接函数决定。分位数回归中的 IRLS 是另一种上界近似，不应与 GLM 的解析 `sample_weight` 混为一谈。

---

## 7. Newton-Raphson

**文件**：`statgpu/solvers/_newton.py`

**用途**：光滑损失 + L2/无惩罚，并且提供 Hessian。在二阶曲率稳定、参数维度适中时通常很有效。

### 算法

1. 计算目标函数的梯度与 Hessian。
2. 求解 Newton 系统。
3. 用 Armijo 回溯线搜索选择可接受步长。
4. 必要时加入小的 ridge 稳定项改善数值条件。

### 解析 `sample_weight`

对明确定义加权曲率的损失函数和模型路径，Newton 使用同一个归一化加权目标：

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

目标函数、梯度、Hessian 和每个 Armijo 线搜索候选点都使用同一组权重。因此，把所有有效权重同时乘以同一个正数不会改变最优解。

均匀权重或与均匀权重等效的情况，在对应兼容路径中继续使用历史无权重数值路径。

---

## 8. L-BFGS / L-BFGS-B

**文件**：`statgpu/solvers/_lbfgs.py`、`statgpu/solvers/_lbfgs_b.py`

**用途**：光滑损失 + 光滑/无惩罚，并希望避免显式形成完整 Hessian 的场景。

### L-BFGS 算法

L-BFGS 使用标准的 limited-memory two-loop recursion 与 Armijo 线搜索。当前目标函数、每个线搜索候选点以及接受新点后的梯度，都必须基于同一个目标函数。

### 解析 `sample_weight`

直接调用 L-BFGS 时，非均匀权重需要由具体损失函数明确支持：

| 直接 L-BFGS 路径 | 非均匀 `sample_weight` |
|---|---|
| 当前维护的 `GLMLoss` | ✅ 支持 |
| 通用稳健 / 分位数 / Cox `LossBase` | ❌ 不能由无权重支持自动推出 |

对受支持的 GLM，初始梯度、当前目标函数、每个线搜索候选点以及接受新点后的梯度都使用同一组归一化权重；NumPy/CuPy/Torch 数值计算也保持在选定计算后端。

均匀权重继续兼容历史无权重 L-BFGS 路径。一个损失函数支持无权重 L-BFGS，并不代表它自动支持真正非均匀的 `sample_weight`。

`L-BFGS-B` 是独立的盒约束实现，不应默认继承 `lbfgs_solver` 的全部权重能力。

---

## 9. ADMM（Alternating Direction Method of Multipliers）

**文件**：`statgpu/solvers/_admm.py`

**用途**：适合变量分裂的受支持可分/近端形式。

### 通用结构

1. 在光滑目标 + 增广二次项下更新主系数变量。
2. 通过声明的近端算子更新分裂变量。
3. 更新缩放后的对偶变量。
4. 根据残差规则调整 penalty parameter。

---

## 10. `exact`（闭式路径）

**实现位置**：`_fit_mixin._solve_exact_*`

**用途**：squared error + L2 且自动分发选择闭式/特征分解路径的场景。

---

## 求解器分发

对普通直接拟合，`solver="auto"` 遵循模型层面的分发表。可以粗略理解为：

```text
direct fit with solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + sparse penalty → FISTA/FISTA-BB
├── smooth non-Gaussian GLM + L2   → Newton
├── SCAD/MCP/adaptive path          → LLA + FISTA-family inner solve
├── quantile                        → quantile-specific FISTA/IRLS path
└── group penalty                   → group-aware FISTA / FISTA-LLA
```

`PenalizedGLM_CV` 有一套相关但有意独立的光滑 L2 分发规则。尤其是 Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 交叉验证与最终重拟合使用 L-BFGS，而 logistic、Poisson、Tweedie 使用 Newton。不要从直接拟合的树状图推断 CV 行为，请以兼容性矩阵为准。

`sample_weight` 不会改变显式指定的 `solver`。如果请求的加权组合不受支持，则直接报错。

## 参考文献

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
