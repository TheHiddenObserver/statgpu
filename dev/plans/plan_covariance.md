# 协方差估计 GPU 实现计划

**状态**: ✅ 已实现 (2026-06-13, PR #57)

> 实现: `EmpiricalCovariance`, `LedoitWolf`, `OAS`

## 1. 现有包调研总结

### 1.1 Python 协方差估计包

| 包名 | 主要功能 | Ledoit-Wolf | 其他估计量 | 统计检验 |
|------|----------|-------------|------------|----------|
| **scikit-learn** (`sklearn.covariance`) | 协方差估计、收缩估计 | ✅ `LedoitWolf` 类 | `OAS`、`EmpiricalCovariance`、`GraphicalLasso` | ❌ |
| **statsmodels** | 统计建模、假设检验 | ❌ | 基础协方差功能 | ✅ 协方差检验 |
| **pingouin** | 统计分析包 | ❌ | 基础协方差 | ✅ 球形检验等 |
| **CuPy** | GPU 加速数值计算 | ❌ (需手动实现) | 基础线性代数 | ❌ |
| **RAPIDS cuML** | GPU 加速机器学习 | ✅ `cuml.covariance.LedoitWolf` | GPU 加速 PCA 等 | ❌ |

### 1.2 R 协方差估计包

| 包名 | 主要功能 | Ledoit-Wolf | 其他估计量 | 统计检验 |
|------|----------|-------------|------------|----------|
| **covShrinkage** | 收缩估计 | ✅ 线性 + 非线性 | `covShrink`、`corShrink` | ❌ |
| **nlshrink** | 非线性收缩 | ✅ 线性 + 非线性 | 特征值收缩 | ❌ |
| **CovEstim** | 协方差估计 | ✅ `cov_estim_lwone`、`cov_estim_lwnl` | 多种 LW 变体 | ❌ |
| **cvCovEst** | 交叉验证协方差估计 | ✅ `linearShrinkLWEst()` | 交叉验证选择 | ❌ |
| **GGMncv** | 高斯图模型 | ✅ `ledoit_wolf()` | 网络估计 | ❌ |
| **covatest** | 协方差函数检验 | ❌ | ❌ | ✅ 对称性、可分性检验 |
| **CovCorTest** | 协方差/相关矩阵检验 | ❌ | ❌ | ✅ 球形检验等 |
| **HDShOP** | 高维收缩组合 | ✅ 高维 LW | 组合优化 | ❌ |
| **spice** | 协方差收缩 | ✅ `ledoit_wolf_est()` | 多种收缩方法 | ❌ |

### 1.3 核心统计方法总结

#### Ledoit-Wolf 估计量类型

1. **线性收缩 (Linear Shrinkage)**
   - 原始 LW2004: `(1-α)S + α*F`，其中 F 为目标矩阵（通常为单位矩阵或恒相关矩阵）
   - 收缩系数 α 通过解析公式计算

2. **非线性收缩 (Nonlinear Shrinkage, LW2020)**
   - 对样本协方差的特征值进行非线性变换
   - 在高维情况下表现更好

3. **OAS (Oracle Approximating Shrinkage)**
   - 类似 LW 但使用不同的收缩系数公式
   - 计算更简单

#### 常见统计检验

| 检验类型 | 用途 | R 包 | Python 包 |
|----------|------|------|-----------|
| **球形检验 (Sphericity Test)** | 检验协方差矩阵是否为单位矩阵的倍数 | `covatest`, `CovCorTest`, `SpatialNP` | `pingouin` |
| **Mauchly 检验** | 重复测量方差分析的球形假设 | 基础 R | `pingouin` |
| **Box's M 检验** | 多组协方差矩阵相等性 | `biotools` | ❌ |
| **协方差对称性检验** | 时空协方差函数对称性 | `covatest` | ❌ |
| **可分性检验** | 时空协方差可分性 | `covatest` | ❌ |

---

## 2. GPU 实现可行性分析

### 2.1 可 GPU 加速的核心操作

Ledoit-Wolf 估计的核心计算包括：

1. **样本协方差矩阵计算**: `S = X.T @ X / (n-1)` — 矩阵乘法，高度可并行
2. **特征值分解**: `eigh(S)` — CuPy 已实现
3. **收缩系数计算**: 涉及迹、Frobenius 范数 — 可并行
4. **收缩组合**: 线性组合 — 可并行

### 2.2 现有 GPU 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **RAPIDS cuML** | 原生 GPU 支持、API 友好 | 依赖 RAPIDS 生态、仅限 NVIDIA GPU |
| **CuPy 手动实现** | 灵活、轻量 | 需自行维护 |
| **PyTorch/TensorFlow** | 自动求导、多 GPU | 过度复杂、依赖重 |

---

## 3. statgpu 协方差模块实现计划

### 3.1 模块结构

```
statgpu/
└── covariance/
    ├── __init__.py              # 导出接口
    ├── _base.py                 # 基础协方差类
    ├── _shrunk_covariance.py    # 收缩估计 (Ledoit-Wolf, OAS)
    ├── _graphical_lasso.py      # 稀疏逆协方差估计
    └── _hypothesis_tests.py     # 协方差假设检验
```

### 3.2 阶段性计划

#### Phase 1: 基础实现 (1-2 周)

**目标**: 实现核心 Ledoit-Wolf 收缩估计

- [ ] **1.1** 创建 `statgpu/covariance/` 目录结构
- [ ] **1.2** 实现 `_base.py` - `EmpiricalCovariance` 基类
  - `fit(X)` - 计算经验协方差
  - `score(X)` - 计算对数似然
- [ ] **1.3** 实现 `_shrunk_covariance.py` - `LedoitWolf` 类
  - GPU 版本使用 CuPy 后端
  - CPU 版本使用 NumPy 后端
  - 实现 `fit()` 和 `ledoit_wolf_shrinkage()` 方法
- [ ] **1.4** 实现 `OAS` (Oracle Approximating Shrinkage) 类
- [ ] **1.5** 添加基础单元测试

**验收标准**:
- 与 scikit-learn 的 `LedoitWolf` 输出一致 (允许数值误差)
- GPU 加速版本在 p > 100 时显著快于 CPU

#### Phase 2: 扩展功能 (2-3 周)

**目标**: 实现非线性收缩和交叉验证

- [ ] **2.1** 实现非线性收缩 (Ledoit-Wolf 2020)
  - 特征值非线性收缩
  - 支持高维数据 (p > n)
- [ ] **2.2** 实现交叉验证选择收缩参数
  - `cvCovEst` 风格的 K 折交叉验证
  - 支持自定义损失函数
- [ ] **2.3** 添加目标矩阵选项
  - 单位矩阵收缩
  - 恒相关矩阵收缩
  - 自定义目标矩阵
- [ ] **2.4** 性能基准测试
  - 与 R `covShrinkage`、`nlshrink` 对比
  - 与 scikit-learn 对比

**验收标准**:
- 非线性收缩与 R `nlshrink` 包结果一致
- 交叉验证能正确选择最优收缩参数

#### Phase 3: 统计检验 (2-3 周)

**目标**: 实现协方差假设检验

- [ ] **3.1** 实现球形检验 (Sphericity Test)
  - 检验 H₀: Σ = σ²I
  - 基于 John (1971) 或 Ledoit-Wolf (2002) 统计量
- [ ] **3.2** 实现协方差矩阵相等性检验 (Box's M Test)
  - 多组协方差矩阵齐性检验
- [ ] **3.3** 实现恒等性检验
  - 检验 H₀: Σ = I
- [ ] **3.4** 添加 GPU 加速版本
  - 大规模数据的快速检验

**验收标准**:
- 检验结果与 R `CovCorTest`、`pingouin` 一致
- p 值计算准确

#### Phase 4: 高级功能 (3-4 周)

**目标**: 实现 Graphical Lasso 和高维估计

- [ ] **4.1** 实现 Graphical Lasso (稀疏逆协方差估计)
  - 基于玻璃索算法
  - GPU 加速迭代求解
- [ ] **4.2** 实现最小体积椭圆估计 (MVE)
  - 稳健协方差估计
- [ ] **4.3** 实现 PCA 白化功能
  - 基于协方差特征分解
- [ ] **4.4** 文档和示例

**验收标准**:
- Graphical Lasso 与 sklearn 结果一致
- 完整的 API 文档和示例代码

---

## 4. 技术细节

### 4.1 Ledoit-Wolf 收缩公式

**线性收缩**:
```
Σ̂ = (1 - α) * S + α * F
```
其中:
- `S` = 样本协方差矩阵
- `F` = 目标矩阵 (通常为 `μI`，`μ = trace(S)/p`)
- `α` = 收缩系数，通过以下公式计算:

```
α = (∑(i≠j) Var(x_i * x_j)) / (∑(i≠j) Cov(x_i, x_j)²)
  ≈ (1/n) * (||S||_F² / ||S - μI||_F²)
```

**OAS 收缩系数**:
```
α_OAS = min(1, (1 - 2/p) * (trace(S²) / (trace(S)²))) / (n * (1 + (trace(S²) / (trace(S)²)) - 2*(trace(S³) / (trace(S)² * trace(S)))))
```

### 4.2 GPU 实现要点

```python
import cupy as cp

def ledoit_wolf_gpu(X, assume_centered=False):
    """GPU 加速 Ledoit-Wolf 估计"""
    n_samples, n_features = X.shape
    
    # 中心化
    if not assume_centered:
        X = X - X.mean(axis=0)
    
    # 计算样本协方差 (GPU)
    S = (X.T @ X) / n_samples
    
    # 计算收缩目标
    mu = cp.trace(S) / n_features
    F = mu * cp.eye(n_features)
    
    # 计算收缩系数 (Ledoit-Wolf 公式)
    delta = S - F
    shrinkage = cp.sum(delta ** 2) / (n_samples * cp.sum((X ** 2 @ X.T / n_samples - S) ** 2))
    shrinkage = cp.clip(shrinkage, 0, 1)
    
    # 收缩估计
    sigma = (1 - shrinkage) * S + shrinkage * F
    
    return sigma, shrinkage
```

### 4.3 后端抽象

利用 statgpu 现有的后端架构:

```python
from statgpu.backends import get_backend

def fit(self, X):
    xp = get_backend(X)  # 自动选择 numpy 或 cupy
    
    # 通用实现，xp 可以是 numpy 或 cupy
    S = xp.cov(X, rowvar=False)
    # ...
```

---

## 5. 参考资料

### 论文

1. Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices". *Journal of Multivariate Analysis*.

2. Ledoit, O., & Wolf, M. (2020). "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection". *Review of Financial Studies*.

3. Ledoit, O., & Wolf, M. (2012). "Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices". *Annals of Statistics*.

### R 包文档

- covShrinkage: https://github.com/MikeWolf007/covShrinkage
- nlshrink: https://cran.r-project.org/web/packages/nlshrink/
- CovEstim: https://rdrr.io/github/antshi/CovEstim/
- cvCovEst: https://cran.r-project.org/web/packages/cvCovEst/

### Python 实现参考

- scikit-learn: https://scikit-learn.org/stable/modules/covariance.html
- RAPIDS cuML: https://docs.rapids.ai/api/cuml/stable/api/
- CuPy: https://cupy.dev/

---

## 6. 风险与注意事项

1. **数值稳定性**: 高维数据 (p >> n) 时特征值分解可能不稳定
2. **内存限制**: GPU 内存限制大矩阵处理 (考虑分块/流式处理)
3. **测试覆盖**: 需要与 R/Python 现有包进行广泛对比验证
4. **文档完整性**: 统计方法需要清晰的数学公式和使用示例

---

## 7. 时间估算

| 阶段 | 内容 | 预估时间 |
|------|------|----------|
| Phase 1 | 基础 Ledoit-Wolf + OAS | 1-2 周 |
| Phase 2 | 非线性收缩 + 交叉验证 | 2-3 周 |
| Phase 3 | 统计检验 | 2-3 周 |
| Phase 4 | Graphical Lasso + 高级功能 | 3-4 周 |
| **总计** | | **8-12 周** |

---

*文档创建日期：2026-04-20*
*statgpu 项目 - 协方差估计模块开发计划*
