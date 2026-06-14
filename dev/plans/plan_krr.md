# Kernel Ridge Regression GPU 实现计划

**状态**: ✅ 接近完整 (~60%) (2026-06-13, PR #57)

> 已实现: `KernelRidge`, `KernelRidgeCV`, `pairwise_kernels` — 6 核函数 (rbf/polynomial/linear/laplacian/sigmoid/cosine), CPU/CuPy/Torch
> 缺失: chi2 核, Nystroem 近似, KernelPCA, sample_weight 支持

## 1. 背景与目标

### 1.1 什么是 Kernel Ridge Regression (KRR)

核岭回归 (Kernel Ridge Regression, KRR) 是一种结合了岭回归 (L2 正则化) 和核技巧 (Kernel Trick) 的非参数回归方法。其核心思想是：

1. 将原始特征空间的数据映射到高维特征空间
2. 在高维空间中进行岭回归
3. 通过核函数避免显式计算高维映射

**数学形式**:
```
给定训练数据 {(x_i, y_i)}_{i=1}^n，核函数 K(·,·)

KRR 的预测公式为：f(x) = K(x, X) @ (K(X, X) + αI)^{-1} @ y

其中：
- K(X, X) 是 n×n 的核矩阵
- α > 0 是正则化参数
- K(x, X) 是测试点 x 与所有训练点的核函数值
```

### 1.2 现有实现调研

#### Python 生态

| 包名 | 主要功能 | GPU 支持 | 特点 |
|------|----------|----------|------|
| **scikit-learn** (`KernelRidge`) | 基础 KRR 实现，多种核函数 | 否 (CPU only) | 行业标准，API 成熟 |
| **NVIDIA cuML** | sklearn 兼容的 GPU 加速 | 是 (CUDA) | 零代码修改加速 |
| **Himalaya** | 多核岭回归，神经编码 | 是 (PyTorch CUDA) | 专门优化 KRR |
| **GPyTorch** | 高斯过程 + 核方法 | 是 (PyTorch) | 支持深度核学习 |

#### R 生态

| 包名 | 主要功能 | GPU 支持 | 特点 |
|------|----------|----------|------|
| **kernlab** (`ksrr`) | 核方法统一框架 | 否 | S4 类设计，可扩展 |
| **caret** | 统一建模接口 | 否 | 集成多个包 |
| **listdtr** (`krr`) | 专用 KRR 实现 | 否 | 简洁 API |
| **DRR** | 快速 KRR | 否 | 适合大数据集 |

### 1.3 功能对比分析

| 功能模块 | scikit-learn | kernlab | cuML | Himalaya | statgpu (计划) |
|----------|--------------|---------|------|----------|----------------|
| **模型估计** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **交叉验证** | GridSearchCV | train() | ✓ | ✓ | ✓ |
| **多核函数** | 6 种 | 10+ 种 | 有限 | 5 种 | 6+ 种 |
| **多输出回归** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **稀疏核方法** | ✗ | ✗ | ✗ | ✓ | 计划中 |
| **GPU 加速** | 有限 | ✗ | ✓ | ✓ | ✓ |
| **统计推断** | ✗ | ✗ | ✗ | ✗ | 计划中 |

---

## 2. 核心功能设计

### 2.1 模型估计 (Model Estimation)

#### 2.1.1 闭式解 (Closed-form Solution)

```
系数求解：α = (K + λI)^{-1} y

预测：f(x*) = Σ α_i K(x*, x_i)
```

**实现要点**:
- Cholesky 分解：`K + λI = L @ L.T`
- 前代 + 回代求解
- GPU 并行核矩阵计算

#### 2.1.2 支持的核函数

| 核函数 | 公式 | 参数 |
|--------|------|------|
| Linear | `K(x,x') = x·x'` | - |
| Polynomial | `K(x,x') = (γ x·x' + c)^d` | γ, c, d |
| RBF/Gaussian | `K(x,x') = exp(-γ||x-x'||²)` | γ |
| Laplacian | `K(x,x') = exp(-γ||x-x'||₁)` | γ |
| Sigmoid | `K(x,x') = tanh(γ x·x' + c)` | γ, c |
| Cosine | `K(x,x') = (x·x')/(||x||||x'||)` | - |

### 2.2 模型选择 (Model Selection)

#### 2.2.1 交叉验证 (Cross-Validation)

```python
class KernelRidgeCV:
    """K-fold 交叉验证选择超参数"""
    
    def fit(self, X, y):
        # 1. 生成超参数网格
        # 2. K-fold 划分
        # 3. 每折训练 + 验证
        # 4. 选择最优参数
        pass
```

**优化策略**:
-  Leave-One-Out CV (LOOCV) 高效实现：利用 Sherman-Morrison 公式
-  Generalized Cross-Validation (GCV)
-  K-fold 的批量 GPU 求解

#### 2.2.2 超参数网格

```python
param_grid = {
    'alpha': np.logspace(-4, 4, 20),  # 正则化参数
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': np.logspace(-3, 3, 10),  # 核参数
    'degree': [2, 3, 4],               # 多项式次数
}
```

### 2.3 统计推断 (Statistical Inference) - *特色功能*

#### 2.3.1 系数推断

```
var(α̂) = σ² (K + λI)^{-1} K (K + λI)^{-1}

标准误：SE(α̂_i) = √var(α̂_i)

t 统计量：t_i = α̂_i / SE(α̂_i)

p 值：p_i = 2 * (1 - Φ(|t_i|))
```

#### 2.3.2 预测区间

```
给定新样本 x*，预测区间为：

f(x*) ± t_{α/2, n-p} * √(σ² + var(f(x*)))
```

#### 2.3.3 模型诊断

- 残差分析
- 异方差检验
- 核函数选择准则 (GCV, AIC, BIC)

---

## 3. GPU 加速策略

### 3.1 核矩阵计算的 GPU 优化

#### 3.1.1 RBF 核矩阵的 GPU 实现

```python
#  naive 实现：O(n²d) 全局内存访问
#  K[i,j] = exp(-γ ||x_i - x_j||²)

# 优化方案 1：利用 ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2x_i·x_j
def rbf_kernel_gpu(X, gamma):
    # X: (n, d)
    X_norm_sq = xp.sum(X**2, axis=1)  # (n,)
    K = -2.0 * X @ X.T                # (n, n) 矩阵乘法
    K += X_norm_sq[:, None]           # 广播
    K += X_norm_sq[None, :]
    K *= -gamma
    xp.exp(K, out=K)
    return K

# 优化方案 2：分块计算 (处理大矩阵)
def rbf_kernel_blocked(X, gamma, block_size=4096):
    n = X.shape[0]
    K = xp.empty((n, n), dtype=xp.float64)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            X_i = X[i:i+block_size]
            X_j = X[j:j+block_size]
            K[i:i+block_size, j:j+block_size] = rbf_kernel_gpu(X_i, gamma)
    return K
```

### 3.2 线性系统求解的 GPU 优化

#### 3.2.1 Cholesky 分解 (cuSOLVER)

```python
import cupy as cp
from cupyx.scipy.linalg import cho_factor, cho_solve

# GPU 上的 Cholesky 分解
L, lower = cho_factor(K + alpha * I)
coefficients = cho_solve((L, lower), y)
```

#### 3.2.2 特征分解 (多 α 同时求解)

```python
# 一次性求解多个 α 值（用于交叉验证）
# K = Q @ Λ @ Q.T (特征分解)
# α = Q @ diag(1/(λ_i + α)) @ Q.T @ y

def solve_krr_path_eig(K, y, alphas):
    eigvals, Q = xp.linalg.eigh(K)
    QTy = Q.T @ y
    solutions = []
    for alpha in alphas:
        inv_diag = 1.0 / (eigvals + alpha)
        sol = Q @ (inv_diag * QTy)
        solutions.append(sol)
    return xp.stack(solutions, axis=1)
```

### 3.3 内存优化

#### 3.3.1 低精度计算 (Mixed Precision)

```python
# 核矩阵计算使用 float32，最终求解用 float64
def compute_kernel_mixed_precision(X, gamma):
    X_fp32 = X.astype(xp.float32)
    K_fp32 = rbf_kernel_gpu(X_fp32, gamma)
    return K_fp32.astype(xp.float64)  # 转回 double 用于求解
```

#### 3.3.2 核矩阵近似 (Nyström 方法)

```python
# 适用于大规模数据集 (n > 10000)
def nystrom_approximation(X, m=500, gamma=None):
    """Nyström 低秩近似"""
    n = X.shape[0]
    idx = xp.random.choice(n, m, replace=False)
    X_landmark = X[idx]
    
    C = rbf_kernel_gpu(X, X_landmark, gamma)  # (n, m)
    W = rbf_kernel_gpu(X_landmark, gamma)     # (m, m)
    
    # K ≈ C @ W^{-1} @ C.T
    return C, W
```

---

## 4. 实现计划

### 4.1 阶段划分

| 阶段 | 内容 | 预计时间 | 交付物 |
|------|------|----------|--------|
| **Phase 1** | 基础 KRR 实现 | 1-2 周 | `KernelRidge` 类 (CPU+GPU) |
| **Phase 2** | 交叉验证 | 1-2 周 | `KernelRidgeCV` 类 |
| **Phase 3** | 统计推断 | 2-3 周 | 标准误、置信区间 |
| **Phase 4** | 大规模优化 | 2-3 周 | Nyström 近似、稀疏 KRR |
| **Phase 5** | 测试与基准 | 1-2 周 | 单元测试、性能对比 |

### 4.2 文件结构

```
statgpu/
├── kernel_methods/
│   ├── __init__.py
│   ├── _krr.py              # 核心 KRR 实现
│   ├── _krr_cv.py           # 交叉验证
│   ├── _krr_inference.py    # 统计推断
│   ├── _kernels.py          # 核函数实现
│   └── _nystrom.py          # Nyström 近似
├── backends/
│   ├── _cupy_kernels.py     # CuPy 核函数
│   └── _torch_kernels.py    # PyTorch 核函数
└── tests/
    └── test_kernel_methods/
        ├── test_krr.py
        ├── test_krr_cv.py
        └── test_kernels.py
```

### 4.3 Phase 1 详细任务清单

#### 任务 1.1: 核函数模块 (`_kernels.py`)

- [ ] 实现 6 种核函数 (linear, poly, rbf, laplacian, sigmoid, cosine)
- [ ] GPU 优化的 RBF 核 (利用矩阵乘法)
- [ ] 核函数梯度计算 (用于带宽选择)
- [ ] 单元测试 (与 sklearn kernlab 对比)

#### 任务 1.2: 核心 KRR 类 (`_krr.py`)

- [ ] `KernelRidge` 类 (sklearn 兼容 API)
- [ ] 拟合方法：`fit(X, y)`
- [ ] 预测方法：`predict(X)`
- [ ] 评分方法：`score(X, y)` (R²)
- [ ] GPU 后端支持 (CuPy + PyTorch)
- [ ] 多输出回归支持

#### 任务 1.3: 后端集成

- [ ] CuPy 后端核矩阵计算
- [ ] PyTorch 后端核矩阵计算
- [ ] 自动后端选择 (基于输入类型)
- [ ] 内存管理 (GPUMemoryCleanup)

---

## 5. 基准测试设计

### 5.1 正确性验证

```python
# 与 scikit-learn 对比
from sklearn.kernel_ridge import KernelRidge as SKLearnKRR

def test_correctness():
    X_train, y_train = make_regression(n_samples=500, n_features=20)
    X_test = X_train[:100]
    
    # statgpu
    model_statgpu = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1, device='cuda')
    model_statgpu.fit(X_train, y_train)
    pred_statgpu = model_statgpu.predict(X_test)
    
    # sklearn
    model_sklearn = SKLearnKRR(alpha=1.0, kernel='rbf', gamma=0.1)
    model_sklearn.fit(X_train, y_train)
    pred_sklearn = model_sklearn.predict(X_test)
    
    # 验证 (允许数值误差)
    assert np.allclose(pred_statgpu, pred_sklearn, rtol=1e-5)
```

### 5.2 性能基准

| 数据集规模 | CPU (numpy) | GPU (cupy) | 加速比 |
|------------|-------------|------------|--------|
| n=1,000 | - | - | - |
| n=5,000 | - | - | - |
| n=10,000 | - | - | - |
| n=50,000 | - | - | - |

### 5.3 交叉验证效率

比较不同 CV 策略的时间:
- 标准 K-fold
- LOOCV (高效实现)
- GCV

---

## 6. 与 R 实现的对比验证

### 6.1 kernlab 对比

```R
# R kernlab 实现
library(kernlab)
model <- ksrr(x ~ ., data=train, kernel=rbfdot(sigma=0.1), C=1.0)
predictions <- predict(model, newdata=test)
```

### 6.2 验证协议

1. 使用相同随机种子生成数据
2. 相同的核函数参数
3. 相同的正则化参数
4. 对比预测结果 (RMSE < 1e-5)

---

## 7. 风险与挑战

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| 核矩阵内存溢出 (O(n²)) | 高 | Nyström 近似、分块计算 |
| GPU 数值稳定性 | 中 | Mixed precision、条件数检查 |
| 大规模 CV 时间过长 | 中 | 批量求解、特征分解路径 |
| 与 sklearn API 兼容性 | 低 | 严格遵循 sklearn 接口规范 |

---

## 8. 参考资源

### 8.1 文档与教程

- [scikit-learn KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html)
- [kernlab vignette](https://cran.r-project.org/web/packages/kernlab/vignettes/kernlab.pdf)
- [Himalaya documentation](https://gallantlab.org/himalaya/)
- [NVIDIA cuML](https://developer.nvidia.com/blog/nvidia-cuml-brings-zero-code-change-acceleration-to-scikit-learn/)

### 8.2 关键论文

1. Saunders, C., Gammerman, A., & Vovk, V. (1998). Ridge regression learning algorithm in dual variables. ICML.
2. Williams, C. K., & Seeger, M. (2001). Using the Nyström method to speed up kernel machines. NeurIPS.
3. Rakotomamonjy, A. (2022). Efficient Hyperparameter Tuning for Large Scale Kernel Ridge Regression.

### 8.3 代码参考

- [scikit-learn kernel_ridge.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/kernel_ridge.py)
- [Himalaya kernel ridge](https://github.com/gallantlab/himalaya)
- [kernlab source](https://cran.r-project.org/src/contrib/kernlab_0.9-33.tar.gz)

---

## 9. 附录：API 设计草稿

```python
from statgpu.kernel_methods import KernelRidge, KernelRidgeCV

# 基础使用
model = KernelRidge(
    alpha=1.0,
    kernel='rbf',
    gamma=0.1,
    device='cuda'  # 或 'cpu', 'auto'
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)

# 交叉验证
model_cv = KernelRidgeCV(
    alphas=np.logspace(-4, 4, 20),
    kernels=['rbf', 'poly'],
    gammas=np.logspace(-3, 3, 10),
    cv=5,
    device='cuda'
)
model_cv.fit(X_train, y_train)
print(f"Best alpha: {model_cv.alpha_}")
print(f"Best kernel: {model_cv.kernel_}")

# 统计推断
from statgpu.kernel_methods import KernelRidgeInference

inference = KernelRidgeInference(model_cv)
summary = inference.summary()
confidence_intervals = inference.predict_confidence(X_test, alpha=0.05)
```

---

*文档创建日期：2026-04-20*
*最后更新：2026-04-20*
