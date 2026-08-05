# Elastic Net 弹性网络

> Language: Chinese (中文)  
> Last updated: 2026-07-24<br>
> This page: 模型文档  
> Language switch: [English](../../en/models/elastic-net.md)

## 概述

`ElasticNet` 结合了 L1 和 L2 正则化，在稀疏特征选择（Lasso）和系数收缩（Ridge）之间取得平衡。支持 CPU、CuPy GPU 和 PyTorch GPU 后端，可灵活选择设备。

## 路径

`statgpu.linear_model.ElasticNet`

## 目标函数

Elastic Net 优化问题为：

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \cdot \lambda \cdot \|\beta\|_1 + \frac{\alpha}{2} \cdot (1 - \lambda) \cdot \|\beta\|_2^2
$$

其中：
- `alpha` (α) 控制整体正则化强度
- `l1_ratio` (λ) 混合 L1 和 L2：λ=1 为 Lasso，λ=0 为 Ridge
- 损失函数缩放因子 `1/(2n)` 使 `alpha` 的解释与样本量无关

**正则化缩放说明**：当 `l1_ratio=0` 时，`ElasticNet(alpha)` 等价于 `Ridge(n_samples * alpha)`，这是由于损失函数的缩放约定。

## 估计方程

Elastic Net 估计量求解以下一阶最优性（KKT）条件：

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0
$$

其中 $\partial\|\hat{\beta}\|_1$ 是 L1 范数的次微分：
- 当 $\hat{\beta}_j \neq 0$ 时：$\text{sign}(\hat{\beta}_j)$
- 当 $\hat{\beta}_j = 0$ 时：$[-1, 1]$ 中的任意值

收敛时，KKT 残差（次梯度违反）满足：
$$
\left| \frac{1}{n} X_j^\top(y - X\hat{\beta}) - \alpha(1-\lambda)\hat{\beta}_j \right| \leq \alpha\lambda \quad \forall j
$$

## 估计算法

Elastic Net 通过 **FISTA**（快速迭代收缩阈值算法）求解，这是一种带有 Nesterov 动量加速的 proximal gradient 方法。

### 关键优化洞察

L2 正则化项**仅在 proximal 步中处理**，不在梯度计算中处理：

```python
# 仅计算 RSS 的梯度（L2 单独处理）
grad = (X.T @ X @ w - X.T @ y) / n

# Proximal 步：软阈值 + L2 缩放
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (1 + alpha * (1 - l1_ratio) * step)
```

这避免了冗余计算并提高了数值稳定性。

### 收敛判据

通过 `stopping` 参数提供两种停止模式：

| 模式 | 说明 |
|------|------|
| `coef_delta` | 当 `||w_new - w_old||_∞ < tol` 时停止 |
| `kkt` | 当 KKT 次梯度违反 < tol 时停止 |

对于 `kkt` 模式，最优性条件为：
- 对于非零系数：`|∇f + α(1-λ)w + αλ·sign(w)| < tol`
- 对于零系数：`|∇f + α(1-λ)w| ≤ αλ`

**注意**：数值优化中 KKT 违反 ~1e-2 是可接受的；不需要精确为零。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | `1.0` | 总体正则化强度 |
| `l1_ratio` | `0.5` | L1 混合比例：0=Ridge，1=Lasso |
| `fit_intercept` | `True` | 拟合不受惩罚的截距 |
| `max_iter` | `1000` | 最大求解迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | `"coef_delta"` 或 `"kkt"` 停止准则 |
| `device` | `"auto"` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"` |
| `n_jobs` | `None` | 适用 CPU 路径的并行度 |
| `solver` | `"fista"` | 后端感知的优化方法 |
| `cpu_solver` | `"fista"` | CPU 求解器覆盖选项 |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开 wrapper 不接受单独的 `backend`、`warm_start` 或 `random_state`
构造参数。后端由 `device` 控制；单次拟合的 warm start 可通过
`fit(initial_coef=...)` 提供。

## CPU/GPU 示例

```python
from statgpu.linear_model import ElasticNet

# CPU (NumPy)
model_cpu = ElasticNet(alpha=0.1, l1_ratio=0.5, device="cpu")
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU (CuPy)
model_gpu_cupy = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="cuda",
    gpu_memory_cleanup=True
)
model_gpu_cupy.fit(X, y)

# GPU (PyTorch，推荐用于 n >= 10,000)
model_gpu_torch = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="torch"
)
model_gpu_torch.fit(X, y)
```

### 按数据规模选择后端

| 数据规模 | 推荐后端 | 相对 sklearn 加速比 |
|----------|----------|---------------------|
| n < 1,000 | CPU (NumPy) | 0.7x - 1.0x |
| 1,000 ≤ n < 10,000 | CPU (NumPy) | 1.5x - 4x |
| 10,000 ≤ n < 50,000 | GPU (Torch) | 2x - 3x |
| n ≥ 50,000 | GPU (Torch) | 3x - 4.4x |

## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，将通过共享的
penalized-linear 推断引擎执行拟合后推断。默认
`inference_method="debiased"` 使用 nodewise Lasso 构造偏误校正估计量、标准误、
z 统计量、p 值与 95% 置信区间；推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

NumPy、CuPy 与 Torch 拟合路径均已实现 debiased 推断。CPU 验证属于托管测试套件；
每个精确发布候选仍必须通过 CuPy 与 Torch 的物理 CUDA 远程验证。
Post-selection OLS 只是启发式方法，不保证有效的选择后覆盖率。推断以已选定的正则化
参数为条件，不会改变原 penalized coefficient。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio`
选定后的全数据最终重拟合；各折模型仍仅用于估计和评分。

## 求解器与推断语义

默认估计器使用 FISTA 优化声明的 Elastic Net 目标函数。`stopping` 仅改变
收敛诊断（`coef_delta` 或 KKT violation），并不定义不同的统计近似模式。

`compute_inference=False` 只返回 penalized estimate。设置
`compute_inference=True` 后，原拟合系数保持不变，并在拟合完成后运行所选推断方法。
独立 `ElasticNet` wrapper 与 `ElasticNetCV` 的最终全数据重拟合都直接支持该契约；
用户不需要仅为了 debiased inference 而切换到其他估计器类。

## 输出属性

拟合后可用以下属性：

| 属性 | 说明 |
|------|------|
| `coef_` | 估计的系数 (形状：n_features) |
| `intercept_` | 拟合的截距 |
| `n_iter_` | 收敛所需迭代次数 |
| `aic` | Akaike 信息准则（如可用） |
| `bic` | Bayesian 信息准则（如可用） |

方法：`fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## 数值一致性

所有 statgpu 后端（CPU、CuPy、Torch）产生数值一致的结果：

| 后端对比 | 最大系数差异 |
|----------|--------------|
| CPU vs CuPy | < 3e-8 |
| CPU vs Torch | < 3e-8 |
| 全部 vs sklearn | < 3e-8 |

## 性能基准测试

### vs sklearn (Python)

| 数据集 | n | p | sklearn (ms) | statgpu CPU (ms) | 加速比 |
|--------|---|---|--------------|------------------|--------|
| small | 200 | 20 | 0.77 | 1.10 | 0.70x |
| medium | 1,000 | 50 | 10.42 | 2.37 | **4.40x** |
| large | 5,000 | 100 | 6.01 | 4.13 | **1.45x** |

### vs glmnet (R)

| 数据集 | n | p | R glmnet (ms) | statgpu CPU (ms) | 胜者 |
|--------|---|---|---------------|------------------|------|
| small | 200 | 20 | 8.51 | **1.10** | statgpu |
| medium | 1,000 | 50 | 6.27 | **2.06** | statgpu |
| large | 5,000 | 100 | 10.70 | **6.14** | statgpu |

statgpu CPU 在与 R glmnet 的 6 项对比中赢得 4 项。

### 大规模性能 (n ≥ 10,000)

| 数据集 | n | p | sklearn | statgpu CPU | statgpu Torch | Torch 加速比 |
|--------|---|---|---------|-------------|---------------|--------------|
| n_10k_p100 | 10,000 | 100 | 11.39 | 11.24 | 12.02 | 0.95x |
| n_10k_p500 | 10,000 | 500 | 82.03 | 100.51 | **30.52** | **2.69x** |
| n_50k_p100 | 50,000 | 100 | 69.74 | 52.01 | **21.74** | **3.21x** |
| n_50k_p500 | 50,000 | 500 | 310.34 | 145.31 | **79.77** | **3.89x** |
| n_100k_p100 | 100,000 | 100 | 118.94 | 60.85 | **33.23** | **3.58x** |
| n_100k_p500 | 100,000 | 500 | 615.59 | 269.45 | **141.05** | **4.36x** |

**关键发现**：
- statgpu Torch 在 6 项大规模测试中 5 项最快 (83%)
- 最大加速比：**4.36x** (n=100k, p=500 对比 sklearn)
- GPU 加速在 n ≥ 10,000 时开始显现优势

## 常见问题

**Q: 如何选择 l1_ratio？**
- `l1_ratio=1.0`: 纯 Lasso（稀疏解）
- `l1_ratio=0.0`: 纯 Ridge（密集收缩）
- `l1_ratio=0.5`: 平衡（默认值）
- 可通过交叉验证选择最优预测性能

**Q: 为什么 CPU 和 GPU 的迭代次数不同？**
不同的数值路径和浮点运算可能导致略微不同的收敛轨迹。应比较最终系数和 R²而非迭代次数。

**Q: 何时使用 GPU vs CPU？**
- n < 10,000: CPU 更快（无数据传输开销）
- n ≥ 10,000: GPU (Torch 后端) 显示 2-4 倍加速
- n ≥ 50,000: CuPy 和 Torch 都显示显著优势

**Q: 为什么系数与 sklearn 有 ~1e-8 的差异？**
这在浮点运算的数值精度范围内。所有后端都以 1e-6 的容忍度求解相同的优化问题。

**Q: alpha 与 Ridge/Lasso 的关系？**
- `ElasticNet(l1_ratio=1.0, alpha=X)` ≈ `Lasso(alpha=X)`
- `ElasticNet(l1_ratio=0.0, alpha=X)` ≈ `Ridge(alpha=n_samples * X)`

## 外部验证

基准测试脚本：
- `dev/benchmarks/benchmark_elasticnet_sklearn.py` - sklearn 对比
- `dev/benchmarks/benchmark_glmnet_full.R` - R glmnet 对比
- `dev/benchmarks/benchmark_large_scale.py` - 大规模性能测试
- `dev/benchmarks/run_full_benchmark.py` - 统一基准运行器

测试脚本：
- `dev/scripts/remote_elasticnet_smoke.py` - 基础验证
- `dev/scripts/remote_stability_en.py` - 数值稳定性测试

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320. [https://doi.org/10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Mathematical Programming*, 103(1), 127-152. [https://doi.org/10.1007/s10107-004-0552-5](https://doi.org/10.1007/s10107-004-0552-5)
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202. [https://doi.org/10.1137/080716542](https://doi.org/10.1137/080716542)
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://www.jstatsoft.org/v33/i01/](https://www.jstatsoft.org/v33/i01/)
