# 有序广义线性模型 (Ordered Logit/Probit)

> 语言: 中文  
> 最后更新: 2026-04-26  
> 切换: [English](../en/models/ordered.md)

有序响应模型，适用于目标变量为序数类别（如"低/中/高"）的场景。

## 模型形式

P(y <= j | X) = F(theta_j - X * beta)

其中：
- `j = 1, ..., K-1` 为类别阈值
- `F` 为累积分布函数（Logit 或 Probit）
- `theta_j` 为阈值参数（严格递增）
- `beta` 为系数向量（比例优势假设：所有类别共享同一系数）

## 当前实现

### OrderedLogitRegression

比例优势模型（proportional odds），使用 Logit 链接函数。

```python
from statgpu.linear_model import OrderedLogitRegression

model = OrderedLogitRegression(
    n_categories=3,        # 类别数
    fit_intercept=True,    # 是否拟合截距
    max_iter=100,          # 最大迭代数
    tol=1e-4,              # 收敛容差
    C=1.0,                 # 逆正则化强度
    device='auto',         # 'auto' | 'cpu' | 'cuda' | 'torch'
)
model.fit(X, y)
print(model.coef_)         # 系数
print(model.thresholds_)   # 阈值 [-inf, theta_1, ..., theta_{K-1}, +inf]
```

### OrderedProbitRegression

使用 Probit 链接函数的有序模型。

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, device='cuda')
model.fit(X, y)
```

## 后端支持

| 后端 | 优化器 | 说明 |
|------|--------|------|
| numpy (CPU) | scipy L-BFGS-B | 参考实现 |
| cupy (GPU) | 手写 L-BFGS + Armijo line search | 需要 CuPy |
| torch (GPU) | torch.optim.LBFGS + strong_wolfe | 需要 PyTorch >= 1.13 |

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| n_categories | int | 3 | 序数类别数（>= 2） |
| fit_intercept | bool | True | 是否拟合截距项 |
| max_iter | int | 100 | L-BFGS 最大迭代数 |
| tol | float | 1e-4 | 收敛容差 |
| C | float | 1.0 | 逆正则化强度（C 越大正则化越弱） |
| device | str | 'auto' | 计算设备 |
| gpu_memory_cleanup | bool | False | 拟合后清理 GPU 显存 |

## 属性（拟合后）

| 属性 | 说明 |
|------|------|
| coef_ | 系数向量 (p,) |
| thresholds_ | 阈值向量 [-inf, theta_1, ..., +inf] (K+1,) |
| n_iter_ | 实际迭代次数 |
| _bse | 标准误 |
| _pvalues | p 值 |

## 跨后端精度

2026-04-26 修复后，跨后端 coef 最大差异 < 1e-2：

| 对比 | Max |Delta Coef| |
|------|---------------------------|
| CPU vs CuPy | < 1e-2 |
| CPU vs Torch | < 1e-2 |
| CuPy vs Torch | < 1e-2 |

## 参考文献

- McCullagh, P. (1980). Regression models for ordinal data.
- Agresti, A. (2010). Analysis of Ordinal Categorical Data.
