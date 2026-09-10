# 自适应 Lasso（Adaptive Lasso）

> 语言：中文  
> 最后更新：2026-09-04
> 页面定位：模型文档  
> 切换：[English](../../en/models/adaptive-lasso.md)

语言切换：[English](../../en/models/adaptive-lasso.md)

## 概述

`AdaptiveLasso` 提供自适应 L1 惩罚线性回归（Zou, 2006）。与标准 Lasso 使用统一惩罚不同，Adaptive Lasso 根据初始估计为每个坐标分配数据驱动的权重 $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$，具有 **oracle property**——渐近表现如同已知真实模型。

## 路径

`statgpu.linear_model.AdaptiveLasso`

## 目标函数

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \sum_{j=1}^p w_j |\beta_j|
$$

其中 $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ 为自适应权重，由初始估计（默认为岭回归）计算。

## 算法

1. **初始化**：通过岭惩罚坐标下降计算初始系数估计（匹配 R glmnet 的岭求解器）
2. **权重计算**：$w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$，默认 $\nu = 1$
3. **加权 L1 求解**：使用 FISTA 求解加权 Lasso 问题

## 预言性质（Oracle property）

在正则条件下（Zou 2006, Theorem 1）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

## 推断

- 默认 `compute_inference=False`（adaptive_l1 不支持 debiased 推断）
- 对选中变量做推断可用 oracle 方法：在选中的 support set 上重新拟合 OLS

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度 |
| `nu` | `1.0` | 权重计算指数（1 或 2，Zou 2006） |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## 求解器支持

Adaptive Lasso 先构造数据驱动的惩罚权重，再用加权 L1 FISTA 求解；这条模型专属分发优先于通用光滑求解器。

| 选择 | 支持情况 | 实际计算 |
|---|:---:|---|
| `auto`（默认） | 支持 | 加权 L1 FISTA |
| `fista` | 支持 | 加权 L1 FISTA |
| `newton`、`lbfgs`、`irls`、`exact` | 不支持 | adaptive-L1 是非光滑惩罚，因此会被拒绝 |
| `fista_bb`、`admm`、`coordinate_descent` | 没有独立 Adaptive-Lasso 路径 | 当前 wrapper 仍使用加权 L1 FISTA |

用于构造权重的初始 Ridge 估计属于内部初始化步骤，不是第二个公开求解器选择。FISTA 本身见[求解器算法](../guides/solver-algorithms.md#1-fista)。

## CPU 与 GPU 示例

```python
from statgpu.linear_model import AdaptiveLasso

model = AdaptiveLasso(alpha=0.1, nu=1.0)
model.fit(X, y)
print(model.coef_)        # 稀疏系数
print(model.score(X, y))  # R-squared

# GPU 加速
model_gpu = AdaptiveLasso(alpha=0.1, device="cuda")
model_gpu.fit(X, y)
```

## 输出

- 系数：`intercept_`、`coef_`
- 方法：`fit`、`predict`、`score`
- 限制：adaptive L1 当前不支持 `compute_inference=True`

## 外部验证

- `dev/tests/test_refactor_safety_net.py` 覆盖求解器收敛回归测试。

## 参考文献

- Zou, H. (2006). [The adaptive lasso and its oracle properties](https://doi.org/10.1198/016214506000000735). *Journal of the American Statistical Association*, 101(476), 1418-1429.
- Wang, H., Li, B., & Leng, C. (2009). Shrinkage tuning parameter selection with a diverging number of parameters. *Journal of the Royal Statistical Society: Series B*, 71(3), 671-683.
