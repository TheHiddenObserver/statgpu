# 损失函数 (LossBase)

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../../en/models/losses.md)

## 概述

`LossBase` 是 statgpu 中所有损失函数的通用基类。它为优化求解器（FISTA、Newton、L-BFGS、ADMM）和惩罚函数（L1、L2、ElasticNet、SCAD、MCP 等）提供统一接口。

三种新损失类型扩展了 `LossBase`：

| 损失 | 类 | R 对应 | 用途 |
|------|------|--------|------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健回归（M-估计器） |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

## 路径

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## 求解器兼容性

| 求解器 | Quantile | Huber | Cox PH |
|--------|----------|-------|--------|
| FISTA | ✅ | ✅ | ✅ |
| Newton | ❌ | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ |

## 示例

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# 分位数回归
loss = QuantileLoss(quantile=0.5)
coef, _ = lbfgs_solver(loss, None, X, y)

# 稳健回归
loss = HuberLoss(delta=1.345)
coef, _ = lbfgs_solver(loss, None, X, y)
```

## 注意事项

- `CoxPartialLikelihoodLoss` 支持 GPU（CuPy CUDA / PyTorch-CUDA kernel），涵盖 Breslow 和 Efron。显式 GPU 输入在 GPU 路径不可用时将 `raise RuntimeError`；CPU 输入使用 numpy 实现。
- `HuberLoss` 的 `has_hessian = True`，proximal Newton 为首选求解器。
- `BisquareLoss` 可通过 `statgpu.losses.BisquareLoss` 获取，支持 redescending M-估计。
- 所有损失自动继承 10 种惩罚类型和 6 种求解器。
