# Knockoff 特征选择

> 语言: 中文  
> 最后更新: 2026-04-17  
> 页面定位: 方法文档  
> 切换: [English](../en/models/knockoff.md)

语言切换：[English](../en/models/knockoff.md)

## 概览（Overview）

Knockoff 方法用于控制特征选择中的 FDR。当前实现包含 `fixed_x` 与 `model_x` 两条路径，统一入口为 `knockoff_filter`。`fixed_x` 通常要求 `n >= 2p`；`model_x` 基于高斯二阶近似（协方差估计 + S-matrix），支持多次 draw 聚合 W 统计量。

## 路径（Path）

主路径：
- `statgpu.feature_selection.knockoff_filter`
- `statgpu.feature_selection.fixed_x_knockoff_filter`
- `statgpu.feature_selection.model_x_knockoff_filter`
- `statgpu.feature_selection.KnockoffSelector`
- `statgpu.feature_selection.FixedXKnockoffSelector`

顶层别名：
- `statgpu.knockoff_filter`
- `statgpu.fixed_x_knockoff_filter`
- `statgpu.model_x_knockoff_filter`
- `statgpu.KnockoffSelector`
- `statgpu.FixedXKnockoffSelector`

## 目标函数（Objective Function）

目标是在给定目标 FDR 水平 `q` 下，通过 knockoff 统计量 `W` 与阈值规则（`knockoff_plus` 或 `knockoff`）选择特征集合，同时控制期望误发现率。

## 估计方程（Estimating Equation）

核心步骤：
- 构造 knockoff 特征（`fixed_x` 或 `model_x`）
- 计算特征统计量 `W`（`corr_diff` / `ols_coef_diff` / `lasso_coef_diff`）
- 按 FDR 规则求阈值并输出入选特征

`model_x` 可通过 `modelx_draws` 进行多次采样并聚合统计量。

## 协方差与推断（Covariance/Inference）

Knockoff 为选择推断框架，不采用回归模型中的 `cov_type` 协方差配置。其统计保证依赖 knockoff 构造假设与阈值规则。

`compat_mode` 支持：
- `statgpu`：默认实现
- `knockpy`：兼容路径（部分 sampler 分发入口仍为占位）

## 参数（Parameters）

统一入口 `knockoff_filter` 关键参数如下：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `knockoff_type` | `fixed_x` | `fixed_x` 或 `model_x` |
| `q` | `0.1` | FDR 控制目标，必须在 `(0, 1)` |
| `method` | `corr_diff` | `corr_diff` / `ols_coef_diff` / `lasso_coef_diff` |
| `fdr_control` | `knockoff_plus` | `knockoff_plus` 或 `knockoff` |
| `random_state` | `None` | 随机种子 |
| `backend` | `auto` | 计算后端：`auto` / `numpy` / `cupy` / `torch` |
| `Xk` | `None` | 外部提供 knockoff 设计矩阵，形状需与 `X` 相同 |
| `compat_mode` | `statgpu` | `statgpu` 或 `knockpy` |
| `lasso_cv_impl` | `auto` | `auto` / `statgpu` / `sklearn` |
| `lasso_fast_profile` | `off` | lasso 快速配置开关 |
| `modelx_covariance_shrinkage` | `0.20` | model-X 协方差收缩系数 |
| `modelx_s_scale` | `0.999` | model-X `S` 缩放系数 |
| `modelx_draws` | `None` | model-X draw 次数 |
| `modelx_shrinkage` | `ledoitwolf` | knockpy 兼容路径协方差估计策略 |
| `modelx_smatrix_method` | `mvr` | knockpy 兼容路径 `S`-matrix 方法 |
| `knockpy_sampler` | `None` | 可选分发入口 |
| `knockpy_sampler_method` | `None` | `sampler=gaussian` 时子方法 |

## Torch Backend 性能

**Torch 后端性能对比** (20 次实验基准):
- **大型数据集 (n=1000, p=200)**: 相对 NumPy 约 1.14x 加速
- **超大型数据集 (n=2000, p=500)**: 相对 NumPy 约 1.33x 加速
- **小型数据集 (<500 样本)**: NumPy 可能更快（GPU 开销较大）

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu import fixed_x_knockoff_filter, knockoff_filter

# CPU: fixed-X
res_cpu = fixed_x_knockoff_filter(
    X,
    y,
    q=0.1,
    method="ols_coef_diff",
    backend="numpy",
)

# GPU: model-X
res_gpu = knockoff_filter(
    X,
    y,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="cupy",
    modelx_draws=3,
)

# GPU Torch: fixed-X
import torch
X_torch = torch.from_numpy(X).to('cuda')
y_torch = torch.from_numpy(y).to('cuda')

res_torch = fixed_x_knockoff_filter(
    X_torch, y_torch,
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
)

# GPU Torch: model-X
res_torch_mx = knockoff_filter(
    X_torch, y_torch,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
    modelx_draws=3,
)
```

## strict/approx 差异（strict/approx difference）

本模块不使用 `strict/approx` 推断口径开关。性能与精度权衡主要体现在 `fixed_x`/`model_x` 选择、`modelx_draws` 次数和后端（`numpy`/`cupy`）选择上。

## 输出（Outputs）

核心返回对象为 `KnockoffResult`，常用字段：

- `selected_features`
- `W`
- `threshold`
- `estimated_fdr`
- `q_trajectory`
- `metadata`（例如 draw 次数、兼容模式、`Xk` 来源）

## 常见问题（FAQ）

- **哪些输入会直接报错？**  
  `q` 不在 `(0,1)`、`X` 非二维、`y` 长度与 `X` 不一致、`Xk` 形状不匹配都会报错。
- **`knockpy_sampler` 可以直接用吗？**  
  当前显式传入会触发 `NotImplementedError`（占位保护）；不传时走稳定路径。
- **`fixed_x` 的主要约束是什么？**  
  通常要求样本规模与矩阵秩满足构造条件（常见约束为 `n >= 2p`）。

## 外部验证（External Validation）

推荐验证脚本：

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`

## 参考（References）

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)
