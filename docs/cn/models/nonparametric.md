# 非参数方法

> 语言: 中文  
> 最后更新: 2026-04-17  
> 页面定位: 非参数方法总览  
> 切换: [English](../../en/models/nonparametric.md)

语言切换：[English](../../en/models/nonparametric.md)

## 相关页面

- [核岭回归](kernel-methods.md) — KernelRidge, KernelRidgeCV
- [样条基函数](splines.md) — bspline_basis, natural_cubic_spline_basis
- [GAM（半参数）](semiparametric.md) — 广义可加模型

## 概览（Overview）

statgpu 非参数模块提供核平滑方法：
- **KDE**：密度估计（`fit_kde`、`kde_pdf`、bootstrap 置信区间）。
- **核回归**：Nadaraya-Watson（`nw`）和局部线性（`local_linear`）回归，提供函数式 API 和 sklearn 风格封装。

## 路径（Path）

核心路径（按功能）：

- KDE：`statgpu.nonparametric.fit_kde`、`statgpu.nonparametric.kde_pdf`、`statgpu.nonparametric.kde_bootstrap_confidence_interval`
- Kernel Regression：`statgpu.nonparametric.fit_kernel_regression`、`statgpu.nonparametric.kernel_regression_predict`
- sklearn 风格类：`KernelDensityEstimator`、`KernelRegressionRegressor`

## 目标函数（Objective Function）

- KDE：估计未知概率密度函数 `f(x)`。
- Kernel Regression：估计条件均值函数 `m(x)=E[Y|X=x]`，支持 `nw` 与 `local_linear`。

## 估计方程（Estimating Equation）

- KDE：基于核加权求和构造密度估计，支持权重样本与多种带宽规则。
- Kernel Regression：
  - `nw`：Nadaraya-Watson 核加权平均
  - `local_linear`：局部线性加权回归
- `kernel_metric='full'|'diagonal'`；在 `diagonal` 下可用 `bandwidth_per_feature` 按特征控制带宽。

## 协方差与推断（Covariance/Inference）

该模块不使用回归模型中的 `cov_type` 协方差推断。KDE 提供 `kde_bootstrap_confidence_interval` 作为区间估计工具；Kernel Regression 以预测与对齐验证为主。

## 参数（Parameters）

常用参数摘要：

| 模块 | 参数 | 说明 |
|---|---|---|
| KDE | `bandwidth` | 支持 `scott`、`silverman`、`nrd0`、`nrd`、`ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi` |
| KDE | `kernel` | `gaussian` / `rectangular` / `triangular` / `epanechnikov` / `biweight` / `triweight` / `cosine` / `optcosine` |
| KDE | `weights` | 加权样本 |
| KDE/Kernel Regression | `backend` | `auto` / `numpy` / `cupy` |
| Kernel Regression | `regression` | `nw` / `local_linear` |
| Kernel Regression | `kernel_metric` | `full` / `diagonal` |
| Kernel Regression | `bandwidth_per_feature` | 对角核按特征带宽设置 |

## CPU+GPU 示例（CPU+GPU Examples）

```python
import numpy as np
from statgpu.nonparametric import kde_pdf, kernel_regression_predict

x = np.random.randn(500)
grid = np.linspace(-4, 4, 200)
X = np.random.randn(500, 2)
y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * np.random.randn(500)
X_query = np.random.randn(64, 2)

# CPU (NumPy)
density_cpu = kde_pdf(x, grid, bandwidth="scott", kernel="gaussian", backend="numpy")

# GPU (CuPy, auto fallback)
pred_gpu = kernel_regression_predict(
    samples=X,
    targets=y,
    points=X_query,
    regression="local_linear",
    kernel_metric="diagonal",
    backend="cupy",
)
```

## strict/approx 差异（strict/approx difference）

模块未定义统一 `strict/approx` 开关。精度与性能权衡主要通过带宽规则、`kernel_metric`（`full`/`diagonal`）以及后端选择（CPU/GPU）实现。

## 输出（Outputs）

- KDE 输出：密度估计向量、（可选）bootstrap 区间
- Kernel Regression 输出：在给定 `points` 的回归预测值
- sklearn 风格类支持 `fit/predict`，其中 `KernelDensityEstimator` 额外支持 `score_samples`

## 常见问题（FAQ）

- **`full` 与 `diagonal` 如何选？**  
  `diagonal` 更快且便于按特征调参；`full` 在特征相关性强时可能更稳健。
- **CPU/GPU 结果是否应完全一致？**  
  允许机器精度范围内差异，建议固定带宽与 kernel 后对比。
- **KDE 带宽优先选哪种？**  
  可先用 `scott/silverman` 做基线，再用 `ucv/sj` 做精调。

## 外部验证（External Validation）

已记录的验证结果：
- `run_id=20260415_103036`：Kernel Regression 对角核与 statsmodels 机器精度对齐
- `run_id=20260415_120903`：多维 local-linear 批处理优化后，CPU/GPU 均显著提速且保持精度

推荐基准脚本：
- `dev/benchmarks/benchmark_kde_vs_scipy.py`
- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- `dev/benchmarks/benchmark_nonparametric_vs_r.py`
- `dev/benchmarks/benchmark_nonparametric_comparison_suite.py`

## 参考（References）

- Rosenblatt, M. (1956). Remarks on some nonparametric estimates of a density function. *Annals of Mathematical Statistics*, 27(3), 832-837. [https://doi.org/10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
- Parzen, E. (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*, 33(3), 1065-1076. [https://doi.org/10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141-142. [https://doi.org/10.1137/1109020](https://doi.org/10.1137/1109020)
- Watson, G. S. (1964). Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series A*, 26(4), 359-372.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
