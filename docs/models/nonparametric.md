# 非参数方法

> 语言: 中文  
> 最后更新: 2026-04-15  
> 页面定位: 非参数方法总览  
> 切换: [English](../en/models/nonparametric.md)

语言切换：[English](../en/models/nonparametric.md)

## 概览

statgpu 的非参数方法当前包含两条主线：

- KDE：`fit_kde` / `kde_pdf` / `kde_bootstrap_confidence_interval`
- Kernel Regression：`fit_kernel_regression` / `kernel_regression_predict`

## KDE

KDE 支持 NumPy/CuPy 后端，提供加权样本、带宽选择和自助法区间估计。

### 主要 API

- `fit_kde(samples, bandwidth='scott', weights=None, kernel='gaussian', backend='auto')`
- `kde_pdf(samples, points, ...)`
- `kde_bootstrap_confidence_interval(samples, points, ...)`
- sklearn 风格：`KernelDensityEstimator(...).fit(X).predict(points)` / `score_samples(points)`

### 支持的 kernel

- `gaussian`
- `rectangular`
- `triangular`
- `epanechnikov`
- `biweight`
- `triweight`
- `cosine`
- `optcosine`

### 带宽规则

- 标准规则：`scott`、`silverman`
- R 风格规则：`nrd0`、`nrd`
- 选择器：`ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi`

### 示例

```python
import numpy as np
from statgpu.nonparametric import fit_kde, kde_pdf

x = np.random.randn(500)
grid = np.linspace(-4, 4, 200)

kde = fit_kde(x, bandwidth="scott", kernel="gaussian")
density = kde_pdf(x, grid, bandwidth="scott")
```

## Kernel Regression

Kernel Regression 支持 `nw` 与 `local_linear` 两种回归器，并提供与 statsmodels 对齐的对角核模式。

### 主要 API

- `fit_kernel_regression(samples, targets, regression='nw', kernel='gaussian', kernel_metric='full'|'diagonal', ...)`
- `kernel_regression_predict(model, points, ...)`
- sklearn 风格：`KernelRegressionRegressor(...).fit(X, y).predict(points)`

### 关键参数

- `regression='nw'|'local_linear'`
- `kernel_metric='full'|'diagonal'`
- `bandwidth_per_feature`：对角核下按特征控制带宽

### 已验证结果

- `run_id=20260415_103036`：对角核模式下与 statsmodels 达到机器精度对齐
- `run_id=20260415_120903`：多维 local-linear 批处理优化后，CPU 和 GPU 均显著提速且保持精度

## 相关页面

- [模型总览](README.md)
- [基准脚本索引](../benchmarks.md)

## 对比基准建议

- Python 对标：
	- `dev/benchmarks/benchmark_kde_vs_scipy.py`
	- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- R 对标：
	- `dev/benchmarks/benchmark_nonparametric_vs_r.py`
- 一键汇总（Python+R）：
	- `dev/benchmarks/benchmark_nonparametric_comparison_suite.py`
