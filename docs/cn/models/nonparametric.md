# 非参数方法

> 语言: 中文  
> 最后更新: 2026-09-21  
> 页面定位: 非参数方法总览  
> 切换: [English](../../en/models/nonparametric.md)

## 相关页面

- [核岭回归](kernel-methods.md) — KernelRidge, KernelRidgeCV
- [样条基函数](splines.md) — bspline_basis, natural_cubic_spline_basis
- [GAM（半参数）](semiparametric.md) — 广义可加模型

## 概览

statgpu 非参数模块提供核平滑方法：
- **KDE**：密度估计（`fit_kde`、`kde_pdf`、bootstrap 置信区间）。
- **核回归**：Nadaraya-Watson（`nw`）和局部线性（`local_linear`）回归，提供函数式 API 和 sklearn 风格封装。

## 导入与主要接口

核心路径（按功能）：

- KDE：`statgpu.nonparametric.fit_kde`、`statgpu.nonparametric.kde_pdf`、`statgpu.nonparametric.kde_bootstrap_confidence_interval`
- 核回归：`statgpu.nonparametric.fit_kernel_regression`、`statgpu.nonparametric.kernel_regression_predict`
- sklearn 风格类：`KernelDensityEstimator`、`KernelRegressionRegressor`

## 目标

- KDE：估计未知概率密度函数 `f(x)`。
- 核回归：估计条件均值函数 `m(x)=E[Y|X=x]`，支持 `nw` 与 `local_linear`。

## 估计方法

- KDE：基于核加权求和构造密度估计，支持权重样本与多种带宽规则。
- 核回归：
  - `nw`：Nadaraya-Watson 核加权平均
  - `local_linear`：局部线性加权回归
- `kernel_metric='full'|'diagonal'`；在 `diagonal` 下可用 `bandwidth_per_feature` 按特征控制带宽。

## 区间估计与推断

该模块不使用回归模型中的 `cov_type` 协方差推断。KDE 提供 `kde_bootstrap_confidence_interval` 作为区间估计工具；核回归 以预测与对齐验证为主。

## 参数

常用参数摘要：

| 模块 | 参数 | 说明 |
|---|---|---|
| KDE | `bandwidth` | 支持 `scott`、`silverman`、`nrd0`、`nrd`、`ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi` |
| KDE | `kernel` | `gaussian` / `rectangular` / `triangular` / `epanechnikov` / `biweight` / `triweight` / `cosine` / `optcosine` |
| KDE | `weights` | 加权样本 |
| KDE/核回归 | `backend` | `auto` / `numpy` / `cupy` |
| 核回归 | `regression` | `nw` / `local_linear` |
| 核回归 | `kernel_metric` | `full` / `diagonal` |
| 核回归 | `bandwidth_per_feature` | 对角核按特征设置带宽 |

## CPU/GPU 示例

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

# GPU（CuPy）
pred_gpu = kernel_regression_predict(
    samples=X,
    targets=y,
    points=X_query,
    regression="local_linear",
    kernel_metric="diagonal",
    backend="cupy",
)
```

## 精度与近似

该模块没有统一的严格/近似计算开关。数值精度与计算成本主要由带宽规则、`kernel_metric`（`full`/`diagonal`）以及 CPU/GPU 后端选择共同决定。

## 输出

- KDE 输出：密度估计向量、（可选）bootstrap 区间
- 核回归 输出：在给定 `points` 的回归预测值
- sklearn 风格类支持 `fit/predict`，其中 `KernelDensityEstimator` 额外支持 `score_samples`

## 常见问题（FAQ）

- **`full` 与 `diagonal` 如何选？**  
  `diagonal` 更快且便于按特征调参；`full` 在特征相关性强时可能更稳健。
- **CPU/GPU 结果是否应完全一致？**  
  允许机器精度范围内差异，建议固定带宽与 kernel 后对比。
- **KDE 带宽优先选哪种？**  
  可先用 `scott/silverman` 做基线，再用 `ucv/sj` 做精调。

## 与外部实现的对照

核回归的对角核实现与 `statsmodels` 做过数值对照。在相同核函数和带宽设置下，CPU 与 GPU 路径也会检查结果一致性。性能会随数据规模、带宽选择和硬件变化，实际使用时建议针对自己的工作负载进行基准测试。

## 参考文献

- Rosenblatt, M. (1956). Remarks on some nonparametric estimates of a density function. *Annals of Mathematical Statistics*, 27(3), 832-837. [https://doi.org/10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
- Parzen, E. (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*, 33(3), 1065-1076. [https://doi.org/10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141-142. [https://doi.org/10.1137/1109020](https://doi.org/10.1137/1109020)
- Watson, G. S. (1964). Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series A*, 26(4), 359-372.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
