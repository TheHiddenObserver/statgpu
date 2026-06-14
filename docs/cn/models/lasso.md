# Lasso

> 语言: 中文  
> 最后更新: 2026-04-17  
> 页面定位: 模型文档  
> 切换: [English](../en/models/lasso.md)

语言切换：[English](../en/models/lasso.md)

## 概览（Overview）

`Lasso` 提供 L1 正则线性回归，支持 CPU/GPU 训练与推断。当前实现支持多求解器与多推断模式，重点覆盖工程可用性与 CPU/GPU 对比。

## 路径（Path）

`statgpu.linear_model.Lasso`

## 目标函数（Objective Function）

优化目标为：

\[
\min_{\beta, b}\ \frac{1}{2n}\|y - X\beta - b\|_2^2 + \alpha\|\beta\|_1
\]

其中 `alpha` 控制稀疏化强度。

## 估计方程（Estimating Equation）

- GPU 求解器：`solver="fista"` 或 `solver="admm"`
- CPU 求解器：`cpu_solver="coordinate_descent"` 或 `cpu_solver="fista"`
- 停止准则：`stopping="coef_delta"` 或 `stopping="kkt"`

不同求解器在同一 `tol` 下可能表现出不同迭代步数与运行时间。

## 协方差与推断（Covariance/Inference）

`Lasso` 推断由 `inference_method` 控制：

- `cpu_ols_inference`：CPU 侧 OLS 风格推断
- `gpu_ols_inference`：GPU 侧推断，减少 host/device 大块传输
- `debiased`：去偏 Lasso 推断（de-biased / de-sparsified），输出 z 统计量口径
- `bootstrap`：重采样推断，通常更稳健但更慢

有效性边界说明：
- `cpu_ols_inference` / `gpu_ols_inference` 的区间是 post-selection 启发式区间，不保证严格 selective-inference 覆盖率。
- `debiased` 当前输出的是“单个系数的边际置信区间”（marginal CI），不提供联合（simultaneous）覆盖或多重比较校正保证。

兼容旧名映射：
- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L1 正则强度 |
| `max_iter` | `1000` | 最大迭代数 |
| `tol` | `1e-4` | 收敛阈值 |
| `solver` | `"fista"` | GPU 求解器：`fista` / `admm` |
| `cpu_solver` | `"coordinate_descent"` | CPU 求解器：`coordinate_descent` / `fista` |
| `stopping` | `"coef_delta"` | 停止准则：`coef_delta` / `kkt` |
| `inference_method` | `"cpu_ols_inference"` | `cpu_ols_inference` / `gpu_ols_inference` / `debiased` / `bootstrap` |
| `compute_inference` | `True` | 是否计算推断统计 |
| `enable_simultaneous_inference` | `False` | 是否启用 simultaneous inference（仅 `debiased`） |
| `simultaneous_method` | `"maxz_bootstrap"` | 当前仅支持 `maxz_bootstrap` |
| `simultaneous_alpha` | `0.05` | simultaneous 置信水平参数 |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| multiplier bootstrap 抽样次数 |
| `simultaneous_random_state` | `None` | simultaneous bootstrap 随机种子 |
| `simultaneous_include_intercept` | `False` | simultaneous 目标集合是否包含截距 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.linear_model import Lasso

# CPU
m_cpu = Lasso(
    alpha=0.1,
    device="cpu",
    cpu_solver="coordinate_descent",
    stopping="kkt",
    inference_method="cpu_ols_inference",
)
m_cpu.fit(X, y)

# GPU
m_gpu = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="gpu_ols_inference",
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)
```

simultaneous inference 示例（支持 `device="cpu"` 与 `device="cuda"`，并在对应设备侧完成计算）：

```python
m_sim = Lasso(
    alpha=0.1,
    device="cpu",
    inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=7,
)
m_sim.fit(X, y)
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
```

## strict/approx 差异（strict/approx difference）

`debiased` 是当前 Lasso 的 strict 主路径（默认高一致性推断语义），`cpu_ols_inference` / `gpu_ols_inference` 属于更轻量的近似推断路径；`bootstrap` 通常更稳健但耗时更高。

## 输出（Outputs）

- `fit(X, y) -> self`
- `predict(X)`、`score(X, y)`（`R^2`）
- 主要属性：`intercept_`, `coef_`, `n_iter_`, `aic`, `bic`
- 推断属性（`compute_inference=True`）：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- 当 `inference_method="debiased"` 时，统计口径为 z 统计（`summary()` 中对应 z/P>|z| 展示），`_conf_int` 为单变量边际区间
- 开启 simultaneous inference 后，`_conf_int_simultaneous` 给出目标集合上的联合区间（`maxz_bootstrap`）
- 汇总：`summary()`

## 常见问题（FAQ）

- **为什么同样 `tol` 下 CPU/GPU 迭代数不同？**  
  不同求解器与数值路径导致收敛轨迹不同，建议固定 `solver/stopping` 后再做性能或精度对比。
- **何时优先 `gpu_ols_inference`？**  
  大样本且训练在 GPU 上时优先，可减少 host/device 传输开销。
- **`debiased` 适用于什么场景？**  
  适用于高维稀疏建模下需要可解释推断（标准误、显著性检验）的场景；低维稠密设置下可与 OLS 推断结果做对照验证。
- **`cpu_ols_inference/gpu_ols_inference` 的区间能当严格置信区间吗？**  
  不建议。它们主要用于工程对比与快速诊断，不保证严格 post-selection 置信覆盖。
- **`debiased` 的区间是联合区间吗？**  
  不是。目前实现是单个系数的边际区间；若需要联合推断，请额外做多重比较控制或专门的 simultaneous inference 程序。
- **如何启用联合区间？**  
  使用 `enable_simultaneous_inference=True` 且 `inference_method="debiased"`，并设置 `simultaneous_method="maxz_bootstrap"`。

## 外部验证（External Validation）

建议优先使用以下脚本做外部对齐与性能回归：

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`

## 参考（References）

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
