# Lasso

> 语言: 中文  
> 最后更新: 2026-09-06  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/lasso.md)

语言切换：[English](../../en/models/lasso.md)

## 概览（Overview）

`Lasso` 提供 L1 正则线性回归，支持 CPU/GPU 训练与多种推断模式。直接拟合现在使用**统一、与后端无关的 `solver` 接口**；“在哪个设备上算”和“使用哪种算法”是两个独立选择。

## 路径（Path）

`statgpu.linear_model.Lasso`

## 目标函数（Objective Function）

优化目标为：

$$
\min_{\beta, b}\ \frac{1}{2n}\|y - X\beta - b\|_2^2 + \alpha\|\beta\|_1
$$

其中 `alpha` 控制稀疏化强度。

## 估计方程（Estimating Equation）

Lasso 通过迭代优化求解，而不是闭式 normal equation。停止条件可由系数变化（`coef_delta`）或 KKT 一致性（`kkt`）控制。

对于**单次直接拟合**，真正控制算法的是 `solver`，无论 CPU 还是 GPU：

- CPU coordinate descent：`solver="coordinate_descent"`
- CPU 或 GPU proximal path：`solver="fista"`（或其他当前支持的求解器）
- 后端位置另外通过 `device="cpu"`、`"cuda"` 或 `"torch"` 选择

历史参数 `cpu_solver` 已进入弃用流程。为兼容旧代码它暂时仍可传入，但在统一 solver engine 中**不再决定 direct-fit 算法**。旧代码应迁移到 `solver=...`；参见 [penalized solver API 迁移指南](../guides/penalized-solver-api-migration.md)。

## 协方差与推断（Covariance/Inference）

`Lasso` 推断由 `inference_method` 控制：

- `cpu_ols_inference`：CPU 侧 OLS 风格 post-selection 推断
- `gpu_ols_inference`：GPU 侧推断，减少 host/device 大块传输
- `debiased`：去偏 Lasso 推断（de-biased / de-sparsified），使用 z 统计量语义
- `bootstrap`：重采样推断，通常更慢

有效性边界：
- `cpu_ols_inference` / `gpu_ols_inference` 的区间是 post-selection 启发式区间，不应解释为严格 selective-inference confidence interval。
- 普通 `debiased` `_conf_int` 是单个系数的 marginal interval；需要 family-wise 区间时应显式启用 simultaneous inference。

兼容旧名映射：
- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

## 参数（Parameters）

下表是 `statgpu.linear_model.Lasso` 的完整公开构造参数清单。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L1 正则强度。 |
| `fit_intercept` | `True` | 是否拟合截距。 |
| `max_iter` | `1000` | 优化最大迭代次数。 |
| `tol` | `1e-4` | 收敛容差。 |
| `stopping` | `"coef_delta"` | 停止准则：`coef_delta` / `kkt`。 |
| `inference_method` | `"debiased"` | `cpu_ols_inference` / `gpu_ols_inference` / `debiased` / `bootstrap`。 |
| `n_bootstrap` | `200` | residual-bootstrap 推断的抽样次数。 |
| `bootstrap_random_state` | `None` | residual-bootstrap 随机种子。 |
| `enable_simultaneous_inference` | `False` | 是否启用 simultaneous inference（仅 `debiased`）。 |
| `simultaneous_method` | `"maxz_bootstrap"` | simultaneous inference 方法；当前为 `maxz_bootstrap`。 |
| `simultaneous_alpha` | `0.05` | simultaneous family-wise error level。 |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| multiplier bootstrap 抽样次数。 |
| `simultaneous_random_state` | `None` | simultaneous bootstrap 随机种子。 |
| `simultaneous_include_intercept` | `False` | simultaneous 目标集合是否包含截距。 |
| `device` | `"auto"` | 执行设备：`auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 适用 CPU 路径的并行度。 |
| `compute_inference` | `True` | 是否计算拟合后推断。 |
| `solver` | `"fista"` | 与后端无关的 direct-fit 求解器；CPU coordinate descent 使用 `coordinate_descent`，其他值按当前 compatibility contract。 |
| `cpu_solver` | `"coordinate_descent"` | **Deprecated compatibility parameter**；当前不再决定 direct-fit 算法，请改用 `solver`。 |
| `lipschitz_L` | `None` | 兼容迭代求解器可用的用户指定 Lipschitz 常数。 |
| `admm_rho` | `1.0` | 选择 ADMM 路径时的 penalty 参数。 |
| `gpu_memory_cleanup` | `False` | 支持路径上拟合后的 best-effort GPU 内存清理。 |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.linear_model import Lasso

# CPU coordinate descent：solver 选择算法，device 选择 CPU。
m_cpu = Lasso(
    alpha=0.1,
    device="cpu",
    solver="coordinate_descent",
    stopping="kkt",
)
m_cpu.fit(X, y)

# GPU FISTA：GPU 上仍使用同一个 solver 接口。
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

simultaneous inference 示例：

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

`debiased` 是高维推断主路径；`cpu_ols_inference` / `gpu_ols_inference` 是更轻量的近似 post-selection diagnostic，`bootstrap` 则计算成本更高。它们的统计含义不能互换。

## 输出（Outputs）

- `fit(X, y) -> self`
- `predict(X)`、`score(X, y)`（`R^2`）
- 主要属性：`intercept_`, `coef_`, `n_iter_`, `aic`, `bic`
- 推断属性（`compute_inference=True`）：`_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`
- 当 `inference_method="debiased"` 时，普通 `_conf_int` 为单变量 marginal interval
- 开启 simultaneous inference 后，`_conf_int_simultaneous` 给出配置目标集合上的联合区间
- 汇总：`summary()`

## 常见问题（FAQ）

- **为什么同样 `tol` 下 CPU/GPU 迭代数不同？**  
  不同数值后端和算法实现可能产生不同收敛轨迹；比较时固定 `solver` 与 `stopping`。
- **CPU 用户应该设置 `cpu_solver` 吗？**  
  不应该。直接拟合统一使用 `solver`；`cpu_solver` 是旧 CPU/GPU split API 的 deprecated compatibility 参数。
- **何时优先 `gpu_ols_inference`？**  
  大样本且训练在 GPU 上时可用于减少 host/device 传输。
- **`debiased` 适用于什么场景？**  
  适用于高维稀疏设置下需要系数级推断的场景，但仍依赖对应理论假设。
- **`cpu_ols_inference/gpu_ols_inference` 的区间能当严格置信区间吗？**  
  不建议；它们不保证严格 post-selection coverage。
- **普通 `debiased` 区间是联合区间吗？**  
  不是。普通 `_conf_int` 是 marginal interval；需要联合控制时使用 dedicated simultaneous path。
- **如何启用联合区间？**  
  使用 `enable_simultaneous_inference=True`、`inference_method="debiased"` 和 `simultaneous_method="maxz_bootstrap"`。

## 外部验证（External Validation）

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`
- `dev/tests/test_penalized_solver_api_cleanup.py`

## 参考（References）

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
