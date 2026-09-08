# Lasso

> 语言: 中文  
> 最后更新: 2026-09-08  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/lasso.md)

语言切换：[English](../../en/models/lasso.md)

## 概览（Overview）

`Lasso` 提供 L1 正则线性回归，支持 CPU/GPU 训练与多种推断模式。直接拟合使用统一、与后端无关的 `solver` 接口；设备选择、求解算法和统计推断方法是三个独立维度。

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

- `post_selection_ols`：与硬件无关的 active-set OLS/WLS 重拟合 diagnostic；
- `debiased`：去偏 Lasso 推断（de-biased / de-sparsified），使用 z 统计量语义；
- `bootstrap`：residual bootstrap，计算通常更昂贵，也不是对模型选择不确定性的普适修正。

统一 wrapper 中的 `cpu_ols` 与 `gpu_ols` **同时进入弃用期**。一个兼容周期内仍可传入，但会发出 `FutureWarning` 并统一归一化为 `post_selection_ols`；它们不是“CPU 版”和“GPU 版”两个不同的统计方法。`LassoCV` 还会在其兼容边界接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼法，并映射到同一个 canonical method。

### `post_selection_ols` 实际计算什么？

penalized fit 先选出 active feature set。随后 statgpu 在**成功拟合已经记录的 backend/device** 上，只对这些选中列做无惩罚 OLS；如果传入 `sample_weight`，则做 WLS，并在同一个数值 backend 上计算 Gaussian covariance 与参考分布推断，最后才执行既有的 NumPy reporting snapshot。

原始 penalized `coef_` 保持不变，并继续用于预测；active-set OLS/WLS 重拟合用于推断与报告，保存在 `_params`、`_inference_result`、`_bse`、`_tvalues` / `_zvalues`、`_pvalues`、`_conf_int` 等字段中。

有效性边界：

- `post_selection_ols` 是启发式 post-selection diagnostic。用同一数据先选变量再做普通 OLS/WLS，并不会自动获得一般 selective-inference coverage；
- 普通 `debiased` `_conf_int` 是单个系数的 marginal interval；需要 family-wise 区间时，应显式启用 simultaneous inference；
- active design 秩亏时，refit residual DoF 使用 effective rank，系数重拟合与 covariance bread 使用 design-level Moore-Penrose/SVD，而不是通过 normal equations 把条件数平方。

### 设备/backend 规则

`inference_method` 只描述**计算哪种统计程序**，不选择硬件：

- 显式 `device="cpu"` -> NumPy CPU；
- 显式 `device="cuda"` -> 只允许 CuPy CUDA，不可用时 fail closed；
- 显式 `device="torch"` -> 只允许 Torch CUDA，不可用时 fail closed；
- 只有 estimator 与全局配置都处于真正的 `device="auto"` 时，已经是 CuPy 或 Torch-CUDA 的输入才可以作为自动路由的一部分保留 native backend。

backend 复用保证按推断方法区分：`post_selection_ols` 复用成功拟合记录的 `_selected_backend_name` / `_selected_backend_device`；维护中的 CuPy/Torch `debiased` 路径也会把数值推断保留在实际执行的 GPU backend，包括 normal-reference 的 scalar critical value。residual `bootstrap` 当前仍使用 CPU-native residual refit，因此显式 GPU `device` 会控制 penalized fit，但不会让 bootstrap 变成 GPU-native。

对于 analytic weights，direct Lasso 与 debiased inference 在 NumPy/CuPy/Torch 上都使用同一个 weighted-centered average-loss 约定，因此把所有权重乘以同一个正常数不会改变统计问题。`LassoCV` 的默认 alpha grid、每个 weighted training fold、validation MSE 与 final refit 也遵循同一约定；常数正权重直接走与 unweighted 完全相同的 CV 路径。AUTO 一旦为 CV 解析出具体 backend，最终 selected-alpha `Lasso` refit 也保持在该 backend。

### Debiased intercept ownership

使用 `inference_method="debiased"` 时，prediction 与 inference 会有意暴露两套不同 ownership 的 intercept。公开 `coef_` 与 `intercept_` 始终属于 **penalized prediction fit**；推断/reporting 使用 `theta_db = _params[1:]`，以及与该 slope vector属于同一个原始坐标系参数化的截距 `_params[0] = ybar_w - xbar_w @ theta_db`。

因此 `_bse[0]`、第一个 z-statistic/p-value 与 `_conf_int[0]` 描述的是 debiased reporting intercept，而不是 `intercept_`。若设计矩阵每列平移常数向量 `c`，debiased slope保持不变，而 `_params[0]` 按 `-c @ theta_db` 平移，从而保持一个 coherent parameterization。structured result 的 metadata 会记录 `intercept_estimator="centered_debiased"` 与 `intercept_influence="centered_nodewise"`。

对于 `LassoCV(compute_inference=True, inference_method="debiased")`，外层 CV estimator 会暴露与 final `estimator_` 相同的 `_inference_result` 与匹配的 `_params`/SE/statistic/p-value/CI reporting surface；公开 `coef_`/`intercept_` 仍属于 selected-alpha penalized prediction refit。

### Debiased simultaneous inference

设置 `enable_simultaneous_inference=True` 后，Lasso 使用 multiplier-bootstrap max-|Z| 临界值。普通 `_conf_int` 仍然是 marginal interval；联合区间单独保存在 `_conf_int_simultaneous`。

`simultaneous_include_intercept=False` 时 family 只包含 feature coefficients。设置为 `True` 时，与 marginal debiased SE 相同的 centered-nodewise 原始坐标系 intercept influence **真正进入 bootstrap max-|Z| calibration**；它不再只是一个额外输出行却套用 feature-only 临界值。每次成功 refit 都会先清除上一轮的 simultaneous critical value、target mask、联合区间以及 precision/influence state，再发布新结果。

## 参数（Parameters）

下表是 `statgpu.linear_model.Lasso` 的完整公开构造参数清单。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L1 正则强度。 |
| `fit_intercept` | `True` | 是否拟合截距。 |
| `max_iter` | `1000` | 优化最大迭代次数。 |
| `tol` | `1e-4` | 收敛容差。 |
| `stopping` | `"coef_delta"` | 停止准则：`coef_delta` / `kkt`。 |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`；`cpu_ols` 和 `gpu_ols` 暂时作为 deprecated alias 接受。 |
| `n_bootstrap` | `200` | residual-bootstrap 推断的抽样次数。 |
| `bootstrap_random_state` | `None` | residual-bootstrap 随机种子。 |
| `enable_simultaneous_inference` | `False` | 是否启用 simultaneous inference（仅 `debiased`）。 |
| `simultaneous_method` | `"maxz_bootstrap"` | simultaneous inference 方法；当前为 `maxz_bootstrap`。 |
| `simultaneous_alpha` | `0.05` | simultaneous family-wise error level。 |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| multiplier bootstrap 抽样次数。 |
| `simultaneous_random_state` | `None` | simultaneous bootstrap 随机种子。 |
| `simultaneous_include_intercept` | `False` | 是否把 debiased intercept 同时纳入 simultaneous target set 与 max-|Z| calibration family。 |
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

# GPU FISTA + 同一个与硬件无关的 post-selection inference method。
m_gpu = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="post_selection_ols",
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)

# 预测使用 penalized fit；推断/reporting 使用 active-set OLS/WLS 重拟合。
penalized_coef = m_gpu.coef_
post_selection_params = m_gpu._params
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
    simultaneous_include_intercept=True,
)
m_sim.fit(X, y)
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
```

## strict/approx 差异（strict/approx difference）

`debiased` 是高维逐系数推断的主路径；`post_selection_ols` 是更轻量的 active-set OLS/WLS diagnostic；`bootstrap` 是计算成本更高的重采样路径。三者的统计主张不同，不能互换解释。

## 输出（Outputs）

- penalized prediction fit：`intercept_`, `coef_`, `n_iter_`
- 推断（启用时）：`_params`, `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`, `_inference_result`
- `inference_method="post_selection_ols"` 时，`coef_` 仍为 penalized coefficients，而 `_params` 保存嵌入完整参数布局的 active-set OLS/WLS 重拟合结果；
- `inference_method="debiased"` 时，`_params[1:]` 保存 debiased slopes，`_params[0]` 保存与它们匹配的原始坐标系 debiased intercept；普通 `_conf_int` 对每个 reporting parameter是 marginal interval；
- 开启 simultaneous inference 后，`_conf_int_simultaneous` 给出配置 target family 上的联合区间；debiased intercept被包含时也会参与 max-|Z| 校准；
- 方法：`fit`, `predict`, `score`, `summary`
- 可用时还包括 `aic`、`bic` 等诊断。

## 常见问题（FAQ）

- **为什么同样 `tol` 下 CPU/GPU 迭代数不同？**  
  不同数值后端和算法实现可能产生不同收敛轨迹；比较时固定 `solver` 与 `stopping`。
- **CPU 用户应该设置 `cpu_solver` 吗？**  
  不应该。直接拟合统一使用 `solver`；`cpu_solver` 是旧 CPU/GPU split API 的 deprecated compatibility 参数。
- **应该根据硬件选择 `cpu_ols` 或 `gpu_ols` 吗？**  
  不应该。两者都是 `post_selection_ols` 的 deprecated alias。统计方法由 `inference_method` 选择，执行位置由 `device` 选择。
- **`post_selection_ols` 会改变 `coef_` 吗？**  
  不会。预测继续使用 penalized coefficients；active-set 重拟合保存在 `_params`、`_inference_result` 等推断/reporting 字段中。
- **为什么 `debiased` 下 `intercept_` 可能和 `_params[0]` 不同？**  
  `intercept_` 属于 penalized prediction fit；`_params[0]` 是与 debiased slope vector配套的 inference intercept。
- **`debiased` 适用于什么场景？**  
  适用于高维稀疏设置下需要系数级推断的场景，但仍依赖对应理论假设。
- **`post_selection_ols` 能当严格 selective-inference 置信程序吗？**  
  不能；应把它理解为 post-selection diagnostic。
- **普通 `debiased` 区间是联合区间吗？**  
  不是。普通 `_conf_int` 是 marginal interval；需要联合控制时使用 dedicated simultaneous path。
- **如何把截距纳入联合覆盖？**  
  设置 `simultaneous_include_intercept=True`；debiased intercept会同时进入 bootstrap max-|Z| 校准和最终联合区间 target set。

## 外部验证（External Validation）

- `dev/benchmarks/validate_post_selection_ols_gpu.py`
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`（历史 hardware-bearing API benchmark）
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`
- `dev/tests/test_post_selection_ols_inference_api.py`
- `dev/tests/test_penalized_solver_api_cleanup.py`

physical post-selection OLS validator 要求同时存在 CuPy CUDA 与 Torch CUDA。脚本存在本身不等于 physical GPU evidence；只有在物理 CUDA 环境对 exact head 真正执行并记录结果后，才能作为 GPU acceptance 证据。

## 参考（References）

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing in high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
