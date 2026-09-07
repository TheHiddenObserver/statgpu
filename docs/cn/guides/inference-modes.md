# 推断配置

> 语言: 中文  
> 最后更新: 2026-09-08  
> 页面定位: 指南文档  
> 切换: [English](../../en/guides/inference-modes.md)

语言切换：[English](../../en/guides/inference-modes.md)

## Gaussian 线性模型推断

对于 squared-error L2/Ridge 使用的共享 Gaussian 推断路径，数值协方差与参考分布推断在实际完成模型拟合的 backend 上执行，即 NumPy、CuPy 或 Torch。数值阶段包括 bread/协方差计算、标准误、检验统计量、p 值以及置信区间临界值。

既有公开 reporting 契约保持不变：所有数值推断完成后，推断结果以及 estimator 的 reporting 属性（`_params`、`_bse`、`_tvalues`、`_pvalues`、`_conf_int`）才进行一次最终 NumPy snapshot。这个转换是 reporting boundary，而不是 CPU inference fallback。共享路径会在 `_inference_result.metadata` 中记录 `numerical_backend`、`numerical_device`、`reporting_backend="numpy"` 和 `reporting_boundary="post_numerical_inference"`。

显式 `device="cuda"` 或 `device="torch"` 时，Gaussian inference 不会静默降级到 NumPy。若缺失或出现非法的实际执行 backend provenance，则直接 fail closed。只有 `device="auto"` 允许自动选择可用 backend。

Gaussian 路径支持：

- `nonrobust`：经典协方差，使用 Student-t 参考分布；
- `hc0`、`hc1`、`hc2`、`hc3`：异方差稳健 sandwich 协方差，使用正态参考分布；
- `hac`：Bartlett kernel HAC 协方差，使用正态参考分布。

backend-native reference helper 同时保留残差自由度为 1 和 2 时的稳定 Student-t 恒等式，避免极端但仍可表示的尾概率因减法消去或不必要的 `t**2` overflow 被错误压成 0。

## 稀疏 penalized-linear 推断

对于 `Lasso`、`ElasticNet`，以及公开 generic
`PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")`
入口，**统计方法是什么**与**在哪个设备上执行**是两个独立控制维度。当前维护的推断方法包括：

- `debiased`：去偏 / de-sparsified 系数推断；
- `post_selection_ols`：在 penalized fit 选出的 active set 上做启发式 OLS/WLS 重拟合；
- `bootstrap`：在支持路径上进行 residual bootstrap 推断。

`post_selection_ols` 是新的、与硬件无关的 canonical 拼法。统一 wrapper 中的 `cpu_ols` 与 `gpu_ols` **同时进入弃用期**：一个兼容周期内仍接受，但会发出 `FutureWarning`，并统一归一化为 `post_selection_ols`。`LassoCV` 还会在其兼容边界接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼法，并同样归一化到该方法。

`inference_method` 不负责选择设备。设备/后端遵循 estimator 的统一契约：

- 显式 `device="cpu"`：NumPy CPU；
- 显式 `device="cuda"`：只允许 CuPy CUDA，不可用时 fail closed；
- 显式 `device="torch"`：只允许 Torch CUDA，不可用时 fail closed；
- 只有 estimator 与全局配置都处于真正的 `device="auto"` 时，已经是 CuPy 或 Torch-CUDA 的输入才可以作为自动路由的一部分保留 native backend。

稀疏 Gaussian penalty 使用字符串还是公开 `Penalty` 对象，不会改变上述 migration 与 AUTO routing 契约。

backend 复用保证是**按推断方法区分**的：`post_selection_ols` 始终复用成功拟合记录的 `_selected_backend_name` / `_selected_backend_device`；维护中的 CuPy/Torch `debiased` 路径也会把数值推断留在实际执行的 GPU backend。相比之下，residual `bootstrap` 当前仍使用 CPU-native residual refit。因此显式 GPU `device` 会控制 penalized fit 的执行位置，但不应被理解成 bootstrap 也变成 GPU-native。

对于 analytic `sample_weight`，维护中的 NumPy/CuPy/Torch `debiased` 路径使用同一个 weighted-centered average-loss 工作问题。因此把所有权重同时乘以任意正的常数，不会改变 penalized fit 或 debiased inference。

加权 `LassoCV` 也使用同一 analytic-weight 约定：默认 alpha grid、每个 training fold 的目标函数、加权 validation MSE 与最终 selected-alpha refit 保持在同一尺度。所有权重都等于同一个正常数时，会直接视为与 unweighted 完全相同的统计问题，避免额外浮点漂移。AUTO 一旦为 CV 解析出具体 CPU/CuPy/Torch backend，最终 `Lasso` refit 也保持在同一 backend；显式 CPU 会在进入 dedicated CV selector 前把异构 GPU 输入统一转换为 NumPy。

对于 debiased simultaneous inference，普通 `_conf_int` 仍然是 marginal interval。`enable_simultaneous_inference=True` 使用 multiplier-bootstrap max-|Z| 校准；当 `simultaneous_include_intercept=True` 时，原始坐标系中的截距 influence **真正参与 bootstrap maximum**，而不只是额外出现在最终区间的输出行中。成功 refit 会先清除上一轮的 simultaneous critical value、target mask、联合区间以及 precision/influence state，再计算新结果。

### `post_selection_ols` 实际计算什么？

penalized model 先确定 active set；随后 statgpu 在**同一个 fit-resolved backend** 上，仅使用该 active set 对数据做无惩罚 OLS，存在 sample weights 时做 WLS，再计算对应 covariance 与参考分布推断。

原始 penalized `coef_` 仍然是预测时使用的系数；active-set OLS/WLS 重拟合用于推断与报告，保存在 `_params` / `_inference_result` 等 reporting surface 中。

两次拟合的 diagnostic ownership 也不同。在 `summary()` 中，系数表和 `Post-selection Refit DoF` 属于 active-set refit；R-squared、adjusted R-squared、F statistic、log-likelihood、AIC、BIC 以及 `Penalized-fit Residual DoF` 仍描述 penalized prediction fit。summary 会明确分开标注，避免把 refit 的残差自由度误当成 penalized-fit diagnostics 使用的自由度。若 active design 秩亏，refit residual DoF 使用 `n - effective_rank`，而不是 `n - active_column_count`；系数重拟合与 covariance bread 都使用 design-level Moore-Penrose/SVD 计算，避免通过 normal equations 把条件数平方。metadata 会记录 `refit_rank`、`refit_parameter_count` 与 `refit_rank_deficient`。

在 `cov_type="nonrobust"` 下，这条路径保留既有的经典 **Student-t** 报告语义。estimator 已公开的 robust covariance 选项则复用共享 Gaussian robust-covariance layer，并使用对应的 normal-reference 报告语义。若模型不含截距且 active set 为空，所有参数坐标都只是 inactive compatibility placeholder；result 仍保留调用者请求的 covariance/reference family（`nonrobust` -> Student-t，robust/HAC -> normal），不会把 robust 请求静默改写成 nonrobust。

完整 reporting array 还保留旧 `cpu_ols` surface 的一个兼容细节：**未被 active set 选中的坐标**会以 `SE=0`、统计量 `0`、`p=1`、置信区间 `[0, 0]` 作为占位。这些值**不表示该系数被“精确证明为 0”或方差真的为 0**。应使用 `_inference_result.metadata["selected_feature_indices"]` 判断哪些坐标实际执行了 active-set OLS/WLS 推断。

它仍然只是 post-selection diagnostic：同一数据先做变量选择、再套普通 OLS/WLS 区间，并不能自动得到一般意义上的 selective-inference coverage。

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
)
model.fit(X, y)

# 预测仍使用 penalized fit。
penalized_coef = model.coef_

# 推断/reporting 使用 active-set OLS/WLS 重拟合。
post_selection_params = model._params
```

如果目标是高维场景下更正式的逐系数推断，而不是工程型 post-selection diagnostic，应优先考虑 `inference_method="debiased"`，并检查对应理论假设。Lasso 的 simultaneous max-|Z| 路径与普通 marginal interval、p-value adjustment 也应区分理解。

## 相关模型的稳健协方差

- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `Ridge(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- 稀疏 Gaussian `post_selection_ols` 会在 estimator 已公开的 covariance choice 下复用同一 Gaussian covariance layer；
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
