# 推断配置

> 语言: 中文  
> 最后更新: 2026-09-07  
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

对于 `Lasso` 与 `ElasticNet`，**统计方法是什么**与**在哪个设备上执行**是两个独立控制维度。当前维护的推断方法包括：

- `debiased`：去偏 / de-sparsified 系数推断；
- `post_selection_ols`：在 penalized fit 选出的 active set 上做启发式 OLS/WLS 重拟合；
- `bootstrap`：在支持路径上进行 residual bootstrap 推断。

`post_selection_ols` 是新的、与硬件无关的 canonical 拼法。统一 wrapper 中的 `cpu_ols` 与 `gpu_ols` **同时进入弃用期**：一个兼容周期内仍接受，但会发出 `FutureWarning`，并统一归一化为 `post_selection_ols`。`LassoCV` 还会在其兼容边界接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼法，并同样归一化到该方法。

`inference_method` 不负责选择设备。设备/后端遵循 estimator 的统一契约：

- 显式 `device="cpu"`：NumPy CPU；
- 显式 `device="cuda"`：只允许 CuPy CUDA，不可用时 fail closed；
- 显式 `device="torch"`：只允许 Torch CUDA，不可用时 fail closed；
- 只有 estimator 与全局配置都处于真正的 `device="auto"` 时，已经是 CuPy 或 Torch-CUDA 的输入才可以作为自动路由的一部分保留 native backend。

penalized fit 成功后，拟合后系数推断复用这次 fit 已记录的 `_selected_backend_name` / `_selected_backend_device`，不会再根据原始输入容器重新猜一次 backend。

### `post_selection_ols` 实际计算什么？

penalized model 先确定 active set；随后 statgpu 在**同一个 fit-resolved backend** 上，仅使用该 active set 对数据做无惩罚 OLS，存在 sample weights 时做 WLS，再计算对应 covariance 与参考分布推断。

原始 penalized `coef_` 仍然是预测时使用的系数；active-set OLS/WLS 重拟合用于推断与报告，保存在 `_params` / `_inference_result` 等 reporting surface 中。

在 `cov_type="nonrobust"` 下，这条路径保留既有的经典 **Student-t** 报告语义。estimator 已公开的 robust covariance 选项则复用共享 Gaussian robust-covariance layer，并使用对应的 normal-reference 报告语义。

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
