# CoxPH

> 语言：中文  
> 最后更新：2026-07-24  
> 页面定位：模型文档  
> 切换：[English](../../../en/models/coxph.md)

## 概览

`CoxPH` 实现比例风险回归，支持 NumPy、CuPy CUDA 和 Torch CUDA 后端，以及 Breslow/Efron ties 处理。公开契约包括后端原生预测、明确的优化终止状态、可选稳健/聚类稳健推断、delayed entry，以及 `CoxPHCV` 交叉验证。

重要行为：

- 显式 `device="cuda"` 和 `device="torch"` 不会静默回退 CPU；
- `compute_inference=False` 表示仅估计，推断字段保持未设置；
- 稳健推断默认使用 strict 路径，Efron 近似推断必须显式启用；
- delayed-entry 是否支持取决于后端、惩罚、协方差类型以及是否请求推断，详见下表。

## 路径

```python
from statgpu.survival import CoxPH, CoxPHCV
```

## 目标函数

对协变量 \(x_i\)、事件指标 δ_i 与风险集 \(R_i\)，无惩罚模型最大化

$$
\ell(\beta)=\sum_{i:\delta_i=1}\left(x_i^\top\beta-\log\sum_{j\in R_i}\exp(x_j^\top\beta)\right),
$$

并按 `ties` 使用 Breslow 或 Efron 处理。当 `penalty > 0` 时，`CoxPH` 按项目约定的目标函数尺度加入 L2 惩罚。

## 优化与收敛

Newton 迭代使用 line search，并在最终参数处重新检查 KKT。line search 失败时不会更新系数，也不会误报收敛。公开拟合状态包括：

- `converged_`；
- `termination_reason_`；
- `n_iter_`；
- `final_kkt_inf_`；
- `final_kkt_normalized_`。

log likelihood、Hessian、协方差和推断结果均在最终系数向量处重新计算，而不是复用旧迭代状态。

## 协方差与推断

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 基于观测信息矩阵的模型协方差 |
| `"hc0"` | 稳健 sandwich 协方差 |
| `"hc1"` | 带有限样本修正的稳健 sandwich 协方差 |
| `"cluster"` | 聚类稳健协方差；在 `fit` 时传入 `cluster=` |

`compute_inference=True` 会计算 `_bse`、`_zvalues`、`_pvalues` 与 `_conf_int`。`compute_inference=False` 仅执行估计；即使 `cov_type` 指定为稳健类型，也不会生成协方差或推断字段。

`inference_mode="strict"` 为默认值：

- Breslow 精确 score residual 由内部实现提供；
- Efron 精确 robust residual 需要安装 `survival` extra；
- 只有显式设置 `inference_mode="approx"`，才允许在精确 residual 不可用时使用 event-row Efron sandwich 近似。

推断来源通过以下字段公开：

- `inference_method_`；
- `inference_backend_`；
- `inference_approximate_`；
- `inference_fallback_reason_`；
- `full_host_transfer_performed_`。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | ties 处理：`"breslow"` 或 `"efron"` |
| `tol` | `1e-9` | Newton/KKT 收敛阈值 |
| `max_iter` | `100` | 最大迭代次数 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"` |
| `compute_inference` | `True` | 是否计算协方差与推断输出 |
| `cov_type` | `"nonrobust"` | `"nonrobust"`、`"hc0"`、`"hc1"` 或 `"cluster"` |
| `penalty` | `0.0` | 非负 L2 惩罚 |
| `inference_mode` | `"strict"` | 稳健推断策略：`"strict"` 或 `"approx"` |
| `gpu_memory_cleanup` | `False` | 尝试释放 CuPy/Torch 缓存 |

## Delayed-entry 支持矩阵

关键区别在于是否请求推断。

| Entry | Penalty | 协方差 | `compute_inference` | CPU | CuPy | Torch |
|---|---:|---|---:|---|---|---|
| 无 | 任意 | 支持的 `cov_type` | 任意 | 支持 | 支持 | 支持 |
| 有 | `0` | `nonrobust` | 任意 | 支持；CPU 路径需要 `statgpu[survival]` | 支持 | 支持 |
| 有 | `>0` | `nonrobust` | 任意 | 显式 `NotImplementedError` | 支持 | 支持 |
| 有 | 任意 | `hc0` / `hc1` / `cluster` | `True` | 显式 `NotImplementedError` | 显式 `NotImplementedError` | 显式 `NotImplementedError` |
| 有 | 任意 | `hc0` / `hc1` / `cluster` | `False` | 允许仅估计；推断字段为 `None` | 允许仅估计；推断字段为 `None` | 允许仅估计；推断字段为 `None` |

补充说明：

- CPU delayed-entry 与 Efron 精确稳健推断依赖 `pip install "statgpu[survival]"`；
- Breslow 与 Efron delayed-entry 均遵循上表；
- `CoxPHCV` 在最终 refit 时执行相同的 `compute_inference` guard；
- GPU delayed-entry CV 当前仅支持 `ties="breslow"`；
- CPU delayed-entry CV 可用显式无惩罚网格，如 `penalties=[0.0]`；任意非零 delayed-entry CPU penalty candidate 都会被拒绝；
- `inference_mode` 会传递给最终 estimator；
- `predict` 与 `score` 复用最终 estimator 的后端原生实现。

## CPU 与 GPU 示例

```python
from statgpu.survival import CoxPH

# Efron 精确 cluster robust；需要 statgpu[survival]。
strict_model = CoxPH(
    device="cpu",
    ties="efron",
    cov_type="cluster",
    inference_mode="strict",
    compute_inference=True,
)
strict_model.fit(X, time, event, cluster=cluster_ids)

# delayed-entry + robust cov_type，但仅估计，不计算推断。
estimation_only = CoxPH(
    device="cuda",
    ties="breslow",
    cov_type="hc0",
    compute_inference=False,
)
estimation_only.fit(X_gpu, time_gpu, event_gpu, entry=entry_gpu)
assert estimation_only._bse is None
assert estimation_only._conf_int is None

# 标准 Torch CUDA 推断。
torch_model = CoxPH(
    device="torch",
    ties="efron",
    cov_type="nonrobust",
    compute_inference=True,
)
torch_model.fit(X_torch, time_torch, event_torch)
```

## 输出

- 参数：`coef_`、`hazard_ratios_`；
- 启用推断时：`_bse`、`_zvalues`、`_pvalues`、`_conf_int`；
- 诊断：`log_likelihood`、`aic`、`bic`、`concordance_index`；
- 收敛状态：`converged_`、`termination_reason_`、`n_iter_`、`final_kkt_inf_`、`final_kkt_normalized_`；
- 推断来源：`inference_method_`、`inference_backend_`、`inference_approximate_`、`inference_fallback_reason_`、`full_host_transfer_performed_`；
- 后端原生预测：`predict_risk_score`、`predict_hazard_ratio`、`predict_survival` 与 `predict`。

## 验证

PR #79 已对 NumPy、CuPy CUDA 与 Torch CUDA 的维护中 CoxPH 路径进行验证。exact-head GitHub Actions 覆盖 Python 3.9–3.12；维护中的真实 GPU 测试在 Tesla P100 上通过。canonical accuracy report 只允许从 clean exact-head validated artifact 生成；陈旧的硬编码 PASS 文件不具有权威性。

相关产物：

- `dev/reviews/pr79_physical_gpu_validation.md`；
- `dev/tests/test_pr79_physical_gpu.py`；
- `dev/benchmarks/pr79/`。

## 限制

- delayed-entry robust/cluster covariance 在请求推断时尚未实现；
- CPU delayed entry + 非零 penalty 尚未实现；
- strata、frailty 与 time-varying covariates 当前不在支持范围；
- 可选 `torch.compile` 路径需要 Triton-capable GPU，Tesla P100 不支持该路径。

## 参考文献

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
