# CoxPH

> 语言: 中文  
> 最后更新: 2026-07-23
> 页面定位: 模型文档  
> 切换: [English](../en/models/coxph.md)

语言切换：[English](../en/models/coxph.md)

## 概览（Overview）

`CoxPH` 实现比例风险模型，支持 CPU/GPU、Breslow/Efron ties 处理。向量化 Efron 梯度/Hessian（无 Python 循环）、多块 CUDA kernel、DLPack 桥接 torch-CUDA。

补充：

- **Efron 优化** (v0.2.1)：前缀和向量化路径，n=5000 时比 statsmodels 快 3-6x；已在 CI 中与 statsmodels PHReg 对齐验证。
- `PenalizedCoxRegression` 支持 SCAD/MCP 惩罚，通过 proximal Newton 求解。
- `CoxPH` 的 `entry`（delayed entry）路径在三后端按下方支持矩阵可用。
- 显式 `device='cuda'` 和 `device='torch'` 不会静默回退 CPU；需要 CPU 路径时使用 `device='cpu'`。
- `CoxPHCV` 已可用，支持 penalty 网格搜索 + 全量重训。

## 路径（Path）

`statgpu.survival.CoxPH`

## 目标函数（Objective Function）

最大化 Cox 部分似然（partial likelihood），估计风险系数 `\beta`；风险比定义为 `exp(X\beta)`。

## 估计方程（Estimating Equation）

通过 Newton-Raphson 迭代求解，`tol` 控制收敛阈值，`max_iter` 控制最大迭代次数。`ties` 支持 `breslow` 与 `efron`。

## 协方差与推断（Covariance/Inference）

支持协方差选项：

- `cov_type="nonrobust"`：经典信息矩阵口径
- `cov_type="hc0"`：稳健 sandwich 口径
- `cov_type="hc1"`：带自由度修正的稳健口径
- `cov_type="cluster"`：聚类稳健口径（需在 `fit` 时传入 `cluster` 分组向量）

推断统计以 z 统计量口径输出。

- `inference_mode="strict"` 为默认值，不会静默退化到近似协方差。
- Breslow 精确 score residual 由内部实现提供；Efron 精确 robust residual
  需要安装 `survival` extra。
- 只有显式设置 `inference_mode="approx"` 时，才允许 Efron event-row 近似。
- 推断来源记录在 `inference_method_`、`inference_backend_`、
  `inference_approximate_` 与 `inference_fallback_reason_`。

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | ties 处理：`breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson 收敛阈值 |
| `max_iter` | `100` | 最大迭代数 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `True` | 是否计算推断与部分诊断 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `penalty` | `0.0` | 非负 L2 惩罚 |
| `inference_mode` | `"strict"` | 稳健推断策略：`strict` / `approx` |
| `gpu_memory_cleanup` | `False` | GPU 路径后尝试释放 CuPy/Torch CUDA 缓存 |

## Entry 与设备约束（Entry & Device Notes）

| Entry | Penalty | 协方差 | CPU | CuPy | Torch |
|---|---:|---|---|---|---|
| 无 | 任意 | 支持的 `cov_type` | 支持 | 支持 | 支持 |
| 有 | `0` | `nonrobust` | 支持；需要 statsmodels | 支持 | 支持 |
| 有 | `>0` | `nonrobust` | 显式 `NotImplementedError` | 支持 | 支持 |
| 有 | 任意 | `hc0` / `hc1` / `cluster` | 显式 `NotImplementedError` | 显式 `NotImplementedError` | 显式 `NotImplementedError` |

- Breslow 与 Efron delayed-entry 都遵循该矩阵。
- CPU delayed-entry 和 Efron 精确稳健推断依赖：
  `pip install "statgpu[survival]"`。
- `device='cuda'` 要求可用的 CuPy CUDA 后端。
- `device='torch'` 要求 `torch.cuda.is_available() == True`。
- `CoxPHCV`：
  - GPU 下 `entry` 目前仅支持 `ties='breslow'`
  - CPU delayed-entry CV 遇到任意非零 penalty candidate 会显式失败；安装
    `statgpu[survival]` 后可使用 `penalties=[0.0]` 执行无惩罚拟合
  - delayed-entry robust/cluster covariance 与 `CoxPH` 一样显式抛出
    `NotImplementedError`
  - `inference_mode` 会传递给最终 estimator，`predict`/`score` 复用其后端原生实现
  - `gpu_memory_cleanup=True` 会传递给最终 `CoxPH` estimator，并暴露 CuPy/Torch 清理钩子
- `torch.compile`（若启用）需要 Triton 支持的 GPU（Compute Capability >= 7.0），如 A30/RTX 4090；P100（CC 6.0）不支持。

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.survival import CoxPH

# Efron 精确 cluster robust（需要 statgpu[survival]）
m_cpu = CoxPH(
    device="cpu", cov_type="cluster", ties="efron",
    inference_mode="strict",
)
m_cpu.fit(X, time, event, cluster=cluster_ids)

# GPU
m_gpu = CoxPH(
    device="cuda",
    ties="breslow",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, time, event)
```

## strict/approx 差异（strict/approx difference）

该开关控制稳健 score-residual 推断，不控制 ties 算法。

- `strict`（默认）：不允许静默返回近似协方差。Breslow 使用内部精确
  residual；Efron 精确 residual 需要 `statgpu[survival]` 中的 statsmodels。
- `approx`：精确 Efron residual 不可用时，允许 event-row sandwich 近似。
  报告结果前应检查 `inference_approximate_` 与
  `inference_fallback_reason_`。
- delayed-entry robust/cluster covariance 尚未实现，无论该开关或
  `compute_inference` 如何设置都会显式报错。

## 输出（Outputs）

- `fit(X, time, event, entry=None) -> self`
- 预测：`predict_risk_score(X)`、`predict_hazard_ratio(X)`、`predict_survival(X, times=None)`、`predict(X)`（hazard ratio 别名）
- 模型属性：`coef_`, `hazard_ratios_`
- 推断属性（`compute_inference=True`）：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合指标：`log_likelihood`, `aic`, `bic`, `concordance_index`
- 收敛状态：`converged_`, `termination_reason_`, `n_iter_`,
  `final_kkt_inf_`, `final_kkt_normalized_`
- 推断来源：`inference_method_`, `inference_backend_`,
  `inference_approximate_`, `inference_fallback_reason_`,
  `full_host_transfer_performed_`
- 预测方法返回 estimator 后端原生数组。
- 其他：基线风险相关结果（启用推断时）

## 常见问题（FAQ）

- **`breslow` 与 `efron` 如何选？**  
  ties 较多时优先 `efron`；ties 较少时两者通常接近。
- **GPU 与 CPU 的 C-index 略有差异是否正常？**  
  正常，可能由数值实现与近似路径差异导致。严格评估建议同时报告 CPU 结果。

## 外部验证（External Validation）

建议按生存分析对齐流程，结合 `dev/tests/` 与 `dev/benchmarks/` 中 CoxPH 相关脚本做一致性与性能回归验证。

## 参考（References）

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220. [https://doi.org/10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89-99. [https://doi.org/10.2307/2529620](https://doi.org/10.2307/2529620)
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), 557-565. [https://doi.org/10.1080/01621459.1977.10480613](https://doi.org/10.1080/01621459.1977.10480613)
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *Journal of the American Statistical Association*, 84(408), 1074-1078. [https://doi.org/10.1080/01621459.1989.10478874](https://doi.org/10.1080/01621459.1989.10478874)
