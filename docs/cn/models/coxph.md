# CoxPH

> 语言: 中文  
> 最后更新: 2026-07-01  
> 页面定位: 模型文档  
> 切换: [English](../en/models/coxph.md)

语言切换：[English](../en/models/coxph.md)

## 概览（Overview）

`CoxPH` 实现比例风险模型，支持 CPU/GPU、Breslow/Efron ties 处理。向量化 Efron 梯度/Hessian（无 Python 循环）、多块 CUDA kernel、DLPack 桥接 torch-CUDA。

补充：

- **Efron 优化** (v0.2.1)：前缀和向量化路径，n=5000 时比 statsmodels 快 3-6x；已在 CI 中与 statsmodels PHReg 对齐验证。
- `PenalizedCoxRegression` 支持 SCAD/MCP 惩罚，通过 proximal Newton 求解。
- `CoxPH` 的 `entry`（delayed entry）路径在 `cpu/cuda/torch` 均可用。
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

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | ties 处理：`breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson 收敛阈值 |
| `max_iter` | `100` | 最大迭代数 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `True` | 是否计算推断与部分诊断 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `gpu_memory_cleanup` | `False` | GPU 路径后尝试释放 CuPy/Torch CUDA 缓存 |

## Entry 与设备约束（Entry & Device Notes）

- `CoxPH`：
  - `entry + breslow`：CPU/CUDA/Torch 支持
  - `entry + efron`：CPU/CUDA/Torch 支持（2026-04-22）
  - `device='cuda'`：要求可用的 CuPy CUDA 后端
  - `device='torch'`：要求 `torch.cuda.is_available() == True`
- `CoxPHCV`：
  - GPU 下 `entry` 目前仅支持 `ties='breslow'`
  - `gpu_memory_cleanup=True` 会传递给最终 `CoxPH` estimator，并暴露 CuPy/Torch 清理钩子
- `torch.compile`（若启用）需要 Triton 支持的 GPU（Compute Capability >= 7.0），如 A30/RTX 4090；P100（CC 6.0）不支持。

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.survival import CoxPH

# CPU + cluster robust
m_cpu = CoxPH(device="cpu", cov_type="cluster", ties="efron")
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

当前接口未区分独立 `strict/approx` 开关。默认路径用于高一致性估计与推断；GPU 与 CPU 在 C-index 等指标上可能有轻微数值差异。

## 输出（Outputs）

- `fit(X, time, event, entry=None) -> self`
- 预测：`predict_risk_score(X)`、`predict_hazard_ratio(X)`、`predict_survival(X, times=None)`、`predict(X)`（hazard ratio 别名）
- 模型属性：`coef_`, `hazard_ratios_`
- 推断属性（`compute_inference=True`）：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合指标：`log_likelihood`, `aic`, `bic`, `concordance_index`
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
