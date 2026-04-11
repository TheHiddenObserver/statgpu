# CoxPH

> 语言: 中文  
> 最后更新: 2026-04-11  
> 页面定位: 模型文档  
> 切换: [English](../en/models/coxph.md)

语言切换：[English](../en/models/coxph.md)

路径：`statgpu.survival.CoxPH`

## 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | ties 处理：`breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson 收敛阈值 |
| `max_iter` | `100` | 最大迭代数 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | 是否计算推断与部分诊断 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `ties` (`breslow` / `efron`)
- `tol`
- `max_iter`
- `device` (`cpu` / `cuda` / `auto`)
- `compute_inference`
- `cov_type` (`nonrobust` / `hc0` / `hc1` / `cluster`)
- `gpu_memory_cleanup`

## 快速示例

```python
from statgpu.survival import CoxPH

m = CoxPH(device="cuda", ties="breslow", compute_inference=False, gpu_memory_cleanup=True)
m.fit(X, time, event)
```

当 `cov_type="cluster"` 时，需要传入 `cluster` 分组向量：

```python
m = CoxPH(device="cpu", cov_type="cluster")
m.fit(X, time, event, cluster=cluster_ids)
```

## 输出

- 参数：`coef_`, `hazard_ratios_`
- 推断：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`（`compute_inference=True`）
- 拟合指标：`log_likelihood`, `aic`, `bic`, `concordance_index`
- 其他：基线风险相关结果（当推断开启时）

## 返回与属性

- `fit(X, time, event, entry=None)`：返回 `self`
- `predict_risk_score(X)`：返回线性风险分数（`X @ coef`）
- `predict_hazard_ratio(X)`：返回风险比（`exp(X @ coef)`）
- `predict_survival(X, times=None)`：返回二元组 `(``survival``, ``times``)`，分别表示生存函数预测值及其对应时间点
- `predict(X)`：`predict_hazard_ratio` 的别名
- 常用属性：`coef_`, `hazard_ratios_`, `concordance_index`, `aic`

## 常见问题

- **Q: `breslow` 和 `efron` 怎么选？**  
  A: ties 较多时通常优先 `efron`；ties 较少时二者差异通常较小。
- **Q: GPU 下 C-index 和 CPU 有细微差异？**  
  A: 目前存在近似与数值路径差异，做严格评估时建议同时报告 CPU 结果。

## 说明

- 支持 Breslow/Efron ties 处理。
- 当前以标准 CoxPH 主路径为主，`strata/frailty/time-varying covariates` 等扩展仍在规划中（见 `TO_DO.md`）。
