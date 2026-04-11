# Ridge

> 语言: 中文  
> 最后更新: 2026-04-10  
> 页面定位: 模型文档  
> 切换: [English](../en/models/ridge.md)

语言切换：[English](../en/models/ridge.md)

路径：`statgpu.linear_model.Ridge`

## 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L2 正则强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | 并行任务数 |
| `compute_inference` | `True` | 是否计算推断统计（SE/t/p/CI） |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶；为空时使用 Newey-West 风格默认规则 |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `alpha`: L2 正则强度
- `fit_intercept`
- `device`: `cpu` / `cuda` / `auto`
- `compute_inference`: 是否计算推断统计（SE/t/p/CI）
- `cov_type`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
- `hac_maxlags`: 仅 `cov_type="hac"` 生效；未指定时按样本规模自动推断
- `gpu_memory_cleanup`

## 快速示例

```python
from statgpu.linear_model import Ridge

m = Ridge(
  alpha=1.0,
  device="cuda",
  compute_inference=True,
  cov_type="hc3",
  gpu_memory_cleanup=True,
)
m.fit(X, y)
m.summary()
```

## 推断与稳健协方差

- `cov_type="nonrobust"`：经典 ridge 协方差
- `cov_type="hc0"`：White/sandwich 稳健协方差
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 的稳健修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差，可通过 `hac_maxlags` 控制滞后阶

## 输出

- 系数：`intercept_`, `coef_`
- 推断：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- 诊断：`r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- 预测：`predict(X)`
- 评分：`score(X, y)`
- 汇总：`summary()`

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict(X)`：返回预测值向量
- `score(X, y)`：返回 `R^2`
- 常用属性：`coef_`, `intercept_`, `aic`, `bic`

## 常见问题

- **Q: `alpha` 怎么选？**  
  A: 先用对数网格（如 `1e-4` 到 `1e2`）做外部 CV，再固定到目标任务。
- **Q: 什么时候用 `hac_maxlags`？**  
  A: 当 `cov_type="hac"` 且数据存在时序相关时，建议显式设置；不设时使用默认滞后规则。

## 说明

- Ridge 当前已支持与 `LinearRegression` 对齐的推断能力与稳健协方差选项（含 `HC2/HC3/HAC`）。
