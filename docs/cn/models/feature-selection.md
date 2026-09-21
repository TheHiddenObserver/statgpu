# 特征选择

> 语言：中文  
> 最后更新：2026-09-21  
> 切换：[English](../../en/models/feature-selection.md)

## 概览与路径

`StepwiseSelector` 与 `stepwise_selection` 基于 AIC/BIC 执行 `forward`、
`backward` 或 `both` 子集搜索。`knockoff_filter` 及选择器封装提供 fixed-X
和高斯二阶近似 model-X FDR 控制；详见 [Knockoff](knockoff.md)。

```python
from statgpu import LinearRegression, StepwiseSelector

selector = StepwiseSelector(
    LinearRegression,
    criterion="bic",
    direction="both",
    max_features=10,
    compute_inference=False,
).fit(X, y)
X_selected = selector.transform(X)
```

## Stepwise 契约

- 候选特征按确定顺序拟合，`predict` 与 `transform` 使用同一顺序；
- 后向选择从全模型开始，先强制满足 `max_features`，再要求信息准则改善；
- 仅截距模型或空模型可以胜出，不强制保留任何特征；
- 重复调用 `fit()` 会清空历史记录与缓存；
- `n_jobs` 使用线程，避免将设备数组序列化并传入其他进程；
- 后端与推断能力由 `model_class` 及其参数决定。

## 输出与边界

拟合后提供 `selected_features_`、`best_model_`、`aic_history_`、`bic_history_`、
`selection_history_`、`predict`、`transform` 与 `fit_transform`。信息准则搜索具有
组合复杂度，适合中等特征数；高维 FDR 控制优先使用 knockoff。
