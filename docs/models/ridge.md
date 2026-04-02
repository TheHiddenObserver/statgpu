# Ridge

> 语言: 中文  
> 最后更新: 2026-04-02  
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
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `alpha`: L2 正则强度
- `fit_intercept`
- `device`: `cpu` / `cuda` / `auto`
- `gpu_memory_cleanup`

## 快速示例

```python
from statgpu.linear_model import Ridge

m = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
m.fit(X, y)
```

## 输出

- 系数：`intercept_`, `coef_`
- 预测：`predict(X)`
- 评分：`score(X, y)`

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict(X)`：返回预测值向量
- `score(X, y)`：返回 `R^2`
- 常用属性：`coef_`, `intercept_`

## 常见问题

- **Q: `alpha` 怎么选？**  
  A: 先用对数网格（如 `1e-4` 到 `1e2`）做外部 CV，再固定到目标任务。
- **Q: 为什么没有完整推断统计？**  
  A: 当前 Ridge 文档侧重估计与预测；更完整推断能力是后续计划项。

## 说明

- 当前 Ridge 重点是估计与预测路径；高级推断选项相对 `LinearRegression` 更少。
- 如果你重点关注推断统计，可优先参考 `LinearRegression` 或先在 Ridge 上做 post-hoc 分析。
