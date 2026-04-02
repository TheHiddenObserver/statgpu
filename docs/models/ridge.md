# Ridge

路径：`statgpu.linear_model.Ridge`

## 主要参数
- `alpha`
- `fit_intercept`
- `device`
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.linear_model import Ridge

m = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
m.fit(X, y)
```
