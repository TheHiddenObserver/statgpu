# LogisticRegression

路径：`statgpu.linear_model.LogisticRegression`

## 主要参数
- `fit_intercept`
- `C`
- `max_iter`
- `tol`
- `device`
- `compute_inference`
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=True)
m.fit(X, y_binary)
proba = m.predict_proba(X)
```
