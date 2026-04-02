# CoxPH

路径：`statgpu.survival.CoxPH`

## 主要参数
- `ties` (`breslow` / `efron`)
- `tol`
- `max_iter`
- `device`
- `compute_inference`
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.survival import CoxPH

m = CoxPH(device="cuda", ties="breslow", compute_inference=False, gpu_memory_cleanup=True)
m.fit(X, time, event)
```
