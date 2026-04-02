# Device and GPU Memory

> Language: English  
> Last updated: 2026-04-02  
> This page: Guide  
> Switch: [中文](../../guides/device-and-memory.md)

Language switch: [中文](../../guides/device-and-memory.md)

## Device Selection

Every estimator supports:
- `device="cpu"`
- `device="cuda"`
- `device="auto"` (default)

## GPU Memory Cleanup: `gpu_memory_cleanup`

Supported by all current models:
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LogisticRegression`
- `CoxPH`

Behavior:
- `gpu_memory_cleanup=False` (default): better repeated-fit throughput due to CuPy pool reuse.
- `gpu_memory_cleanup=True`: frees pool blocks after each fit, usually lower steady VRAM usage.

Example:

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
model.fit(X, y)
```
