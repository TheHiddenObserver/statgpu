# Device and GPU Memory

> Language: English  
> Last updated: 2026-04-25  
> This page: Guide  
> Switch: [Chinese](../../guides/device-and-memory.md)

Language switch: [Chinese](../../guides/device-and-memory.md)

## Device Selection

Every estimator supports:
- `device="cpu"`
- `device="cuda"`
- `device="torch"`
- `device="auto"` (default)

Device purity rules:
- `device="cpu"` keeps core fit/predict/score computation on NumPy.
- `device="cuda"` uses CuPy for core computation. If CuPy/CUDA is unavailable, statgpu raises an error instead of falling back to CPU.
- `device="torch"` uses Torch CUDA for core computation. If Torch CUDA is unavailable, statgpu raises an error instead of using Torch CPU.
- `device="auto"` is the only mode allowed to choose another available backend automatically.
- Formula/DataFrame parsing can run on CPU as preprocessing, but model computation is converted to the selected backend.

Solver coverage for GLM-style estimators:

| Solver | NumPy | CuPy | Torch |
|---|---|---|---|
| `exact` | yes | yes | yes |
| `fista` | yes | yes | yes |
| `irls` | yes | yes | yes |
| `newton` | smooth objectives | smooth objectives | smooth objectives |
| `lbfgs` | smooth objectives | smooth objectives | smooth objectives |

Non-smooth penalties such as L1 and ElasticNet use FISTA. Newton/L-BFGS with non-smooth penalties raises `ValueError`.

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
