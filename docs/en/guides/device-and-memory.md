# Device and GPU Memory

> Language: English  
> Last updated: 2026-06-01
> This page: Guide  
> Switch: [Chinese](../../cn/guides/device-and-memory.md)

Language switch: [Chinese](../../cn/guides/device-and-memory.md)

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

## CPU/GPU Transfer Optimizations

statgpu now keeps common GPU-to-GPU conversions off the host when possible:

- CuPy -> Torch CUDA conversions prefer DLPack zero-copy sharing.
- Torch CUDA -> CuPy conversions prefer DLPack zero-copy sharing.
- NumPy -> Torch CUDA transfers use pinned host memory with `non_blocking=True` when PyTorch accepts it.
- If a DLPack or pinned-memory path is unavailable in the current environment, statgpu falls back to the existing safe conversion path.

These optimizations are implementation details and do not change device purity:
explicit `device="cuda"` still requires CuPy/CUDA, and explicit `device="torch"`
still requires Torch CUDA.

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
- `CoxPHCV`

Behavior:
- `gpu_memory_cleanup=False` (default): better repeated-fit throughput due to CuPy pool reuse.
- `gpu_memory_cleanup=True`: frees CuPy pool blocks and asks Torch CUDA to release cached blocks after public GPU work, usually lowering steady VRAM usage.

Example:

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
model.fit(X, y)
```
