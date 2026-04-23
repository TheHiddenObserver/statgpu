# Cox Entry+Efron GPU Update (2026-04-22)

## Scope

- Module: `statgpu/survival/_cox.py`
- Focus: enable and benchmark `entry + efron` on GPU backends (`cuda`, `torch`)
- Benchmark type: non-CV (`CoxPH`) with delayed entry enabled

## Implemented Changes

- Removed hard blocking for `entry + efron` in GPU fit paths.
- Added entry-group failure pointer context (`fail_ptr`) for grouped Efron terms.
- Implemented Efron-adjusted log-likelihood under delayed entry for CUDA and Torch.
- Implemented Efron-adjusted gradient/Hessian under delayed entry for CUDA and Torch:
  - Per-group correction over `k=0..d-1`
  - Denominator: `S0 - (k/d) * E0_fail`
  - First moment correction: `S1 - (k/d) * E1_fail`
  - Second moment correction: `S2 - (k/d) * E2_fail`
- Kept existing `entry + breslow` optimized path unchanged.

## Validation Summary

### Large (n_train=6400, p=80, entry=True, ties=efron)

- `cpu`: 11.31s
- `cuda`: 16.09s
- `torch`: 10.86s
- `lifelines`: 22.95s

Accuracy vs lifelines:

- `coef_linf`: ~1.37e-08 (cpu/cuda/torch)
- `cindex_abs`: 0.0
- `loglik_abs`: ~1e-11

### XLarge (n_train=16000, p=80, entry=True, ties=efron)

- `cpu`: 19.52s
- `cuda`: 41.31s
- `torch`: 23.67s
- `lifelines`: 114.96s

Accuracy vs lifelines:

- `coef_linf`: ~1.86e-07 (cuda/torch)
- `cindex_abs`: consistent

## GPU Flag Ablation (Large)

- Baseline: `cuda 16.41s`, `torch 10.76s`
- `cupy_fused` on: `cuda 15.63s`, `torch 8.41s` (best stable setting on current host)
- `torch.compile`:
  - Not usable on current remote host (Tesla P100, CC 6.0)
  - Triton backend requires Compute Capability >= 7.0

## Decisions

- Keep `cupy_fused` path and continue tuning thresholds.
- Do not default-enable `torch.compile` on unknown hardware.
- Add capability check + eager fallback for `torch.compile`.

## Next Actions

1. Optimize CUDA `entry + efron` inner `k` loop and small-matrix update overhead.
2. Add automatic `torch.compile` gating by GPU capability.
3. Re-run xlarge A/B after CUDA-side loop fusion/block optimization.

