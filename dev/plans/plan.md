# StatGPU Plan Delta (2026-04-05)

> Canonical merged planning entry: `PLAN_UNIFIED.md` in workspace root.
> This file is retained for history and quick delta tracking.

## Priority Queue (after bootstrap phase)

1. P0 (highest): Multiple-testing expansion batch 1
- Add global p-value combination methods: Fisher combination and Cauchy combination (ACAT alias).
- Provide unified API in statgpu.inference and thin wrappers on BaseEstimator.
- Add NumPy/CuPy consistency tests and axis-behavior tests.
- Status: Completed in this iteration.

2. P1 (high): Multiple-testing expansion batch 2
- Add knockoff roadmap (fixed-X first, model-X next) for linear-family feature selection.
- Standardize outputs: selected set, W statistics, q/FDR trajectory, random-state trace.

3. P2 (medium): Multiple-testing expansion batch 3
- Add correlation-aware/global methods as needed (Brown/Kost style Fisher correction, HMP, weighted variants).

4. P3 (lower): Time-series methods
- Add time-series inference/diagnostic methods after multiple-testing roadmap reaches stable v1.

5. P3 (lower): Spatial econometrics methods
- Add spatial lag/error model roadmap (SAR/SEM) after the same v1 checkpoint.

## Item Completed Now (from high-priority queue)

- Implemented combine_pvalues API with Fisher and Cauchy methods.
- Added BaseEstimator.combine_pvalues wrapper.
- Added/extended targeted tests for CPU/GPU and wrappers.
