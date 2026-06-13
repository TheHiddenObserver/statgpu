# Entry Benchmark Report (CPU/CuPy/Torch + Frameworks)

## Scope
- Objective: compare delayed-entry (`entry`) accuracy and runtime across `statgpu` backends and common frameworks.
- Environment: remote server `myconda`.
- Dataset: synthetic survival data with delayed entry.
- Setting: `ties="breslow"`, 3-fold CV, penalty grid `geomspace(1.0, 0.01, 8)`.
- Split: train `n=1440`, test `n=360`, features `p=60`.

## Results
- statgpu cpu: fit `350.214s`, test c-index `0.826707`, best penalty `1.0`
- statgpu cuda: fit `324.037s`, test c-index `0.826707`, best penalty `1.0`
- statgpu torch: fit `166.908s`, test c-index `0.826707`, best penalty `1.0`
- lifelines: fit `29.613s`, test c-index `0.826523`, best penalty `0.01`
- scikit-survival Coxnet: unavailable for delayed entry (left truncation unsupported)

## Interpretation
- Accuracy is aligned across `statgpu` backends and lifelines (difference in c-index is small).
- Runtime ranking on this delayed-entry case: `lifelines < statgpu_torch < statgpu_cuda < statgpu_cpu`.
- Current delayed-entry path in `CoxPH` is not a fully native GPU solver for CUDA/Torch; therefore additional backend-specific optimization is still needed for true full-GPU entry acceleration.

## Notes
- During run, `_cox_cv.py` emitted `RuntimeWarning: Mean of empty slice` in the two-stage CV section; this does not invalidate the reported fit outputs but indicates edge handling can be improved for entry-heavy folds.
