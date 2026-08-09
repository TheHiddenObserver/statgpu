# PR #122 Panel Stage B physical GPU validation

## Final acceptance status

Panel Tier-1 Stage B has current physical GPU acceptance for exact numerical implementation head `a57efcea29b0e87ecb89865c5a6902d5773812c6`.

Both maintained P100 gates were executed from that exact implementation with a clean working tree and completed successfully on CuPy and Torch CUDA:

1. `dev/benchmarks/validate_panel_stage_b_gpu.py`
2. `dev/benchmarks/validate_panel_stage_b_disconnected_fe_gpu.py`

The raw outputs were committed unchanged in repository commit `72b3279d2028e8ec2af30e138e123aceb611ae8c`. The compare from `a57efcea...` to `72b3279d...` contains only those two JSON evidence files; no numerical implementation or validation-runner code changed.

## Raw evidence

Full Stage-B matrix:

- path: `results/pr122_p100/panel_stage_b_gpu_validation_a57efcea.json`
- Git blob: `254b64776bff4e3b2b642bb4a2ae1eea25f4751c`
- schema version: 2
- top-level `git_sha`: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- `working_tree_clean = true`
- `status = success`
- CuPy: all 17 estimator cases and all four Hausman diagnostics succeeded with the requested CuPy backend
- Torch: all 17 estimator cases and all four Hausman diagnostics succeeded with the requested Torch backend
- no timing or speedup measurement was collected

Focused disconnected two-way FE gate:

- path: `results/pr122_p100/panel_stage_b_disconnected_fe_gpu_validation_a57efcea.json`
- Git blob: `3bda0b2040479ba8201e2722eb990ba086c3f3b9`
- schema version: 1
- top-level `git_sha`: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- `working_tree_clean = true`
- `status = success`
- fixture: two disconnected 2x2 incidence blocks plus one singleton
- `nobs = 9`
- legacy residual df = 0
- component-aware/public residual df = 1
- effect rank = 7
- `rank_x = 1`
- incidence components = 3
- CuPy and Torch both executed on the requested backend and matched NumPy within the validator tolerances

The NumPy reference confidence interval for the one-slope fixture is approximately `[-4.853102368087349, 7.853102368087349]`. CuPy and Torch both return approximately `[-4.853102368087348, 7.853102368087348]`; the recorded maximum absolute CI difference is `8.881784197001252e-16` for each GPU backend.

This focused result closes the physical regression that previously produced a Torch interval of roughly `[-49998.5, 50001.5]` at residual df 1. The repaired panel inference uses the exact identity `Student-t(df=1) == standard Cauchy`; the successful P100 rerun confirms the corrected boundary on physical Torch CUDA as well as CuPy.

## Environment

Both artifacts report:

- GPU: Tesla P100-SXM2-16GB
- Python: 3.9.16
- statgpu: 0.2.4
- NumPy: 1.24.2
- SciPy: 1.10.1
- PyTorch: 2.0.0

The runner could not resolve the installed CuPy distribution version through `importlib.metadata`, so the artifact records that package field as `null`. Physical CuPy execution is instead established by the per-backend `executed_backend = "cupy"` provenance checks.

## Canonical frontend evidence

Current canonical validation source:

- path: `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260809.json`
- canonical SHA-256: `a6e47b9dec9c35040d2715b573aa2a93b084d83c574078bdb430861f0a9aa9f9`
- source id: `panel-stage-b-pr122-20260809-a6e47b9dec9c`
- measurement implementation SHA: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- parser: `panel_stage_b_physical_validation` v1.0
- validation-only evidence; no timing or speedup is inferred

The source registers the full 17-case matrix and four Hausman parameterizations per backend, producing 42 validation rows. It also records the focused disconnected-FE gate as supporting provenance rather than inventing timing rows for it.

The earlier canonical source measured at `faa95ce7fb5cb204088957fbda5544c20a06fbfc` is historical evidence for that older implementation and is no longer the current dashboard source. Its immutable raw artifact remains preserved in `results/pr122_p100/`.

## Applicability after evidence-only commits

The physical acceptance is anchored to the exact numerical implementation `a57efcea...`. Subsequent commits may add or promote raw evidence, canonical metadata, frontend contract tests, generated benchmark assets, and review documentation without invalidating the numerical measurement. Physical applicability is preserved provided a repository compare confirms that no numerical production implementation relevant to the measurement and neither physical validation runner changed after `a57efcea...`.

Frontend/evidence tests may change to assert the promoted source identity or provenance; those tests do not alter the physically measured numerical implementation. If numerical production code or either physical validation runner changes later, a new exact-head physical validation is required before PR #122 can again be considered physically accepted.

## Acceptance conclusion

**PHYSICAL_GPU_ACCEPTED** for Panel Tier-1 Stage B numerical implementation `a57efcea29b0e87ecb89865c5a6902d5773812c6`.

The remaining PR lifecycle work is evidence/frontend staleness synchronization, hosted final-head validation, and the normal review-thread/fresh-review process. No additional P100 rerun is required unless numerical production or physical validation-runner code changes.
