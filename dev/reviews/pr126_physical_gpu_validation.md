# PR #126 Panel Stage C physical GPU validation

## Physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the Stage-C correctness and performance runners measured at exact clean implementation head `9c0b3050dd143c43a06bb6393d69f4f83e861637`.

The immutable artifacts were added by repository commit `85d710bddf633134624501a9e27f03c30bc04ead`. Comparing `9c0b3050dd143c43a06bb6393d69f4f83e861637` to `85d710bddf633134624501a9e27f03c30bc04ead` changes only the two raw JSON evidence files; no numerical implementation, physical validator, performance runner, test, or documentation file changed before measurement was recorded. Therefore the evidence remains applicable under `RELEASING.md`.

## Correctness/backend-provenance artifact

- path: `results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json`
- measurement SHA: `9c0b3050dd143c43a06bb6393d69f4f83e861637`
- artifact repository commit: `85d710bddf633134624501a9e27f03c30bc04ead`
- Git blob: `fe015f03fd488e22d40beccfe4cbe1984b25f5b7`
- SHA-256: `a0d258f6d6b8243e82684a29305606e5f6bd91bbe271c3ed335b32b5ec973665`
- schema version: 1
- working tree clean: true
- top-level status: success
- GPU: Tesla P100-SXM2-16GB
- Python: 3.9.16
- NumPy: 1.24.2
- SciPy: 1.10.1
- Torch: 2.0.0

For both CuPy and Torch, all 26 estimator cases passed with requested/executed backend identity. Each backend also passed two direct public primitive calls with `xp` omitted, physically validating public backend auto-detection rather than only estimator-mediated routing.

Direct primitive maximum absolute differences versus NumPy:

- CuPy `cluster_group_debias`: `8.673617379884035e-19`; `driscoll_kraay_qs`: `4.336808689942018e-19`.
- Torch `cluster_group_debias`: `8.673617379884035e-19`; `driscoll_kraay_qs`: `8.673617379884035e-19`.

The QS estimator path records `n_periods=8`, `bandwidth=2`, `max_weighted_lag=7`, and `all_observed_lags_weighted=true`. The legacy Pooled HAC path remains explicitly `row_order_hac=true`.

## Synchronized performance artifact

- path: `results/pr126_p100/panel_stage_c_performance_9c0b3050.json`
- measurement SHA: `9c0b3050dd143c43a06bb6393d69f4f83e861637`
- artifact repository commit: `85d710bddf633134624501a9e27f03c30bc04ead`
- Git blob: `53a06d82764bdeeb962eda5096fb6ee79220d5b0`
- SHA-256: `214284f02a5e21e775e58deaf2fa3cc9b6384d392b96c6f300f31f4a02953b1c`
- schema version: 2
- working tree clean: true
- timing scope: synchronized end-to-end estimator fit
- input residency: X/y/entity/time preloaded on selected GPU backend; cluster labels remain CPU metadata
- rows: 58 = 54 base rows + 4 high-T QS rows
- repeats per row: 3

High-T `N=10,000`, `k=2`, `T=200` medians:

- CuPy: Pooled QS `23.5766 ms`; Panel entity-FE QS `32.7507 ms`.
- Torch: Pooled QS `15.9364 ms`; Panel entity-FE QS `23.5517 ms`.

The performance artifact is accepted as bounded synchronized timing evidence. It does **not** contain or imply a speedup claim or CPU/external baseline. No pathological complexity or transfer-dominated behavior was observed in the required high-T gate.

## Canonical benchmark promotion

The two immutable raw artifacts are registered directly as SHA-256-protected canonical frontend sources rather than copied into a second normalized JSON.

- validation source id: `panel-stage-c-validation-pr126-20260810-a0d258f6d6b8` — 56 validation-only rows = `(26 estimator + 2 public primitive) x 2 backends`; no timing or speedup.
- performance source id: `panel-stage-c-performance-pr126-20260810-214284f02a5e` — 58 timing rows; no speedup.
- source date: `2026-08-10`
- environment: `remote-p100-pr126-20260810`

The Stage-C parsers fail closed on measurement-SHA drift, backend/case identity drift, public-primitive matrix drift, non-finite/out-of-tolerance correctness differences, malformed timing samples, median/sample inconsistency, and high-T QS contract drift.

## Physical conclusion

Stage-C physical correctness/backend provenance and bounded synchronized performance gates are **ACCEPTED**. Exact-final-head hosted CI and a fresh `.claude/skills/code-review.md` review remain post-promotion lifecycle gates. Any later change to `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be audited for whether a new physical run is required.
