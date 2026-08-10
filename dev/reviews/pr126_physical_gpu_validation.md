# PR #126 Panel Stage C physical GPU validation

## Physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the Stage-C correctness and performance runners measured at post-review exact clean implementation head `c151550ab17bd9533a51599f86b6a4ea12a292e9`.

The immutable rerun artifacts were added by repository commit `a8dc976208d3997140829491be7d3dbfddfcedd6`. Comparing `c151550ab17bd9533a51599f86b6a4ea12a292e9` to `a8dc976208d3997140829491be7d3dbfddfcedd6` changes only the two new raw JSON evidence files; no numerical implementation, physical validator, performance runner, test, or documentation file changed before measurement was recorded. Therefore the rerun evidence is applicable under `RELEASING.md`.

The earlier exact-clean measurement `9c0b3050dd143c43a06bb6393d69f4f83e861637` and artifact commit `85d710bddf633134624501a9e27f03c30bc04ead` remain immutable **historical** evidence. They were superseded for current acceptance because the Ready-triggered ordered-categorical Driscoll-Kraay chronology fix changed `statgpu/panel/_covariance.py` and `statgpu/panel/_formula.py` after that run.

## Current correctness/backend-provenance artifact

- path: `results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json`
- measurement SHA: `c151550ab17bd9533a51599f86b6a4ea12a292e9`
- artifact repository commit: `a8dc976208d3997140829491be7d3dbfddfcedd6`
- Git blob: `ecaf21a96ee60aea5a446e1cc40a17ba5c2a205f`
- SHA-256: `91e9ec53b6ee9516064797fcce0a23ba6a198037b639fd9965135bae3b3c7b63`
- schema version: 1
- working tree clean: true
- top-level status: success
- GPU: Tesla P100-SXM2-16GB
- Python: 3.9.16
- NumPy: 1.24.2
- SciPy: 1.10.1
- Torch: 2.0.0

For both CuPy and Torch, all 26 estimator cases passed with requested/executed backend identity. Each backend also passed the two direct public primitive calls (`cluster_group_debias` and `driscoll_kraay_qs`) with `xp` omitted, physically validating public backend auto-detection rather than only estimator-mediated routing.

Direct primitive maximum absolute differences versus NumPy are `8.673617379884035e-19` for `cluster_group_debias` and `4.336808689942018e-19` for `driscoll_kraay_qs` on both GPU backends. The QS estimator path records `n_periods=8`, `bandwidth=2`, `max_weighted_lag=7`, and `all_observed_lags_weighted=true`; legacy Pooled HAC remains `row_order_hac=true`.

The ordered-categorical chronology regression is a CPU metadata-to-time-code contract and is maintained separately in hosted tests: ordered categories such as `t1,t2,t10` are shown equivalent to explicit numeric chronology before the compact codes enter backend-native grouped-score accumulation. The physical runner then validates those backend-native covariance operations on both CuPy and Torch.

## Current synchronized performance artifact

- path: `results/pr126_p100/panel_stage_c_performance_c151550a.json`
- measurement SHA: `c151550ab17bd9533a51599f86b6a4ea12a292e9`
- artifact repository commit: `a8dc976208d3997140829491be7d3dbfddfcedd6`
- Git blob: `e6495802d47f9db01ce1b379cc13845c5aaee09e`
- SHA-256: `1861fc4f1817789484d4b2fb02c205c302199b7b4ce167d75d409d8c54d81931`
- schema version: 2
- working tree clean: true
- timing scope: synchronized end-to-end estimator fit
- input residency: X/y/entity/time preloaded on selected GPU backend; cluster labels remain CPU metadata
- rows: 58 = 54 base rows + 4 high-T QS rows
- repeats per row: 3

High-T `N=10,000`, `k=2`, `T=200` medians:

- CuPy: Pooled QS `23.4445 ms`; Panel entity-FE QS `32.3719 ms`.
- Torch: Pooled QS `16.1504 ms`; Panel entity-FE QS `23.0404 ms`.

The performance artifact is accepted as bounded synchronized timing evidence. It does **not** contain or imply a speedup claim or CPU/external baseline.

## Canonical benchmark promotion

The current raw rerun artifacts are registered directly as SHA-256-protected canonical frontend sources; no normalized duplicate is created.

- validation source id: `panel-stage-c-validation-pr126-20260810-91e9ec53b6ee` — 56 validation-only rows = `(26 estimator + 2 public primitive) x 2 backends`; no timing or speedup.
- performance source id: `panel-stage-c-performance-pr126-20260810-1861fc4f1817` — 58 timing rows; no speedup.
- source date: `2026-08-10`
- environment: `remote-p100-pr126-20260810`

The Stage-C parsers fail closed on measurement-SHA drift, backend/case identity drift, public-primitive matrix drift, non-finite/out-of-tolerance correctness differences, malformed timing samples, median/sample inconsistency, exact 54-row base matrix drift, and exact four-row high-T QS matrix drift.

## Historical artifacts retained

The following raw files remain in the repository for audit history but are no longer current required canonical sources:

- `results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json` — SHA-256 `a0d258f6d6b8243e82684a29305606e5f6bd91bbe271c3ed335b32b5ec973665`.
- `results/pr126_p100/panel_stage_c_performance_9c0b3050.json` — SHA-256 `214284f02a5e21e775e58deaf2fa3cc9b6384d392b96c6f300f31f4a02953b1c`.

## Physical conclusion

Stage-C post-review physical correctness/backend provenance and bounded synchronized performance gates are **ACCEPTED**. Exact-final-head hosted CI and a fresh `.claude/skills/code-review.md` review remain post-promotion lifecycle gates. Any later change to `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be audited for whether a new physical run is required.
