# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED / POST_PROMOTION_REVIEW_PENDING**

Validation tier: `remote-full`.

Fresh Tesla P100 evidence was measured from exact clean strict-review checkpoint `ec511f539adeaaedf310f92248200d0868577532`. The raw artifact commit `e1a155bf77b416e0873a037015aaafd22371ab11` differs from the measurement checkpoint only by `results/pr126_p100/panel_stage_c_gpu_validation_ec511f53.json` and `results/pr126_p100/panel_stage_c_performance_ec511f53.json`. Canonical promotion changes only parser/source metadata, coverage/tests/docs/review records, and deterministic frontend benchmark assets; it does not change `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py`. Therefore the physical measurement remains applicable under `RELEASING.md`.

## Correctness and backend-provenance evidence

- path: `results/pr126_p100/panel_stage_c_gpu_validation_ec511f53.json`
- measurement SHA: `ec511f539adeaaedf310f92248200d0868577532`
- raw artifact commit: `e1a155bf77b416e0873a037015aaafd22371ab11`
- Git blob: `a02fcad0eefd5993d2ae05b8d00a55e5ca1d885f`
- SHA-256: `af2227efe3cd0ab77472ff1d6584233d475f6ed5c4c4d36d318efc127d143f63`
- GPU: Tesla P100-SXM2-16GB
- result: CuPy **32/32**, Torch **32/32** = 26 estimator covariance cases + 6 direct public primitives per backend
- direct primitives: `cluster_group_debias`, `driscoll_kraay_qs`, `ill_conditioned_hc0`, `ill_conditioned_hc2`, `ill_conditioned_hc3`, `ill_conditioned_dk`
- every estimator case and public primitive records the requested backend as the executed backend; no numerical CPU fallback was observed

## Synchronized performance evidence

- path: `results/pr126_p100/panel_stage_c_performance_ec511f53.json`
- measurement SHA: `ec511f539adeaaedf310f92248200d0868577532`
- raw artifact commit: `e1a155bf77b416e0873a037015aaafd22371ab11`
- Git blob: `76d6eabeefc6e04095270a5df8a231cb150ea220`
- SHA-256: `4099740700221ffdae2770427a5ad0fca7dc3c1ec0f47173caf166aa56a1fca0`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: **58** = 54 base + 4 high-T QS
- timing scope: synchronized end-to-end estimator fit
- high-T matrix: CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000,k=2,T=200`
- every timing sample and stored median is finite and positive; each stored median equals the median of its raw samples
- no speedup claim or CPU-baseline claim is made

## Canonical promotion

- correctness source id: `panel-stage-c-validation-pr126-20260811-af2227efe3cd`
- performance source id: `panel-stage-c-performance-pr126-20260811-409974070022`
- source date: `2026-08-11`
- environment: `remote-p100-pr126-20260811`

The parser fails closed on measurement SHA, clean-tree flag, exact 26+6 correctness identity, requested/executed backend identity, exact 58-row base/high-T matrix, and positive finite synchronized timing samples.

## Superseded historical evidence

The `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` Stage-C artifacts remain immutable audit history and are not current canonical acceptance sources.

## Remaining merge-readiness boundary

The physical gate is closed. Merge readiness still requires permanent hosted workflows green on the final post-promotion checkpoint and a final `.claude/skills/code-review.md` re-review with no new CRITICAL/HIGH/relevant-MEDIUM finding. PR #126 remains Draft until an explicit Ready transition is requested.
