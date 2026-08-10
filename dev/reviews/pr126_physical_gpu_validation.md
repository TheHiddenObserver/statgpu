# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the repaired numerical/validator contract.

Fresh Tesla P100 evidence was measured from exact clean implementation SHA `aad53587c9611da0e71a676e86ef32d9f6403f5c`. The subsequent artifact commit `fdf88e8990b30502958cc357099342d271792ec2` adds only the two raw evidence files; the canonical-promotion commits change parser/manifest/coverage/tests/docs/generated benchmark assets only. They do not change `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py`. Therefore the measurement remains applicable under `RELEASING.md`.

## Correctness and backend-provenance evidence

- path: `results/pr126_p100/panel_stage_c_gpu_validation_aad53587.json`
- measurement SHA: `aad53587c9611da0e71a676e86ef32d9f6403f5c`
- artifact repository commit: `fdf88e8990b30502958cc357099342d271792ec2`
- Git blob: `4fc52f29021321a9d7b1f87fc76950c843ef4059`
- SHA-256: `aab3ac61315b78ecaf04351f2922a444e3f44c587b31f9832ccad32839601b6c`
- GPU: Tesla P100-SXM2-16GB
- result: CuPy 32/32 and Torch 32/32 = 26 estimator covariance cases + 6 direct public primitives per backend
- direct primitives: `cluster_group_debias`, `driscoll_kraay_qs`, `ill_conditioned_hc0`, `ill_conditioned_hc2`, `ill_conditioned_hc3`, `ill_conditioned_dk`
- requested backend equals the backend persisted at the numerical fit boundary; no numerical CPU fallback was observed

## Synchronized performance evidence

- path: `results/pr126_p100/panel_stage_c_performance_aad53587.json`
- measurement SHA: `aad53587c9611da0e71a676e86ef32d9f6403f5c`
- artifact repository commit: `fdf88e8990b30502958cc357099342d271792ec2`
- Git blob: `fd7b06e0bb3a51e6a74e524e5bc059063b5f75eb`
- SHA-256: `99208f9276b92abf212d76655616718980095ba5c8ff9d3356a795a15ffa50c6`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: 58 = 54 base rows + 4 high-T QS rows
- timing scope: synchronized end-to-end estimator fit
- high-T scenario: `N=10,000`, `k=2`, `T=200`
- no speedup claim or CPU-baseline claim is made

## Canonical promotion audit

The fresh evidence is now the registered Stage-C canonical source contract:

- correctness source: `panel-stage-c-validation-pr126-20260810-aab3ac61315b`;
- performance source: `panel-stage-c-performance-pr126-20260810-99208f9276b9`;
- the physical parser expects 26 estimator cases plus all 6 direct primitives and emits 64 validation rows across CuPy/Torch;
- the canonical benchmark bundle contains 1984 runs after adding the eight newly promoted primitive rows;
- recorded maximum absolute differences are treated as finite diagnostics, while the physical runner's `status=success` remains authoritative for its `rtol+atol` NumPy parity check because the artifact does not store the reference magnitudes needed to reconstruct the relative-tolerance term;
- the prior `c151550a...` and `9c0b3050...` artifacts remain immutable historical/non-current evidence;
- deterministic benchmark frontend assets were regenerated after promotion, and temporary promotion helpers are absent from the final tree;
- the promotion audit also verified and restored the pre-existing PR #122 canonical SHA-256 assertion after a broad run-count replacement accidentally touched the `1976` digit sequence inside that immutable hash; no Stage-B source, parser, or artifact content was changed.

## Superseded historical evidence

The prior `c151550a...` and `9c0b3050...` correctness/performance artifacts remain immutable audit history but are not current acceptance sources. The current canonical source contract is the fresh `aad53587...` measurement above.

## Merge-readiness boundary

The physical gate is closed. Merge readiness still requires all permanent hosted workflows to be green on the final metadata/promotion head and a fresh review with no unresolved CRITICAL/HIGH/relevant-MEDIUM finding.
