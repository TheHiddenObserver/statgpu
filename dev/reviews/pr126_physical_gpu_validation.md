# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**

Validation tier: `remote-full`.

The accepted numerical measurement is exact clean head `3dc7df19176f8fb881a8d37e9d75b4f75e71b058` on Tesla P100-SXM2-16GB. Raw evidence was committed in `be679c13357475b8d26db42661b557fbea2f327f`; comparison from the measurement head to that raw commit changes only the two `results/pr126_p100/*_3dc7df19.json` artifacts. Canonical promotion at `d771a7432e3b3da3299b6af2b05c5171579092ce` changes parser/source metadata, maintained benchmark tests, docs/review records, and deterministic frontend benchmark assets only; it does not change `statgpu/panel/**` or either physical runner. The post-promotion review checkpoint `d38feab5aca8fa76b9be9f08d753be8bf85b7203` passed all seven permanent hosted workflows, including the maintained Torch matrix and the R `plm`/`sandwich` external alignment.

## Current correctness and backend provenance

- path: `results/pr126_p100/panel_stage_c_gpu_validation_3dc7df19.json`
- measurement SHA: `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`
- raw artifact commit: `be679c13357475b8d26db42661b557fbea2f327f`
- Git blob: `5de153023af751ff960969483c11fb77dc115456`
- SHA-256: `c67ada7ec59ff85d6a652714dc45cdae81efc67a4c13fe853c85d0712fc689ad`
- GPU: Tesla P100-SXM2-16GB
- CuPy package provenance: `13.6.0`
- CuPy: **39/39** = 27 estimator integrations + 12 direct public primitives
- Torch: **39/39** = 27 estimator integrations + 12 direct public primitives
- rank-boundary primitives: `rank_boundary_nonrobust`, `rank_boundary_hc0`, `rank_boundary_hc2`, `rank_boundary_hc3`, `rank_boundary_cluster`, `rank_boundary_dk`
- estimator integration: `panel_rank_boundary_dk` passes on both backends with numerical `design_rank=2`, `design_columns=3`, and `rank_deficient_extension=true`
- every estimator/primitive records requested backend == executed backend; no numerical CPU fallback is accepted

## Current synchronized performance

- path: `results/pr126_p100/panel_stage_c_performance_3dc7df19.json`
- measurement SHA: `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`
- raw artifact commit: `be679c13357475b8d26db42661b557fbea2f327f`
- Git blob: `5e903cf0bd9a5f73e9273d9cc3488cb5e13219da`
- SHA-256: `f27bef0b7c55db8a8d31f0574e3133760649c28a9089b3c7d3cfcc121009e152`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: **58/58** = 54 base + 4 high-T QS
- timing scope: synchronized end-to-end estimator fit
- high-T matrix: CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000,k=2,T=200`
- every timing sample is finite/positive and each stored median equals the median of its three raw samples
- no speedup or CPU-baseline claim is made

The performance artifact has `environment.packages.cupy=null` because that runner queried the distribution name `cupy` while the host uses a CUDA-suffixed CuPy distribution. This is a metadata-only limitation: the paired required exact-head correctness artifact records CuPy `13.6.0`, and the timing runner could only produce CuPy rows after importing CuPy and additionally failed closed unless every timed fit persisted the requested backend. The canonical parser does not synthesize a CuPy version into the timing rows; this cross-artifact fact is retained only as provenance.

## Immutable canonical identities

- correctness source: `panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f`
- performance source: `panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55`
- parser generation: v2 rank-policy parser identities, separate from the frozen historical v1 `ec511f53...` parsers
- environment: `remote-p100-pr126-20260811`
- source date: `2026-08-11`

The canonical generator/promotion gate produced **2120 runs / 47 models**, parsed **15/15** registered sources, reported **0 validation errors and 0 errors/warnings**, passed the maintained parser/catalog/inventory/domain suite **59/59**, passed strict-source regeneration, and passed frontend typecheck/build. The exact-source applicability gate confirmed that no `statgpu/panel/**` file and neither physical runner changed after the numerical measurement.

## Historical evidence

The prior canonical sources remain registered and immutable but are historical for current acceptance:

- `panel-stage-c-validation-pr126-20260811-af2227efe3cd` (`ec511f53...`, 32/32 per backend)
- `panel-stage-c-performance-pr126-20260811-409974070022` (`ec511f53...`, 58 rows)

The earlier `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` raw artifacts also remain audit history. None is overwritten.

## Final hosted and review gate

The post-promotion review checkpoint `d38feab5...` completed all seven permanent workflows successfully:

1. Tests
2. Panel Stage C Torch CPU
3. Panel Stage C external covariance
4. Maintenance compatibility
5. Release notes validation
6. Release package validation
7. Benchmark Frontend CI

The maintained Torch 2.0.1 CPU suite passed **31/31** and the R `Run R plm and sandwich alignment` step completed successfully. The final strict review found no new CRITICAL, HIGH, or relevant MEDIUM issue after the promotion/artifact fixes. The only final-head delta after that checkpoint is terminal review/status documentation; it does not change production source, parsers, manifests, generated benchmark assets, or physical runners.

The technical Stage-C status is therefore **PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**. PR #126 intentionally remains Draft and unmerged; Ready-for-review or merge requires explicit user instruction.
