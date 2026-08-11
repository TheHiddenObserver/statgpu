# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED / CANONICAL_PROMOTION / HOSTED_FINAL_PENDING**

Validation tier: `remote-full`.

The current accepted numerical measurement is exact clean head `3dc7df19176f8fb881a8d37e9d75b4f75e71b058` on Tesla P100-SXM2-16GB. Raw evidence was committed in `be679c13357475b8d26db42661b557fbea2f327f`; comparison from the measurement head to that raw commit changes only the two `results/pr126_p100/*_3dc7df19.json` artifacts. Promotion changes parser/source metadata, tests, docs/review records, and deterministic frontend benchmark assets only; it does not change `statgpu/panel/**` or either physical runner.

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

## New immutable canonical identities

- correctness source: `panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f`
- performance source: `panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55`
- parser generation: v2 rank-policy parser identities, separate from the frozen historical v1 `ec511f53...` parsers
- environment: `remote-p100-pr126-20260811`
- source date: `2026-08-11`

## Historical evidence

The prior canonical sources remain registered and immutable but are historical for current acceptance:

- `panel-stage-c-validation-pr126-20260811-af2227efe3cd` (`ec511f53...`, 32/32 per backend)
- `panel-stage-c-performance-pr126-20260811-409974070022` (`ec511f53...`, 58 rows)

The earlier `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` raw artifacts also remain audit history. None is overwritten.

## Remaining lifecycle gate

Physical evidence is accepted. The remaining gate is canonical promotion validation, all permanent hosted workflows on the post-promotion checkpoint, and one final `.claude/skills/code-review.md` re-review. PR #126 remains Draft and unmerged; Ready-for-review or merge requires explicit user instruction.
