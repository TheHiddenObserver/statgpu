# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the final post-review numerical/validator contract.

Fresh Tesla P100 evidence was measured from exact clean review/status SHA `5ed763be2a331e6dc988ac133e79f0484d4cdebd`. Its production-source parent checkpoint is `86bed6feb8f97ba80dbb58876238b972e60711f8`; the only commit between that production checkpoint and `5ed763be2a331e6dc988ac133e79f0484d4cdebd` changes this review document. The artifact commit `07ab76db7b9454e51683bbe4c1a2c2dc54ce58c2` adds only the two raw evidence files. The canonical-promotion commit `54b664c34aa6a150b7431f6289c83546502b2fd1` changes frontend parser metadata, manifest/coverage/tests/docs/generated benchmark assets only; it does not change `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py`. Therefore the measurement remains applicable under `RELEASING.md`.

## Correctness and backend-provenance evidence

- path: `results/pr126_p100/panel_stage_c_gpu_validation_5ed763be.json`
- measurement SHA: `5ed763be2a331e6dc988ac133e79f0484d4cdebd`
- artifact repository commit: `07ab76db7b9454e51683bbe4c1a2c2dc54ce58c2`
- Git blob: `d5aa76dd8d3305792c643bb29856610d784af3f0`
- SHA-256: `7d8777fabe32a012c91e8eb68914daca3e981cf6c6efeaa092b2172746b0d063`
- GPU: Tesla P100-SXM2-16GB
- result: CuPy 32/32 and Torch 32/32 = 26 estimator covariance cases + 6 direct public primitives per backend
- direct primitives: `cluster_group_debias`, `driscoll_kraay_qs`, `ill_conditioned_hc0`, `ill_conditioned_hc2`, `ill_conditioned_hc3`, `ill_conditioned_dk`
- requested backend equals the backend persisted at the numerical fit boundary for every correctness case; no numerical CPU fallback was observed
- the runner's package metadata records `cupy: null`; this is the same pre-existing package-name discovery limitation present in the previously accepted `aad53587...` artifact and does not replace executed-backend provenance

## Synchronized performance evidence

- path: `results/pr126_p100/panel_stage_c_performance_5ed763be.json`
- measurement SHA: `5ed763be2a331e6dc988ac133e79f0484d4cdebd`
- artifact repository commit: `07ab76db7b9454e51683bbe4c1a2c2dc54ce58c2`
- Git blob: `1e735274c76f3bad3aa747b25a36f449fb764cc0`
- SHA-256: `75da75c0405cce6e842c64c1b07b08a7b1cfd030fa08eac10916bd118ea33524`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: 58 = 54 base rows + 4 high-T QS rows
- timing scope: synchronized end-to-end estimator fit
- high-T scenario: `N=10,000`, `k=2`, `T=200`
- no speedup claim or CPU-baseline claim is made

## Canonical promotion audit

The fresh sources are registered as:

- correctness: `panel-stage-c-validation-pr126-20260810-7d8777fabe32`;
- performance: `panel-stage-c-performance-pr126-20260810-75da75c0405c`.

The promotion regenerated deterministic frontend assets and passed the maintained Stage-C parser, benchmark catalog, inventory-v2, domain-coverage, and benchmark-data validation suite. The generated inventory remains 13 registered / 13 available / 13 parsed sources with zero unclassified artifacts. The prior `aad53587...` source is no longer registered and remains immutable historical evidence.

## Strict review status

The final production-source re-review found no unresolved CRITICAL, HIGH, or relevant MEDIUM issue in Stage-C scope. The last in-scope fixes covered RandomEffects formula matrix depth, transformed-fit-space external HC baselines, PooledOLS/BetweenOLS intercept metadata and explicit no-intercept rejection, documentation status, and robust-summary `z` labels. One older unresolved P2 inline thread about backend provenance is outdated because the implemented fit/runner contract now persists and validates the actually selected backend. A pre-existing FamaMacBeth formula-summary naming limitation remains outside Stage-C scope.

## Final promoted-head checkpoint

This user-authored review/status-only commit follows canonical promotion specifically to trigger the permanent PR workflows after GitHub's `GITHUB_TOKEN` recursion guard on the bot-authored promotion commit. It changes no numerical source, physical validator, performance runner, parser contract, manifest, benchmark data, or raw evidence. Final acceptance requires those permanent workflows to complete green on this checkpoint and one final review of the post-measurement/promotion delta.

## Superseded historical evidence

The prior `aad53587...`, `c151550a...`, and `9c0b3050...` correctness/performance artifacts remain immutable audit history but are not current acceptance sources. The current canonical source contract is the fresh `5ed763be...` measurement above.

## Merge-readiness boundary

The physical gate is closed. Merge readiness still requires all permanent hosted workflows to be green on the final promoted-head checkpoint and a final post-promotion review with no new CRITICAL/HIGH/relevant-MEDIUM finding. The PR remains Draft until an explicit Ready transition is requested.
