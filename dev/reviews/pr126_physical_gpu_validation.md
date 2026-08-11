# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PARTIAL_REMOTE_PENDING** after the 2026-08-11 `.claude/skills/code-review.md` re-audit.

Validation tier target: `remote-full`.

The previously accepted `5ed763be...` physical evidence remains immutable historical evidence, but it is no longer exact-source acceptance evidence after the re-audit changes `statgpu/panel/_fixed_effects.py` to repair formula/inference presentation. Fresh physical GPU correctness and synchronized performance evidence are required on the final review-fix source head before this record can return to `PHYSICAL_GPU_ACCEPTED`.

Historical Tesla P100 evidence was measured from exact clean review/status SHA `5ed763be2a331e6dc988ac133e79f0484d4cdebd`. Its production-source parent checkpoint is `86bed6feb8f97ba80dbb58876238b972e60711f8`; the artifact commit `07ab76db7b9454e51683bbe4c1a2c2dc54ce58c2` adds only the two raw evidence files, and the prior canonical-promotion commit `54b664c34aa6a150b7431f6289c83546502b2fd1` changed only parser metadata, manifest/coverage/tests/docs/generated benchmark assets. That evidence was applicable to the pre-re-audit source through the prior promoted checkpoint, but the later `PanelOLS.summary()` formula/inference presentation fix changes `statgpu/panel/**`; under `RELEASING.md` it is therefore no longer exact-source acceptance evidence for the current branch.

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

The 2026-08-11 `.claude/skills/code-review.md` re-audit found and fixed a HIGH formula/inference issue: `PanelOLS.summary()` had overwritten Patsy term names with generic `x1`, `x2`, ... labels. A new categorical/interaction/transform regression now requires `summary().feature_names`, `_feature_names`, and `_inference_result.feature_names` to match the Patsy term order. The same re-audit made benchmark capability decisions and the memory/no-host-transfer acceptance contract explicit. The current review/fix loop has no unresolved CRITICAL/HIGH source finding at this checkpoint; acceptance remains `PARTIAL_REMOTE_PENDING` until the loop reaches a no-new-findings review, hosted gates are green, and fresh exact-head physical GPU evidence is recorded. One older unresolved P2 inline thread about backend provenance is outdated because the implemented fit/runner contract now persists and validates the actually selected backend. A pre-existing FamaMacBeth formula-summary naming limitation remains outside Stage-C scope.

## Prior promoted-head checkpoint

The prior acceptance cycle reached promoted checkpoint `102d58b0ee239902804a282fe9747fdf713ad3ab` with permanent hosted workflows green. That checkpoint predates the current `PanelOLS.summary()` production-source fix and is retained only as audit history; it is not the current final acceptance head.

## Superseded historical evidence

The prior `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` correctness/performance artifacts remain immutable audit history. The `5ed763be...` sources remain registered benchmark provenance until replacement evidence is promoted, but none of these artifacts is exact-source physical acceptance evidence for the current post-re-audit source.

## Merge-readiness boundary

The previous physical gate is superseded by the current production-source review fix. Merge readiness now requires fresh physical GPU correctness/performance evidence on the final source head, canonical promotion of any replacement artifacts, permanent hosted workflows green on the resulting final checkpoint, and a final `.claude/skills/code-review.md` re-review with no unresolved CRITICAL/HIGH/relevant-MEDIUM finding. The PR remains Draft until an explicit Ready transition is requested.
