# PR #122 Panel Stage B physical GPU validation

## Physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the Stage-B runner measured at exact clean implementation head `2701aa9feb3796c33c94e6480fcb78c80c6a809c`.

This status covers the active physical correctness/backend-provenance gate only. Exact-final-head hosted CI and a fresh `.claude/skills/code-review.md` review remain lifecycle gates after evidence promotion; the physical runner itself must not change without another P100 rerun.

## Accepted full P100 artifact

- path: `results/pr122_p100/panel_stage_b_gpu_validation_2701aa9f.json`
- measurement SHA: `2701aa9feb3796c33c94e6480fcb78c80c6a809c`
- artifact repository commit: `0d0d654d825cea872672f27d02107a58048b345f`
- Git blob: `fa3a253e6d882a4e69be29e7e3b1dce7b223b9a9`
- schema version: 2
- working tree clean: true
- top-level status: success
- GPU: Tesla P100-SXM2-16GB
- Python: 3.9.16
- NumPy: 1.24.2
- SciPy: 1.10.1
- Torch: 2.0.0
- timing collected: false

For both CuPy and Torch, all 17 estimator cases passed with the requested backend actually executed and no CPU fallback.

## Hausman branch coverage

Each backend passed five Hausman diagnostics:

1. `hausman_balanced` — structured `applicable=false` / non-PSD covariance difference;
2. `hausman_explicit_re_constant_balanced` — structured `applicable=false`;
3. `hausman_unbalanced` — structured `applicable=false`;
4. `hausman_explicit_re_constant_unbalanced` — structured `applicable=false`;
5. `hausman_applicable_nonzero_effect` — **`applicable=true`**, df=1.

For the dedicated applicable fixture, CuPy recorded statistic `1.1965942530851057` and p-value `0.27400344142676447`; Torch recorded statistic `1.1965942530849238` and p-value `0.2740034414268009`; the NumPy reference is statistic `1.196594253085033`, p-value `0.274003441426779`, df=1. Maximum statistic differences are `7.26e-14` (CuPy) and `1.09e-13` (Torch), and all FE/RE coefficient/covariance differences are below `1e-12`.

The fixture uses seed 20260810, 12 entities, 4 observations per entity, one slope, entity-effect scale 0.005, and noise scale 0.1. Its NumPy FE-minus-RE diagnostic covariance margin is `4.6413153162319366e-05`, safely above the hosted stability guard.

## Supplementary disconnected-FE evidence

The focused disconnected two-way FE artifact remains valid supplementary evidence for an unchanged numerical path:

- path: `results/pr122_p100/panel_stage_b_disconnected_fe_gpu_validation_a57efcea.json`
- measurement SHA: `a57efcea29b0e87ecb89865c5a6902d5773812c6`
- artifact repository commit: `72b3279d2028e8ec2af30e138e123aceb611ae8c`
- Git blob: `3bda0b2040479ba8201e2722eb990ba086c3f3b9`
- legacy residual df: 0
- component-aware/public residual df: 1
- effect rank: 7
- incidence components: 3
- CuPy/Torch confidence intervals agree with NumPy at floating-point noise.

It is deliberately identified as historical supplementary evidence rather than relabeled as a `2701aa9feb3796c33c94e6480fcb78c80c6a809c` measurement.

## Promoted canonical evidence

- canonical path: `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260809_2701aa9f.json`
- canonical SHA-256: `2056f836bfe2a708b3131becca42dbba15e519762c8bafb582000b19f81120bf`
- source id: `panel-stage-b-pr122-20260809-2056f836bfe2`
- validation rows: 44 = 17 estimator cases x 2 backends + 5 Hausman diagnostics x 2 backends
- timing/speedup: absent by contract

The parser requires the dedicated applicable row to preserve finite nonnegative statistic, p-value in [0,1], and positive df; missing or invalid numeric evidence fails closed. The older 42-row canonical source remains in the repository as explicitly superseded historical audit evidence and is no longer the registered source.

## Physical conclusion

The P2 finding “Require an applicable Hausman case in the GPU gate” is physically closed. Both physical backends now exercise the applicable statistic/p-value/df path and the structured-inapplicable path without fallback.

**Physical gate: ACCEPTED.**
