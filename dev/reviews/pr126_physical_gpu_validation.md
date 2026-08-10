# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PARTIAL_REMOTE_PENDING**.

The previously accepted Tesla P100 evidence was measured at exact clean source SHA `c151550ab17bd9533a51599f86b6a4ea12a292e9` and recorded by repository commit `a8dc976208d3997140829491be7d3dbfddfcedd6`. That evidence remains immutable audit history, but it is **not applicable to the current repaired candidate** because post-acceptance review changed both the numerical covariance implementation and the physical-validation runners.

In particular, the repaired branch now:

- derives Stage-C covariance breads and observation influence rows from `X+` instead of normal equations, including full-rank ill-conditioned designs;
- computes HC2/HC3 leverage from `diag(X X+)`;
- fails closed on materially negative covariance diagonals before BSE/t/z storage;
- removes the CuPy numerical host fallback in panel group scatter-add;
- persists the backend selected at the numerical fit boundary instead of reconstructing backend identity from the requested device;
- extends physical parity coverage to full-rank ill-conditioned HC0/HC2/HC3 and Driscoll-Kraay cases.

These are physical-applicability changes under `RELEASING.md`, so a fresh clean CuPy/Torch GPU run is mandatory before this PR can return to `PHYSICAL_GPU_ACCEPTED` or `COMPLETE`.

## Historical correctness/backend-provenance artifact

The following artifact is retained as historical evidence only:

- path: `results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json`
- measurement SHA: `c151550ab17bd9533a51599f86b6a4ea12a292e9`
- artifact repository commit: `a8dc976208d3997140829491be7d3dbfddfcedd6`
- Git blob: `ecaf21a96ee60aea5a446e1cc40a17ba5c2a205f`
- SHA-256: `91e9ec53b6ee9516064797fcce0a23ba6a198037b639fd9965135bae3b3c7b63`
- GPU: Tesla P100-SXM2-16GB
- result at that historical source: CuPy 26/26 estimator cases + 2/2 direct primitives; Torch 26/26 estimator cases + 2/2 direct primitives.

The artifact validated the implementation that existed at `c151550a...`; it must not be cited as evidence for the repaired pseudoinverse/provenance implementation.

## Historical synchronized-performance artifact

The following timing artifact is likewise historical until rerun on the repaired source:

- path: `results/pr126_p100/panel_stage_c_performance_c151550a.json`
- measurement SHA: `c151550ab17bd9533a51599f86b6a4ea12a292e9`
- artifact repository commit: `a8dc976208d3997140829491be7d3dbfddfcedd6`
- Git blob: `e6495802d47f9db01ce1b379cc13845c5aaee09e`
- SHA-256: `1861fc4f1817789484d4b2fb02c205c302199b7b4ce167d75d409d8c54d81931`
- rows: 58 = 54 base rows + 4 high-T QS rows.

It remains useful for audit/history and makes no speedup claim, but the performance runner now also validates persisted executed-backend provenance. A new exact-source run is required before timing evidence can be promoted again.

## Earlier historical artifacts

The older `9c0b3050dd143c43a06bb6393d69f4f83e861637` measurement remains historical as before:

- `results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json` — SHA-256 `a0d258f6d6b8243e82684a29305606e5f6bd91bbe271c3ed335b32b5ec973665`;
- `results/pr126_p100/panel_stage_c_performance_9c0b3050.json` — SHA-256 `214284f02a5e21e775e58deaf2fa3cc9b6384d392b96c6f300f31f4a02953b1c`.

Neither old measurement is current acceptance evidence.

## Required fresh physical closure

On the final repaired source SHA, run the maintained physical correctness validator and synchronized performance runner from a **clean tree** on both CuPy CUDA and Torch CUDA. The correctness result must include:

- all maintained estimator covariance cases;
- direct public covariance primitives;
- the new full-rank ill-conditioned HC0/HC2/HC3 and Driscoll-Kraay parity cases;
- requested backend equal to the backend persisted by the actual fit;
- no numerical CPU fallback.

Only exact-head fresh artifacts that satisfy the repository source/manifest contracts may be promoted to current canonical evidence. Until then the lifecycle hard exit is **PARTIAL_REMOTE_PENDING**.
