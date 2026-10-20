# Repair the five benchmark failures / blockers

Baseline: efbfb80570eb1632e0605d68d410715dac4d9be8, with the local benchmark coverage overlay. User authorized repairs on 2026-10-20; no commit, push, new release, or package publication is authorized by this task.

## Scope and contracts frozen before implementation

Existing-capability reconciliation, not new estimator capabilities. Preserve the original statistical targets, backend requests, and correctness thresholds. Do not weaken or skip benchmark cases to obtain a green aggregate.

| Area | Observed discrepancy | Intended repair and consumer scope | Active gates |
|---|---|---|---|
| F native rvs / dtype (#173) | NumPy receives dfn/dfd; shared native sampling ignores dtype | Correct NumPy F keyword names; honor requested dtype during output conversion for 15 native distributions. Preserve CPU RNG semantics and backend-native output; do not claim GPU-native RNG. Inference distribution public facade and CI/resampling consumers preserved. | API, dtype/device, distribution numerical sanity, unavailable-backend behavior |
| IG / NB HC0 | Public HC sandwich promises observed bread; loss.hessian returns solver Fisher curvature | Introduce inference-only observed curvature; preserve solver Fisher Hessian and nonrobust information convention. Preserve normalized analytic weights, squared-weight meat, intercept, smooth penalty curvature and fixed-tuning target. Affect HC0/HC1 direct GLM, penalized analytic/oracle inference, formula and selected-CV refits; do not enable unsupported HC2/HC3/HAC non-Gaussian combinations. | independent Hessian derivatives, sandwich reference, weighted/penalized consumers, CV/refit, backend parity, method metadata |
| Cyclic spline (#216) | Central differences across clamped exterior form incorrect periodic constraints | Exact one-sided endpoint value/derivative constraint rows, normalized coordinates, backend-native null-space projection. Enforce intended periodic value/first/second derivative conditions and intended basis dimension, preserve constant subspace and actual input device. | analytic constraints, scaling/translation, rank/shape, Torch CPU, conditional physical GPU, existing consumers |
| UMAP NNDescent (#191) | NumPy 2 ndarray.device mistaken for Torch tensor | Dispatch by actual array-library identity through all affected fuzzy-graph/RNG branches. Preserve CPU operation without Torch and explicit accelerator requests. | NumPy 1/2 behavior, exact/NNDescent, RNG determinism, no silent fallback, backend-specific paths |

The HC inference change and cyclic constraints receive an independent pre-implementation design review. Worker plans and final consumer tests are part of the review evidence; no statistical convention is changed merely to match same-named external output.

## Validation order

1. Independently reproduce each failure from the unmodified numerical baseline.
2. Add targeted regressions and implement minimal shared-source fixes.
3. Run nearby existing tests; convert stale strict-xfail or known-failure assertions only after their unchanged numerical oracle succeeds.
4. Run all seven benchmark suites through the unchanged dispatcher and inspect all 316 case results, not merely process exit or harness tests.
5. Reconcile active docs/installed help and the manifest. Prior v2 failed artifacts remain historical; do not rewrite them.
6. Independent full-diff review/fix loop, followed by exact-final-source tests and fingerprint checks.

No physical GPU is available in this task environment. Implement and test applicable backend paths with genuine CPU execution / availability-aware tests, but report physical CuPy/Torch CUDA as pending. Do not repurpose the db525d0 release P100 acceptance for these new numerical changes.

## Release boundary

These changes are local and unreleased. Published statgpu 0.3.0 retains its historical behavior until a separately authorized release incorporates the fixes. Package version remains 0.3.0; no silent version bump or retroactive release-note rewrite.


## Local repair outcome (2026-10-20)

The unchanged numerical benchmark oracles now pass all316 CPU smoke cases (seven suites). The previous three failures and two preemptive/known-defect blocks are no longer present. The dispatcher completes with exit0 and source_changed=false; the final distributed artifact records exact source and manifest fingerprints.

The final targeted runner suite passes424 tests (14 known deprecation warnings). Independent production regressions pass393 tests, with284 unavailable CuPy/TorchCUDA skips; real TorchCPU executes successfully. Separate documentation and existing neighboring suites are recorded in the repair verification artifact. No physicalGPU acceptance is claimed.

The wider PR168 regression selection contains an unrelated strictXPASS for the existing Nystroem polynomial-overflow issue#236. Running that exact case on a separate clean efbfb805 checkout with the same environment reproduces the sameXPASS. This task does not change that test or claim the entire repository suite is green.

Prior v2 artifacts remain immutable historical failure evidence. The numerical fixes are local/unreleased; commit,push,PRcreation and publication were not performed. Final workflow status is PARTIAL_REMOTE_PENDING for physicalCUDA acceptance, not a published0.3.0 fix.

## Branch publication authorization — 2026-10-20

The earlier no-commit/no-push statements above describe the completed local repair stage. The user subsequently authorized publishing the reviewed repairs and full benchmark coverage on an independent branch. This authorization does not include PR creation, merge, tags, release, package publication, or issue closure. Physical P100/CUDA acceptance remains pending; the CPU validation results above are unchanged.
