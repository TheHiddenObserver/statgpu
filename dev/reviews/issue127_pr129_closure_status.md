# Issue #127 / PR #129 closure status

Recorded: 2026-08-28

This checkpoint is part of the `.claude/skills/code-review.md` review-fix closure for draft PR #129.

## Reviewed source before this checkpoint

- reviewed head: `dc0f63f7eec1b2540e22e125b3f3c795d1e61dac`;
- baseline: `master` at `84f8bc7e17f66466b3a325cbb007b6cb41843821`;
- PR remains draft until all acceptance gates are satisfied.

## Findings closed in the final device/backend rounds

- `LinearRegression` CuPy/Torch critical-value evaluation now selects the executed numerical backend explicitly;
- `LinearRegression` records concrete executed device provenance (`cuda:N`) rather than an ambient `cuda` alias;
- Torch inference derives helper/scalar placement from the concrete fitted `X.device`;
- CuPy response, weight, intercept, critical-value, AIC/BIC, and F-statistic helper placement is bound to the executed `X` device;
- CuPy Cholesky recovery now falls back to pseudoinverse only for recognized rank/definiteness failures; unrelated runtime failures propagate;
- physical validator schema 5 covers `LinearRegression` nonrobust/HC3/HAC on CuPy and Torch, concrete-device provenance, and a CuPy non-rank failure negative control;
- focused Gaussian CI tracks and statically checks `statgpu/backends/_gpu_inference_cupy.py`;
- shared Gaussian/PGLM CuPy host-to-device conversion, intercept construction, and Ridge penalty allocations now use the reference `X.device`; concrete-device provenance fails closed instead of degrading to an ambiguous `cuda` label.

## Acceptance state at checkpoint creation

The source review above did not identify another open CRITICAL/HIGH/actionable MEDIUM finding in the reviewed device/fail-closed surface. This checkpoint commit intentionally re-triggers PR workflows with a normal repository actor because the immediately preceding production commit was authored by `github-actions[bot]` and its pull-request workflow runs were marked `action_required` before jobs executed.

Do not claim `COMPLETE` from this checkpoint. Required remaining gates are:

1. hosted PR workflows must pass on the exact post-checkpoint head;
2. perform one final complete-diff review on that exact source/validator contract;
3. freeze the physical validator contract;
4. run exact clean-head CuPy + Torch CUDA acceptance and persist canonical evidence.

If gates 1-3 close with no new CRITICAL/HIGH/actionable MEDIUM finding, the correct hard-exit state is `PARTIAL_REMOTE_PENDING` until gate 4 passes.
