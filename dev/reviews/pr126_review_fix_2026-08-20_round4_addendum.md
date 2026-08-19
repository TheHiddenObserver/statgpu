# PR #126 fourth independent review-fix addendum — 2026-08-20

Review standard: `.claude/skills/code-review.md`.

This addendum supersedes the **fourth-pass wording only** in
`dev/reviews/pr126_review_fix_2026-08-19.md`, especially its earlier item 27. The
first three review passes remain historical audit context. The final fourth-pass
precision rule is narrower than the intermediate rule recorded there: a raw
nonconstant RHS equal to zero is **not** sufficient evidence of lost information.
The fallback compares the ordinary RHS against a magnitude-tiered RHS and fails
closed only when those two reductions disagree at zero.

## Final fourth-pass findings

1. **CRITICAL — nonconstant Fama-MacBeth Gram RHS could round to zero and bypass
   coefficient-resolution certification.**

   The exact period intercept had already received a stable Gram RHS, but the other
   coordinates could still be formed by ordinary BLAS. A condition-one period with
   `x=[1,-1,1,-1]` and `y=[-3,2**55,2,-2**55]` has exact sums over the supplied
   float64 values `sum(y)=sum(x*y)=-1`, so intercept and slope are both `-1/4`.
   Ordinary matrix multiplication can nevertheless form the nonconstant RHS as
   exact zero. The Gram resolution bound now treats a zero nonconstant candidate as
   uncertified when it lies inside a nonzero RHS-roundoff interval, while the exact
   first-column intercept remains separately protected by the stable response sum.

2. **HIGH — the first zero-RHS fix over-rejected genuine zero coefficients.**

   The intermediate rule `raw_rhs == 0 && absolute_summand_mass > 0` was too broad:
   a condition-one period with `x=[-1,1,-1,1]` and `y=[1,-1,-1,1]` has a genuine
   exact-zero intercept and slope even though both positive and negative summands
   are present. The final fallback therefore recomputes the original normal-equation
   RHS with the magnitude-tiered reducer. Only `raw_rhs == 0` **and**
   `stable_rhs != 0` proves that a representable cancellation tail was lost and
   triggers the coefficient-resolution sentinel. If `raw_rhs == stable_rhs == 0`,
   the coordinate is treated as a genuine-zero candidate and ordinary SVD/rank/
   stationarity certification may continue.

3. **CRITICAL — a stable SVD response projection is not, by itself, a valid rescue
   for a demonstrated lost nonconstant RHS tail.**

   On the lost-tail condition-one fixture above, compensating the `U' y` reduction
   does not restore the true `-1/4` slope because the singular vectors themselves are
   rounded. Once the raw-versus-stable RHS comparison proves that ordinary BLAS lost
   a nonzero tail, the period therefore fails closed with the existing
   `FloatingPointError` rather than publishing a finite but uncertified SVD
   coefficient. This is reported as coefficient resolution, not rank deficiency.

4. **MEDIUM — stale Fama-MacBeth solver-provenance assertion.**

   The maintained period-intercept fixture `[2**55,1,-2**55]` now retains intercept
   `1/3` directly in the Gram RHS. Its correct optimized provenance is therefore
   `gram-certified` with zero SVD fallbacks. Only the obsolete path assertion was
   changed; coefficient and slope oracles were not relaxed.

5. **MEDIUM — focused three-backend physical and hosted-runner coverage.**

   `dev/tests/test_panel_lstsq_resolution_precision.py` covers the stable intercept
   fast path, demonstrated lost nonconstant tail failure, and genuine-zero success
   on NumPy and maintained Torch CPU. The focused physical runner
   `dev/benchmarks/validate_fama_macbeth_rhs_cancellation_gpu.py` requires the same
   three outcomes on CuPy CUDA and Torch CUDA and records exact SHA, clean worktree,
   GPU/package versions, requested/executed backend, solver mode, and fallback
   provenance. `dev/tests/test_fama_macbeth_rhs_cancellation_runner_contract.py`
   locks the runner fixtures/provenance fields in hosted CI. English and Chinese
   Fama-MacBeth documentation describe the final raw-versus-stable RHS rule.

## Rejected approaches

The fourth pass explicitly rejected the following shortcuts:

- **Raw zero plus nonzero absolute summand mass as proof of cancellation loss.** It
  rejects genuine zero coefficients and is superseded by the raw-versus-stable RHS
  comparison.
- **Stable `U' y` alone as proof of a correct fallback coefficient.** Rounded SVD
  basis vectors can still contaminate a low-order coefficient even when the response
  reduction itself is compensated.
- **Magnitude-tiered computation of every Fama-MacBeth Gram-RHS coordinate on the
  ordinary fast path.** This would broaden a rare precision fix into a routine
  CPU/GPU performance cost. Only the exact intercept is always stabilized; full
  nonconstant stable RHS evaluation is restricted to already-unsafe fallback cases.
- **Relaxing the analytic coefficient oracle.** The condition-one lost-tail fixture
  remains a hard failure rather than accepting a finite rounded answer.

## Validation evidence before final synchronization

The source/test combination at `4b6157376e40b05727b8671176d701a4b0a847c8`
completed both the full `Tests` workflow and the independent `Panel Stage C Torch
CPU` workflow successfully. That includes the full CPU tree, maintained Torch 2.0,
Python 3.9/3.10/3.11/3.12 regression matrix, static/docs contracts, and current
`linearmodels` alignment. The final refined source is
`7919f0184853543b726e8d524e93d070b192abbf`; the final focused physical runner is
`ba49d6cfaed6b9071a193a86b127e2a8ae04ce47`; subsequent commits synchronize tests,
bilingual documentation, runner contract, and review records.

Final exact-head hosted CI must be rerun after this addendum. Fresh physical GPU
acceptance remains required on **both CuPy CUDA and Torch CUDA**. The focused runner
must be executed on the final exact SHA in addition to the broader Panel Stage-C /
shared least-squares physical gates. Historical P100 artifacts remain historical-only
and do not satisfy current-head acceptance.

## Current review exit

- actionable CRITICAL: **0 open**
- actionable HIGH: **0 open**
- relevant locally actionable MEDIUM: **0 open**
- merge performed: **no**
- remaining hard exit: **final exact-head hosted CI + fresh CuPy/Torch physical GPU**
