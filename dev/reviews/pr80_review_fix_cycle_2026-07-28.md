# PR #80 Review-Fix Cycle Addendum — 2026-07-28

This addendum supersedes the `Current ... SHA-256` metadata and the unconditional
`COMPLETE` status at the top of `dev/reviews/pr80_review_fix.md` for source
changes made after its recorded boundary/workspace artifact.

## Reviewed source

- User-updated head reviewed: `f59815b440ed385275fc4ad75530663bb1fa89e3`
- Final production/test head from this cycle: `86755ce6fedc65370e09dae56031bce8eee44df7`
- Base used for the incremental review: `da3536605d51c2f7b72a7f03ff251ecdd2850ca2`

## Findings closed in this cycle

1. **Constructor boolean coercion.** `CoxPH` converted truthy strings such as
   `compute_cindex="False"` and `gpu_memory_cleanup="False"` with `bool(...)`,
   making them `True` before fit-time validation could reject them. Public
   `CoxPH` and `CoxPHCV` constructors now accept only actual booleans or integer
   `0`/`1` controls and reject truthy strings before constructor coercion.
2. **Wide delayed-entry workspace underestimation.** The dense Breslow/Efron
   workspace estimator counted row-scalar and `p x p` tensors but omitted the
   possible `n x p` weighted-design intermediate used by optimized three-operand
   `einsum` contraction paths. The estimate now conservatively includes two
   row-feature buffers so wide models select the row-streaming fallback before
   exceeding `STATGPU_COX_GROUP_MAX_BYTES`.
3. **Coverage gaps.** New regression gates cover constructor boundaries,
   signature preservation, the wide-model estimate, and forced row-streaming
   parity for Breslow/Efron with delayed entry, multiple failure times, multiple
   strata, score residuals, and log-likelihood-only evaluation.

## Validation

GitHub Actions run `30328570573` (run number 708) passed on the final
production/test head:

- full CPU test tree;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices;
- static/compile contracts and complete test collection;
- documentation contracts.

The temporary write-enabled patch workflow used to apply the large source-file
edit was removed before the final validation head.

## Physical-GPU evidence status

The existing Tesla P100 boundary/workspace artifact was generated from source
commit `16695feec8d4187b591d8a24d8977de543fd33c3`. It already validates the same
CuPy/Torch row-streaming mathematics and the tested `p=3` case selects streaming
both before and after the estimator correction. However, its recorded source
hash predates the conservative wide-model routing estimate and the constructor
boundary wrapper.

Therefore the status after this cycle is:

**SOURCE REVIEW AND CPU MATRIX COMPLETE; TARGETED PHYSICAL-GPU HASH REFRESH
PENDING.**

The pending GPU action is evidence refresh rather than a known numerical or
backend correctness defect. It should rerun the boundary/workspace benchmark on
CuPy and Torch, including at least one wider `p` case whose old estimate would
have selected the dense path and whose corrected estimate selects streaming.

PR merge and release remain outside this review-fix cycle.
