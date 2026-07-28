# PR #80 Review-Fix Cycle Addendum — 2026-07-28

This addendum supersedes the `Current ... SHA-256` metadata and the unconditional
`COMPLETE` status at the top of `dev/reviews/pr80_review_fix.md` for source
changes made after its recorded boundary/workspace artifact.

## Reviewed source

- Latest user-updated head reviewed: `c967e6f08b976f7f1df0df63ec58efda528df438`
- Final production/test head from the remote cycle: `86755ce6fedc65370e09dae56031bce8eee44df7`
- Exact-source evidence runner head: `b42ab0ade37e9fc7c5abf159089da195220680df`
- Base used for this incremental review: `f59815b440ed385275fc4ad75530663bb1fa89e3`

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

## Physical-GPU evidence refresh

- [MEDIUM][ARTIFACT][fixed]
  `results/benchmark_frontend_sources/coxph_boundary_workspace_pr80_20260728_refresh.json` -
  prior P100 evidence predated the conservative wide-model workspace estimate
  and constructor boundary wrapper.
  Impact: the row-streaming mathematics was already covered, but the final
  source hashes and the newly active wide-model routing branch were not
  independently auditable on physical CUDA.
  Fix: schema-v2 evidence now checks constructor truthy-string rejection and a
  deterministic `n=4096`, `p=128` Efron case under an 8 MiB workspace limit.
  The recorded pre-fix estimate is 1,056,768 bytes and selects dense; the
  corrected estimate is 9,445,376 bytes and selects streaming. Both CuPy and
  Torch recorded exactly one streaming call and matched NumPy with maximum
  absolute difference `3.997e-15`.
  Evidence: clean detached P100 source commit
  `b42ab0ade37e9fc7c5abf159089da195220680df`, `gate_failures=[]`, plus **104
  passed** physical-GPU boundary/workspace tests. Artifact SHA-256 is
  `ec874ad3059b2044a9b12403763847fa9a05d254a24f89bec2763353258c2bea`;
  `_risk_sets.py`, `_cox_fit_adapter.py`, and the runner hashes are respectively
  `08a9f9c5f447d139cb143d8d715638f6e3db742ae2ba6485544a3e26e7fd657d`,
  `8d34ab12ae5f136249cc597463868ae6af35968c7fad5896afe49dfccf1b3134`,
  and `312250ba5b489d8b24ca8de8d4e2193c074b9b05f735d6c19391f382e753b9ea`.

Exit status: **COMPLETE**. No unresolved CRITICAL/HIGH finding remains, and the
targeted physical-GPU exact-source evidence gap is closed.

PR merge and release remain outside this review-fix cycle.
