# PR #126 fifth independent review-fix addendum — 2026-08-20

Review standard: `.claude/skills/code-review.md`.

This pass independently re-reviewed the current PR rather than relying on the prior
`REVIEW CLEAN` record. It found one inference-precision bug plus one validation
coverage gap introduced by the necessary fix. Both are closed in source/hosted
contracts below; fresh exact-head CuPy/Torch CUDA execution remains the physical
acceptance gate.

## Fifth-pass findings

1. **HIGH — exact Student-t(2) tails could collapse to zero while still
   representable in float64.**

   The df=2 exact identity was already rationalized to avoid subtractive
   cancellation, but it evaluated
   `2 / {root * (root + x)}` with `root = sqrt(x*x + 2)`. For an extreme but finite
   statistic such as `x=1e154`, the true two-sided tail is approximately `1e-308`
   and remains representable, while the denominator product overflows and the
   implementation returns zero. Squaring `x` also becomes a separate overflow risk
   at larger finite statistics.

   The fix keeps the same exact statistical identity while changing only its
   numerical evaluation: `root = hypot(x, sqrt(2))` avoids the unnecessary square,
   and `(2 / root) / (root + x)` avoids materializing the overflowing denominator
   product. A maintained Torch-2.0 regression now covers both the prior `1e10`
   cancellation case and the `1e154` denominator-overflow case. A NumPy contract
   independently checks that the subnormal tail remains positive and matches the
   stable oracle.

2. **MEDIUM — the existing physical GPU runner did not specifically exercise the
   newly fixed extreme df=2 tail.**

   General Fama-MacBeth nonrobust inference was already present in the physical
   runner, but its ordinary-scale fixture could not detect a backend-specific
   regression in the new extreme-tail evaluation. A focused exact-head runner,
   `dev/benchmarks/validate_fama_macbeth_t2_tail_gpu.py`, now requires both CuPy
   CUDA and Torch CUDA, checks backend-native p-value/critical-value residency, and
   verifies the representable `1e-308` tail at `|t|=1e154`. Its hosted contract is
   `dev/tests/test_fama_macbeth_t2_tail_gpu_runner_contract.py`. The maintained
   Stage-C Torch-2.0 workflow routes both new files and executes the contract test.

## Validation evidence obtained during this pass

Before the focused physical-runner additions, exact source head
`e595618a47db7a1e80d68491cf5d4757d7c69e22` completed the maintained
`Panel Stage C Torch CPU` workflow successfully: **155 passed, 12 skipped** on
Torch `2.0.1+cpu`, including the new `1e154` regression in
`test_fama_macbeth_inference_matrix.py`. Maintenance compatibility, release notes,
and release package validation were also green at that head.

The subsequent runner/contract/workflow commits intentionally change the exact
source head. Therefore all final hosted evidence must be taken from the final head
that contains this addendum; the earlier successful runs are regression evidence,
not final-head acceptance.

## Current review exit

- actionable CRITICAL: **0 open**
- actionable HIGH: **0 open**
- relevant locally actionable MEDIUM: **0 open**
- merge performed: **no**
- remaining hard exit: **final exact-head hosted CI + fresh CuPy/Torch CUDA physical GPU**

The fresh physical execution must include the existing PR126 Stage-C/Fama-MacBeth
GPU gates **and** `validate_fama_macbeth_t2_tail_gpu.py` on the final exact SHA.
Historical GPU artifacts do not satisfy current-head acceptance.
