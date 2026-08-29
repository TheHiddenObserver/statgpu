from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "CHANGELOG.md",
    "- Validation remains intentionally incomplete while PR #129 is draft: hosted PR gates and exact clean-head CuPy/Torch CUDA acceptance must pass before #127 can be marked `COMPLETE`. No GPU speedup claim is made.\n",
    "- Validation remains intentionally incomplete: all hosted gates and the fresh complete-diff review are green on the reviewed implementation head, but final acceptance still requires exact clean-head CuPy/Torch CUDA validation on the final source SHA. PR #129 remains open/unmerged and #127 is not yet `COMPLETE`. No GPU speedup claim is made.\n",
)

replace_once(
    "docs/en/changelog.md",
    "> Last updated: 2026-08-28<br>\n",
    "> Last updated: 2026-08-30<br>\n",
)
replace_once(
    "docs/en/changelog.md",
    "- PR #129 remains draft until hosted gates and exact clean-head CuPy/Torch CUDA acceptance pass. No GPU speedup claim is made, and #127 is not yet `COMPLETE`.\n",
    "- All hosted gates and the fresh complete-diff review are green on the reviewed implementation head. Final acceptance still requires exact clean-head CuPy/Torch CUDA validation on the final source SHA; PR #129 remains open/unmerged and #127 is not yet `COMPLETE`. No GPU speedup claim is made.\n",
)

replace_once(
    "docs/cn/changelog.md",
    "> 最后更新：2026-08-28<br>\n",
    "> 最后更新：2026-08-30<br>\n",
)
replace_once(
    "docs/cn/changelog.md",
    "- PR #129 仍保持 draft；只有 hosted gates 与 exact clean-head CuPy/Torch CUDA acceptance 都通过后，#127 才能标记为 `COMPLETE`。本变更不做 GPU speedup 声明。\n",
    "- reviewed implementation head 的 hosted gates 与 fresh complete-diff review 已全部通过；最终 acceptance 仍要求在 final source SHA 上执行 exact clean-head CuPy/Torch CUDA validation。PR #129 当前保持 open/unmerged，#127 尚不能标记为 `COMPLETE`。本变更不做 GPU speedup 声明。\n",
)

plan = Path("dev/plans/gaussian_inference_backend_native_plan.md")
text = plan.read_text()
start = text.index("## 19. Production implementation status — 2026-08-28")
new_section = '''## 19. Production implementation status — 2026-08-30

The production implementation is present on `agent/post-v0.2.5-next-phase-plan` and is being validated through open PR #129.

Implemented and review-fixed before final acceptance:

- backend-native Gaussian working state and covariance/inference routing for NumPy/CuPy/Torch;
- final reporting conversion only after numerical inference;
- Gaussian/L2 public consumer routing, including `LinearRegression`, `Ridge`, bounded penalized squared-error L2 paths, and `RidgeCV` final refit;
- fail-closed executed-backend and concrete-device provenance instead of an implicit NumPy/default-device fallback;
- maintained shared normal/Student-t reference-distribution routing, including stable df=1/df=2 extreme-tail formulas;
- Ridge/L2 `n_eff * alpha` inference mapping for ordinary and weighted fits;
- concrete-device CuPy alignment for response/weight arrays and scoped solver-status rank recovery;
- native fitted-parameter lifetime through GPU Gaussian inference, followed only then by the established NumPy reporting snapshot;
- success/failure cleanup covering both post-fit and exact/precomputed GPU inference, with failed refits invalidated rather than advertising stale fitted state;
- representative hosted coverage for covariance families, weighted/robust, rank-deficient, multi-target, formula, statsmodels alignment, Torch float32, non-L2 delegation, no-host-transfer behavior, cleanup/rollback, and cross-device placement;
- a focused PR CI workflow;
- exact-SHA physical validators covering CuPy/Torch backend/device provenance, clean-tree state, representative covariance/Ridge/weighted/rank/multi-target/small-df cases, numerical error fields, native-fit/reporting-boundary checks, and `RidgeCV` final-refit inference;
- synchronized root/English/Chinese unreleased changelog entries and user-facing inference/device documentation.

The reviewed implementation head is `0adcd63afef6e38ee757a6635b30839d2edfbab2`. On that exact head all seven triggered hosted workflows passed: `Tests`, `Gaussian inference backend-native`, `Release package validation`, `Release notes validation`, `Maintenance compatibility`, `Benchmark Frontend CI`, and `Panel Stage C Torch CPU`. A fresh Codex complete-diff review of that same head reported no major issues, and all inline review threads were resolved.

Historical physical artifacts remain superseded once later source or validator-acceptance semantics changed. In particular, the earlier successful `0c05d17cb7f7520fb703566b1e662d18bf2d4ee0` physical run is diagnostic/historical evidence only because subsequent review findings required source and host-transfer-validator changes.

This finalization step is documentation/evidence-only: it must not alter production numerical code, validator cases, thresholds, provenance checks, or pass/fail semantics. The resulting final source SHA must receive a fresh hosted run, and exact clean-head CuPy plus Torch CUDA acceptance must then be executed against that same SHA.

Until the final exact-head physical CUDA evidence passes, #127 is not `COMPLETE`. Once the docs-only finalization head is hosted-green and no new review finding exists, with physical CUDA as the sole missing gate, the correct hard-exit status is `PARTIAL_REMOTE_PENDING`.
'''
plan.write_text(text[:start] + new_section)

closure = Path("dev/reviews/issue127_pr129_closure_status.md")
closure.write_text('''# Issue #127 / PR #129 closure status

Recorded: 2026-08-30

This checkpoint records the current `.claude/skills/code-review.md` closure state after repeated hosted and physical review/fix cycles for PR #129.

## Superseded physical candidates

The following candidates are historical/diagnostic evidence only and must not be promoted as final acceptance evidence:

1. `3180add336b41017f4cd5a5e6721a6470a797360` — physical execution exposed unsupported CuPy triangular-solve usage and incomplete concrete-device provenance.
2. `11343d87e810cb147bd0b9f821ac5f5e52976ea2` — physical execution exposed silent invalid CuPy Cholesky results on singular normal equations.
3. `7375b36117988e182fb74443a937e6b302eabb99` — physical execution exposed the remaining bare shared CuPy inverse site that could return invalid values without triggering pseudoinverse recovery.
4. `0c05d17cb7f7520fb703566b1e662d18bf2d4ee0` — both canonical Tesla P100 validators passed on that exact source, but a later fresh code review found additional backend-locality/cleanup issues and the physical host-transfer acceptance logic subsequently changed. Its artifacts therefore no longer prove the current acceptance contract.

Later source changes addressed concrete-device CuPy response/weight alignment, native fitted-parameter lifetime, failed-refit invalidation, success/failure allocator cleanup including exact/precomputed inference, and the physical host-transfer fixture. Any earlier artifact predating those changes remains superseded.

## Current review/fix closure

Reviewed implementation head: `0adcd63afef6e38ee757a6635b30839d2edfbab2`.

Closed findings include:

- supported `cupyx.scipy.linalg.solve_triangular` usage for CuPy `LinearRegression`;
- scoped `cupyx.errstate(linalg="raise")` at maintained CuPy Cholesky/solve/inverse boundaries, preserving classified pseudoinverse recovery while unrelated failures remain fatal;
- shared `_cupy_asarray_on_device(...)` relocation when native CuPy values live on a different concrete GPU;
- authoritative X-device alignment for `LinearRegression` response and sample weights;
- fail-closed executed-backend and concrete-device provenance;
- backend-native fitted parameters retained through ordinary GPU squared-error L2 inference, with reporting materialized only afterward;
- final GPU cleanup after post-fit inference and cleanup on backend-fit/exact-precomputed inference failure;
- native fitted references cleared and `_fitted=False` on failed fit/inference transactions;
- a physical host-transfer guard that verifies native fitted state at the inference boundary without rejecting unrelated solver warm-start host work;
- hosted behavioral/static coverage for each repaired boundary.

On `0adcd63afef6e38ee757a6635b30839d2edfbab2`, all seven triggered hosted PR workflows completed successfully:

1. `Tests` — complete CPU test tree, Python 3.9-3.12 regression matrix, static/docs contracts, Torch CPU regressions, and external panel alignment;
2. `Gaussian inference backend-native` — Python 3.9/3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`;
7. `Panel Stage C Torch CPU`.

A fresh Codex review explicitly reviewed commit `0adcd63afe` and reported: `Didn't find any major issues.` All inline review threads are resolved. The complete-diff review state is therefore:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

## Finalization and hard-exit state

The commit containing this checkpoint is documentation/evidence-only. It must not modify production numerical code, physical-validator cases, thresholds, backend/device checks, provenance checks, or pass/fail semantics. Because canonical physical evidence is exact-SHA evidence, the resulting final source SHA still requires both a fresh hosted run and new exact clean-head physical CUDA execution even though the reviewed numerical implementation is unchanged.

After the finalization head is hosted-green and no new review finding exists, the only unresolved acceptance gate is exact clean-head physical CUDA execution on that same final `HEAD`. At that point the correct workflow state is **`PARTIAL_REMOTE_PENDING`**, not `COMPLETE`.

## Canonical physical acceptance

Run both validators from the same exact clean final checkout. `results/` is git-ignored, so the first artifact does not invalidate the second validator's clean-tree precondition.

```bash
SHA="$(git rev-parse HEAD)"

python dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py \\
  --out "results/pr129_gaussian_inference_${SHA:0:8}.json" \\
  --expected-sha "$SHA" \\
  --validation-tier remote-full \\
  --backends cupy,torch

python dev/benchmarks/validate_gaussian_inference_cupy_rank_recovery_gpu.py \\
  --out "results/pr129_gaussian_inference_cupy_rank_${SHA:0:8}.json" \\
  --expected-sha "$SHA" \\
  --validation-tier remote-full
```

Acceptance requires both commands to succeed on the same exact final candidate SHA. The full artifact must report `status: success`, `validation_tier: remote-full`, both required CUDA backends, concrete CUDA-device provenance, clean-tree acceptance, and successful native-fit/reporting-boundary checks. The focused artifact must report `status: success`, the same exact SHA, `validation_tier: remote-full`, concrete CuPy/CUDA provenance, and success for all maintained rank-recovery cases.

PR #129 remains open/unmerged and Issue #127 remains open until both exact-head physical validators pass and the resulting evidence is reviewed against this frozen acceptance contract. No GPU speedup claim is made.
''')

for path in (
    "CHANGELOG.md",
    "docs/en/changelog.md",
    "docs/cn/changelog.md",
    "dev/plans/gaussian_inference_backend_native_plan.md",
    "dev/reviews/issue127_pr129_closure_status.md",
):
    p = Path(path)
    p.write_text(p.read_text().rstrip() + "\n")

print("PR129 evidence docs finalized")
