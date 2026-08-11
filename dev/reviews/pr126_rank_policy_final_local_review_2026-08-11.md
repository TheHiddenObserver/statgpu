# PR #126 final local rank-policy review checkpoint

Status: **PARTIAL_REMOTE_PENDING / LOCAL REVIEW CLEAN / NOT MERGE-READY**

Validation tier: `local-full`. Fresh `remote-full` physical CUDA correctness and synchronized performance evidence remain required.

## Active review axes

- Public API
- Inference
- Backend
- Formula/data alignment
- Benchmark/performance
- Docs/artifacts

Loss, penalty, solver framework, and CV remain inactive: Stage C changes panel covariance/inference, metadata ordering, and the numerical linear-algebra policy of the touched panel fit spaces; it does not add a loss, penalty path, optimizer family, or tunable CV capability.

Capability decisions remain those in `dev/plans/panel_p1_stage_c_covariance_plan.md`: all touched panel families are three-backend and inference/formula supported; PooledOLS, PanelOLS, and RandomEffects require physical benchmark evidence; BetweenOLS, FirstDifferenceOLS, and FamaMacBeth are not-performance-sensitive for Stage C; CV is non-tunable/not applicable.

## Findings closed in this auto-fix/re-review cycle

- `[CRITICAL][INFER][fixed] statgpu/panel/_covariance.py + statgpu/panel/_linalg.py` — pseudoinverse and numerical rank no longer use independent backend defaults. One explicit float64 SVD cutoff, `max(n,k) * eps * s_max`, defines the retained covariance subspace and reported numerical rank on NumPy/CuPy/Torch.
- `[CRITICAL][FORMULA][fixed] statgpu/panel/_first_diff.py` — FirstDifferenceOLS preserves pandas ordered-categorical chronology via `factorize_panel_metadata` before differencing; lexical label order can no longer change the differenced sample and coefficients.
- `[CRITICAL][INFER/BUG][fixed] touched residual-OLS fit spaces` — PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS, and RandomEffects no longer allow a numerical-rank-deficient design to be fitted in one effective subspace and inferred in another. Rank-deficient fits use the same shared minimum-norm SVD policy as covariance.
- `[MEDIUM][API/MAINT][fixed] touched estimator/diagnostic full-rank paths` — the rank fix does not replace normal full-column-rank behavior with an all-SVD implementation. Each estimator retains its historical full-rank coefficient solver and full-rank diagnostic solve; the shared SVD minimum-norm path is entered only for explicit `rank < k` or a historical solver linear-algebra failure.
- `[HIGH][MATRIX][fixed locally / needs remote GPU]` — maintained coverage now contains numerical-cutoff-boundary primitives, a PanelOLS numerical-boundary integration, and an exact-rank-deficient NumPy/Torch estimator matrix spanning PooledOLS, PanelOLS, BetweenOLS, FirstDifferenceOLS, and RandomEffects.
- `[MEDIUM][TEST][fixed] dev/tests/test_panel_stage_c_external.py` — the independent stable-HC reference implements the reviewed explicit SVD cutoff rather than reintroducing NumPy's default `pinv` threshold.
- `[LOW][ARTIFACT][fixed] dev/benchmarks/validate_panel_stage_c_gpu.py` — CuPy provenance checks `cupy`, `cupy-cuda11x`, and `cupy-cuda12x` package names.
- `[MEDIUM][DOC/ARTIFACT][fixed]` — plan/docs/changelogs/physical-review state consistently treat the old `ec511f53...` P100 measurement as immutable historical evidence, and the physical record now attributes supersession to the complete shared fit-space/chronology production change rather than an incomplete file list.

## Reviewed numerical policy

For a fit-space design `X`, rank is determined by a backend-native singular-value calculation with cutoff

```text
max(n, k) * eps(float64) * s_max.
```

The covariance pseudoinverse uses the exact same retained singular-value mask. For coefficient/diagnostic solves:

- if the explicit rank is below the column count, use the shared SVD minimum-norm solve;
- if the design is full column rank, preserve the estimator's historical solver;
- if that historical full-rank solver raises a linear-algebra failure, fail over to the shared SVD solve;
- reuse the fit-space rank in downstream covariance/diagnostic metadata where applicable.

This preserves the Stage-C backward-compatibility contract on ordinary full-rank fits while making the declared numerical-rank extension backend-consistent.

## Local validation completed

Focused auto-fix/re-review suites successively passed 56, 34, 53, 109, 111, and 112-test checkpoints as the issues were discovered and closed.

On technical head `e6aa802b2159803d544c4466e6dc0099dce9bb7c`, all seven permanent hosted workflows completed successfully:

1. Tests
2. Panel Stage C Torch CPU
3. Panel Stage C external covariance
4. Maintenance compatibility
5. Release notes validation
6. Release package validation
7. Benchmark Frontend CI

The maintained Torch 2.0.1 CPU job explicitly executed both `test_panel_stage_c_torch_cpu.py` and `test_panel_stage_c_rank_deficient_matrix.py`: **31/31 passed**. The external workflow executed both Python pinned-reference alignment and the R `plm` / `sandwich` alignment successfully.

The subsequent `79810b074b2895c28f0f85ae9e2c1551bc4d2ec1` change is artifact/plan wording only and does not alter production source, tests, physical runners, or benchmark code.

## Remote-full boundary

The old exact-clean P100 source `ec511f539adeaaedf310f92248200d0868577532` predates the repaired production implementation and remains historical evidence only. Its existing canonical source IDs are immutable and must not be overwritten.

Fresh exact-head physical correctness must execute the maintained expanded matrix on both GPU backends:

- CuPy: **39/39** = 27 estimator cases + 12 direct public primitives;
- Torch: **39/39** = 27 estimator cases + 12 direct public primitives;
- requested backend must equal persisted/executed backend for every estimator case;
- every public primitive result must remain backend-native before snapshot conversion;
- the numerical-rank-boundary nonrobust/HC0/HC2/HC3/cluster/DK primitive cases and `panel_rank_boundary_dk` estimator case must pass.

Because the shared fit-space numerical policy changed, synchronized performance evidence must also be rerun using the maintained Stage-C performance runner. The existing 58-row P100 result is historical and may be used for comparison, not current acceptance.

Fresh physical artifacts must be registered as new immutable benchmark sources/parser identities rather than replacing the historical `ec511...` canonical sources.

## Current review verdict

The latest local `.claude/skills/code-review.md` re-review finds no unresolved CRITICAL, HIGH, or relevant MEDIUM issue in the current source/matrix/docs state. The remaining hard gate is remote physical CUDA correctness plus synchronized performance, so the correct exit remains **PARTIAL_REMOTE_PENDING**.

PR #126 intentionally remains Draft, open, and unmerged.