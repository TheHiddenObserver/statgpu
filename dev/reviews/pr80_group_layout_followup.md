# PR #80 Group Lasso Layout Follow-up

> Implementation commit: `ce2d5b6f74e5f5f63456882da6126a8b0682462e`  
> Hosted validation: GitHub Actions run `#776`  
> Status: `PARTIAL_REMOTE_PENDING`

## Impact Classification

- Numerical coefficients: affected and fixed for explicit nested Group Lasso
  specifications whose members were permuted or interleaved on GPU.
- Selected alpha and final refit: affected and covered for Group Lasso CV.
- Backend placement: unchanged; NumPy, CuPy, and Torch remain the supported
  execution families.
- Public API: nested group lists remain accepted; members inside each explicit
  group are now canonicalized into ascending index order before layout metadata
  and solver routing are computed.
- Inference: unchanged; this follow-up is estimation/CV facing.
- Formula: not formula facing.
- Benchmark evidence: new exact-source physical GPU evidence is required.

## Capability Decisions

| Public family | Backend | CV | Inference | Formula | Benchmark |
|---|---|---|---|---|---|
| Squared-error Group Lasso direct fit | three-backend | n/a | unchanged | not-formula-facing | required; physical refresh pending |
| Squared-error Group Lasso CV | three-backend | supported, including selected-alpha refit | unchanged | not-formula-facing | required; physical refresh pending |
| Group SCAD / Group MCP | unchanged; use their own strict layout metadata | supported as before | unchanged | not-formula-facing | existing evidence remains scoped |

## Findings and Fixes

- [CRITICAL][BUG/BACKEND][fixed locally] The GPU block-coordinate Group Lasso
  path inferred contiguity from each equal-size group's first index. A valid
  specification such as `[[0, 3], [2, 1]]` could therefore be treated as
  contiguous even though its true blocks were interleaved, causing the Gram
  blocks, coefficient reshape, and scatter indices to refer to different
  groups. The public Group Lasso construction boundary now sorts members within
  every explicit nested group while preserving group order. Group penalties are
  invariant to this within-group permutation. After canonicalization, the
  existing first-index fast-path condition can only be true for the actual dense
  contiguous partition; all interleaved layouts retain strict non-contiguous
  metadata and use gather/scatter indices.
- [HIGH][TEST/MATRIX][fixed locally] Regression coverage now includes an
  equal-size non-contiguous layout, the misleading-first-index counterexample,
  and an unequal-size serial layout. It covers direct fit with and without an
  intercept, CPU invariance under within-group permutations, CV score/selected
  alpha/final-refit propagation, NumPy/CuPy/Torch coefficient and prediction
  parity, objective parity, public registry/direct-import identity, and pickle
  round trips.
- [MEDIUM][MAINT/API][fixed locally] The compatibility class retains the
  historical `statgpu.penalties._group_lasso.GroupLassoPenalty` module path and
  rebinds that module symbol, so registry construction, direct imports, and
  serialization resolve to one public class rather than two competing types.
- [HIGH][ARTIFACT][needs remote GPU] The prior schema-21 artifact binds
  `5bb55ede04eecb5ab7689a400e864996fb514240` and only covers standard contiguous
  group IDs. It remains valid historical evidence but cannot certify this
  implementation. `dev/benchmarks/benchmark_group_layout_gpu.py` now records a
  clean source commit and SHA-256 hashes for the solver, CV, penalty boundary,
  test, and runner files; it gates direct-fit and CV parity for all three layout
  categories on both CuPy and Torch and emits `gate_failures` machine-readably.

## Validation

GitHub Actions run `#776` passed:

- complete CPU tree: `1583 passed, 630 skipped, 11 warnings`;
- static contracts, maintained-source/script compilation, high-signal checks,
  Cox behavior checks, and complete test collection;
- documentation contracts;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices.

The CPU run executes canonicalization, public identity, pickle round-trip,
direct-fit invariance, and CV/refit invariance. CuPy/Torch layout cases skip on
the hosted CPU runner by design.

## Remaining Remote Gate

Run the following from a clean physical-GPU checkout of the exact implementation
commit and retain the generated JSON as the evidence artifact:

```bash
python dev/benchmarks/benchmark_group_layout_gpu.py \
  --output results/benchmark_frontend_sources/group_layout_contract_pr80.json
```

Promotion to `COMPLETE` requires both CuPy and Torch sections to pass all direct
and CV layout cases, `source_clean=true`, exact source hashes, and
`gate_failures=[]`. Any runtime, test, or runner change after the audited commit
requires a new exact-source run.
