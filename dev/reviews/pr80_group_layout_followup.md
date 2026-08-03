# PR #80 Group Lasso Layout Follow-up

> Implementation commit: `6aa120a61f0f6da79f5745ba16a5bfaa7f85e137`  
> Hosted validation: GitHub Actions run `#781`  
> Status: `PARTIAL_REMOTE_PENDING`

## Impact Classification

- Numerical coefficients: affected and fixed for explicit nested Group Lasso
  specifications whose members were permuted or interleaved on GPU, including
  objects restored from legacy pickle/joblib state.
- Selected alpha and final refit: affected and covered for Group Lasso CV.
- Backend placement: unchanged; NumPy, CuPy, and Torch remain the supported
  execution families.
- Public API: nested group lists remain accepted; members inside each explicit
  group are canonicalized before layout metadata and solver routing are used.
  The public Adaptive Group Lasso class again inherits the public Group Lasso
  class.
- Serialization: affected. Legacy state no longer supplies trusted derived
  contiguity, gather/scatter, padded-index, or device-cache metadata.
- Inference: no statistical definition changed.
- Formula: not formula facing.
- Benchmark evidence: new exact-source physical GPU evidence is required.

## Capability Decisions

| Public family | Backend | CV | Inference | Formula | Benchmark |
|---|---|---|---|---|---|
| `GroupLassoPenalty` through penalized GLM direct fit | three-backend | supported | supported through the existing bootstrap path; unsupported methods fail explicitly | not-formula-facing for this layout change | remote-pending |
| `GroupLassoPenalty` through `PenalizedGLM_CV` | three-backend | supported, including selected-alpha final refit | final-estimator policy unchanged | not-formula-facing for this layout change | remote-pending |
| `AdaptiveGroupLassoPenalty` public class and LLA inner penalty | three-backend | planned as a standalone tunable family; currently used as an internal/non-registry adaptive group penalty | estimation-only in this follow-up | not-formula-facing | remote-pending |
| Group SCAD / Group MCP | three-backend | supported as before | existing estimator policy unchanged | not-formula-facing for this layout change | existing evidence remains scoped; no solver definition changed here |

## Findings and Fixes

- [CRITICAL][BUG/BACKEND][fixed locally] The GPU block-coordinate Group Lasso
  path historically inferred contiguity from each equal-size group's first
  index. A valid specification such as `[[0, 3], [2, 1]]` could therefore be
  treated as contiguous even though its true blocks were interleaved. New
  construction already canonicalized members within each group, but legacy
  pickles could restore unsorted `_group_indices` together with stale
  `_is_contiguous=True` and `_flat_indices=None`. The public compatibility class
  now implements `__setstate__` and reparses `_group_indices`, rebuilding all
  strict layout metadata instead of trusting serialized derived fields.
- [HIGH][API/MATRIX][fixed locally] Replacing only the public Group Lasso class
  had made the original Adaptive Group Lasso class a sibling of, rather than a
  subclass of, the new public class. The compatibility boundary now defines and
  rebinds both classes. The adaptive class uses a valid cooperative MRO through
  the original adaptive implementation and the canonical public Group Lasso,
  restoring `issubclass`/`isinstance`, direct-import identity, and pickle
  identity.
- [HIGH][TEST/MATRIX][fixed locally] Regression coverage now includes an
  equal-size non-contiguous layout, the misleading-first-index counterexample,
  an unequal-size serial layout, current-object pickle round trips, a simulated
  legacy object with deliberately stale layout metadata, Adaptive Group Lasso
  hierarchy/weights/pickle semantics, direct fit with and without an intercept,
  CPU permutation invariance, CV score/selected-alpha/final-refit propagation,
  and CuPy/Torch coefficient, prediction, and objective parity tests.
- [HIGH][ARTIFACT][needs remote GPU] The previous schema-21 artifact remains
  valid historical evidence but cannot certify this implementation. The
  dedicated runner is now schema 2 and gates the public class hierarchy,
  ordinary layout cases, and legacy-pickle migration on both CuPy and Torch. It
  records a clean source commit and SHA-256 hashes for the solver, CV, penalty
  boundary, test, and runner files and emits `gate_failures` machine-readably.

## Validation

GitHub Actions run `#781` passed at implementation commit
`6aa120a61f0f6da79f5745ba16a5bfaa7f85e137`:

- complete CPU tree: `1585 passed, 634 skipped, 11 warnings`;
- static contracts, maintained-source/script compilation, high-signal checks,
  Cox behavior checks, and complete test collection;
- documentation contracts;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices.

The hosted CPU run executes canonicalization, direct-import and registry
identity, the restored Adaptive Group Lasso hierarchy, current and simulated
legacy pickle round trips, direct-fit invariance, and CV/refit invariance.
CuPy/Torch coefficient tests skip on the hosted CPU runner by design.

## Remaining Remote Gate

Run the following from a clean physical-GPU checkout of the exact implementation
commit and retain the generated JSON as the evidence artifact:

```bash
python dev/benchmarks/benchmark_group_layout_gpu.py \
  --output results/benchmark_frontend_sources/group_layout_contract_pr80_schema2.json
```

Promotion to `COMPLETE` requires:

- the public hierarchy/API contract to pass;
- CuPy and Torch to pass every ordinary direct-fit layout case;
- CuPy and Torch to pass the simulated legacy-pickle direct-fit cases with and
  without an intercept;
- CuPy and Torch to pass CV score, selected-alpha, and final-refit parity;
- `source_clean=true`;
- exact source hashes;
- `gate_failures=[]`.

Any runtime, test, compatibility-boundary, or runner change after the audited
implementation commit requires a new exact-source physical run.
