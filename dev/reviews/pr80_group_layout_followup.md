# PR #80 Group Penalty Review / Fix Cycle

> Audited runtime commit: `af468ffd29442c6c724f47f28e2c92fc062480d3`  
> Hosted validation: GitHub Actions run `#877` (`30821513368`)  
> Review mode: `.claude/skills/code-review.md` audit + auto-fix loop  
> Status: `PARTIAL_REMOTE_PENDING`

## Current Review Decision

No locally reproducible or hosted `CRITICAL` / `HIGH` Group Lasso, Adaptive
Group Lasso, Group MCP, or Group SCAD finding remains open after the latest
independent review pass. The implementation is not yet eligible for
`COMPLETE`/`APPROVE`, because the changed public families execute on NumPy,
CuPy, and Torch and the current implementation has not yet been certified by a
clean exact-source physical-GPU run on both accelerator backends.

The only remaining hard gate is the canonical physical suite described below.
Historical Cox schema-21 and earlier Group Lasso artifacts are context only and
do not certify the current implementation.

## Impact Classification

- **Numerical coefficients and predictions:** affected. The review repaired
  wrong Group Lasso objectives, wrong non-contiguous group mappings, wrong LLA
  surrogate scaling, silent proximal-Newton stalls, and weighted-objective
  inconsistencies.
- **Selected alpha and final refit:** affected. Penalty-object CV previously
  risked evaluating every candidate with the template object's alpha. Candidate
  penalties, selected alpha, and final public/resolved penalty snapshots are now
  explicitly tied together.
- **Sample weights:** affected. The former Group Lasso block path ignored
  `sample_weight`; all public convex group fits now use the actual weighted loss
  gradient and exact group proximal path.
- **Backends:** affected. NumPy is fully exercised in hosted CI. CuPy and Torch
  source paths, import contracts, and skip-aware tests are hosted-covered, but
  physical numerical execution remains remote-pending.
- **Public API / clone:** affected. Class hierarchy, direct imports, registry
  identity, sklearn clone, library `Penalty.clone()`, constructor state, and
  final fitted-estimator penalty snapshots were repaired.
- **Serialization:** affected. Current and simulated legacy pickle/joblib states
  rebuild group layout metadata and discard stale device caches.
- **Formula:** affected. Groups are interpreted against the final patsy-expanded
  feature matrix; formula intercept columns remain unpenalized.
- **Inference:** affected. The generic residual bootstrap refits ordinary L1 and
  therefore cannot represent a group penalty. Every group-family inference
  request now fails before fitting until group-preserving inference exists.
- **Benchmark evidence:** affected. A canonical outer runner now binds all
  specialized Group Lasso/MCP/SCAD and constructor/CV runners to one clean
  source commit and comprehensive SHA-256 manifest.

## Public Capability Decisions

| Public family | Direct fit | CV | `sample_weight` | Formula | Inference | Backend decision |
|---|---|---|---|---|---|---|
| `GroupLassoPenalty` / `group_lasso` / `gl` | supported through exact composite FISTA routing | supported; candidate scores, selection, and final refit share one group contract | supported | supported by the direct estimator after formula expansion | estimation-only; all inference methods explicitly rejected | NumPy hosted-complete; CuPy/Torch physical pending |
| `AdaptiveGroupLassoPenalty` (object-only) | supported with explicit per-group weights | supported as an object penalty; each alpha-grid candidate rebuilds the weighted penalty | supported | supported by the direct estimator | estimation-only; explicitly rejected | NumPy hosted-complete; CuPy/Torch physical pending |
| `GroupMCPPenalty` / `group_mcp` / `gmcp` | supported through group-aware FISTA-LLA | supported; fold candidates and selected-alpha refit use the same surrogate | supported | supported by the direct estimator | estimation-only; explicitly rejected | NumPy hosted-complete; CuPy/Torch physical pending |
| `GroupSCADPenalty` / `group_scad` / `gscad` | supported through group-aware FISTA-LLA | supported; fold candidates and selected-alpha refit use the same surrogate | supported | supported by the direct estimator | estimation-only; explicitly rejected | NumPy hosted-complete; CuPy/Torch physical pending |
| Direct public penalty numerical API | exact one-dimensional grouped feature vector required | not applicable | not applicable | not applicable | not applicable | NumPy/CuPy/Torch implementations retained; private fused LLA alone may carry one trailing free intercept |

## Closed Findings

### Correctness

- **[CRITICAL][Group Lasso objective]** The historical Group Lasso block update
  used Gaussian `X'X/X'y` work for non-quadratic losses, ignored
  `sample_weight`, and applied inverse-Gram-then-Euclidean-threshold updates that
  are not exact for general correlated group blocks. All public Group Lasso and
  Adaptive Group Lasso fits now bypass that branch and use the advertised loss
  gradient plus exact Euclidean group proximal operator. Explicit FISTA,
  FISTA-BB, and ADMM requests are preserved rather than silently rewritten.
- **[CRITICAL][LLA coordinate mapping]** Equal-size non-contiguous Group
  MCP/SCAD derivatives were emitted in grouped order and then consumed in
  original feature order. LLA weights are now scattered through the canonical
  flat indices.
- **[CRITICAL][LLA scaling]** The old factory could multiply a group derivative
  by target alpha and group size again. The exact surrogate is now
  `sum_g D_g ||beta_g||_2`, represented by
  `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/sqrt(p_g))`.
- **[CRITICAL][silent convergence failure]** The generic proximal-Newton inner
  loop could reject all Armijo steps, restore the old iterate, and return no
  failure status. Group MCP/SCAD LLA now uses the group-aware FISTA inner path,
  with objective improvement and tolerance-stability gates.
- **[CRITICAL][CV object alpha]** Penalty-object CV could hold the object's alpha
  fixed across all candidates. Fit-local templates are marked privately and
  rebuilt at each candidate alpha; object and string penalty forms now match.
- **[CRITICAL][bootstrap mismatch]** Group-family `bootstrap` inference entered
  a routine that hard-coded ordinary L1 refits. Group penalties are now
  explicitly estimation-only.

### API, State, and Compatibility

- **[HIGH][legacy layout]** Interleaved groups and legacy state carrying stale
  `_is_contiguous=True` / missing flat indices are canonicalized during
  construction and unpickling.
- **[HIGH][hierarchy/import]** `AdaptiveGroupLassoPenalty` again inherits the
  public `GroupLassoPenalty`; direct imports, public exports, registry aliases,
  pickle identity, `isinstance`, and `issubclass` are aligned.
- **[HIGH][weighted penalty consistency]** Adaptive Group Lasso now applies its
  weights consistently in value, gradient, and proximal operations, uses
  backend-specific caches, and clears caches during state migration.
- **[HIGH][clone]** sklearn clone, old constructor-identity reconstruction,
  estimator-contained clone, library `Penalty.clone()`, pickle, and joblib use
  immutable constructor snapshots rather than descriptive derived fields.
- **[HIGH][fit-time mutation]** Design-width completion no longer mutates a
  caller-owned penalty object or `penalty_kwargs` dictionary. Direct fits clone
  external objects; CV uses temporary object/kwargs state and restores the
  constructor parameters after success or failure.
- **[HIGH][final fitted API]** The selected CV estimator exposes an unmarked
  public penalty snapshot whose alpha, groups, weights, and hyperparameters
  match the actual resolved objective. The top-level CV estimator retains its
  original constructor parameter.
- **[HIGH][transactionality]** A failed refit clears coefficients, intercept,
  params, inference state, formula state, solver/backend selection, CV scores,
  selected alpha, and final estimator. Coefficient and intercept warm starts
  are passed together for exactly one fit and cleared on either outcome.
- **[HIGH][input contract]** Boolean/coercible-string hyperparameters,
  fractional/negative/non-finite indices, signed-int64 overflow, duplicate or
  empty explicit groups, discontinuous flat group IDs, invalid design widths,
  and incomplete adaptive weighted coverage fail before numerical work.
- **[HIGH][Adaptive public family]** `adaptive_group_lasso` is included in the
  shared group/non-smooth categories, exact convex group routing, CV object
  alpha handling, constructor isolation, inference rejection, and smooth-solver
  validation.
- **[MEDIUM][Adaptive FISTA curvature]** The object-only Adaptive Group Lasso
  name was absent from a hand-written solver utility list, adding fictitious
  smooth Lipschitz curvature and shrinking FISTA steps. An import-time contract
  now classifies it as zero smooth curvature before FISTA/FISTA-BB bind the
  helper; L2 and ElasticNet curvature remain unchanged.

## Hosted Validation

GitHub Actions run `#877` passed for audited runtime commit
`af468ffd29442c6c724f47f28e2c92fc062480d3`:

- complete CPU tree: `1834 passed, 662 skipped, 11 warnings`;
- static contracts and maintained Python/script compilation;
- high-signal static checks, Cox behavior checks, and complete test collection;
- documentation contracts and Python 3.9 documentation writer;
- regression gates on Python 3.9, 3.10, 3.11, and 3.12.

Hosted tests cover correlated Group Lasso KKT conditions, weighted objectives,
non-quadratic loss routing, explicit solvers, interleaved/unequal groups,
formula-expanded designs, current and legacy serialization, class hierarchy,
clone methods, Group MCP/SCAD surrogate scaling and convergence, CV scores and
selected-alpha refits, object/string penalty equivalence, constructor-state
isolation, failed-refit cleanup, one-shot warm starts, strict inputs, and
explicit inference failure. Accelerator tests skip on the hosted CPU runner by
design.

## Canonical Physical GPU Gate

Run the following from a **clean checkout of the latest PR head** on a machine
where both CuPy CUDA and Torch CUDA are available:

```bash
python dev/benchmarks/benchmark_pr80_group_gpu_suite.py \
  --output results/benchmark_frontend_sources/pr80_group_gpu_suite_schema1.json
```

The canonical suite records a comprehensive outer SHA-256 manifest and runs:

1. `benchmark_group_layout_gpu.py` — direct layouts, legacy pickle migration,
   intercept/no-intercept, CV selection and refit;
2. `benchmark_group_lasso_objective_gpu.py` — exact correlated and weighted
   Group Lasso objectives/KKT;
3. `benchmark_group_nonconvex_layout_gpu.py` — Group MCP/SCAD LLA coordinates,
   surrogate scaling, direct fit, CV, public API;
4. `benchmark_group_nonconvex_weighted_gpu.py` — weighted Group MCP/SCAD direct
   fit and CV;
5. `benchmark_group_cv_object_gpu.py` — object/string alpha-grid equivalence,
   Adaptive Group Lasso, constructor isolation, fit-local completion, and final
   public penalty snapshots.

Promotion to `COMPLETE` requires all of the following in the outer JSON:

- `source_commit` equals the exact reviewed PR head;
- `source_clean=true` and `source_clean_after=true`;
- no missing manifest source;
- every sub-runner reports the same commit and `source_clean=true`;
- both CuPy and Torch pass every sub-runner case;
- every sub-runner exits zero;
- outer `gate_failures=[]`.

Any runtime, test, runner, compatibility-boundary, or manifest change after the
physical run invalidates the artifact and requires rerunning the canonical
suite.
