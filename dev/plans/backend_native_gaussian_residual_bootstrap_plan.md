# Backend-native Gaussian residual bootstrap plan

Status: PLAN REVIEW-FIX PASS 1 APPLIED / IMPLEMENTATION OPEN
Issue: #145
PR: #147
Current base: `master` at PR #142 merge commit `bc61b18123503fd5132d62ac797a710df0b53e89`
Parent statistical contract: merged PR #142 head `6d3c51c54cb2571b547b730d3c98cdc66071a545`

## 0. Change classification and non-goals

This work is a **shared backend-execution capability closure** for an already-defined statistical inference procedure. It materially changes the execution capability of residual bootstrap, so NumPy + CuPy + Torch closure is a blocking repository-default gate, but it must not broaden the PR #142 statistical DGP.

Active axes:

- backend / concrete-device ownership / host-transfer boundary;
- solver and child-refit ownership;
- CV selected-final-refit behavior;
- inference / resampling;
- formula and public-wrapper consumers of the shared penalized Gaussian path;
- docs / release-boundary / exact-source evidence.

Not active as new statistical-method scope:

- weighted bootstrap semantics;
- HC/wild bootstrap;
- HAC/block/time-series bootstrap;
- non-Gaussian parametric bootstrap;
- Cox bootstrap;
- batched or multi-bootstrap performance optimization.

A correct serial backend-native implementation is sufficient. Do not add a statistical approximation or silently fallback to CPU merely to satisfy backend coverage.

## 1. Phase-0 current capability matrix and consumer graph

### Capability matrix

| Row | PR #142 state | PR #147 target |
| --- | --- | --- |
| Gaussian L1 residual bootstrap, NumPy/CPU | supported | preserve numerically/statistically |
| Gaussian ElasticNet residual bootstrap, NumPy/CPU | supported | preserve numerically/statistically |
| Gaussian SCAD residual bootstrap, NumPy/CPU | supported when explicitly requested | preserve and close backend execution |
| Gaussian MCP residual bootstrap, NumPy/CPU | supported when explicitly requested | preserve and close backend execution |
| Same rows after CuPy fit on `cuda:k` | fail-closed CPU-only guard | CuPy child refits on the same `cuda:k` |
| Same rows after Torch fit on `cuda:k` | fail-closed CPU-only guard | Torch child refits on the same `cuda:k` |
| Weighted Gaussian residual bootstrap | fail closed | unchanged fail-closed |
| robust/HC or HAC residual bootstrap semantics | fail closed | unchanged fail-closed |
| non-Gaussian bootstrap | fail closed | unchanged fail-closed |
| Cox bootstrap | fail closed | unchanged fail-closed |
| `PenalizedGLM_CV` bootstrap | selected-final-refit inference only | preserve exactly; candidates/folds stay estimation-only |

### Maintained consumers to close

The shared post-fit inference hook can be reached by more than one public surface. Review/test the relevant maintained consumers rather than proving only one representative class:

- `PenalizedGeneralizedLinearModel(loss="squared_error", ...)`;
- `PenalizedLinearRegression`;
- specialized sparse-Gaussian wrappers that permit an explicit bootstrap request, including Lasso/ElasticNet surfaces where applicable;
- SCAD/MCP Gaussian fits through the maintained generic/typed penalized surface;
- `PenalizedGLM_CV` selected final refit;
- formula/data entry points that end in the same penalized Gaussian fit path;
- runtime inference/fit-transaction installers and their import/idempotence behavior;
- EN/CN support docs and changelogs that currently describe bootstrap as CPU-only.

Ridge/L2 bootstrap is not added by this PR because the merged PR #142 resolver does not expose residual bootstrap for that row.

## 2. Statistical and public inference contract

For a successful unweighted penalized Gaussian fit with fixed design `X`:

`y_hat = fitted value`

`resid = y - y_hat`

For draw `b`, consume one row from a deterministic residual-index schedule and construct

`y_star[b] = y_hat + resid[index[b]]`.

Each child refits the **same penalized Gaussian estimator contract** on `(X, y_star[b])`. The bootstrap target remains the penalized coefficient distribution.

Public/requested/result identity stays explicit:

- public request: `bootstrap`;
- resolved method: `residual_bootstrap`;
- reported `_inference_result.method`: `residual_bootstrap`;
- `inference_target_`: `penalized_coefficient_distribution`;
- fixed-direct-fit conditioning: fixed penalty/tuning;
- CV conditioning: selected penalty, with `penalty_selection_adjusted_=False` exactly as in PR #142.

Published summaries remain the established PR #142 outputs:

- sample standard deviation (`ddof=1`) of bootstrap parameter vectors;
- sign-based two-sided p-values;
- percentile confidence intervals;
- approximate `params / bse` statistic retained for result-schema compatibility.

This PR does not claim selective-inference or tuning-selection coverage guarantees.

### Bootstrap controls

Do not invent a new public constructor API solely for this backend closure. Preserve the PR #142/bootstrap-owner behavior for `n_bootstrap` and `bootstrap_random_state`. Characterize `PenalizedGLM_CV` separately: its acceptance requirement is that inference executes exactly once on the selected full-data refit and retains selected-penalty conditioning; do not silently rerun bootstrap inside folds/candidates.

If implementation needs to propagate an already-existing bootstrap control across the CV final-refit boundary, preserve that existing control by identity/semantics rather than adding an unrelated new public parameter in this PR.

## 3. Deterministic resampling contract

For a fixed `(n, n_bootstrap, bootstrap_random_state)`, generate one backend-neutral `int64` index matrix with NumPy `Generator` as **control-plane data only**.

Requirements:

- NumPy, CuPy, and Torch consume identical integer draws;
- the schedule may cross host->device because it is small control-plane state;
- raw `X`, `y`, residuals, `y_hat`, `y_star`, and every child optimization stay on the fit-recorded numerical backend/device;
- no backend may substitute a backend-specific random bootstrap DGP;
- the physical validator records enough schedule identity (seed/draw configuration and preferably a stable schedule hash) to prove parity cases consumed the same draws.

## 4. Child-refit ownership and solver semantics

Preserve all applicable estimator/solver controls that can alter the child numerical solution:

- deep-copied resolved penalty object/family;
- `alpha`;
- `l1_ratio`;
- `penalty_kwargs`;
- `fit_intercept` / effective intercept semantics;
- `max_iter` and `tol`;
- `n_jobs` where applicable;
- canonical public `solver` request;
- `lipschitz_L` when supplied;
- `stopping`;
- `lla`, `max_lla_iters`, and `lla_tol` for SCAD/MCP;
- `loss_kwargs` where accepted by the maintained Gaussian wrapper;
- `gpu_memory_cleanup` behavior without changing fitted-state lifetime;
- concrete backend/device;
- `compute_inference=False` on every child.

`cpu_solver` is a legacy CPU-stage control: preserve it on NumPy/CPU child refits where it can own execution, but do not make it an authoritative GPU control.

For an explicit parent solver request, the child must execute the corresponding supported solver. For `solver="auto"`, the child must follow the same canonical dispatch table for the same loss/penalty/backend; record/compare selected solver identity in targeted tests where it is observable.

No child may recursively invoke bootstrap or any other inference.

## 5. Concrete backend/device ownership

The parent fit's recorded `_selected_backend_name` and `_selected_backend_device` are authoritative.

- NumPy/CPU parent -> NumPy/CPU response construction and child refits.
- CuPy parent on `cuda:k` -> CuPy response construction and child refits on **that same `cuda:k`**.
- Torch parent on `cuda:k` -> Torch response construction and child refits on **that same `cuda:k`**.

A post-hoc provenance check is necessary but not sufficient. The child fit itself must be entered under the parent concrete device context:

- CuPy: parse `cuda:k` and execute conversion/index creation/refit under `cp.cuda.Device(k)` (or an equivalent maintained exact-device mechanism);
- Torch: bind the refit to `cuda:k` using an exact Torch device/current-device context plus device-resident inputs so construction/solver allocations cannot drift to another ordinal.

Then verify each child recorded backend/device equals the parent. Any mismatch is a hard failure.

Heterogeneous public input containers follow **executed fit provenance**, not original container type. Include at least one supported Torch->CuPy and CuPy->Torch crossing in physical validation.

Only two host-transfer classes are allowed:

1. the small integer resampling schedule as control-plane H2D data;
2. each fully completed child fit's established NumPy parameter/reporting snapshot, plus the final diagnostic/reporting boundary.

Do not copy `X`, `y`, residuals, or `y_star` to NumPy to reuse a CPU optimizer.

## 6. Failure transaction and fitted diagnostic-state preservation

The outer estimator must remain fail-closed if bootstrap cannot complete. Cover failures that occur:

- before draws (`n_bootstrap < 2`, malformed controls, unsupported covariance/weights/family);
- while constructing backend data/index state;
- during a child refit;
- during child provenance or parameter-shape validation.

A failed inference-enabled refit must not leave a prior successful inference result, coefficient snapshot, or `_fitted=True` state advertised as current. Reuse the merged PR #142 fit-transaction invalidation contract rather than inventing a second transaction system.

For successful fits, preserve the CPU PR #142 diagnostic/reporting behavior needed by maintained Gaussian consumers. Regression-test that moving bootstrap execution to another backend does not newly erase or corrupt the established response/residual/design/nobs state or public diagnostics such as `rsquared`/related summary fields where they were available before this extension.

## 7. CV and formula consumer contract

### `PenalizedGLM_CV`

Bootstrap remains **selected-final-refit-only**:

- folds, alpha candidates, path/grid scoring, and selection run with inference disabled;
- after selecting `alpha`, bootstrap executes exactly once on the full-data selected estimator;
- `selected_alpha` / `alpha_` agree;
- `penalty_conditioning_="cv_selected_penalty"`;
- `penalty_selection_adjusted_=False`;
- final inference backend/device follows the selected final-refit backend, not the input container type;
- no bootstrap work occurs during candidate evaluation.

Do not broaden this PR into a new CV API. If a pre-existing bootstrap owner control must cross the final-refit reconstruction boundary, characterize and preserve it explicitly.

### Formula/data route

Because the shared estimator is formula-facing, add a regression proving that formula and array routes produce the same residual-bootstrap contract after design-matrix construction for at least one representative supported Gaussian penalty. Formula handling itself is not redesigned here; row/intercept/feature-name semantics must remain unchanged.

## 8. Hosted deterministic acceptance matrix

Before physical CUDA acceptance, add deterministic tests for all of the following.

### Resampling and NumPy preservation

- backend-neutral schedule reproducibility;
- identical schedule identity for the same seed and different identity for a different seed;
- NumPy residual-bootstrap output reproduces the merged PR #142 CPU behavior for fixed draws within a frozen tolerance (exact where the path is deterministic enough);
- `n_bootstrap < 2` remains transactional/fail-closed.

### Penalty/refit ownership

- L1 end-to-end bootstrap;
- ElasticNet end-to-end bootstrap, including `l1_ratio`;
- SCAD hosted coverage;
- MCP hosted coverage;
- deep-copied penalty/`penalty_kwargs` preservation;
- explicit `lipschitz_L` preservation where the selected solver consumes it;
- LLA controls preserved for SCAD/MCP;
- child `compute_inference=False` / no recursive inference.

SCAD and MCP are both claimed rows. Do not make hosted coverage optional merely because nonconvex CUDA acceptance may be less stable; only the **physical GPU** nonconvex row may remain optional if documented with reason.

### Unsupported/fail-closed rows

- weighted Gaussian residual bootstrap remains rejected;
- robust/HC bootstrap semantics remain rejected;
- HAC/block semantics remain rejected;
- non-Gaussian bootstrap remains rejected;
- Cox bootstrap remains rejected through its existing estimation-only contract;
- explicit unavailable CuPy/Torch requests do not silently fallback to CPU.

### Backend/device behavior

- Torch host-contract test proves bootstrap responses supplied to child fits remain Torch-native;
- equivalent CuPy behavior is covered when CuPy is available, with deterministic skip only when CUDA is unavailable;
- child backend/device provenance mismatch fails closed;
- exact-device context is exercised/locked by a targeted regression (not only post-fit metadata comparison);
- heterogeneous-container conversion follows fit-recorded backend where host-only mocking can prove routing without pretending to be physical GPU evidence.

### Transaction/consumer closure

- a child fit exception invalidates outer fit/inference state;
- a prior successful fit followed by a failed bootstrap refit does not leak stale inference fields;
- `PenalizedGLM_CV` performs inference exactly once on the selected final refit and preserves selected-penalty metadata;
- formula vs array route parity for one representative supported penalty;
- generic `PenalizedGeneralizedLinearModel` and typed `PenalizedLinearRegression` surfaces remain consistent;
- specialized Lasso/ElasticNet explicit-bootstrap surfaces remain usable where already supported;
- installer is idempotent and import order does not stack duplicate wrappers;
- sklearn reconstruction/clone surfaces that existed before PR #147 are not broken.

## 9. User-facing docs and release boundary

This PR changes a public backend support claim, so documentation is a blocking pre-final-review task.

At minimum reconcile all maintained pages that currently call Gaussian residual bootstrap CPU-only, including as applicable:

- `docs/en/guides/penalized-glm-inference.md`;
- `docs/cn/guides/penalized-glm-inference.md`;
- inference-mode/support-matrix pages if they repeat the CPU-only boundary;
- root `CHANGELOG.md`;
- `docs/en/changelog.md`;
- `docs/cn/changelog.md`.

Update only surfaces actually affected by this capability. Keep EN/CN claims conceptually aligned.

Wording must distinguish:

- published release status;
- behavior merged on `master` / implemented by PR #147;
- target release if known.

Do not claim weighted, robust/HAC, non-Gaussian, Cox, or batched bootstrap support.

## 10. Exact-source physical CUDA validator contract

Add a maintained validator such as

`dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py`

with an artifact destination such as

`results/pr147_gaussian_residual_bootstrap_gpu/pr147_gaussian_residual_bootstrap_gpu.json`.

The validator must require/record:

- clean git worktree;
- exact source SHA;
- validator schema version;
- Python/statgpu/NumPy/CuPy/Torch versions;
- CUDA/runtime/device provenance, including the concrete ordinal actually executed;
- fixed data seeds, bootstrap seed, `n_bootstrap`, and schedule identity/hash;
- named coefficient/parameter and bootstrap-summary tolerances;
- public requested solver and selected solver where relevant;
- numerical backend/device and reporting-boundary metadata;
- explicit evidence that no child refit executed on CPU for GPU rows.

Freeze named tolerance constants **before the first physical acceptance run**. Do not loosen them after a failed P100 run merely to obtain green status. Any later justified tolerance change requires a code-reviewed rationale, schema/version change, and a new physical run.

Physical matrix must include at least:

- NumPy reference under the same fixed draws;
- CuPy CUDA L1 direct fit;
- Torch CUDA L1 direct fit;
- CuPy CUDA ElasticNet direct fit;
- Torch CUDA ElasticNet direct fit;
- same-schedule parameter/bootstrap-summary parity;
- concrete backend/device provenance;
- Torch->CuPy and CuPy->Torch heterogeneous-input crossings where the public fit boundary supports them;
- no CPU numerical fallback.

A representative SCAD/MCP physical row is desirable but not a completion requirement if the maintained nonconvex GPU solver is not stable enough for a deterministic threshold. SCAD and MCP still require hosted contract coverage.

## 11. Validation, review, and evidence-DAG order

Use this order so the final artifact does not repeat PR #142's evidence-freshness ambiguity:

1. finish production implementation and targeted hosted tests;
2. finish user-facing EN/CN docs/changelog updates;
3. add/freeze the physical validator contract and tolerance constants;
4. run targeted + full hosted CI on the resulting source head;
5. run a fresh independent code-review/fix pass over code/tests/docs/validator;
6. if review fixes numerical/backend/validator behavior, rerun invalidated hosted checks and repeat review;
7. once the numerical/validator source is stable, run the physical CUDA validator on that exact clean source;
8. retain the artifact with exact source/schema/environment/provenance;
9. perform a final fresh exact-head/evidence review; any later source change makes earlier review/CI/physical evidence historical unless its validator contract explicitly fingerprints and permits reuse of the unchanged relevant source.

Prefer running the physical gate on the actual final PR head. Avoid a docs-after-physical sequence when possible. If any later docs-only commit is unavoidable, do not silently reuse the artifact: the validator/evidence contract must explicitly support the claimed reuse or rerun the physical gate.

## 12. Plan-review closure criteria

A plan-level `REVIEW CLEAN` may be recorded only when a fresh pass finds:

- the statistical DGP/estimand remains exactly PR #142's unweighted Gaussian residual bootstrap;
- all claimed L1/ElasticNet/SCAD/MCP consumers are accounted for;
- NumPy/CuPy/Torch and concrete-device ownership are explicit;
- refit solver/tuning ownership includes all materially relevant controls;
- CV selected-final-refit and formula consumers are explicit;
- fail-closed unsupported rows and failure transaction are explicit;
- docs/release-boundary work precedes final review;
- physical validator/evidence freshness is explicit and exact-source based.

Plan review cleanliness does **not** imply implementation readiness. PR #147 remains Draft until implementation, hosted validation, exact-source physical CUDA evidence, and final fresh review all close.
