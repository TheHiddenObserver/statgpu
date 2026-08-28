# Gaussian Linear-Model Backend-Native Inference Plan

Issue: #127  
Baseline release: `0.2.5`  
Baseline `master`: `84f8bc7e17f66466b3a325cbb007b6cb41843821`  
Status: planning review/fix in progress; production work not started

## 1. Goal

Migrate the maintained Gaussian linear-model inference path so covariance, standard errors, test statistics, p-values, and confidence intervals execute on the selected NumPy/CuPy/Torch backend without silent full-array GPU-to-CPU fallback.

The migration changes execution locality and backend provenance, not established statistical definitions. A final NumPy reporting snapshot may remain where required by the existing result-container API, but it must occur only after numerical inference has completed.

This work is intentionally narrower than a repository-wide inference rewrite.

## 2. Impact classification

Active axes:

- public backend/device contract;
- inference;
- numerical correctness and precision;
- backend-native array ownership and device locality;
- linear-model and penalized-Gaussian compatibility;
- formula regression compatibility;
- CV final-refit inference where the shared Gaussian path is consumed;
- physical-GPU validation and provenance;
- documentation and evidence artifacts.

Inactive axes:

- new loss definitions;
- new penalties;
- solver redesign;
- new model families;
- new CV selection algorithms;
- sparse-input support;
- repository-wide removal of SciPy.

Eliminating unintended full-array GPU-to-CPU inference transfers is a correctness/backend-contract gate. GPU speedup is not a completion claim for this issue and requires separate synchronized benchmark evidence if advertised.

### 2.1 Capability decisions

The table below is the minimum known public surface before the mandatory consumer inventory. Phase 1 must add any newly discovered public consumer before production edits continue.

| Public capability | backend | CV | inference | formula | benchmark |
| --- | --- | --- | --- | --- | --- |
| `LinearRegression` | `three-backend` | `non-tunable` | `supported` | `supported` | `not-performance-sensitive` |
| `Ridge` / Gaussian squared-error L2 direct fit | `three-backend` | `supported` through the maintained Ridge/unified CV surface; no CV expansion in #127 | `supported` | `supported` | `not-performance-sensitive` |
| `RidgeCV` final refit | `three-backend` | `supported` | `supported` on the final `Ridge` refit when `compute_inference=True` | `not-formula-facing` | `not-performance-sensitive` |
| Other public Gaussian/L2 consumers discovered in Phase 1 | must resolve to `three-backend` or require explicit approved deferral | preserve current supported/non-tunable decision; no new CV capability | preserve current supported/estimation-only contract | preserve current supported/not-formula-facing contract | default `not-performance-sensitive` unless a claim activates benchmark gate |

`planned` is not an acceptable completion state for an already supported inference/backend capability. If inventory reveals a public consumer whose current documented contract conflicts with this table, stop and amend the plan/decision explicitly rather than silently narrowing support.

### 2.2 Objective, penalty, and CV non-change contract

#127 does not change the fitting objective, regularization path, selected hyperparameter, or CV scoring definition.

For Ridge/L2 paths, preserve the repository's current average-loss convention and exact-solve scaling. In particular, current direct/CV implementations solve unnormalized normal equations with the equivalent `n_eff * alpha` L2 contribution, where `n_eff` is the sample count or the declared weight normalization for weighted fits. Inference bread/penalty construction must preserve the same mapping rather than introducing a new `alpha`, `lambda`, or `C` convention.

External Ridge comparisons must state the equivalent penalty mapping explicitly. A mismatch is fixed in the comparison/mapping layer; the statgpu fit objective is not changed merely to match an external package.

`RidgeCV` already selects alpha and performs a final `Ridge(..., compute_inference=...)` refit. #127 may change only the numerical locality/provenance of that final inference, not folds, alpha grid, scoring, selection, tie behavior, or final-refit coefficient semantics.

## 3. Mandatory consumer and data-lifecycle inventory

Before production edits, resolve and record the exact consumer graph.

Known direct/shared locations include:

1. `statgpu/linear_model/_gaussian_inference.py`;
2. `statgpu/linear_model/wrappers/_linear.py`;
3. `statgpu/linear_model/penalized/_inference_mixin.py`.

Known public consumers include at least `LinearRegression`, `Ridge`, the Gaussian/L2 penalized path used by `Ridge`, and `RidgeCV` final refit. Inventory every additional public wrapper and maintained CV final-refit path that reaches these helpers or implements equivalent Gaussian inference behavior.

For every consumer record:

- requested device/backend;
- executed fit backend;
- where design, residuals, parameters, scale, and weights are created;
- every `_to_numpy()` / `np.asarray()` boundary before inference completion;
- covariance implementation reached;
- distribution implementation reached;
- final reporting conversion;
- formula behavior;
- sample-weight behavior;
- scalar versus multi-target behavior;
- CV final-refit ownership where applicable.

The implementation scope is the resolved Gaussian consumer graph. It is not every `scipy.stats` import under `statgpu/linear_model`.

If the inventory discovers another public Gaussian consumer, amend the capability table and consumer matrix before implementation rather than leaving it silently CPU-oriented.

## 4. Backward-compatibility contract

Freeze established ordinary-regime NumPy behavior for:

- nonrobust inference;
- HC0 / HC1 / HC2 / HC3;
- Bartlett HAC and existing lag semantics;
- degrees of freedom;
- Ridge/L2 inference and intercept-penalty treatment;
- weighted inference;
- rank-deficient / pseudoinverse behavior;
- scalar and multi-target output shape/order;
- formula feature names/order;
- `summary()` and inference-result fields;
- sklearn clone, fitted-state, and transactional-refit behavior;
- RidgeCV folds, alpha candidates, selected alpha, scoring, final-refit coefficients, and failure transactionality.

Public result types and established reporting types remain unchanged unless an independent API change is explicitly approved.

Historical output is not frozen when it is demonstrably a numerical precision defect. Small-df and extreme-tail tests must follow the maintained backend-distribution precision contract rather than preserve cancellation-induced artifacts.

## 5. Numerical-state versus reporting-state boundary

Define one explicit boundary.

### Numerical inference state

Remain on the selected backend through:

- design construction after formula parsing;
- weighting;
- residual handling;
- Gram/bread construction;
- inverse/pseudoinverse;
- covariance bread/meat calculations;
- standard errors;
- test statistics;
- distribution CDF/SF/quantile evaluation;
- confidence-interval endpoints.

### Reporting snapshot

Only after numerical inference completes may arrays be converted to the established result/reporting representation.

Do not convert full design, residual, covariance-working, or statistic arrays to NumPy merely to populate inference.

The reporting boundary must be centralized or otherwise machine-auditable so tests can distinguish an allowed final snapshot from an accidental pre-inference host transfer.

## 6. Backend-native fit-state construction

Refactor `GaussianFitState` or replace it with an equivalent backend-neutral numerical-state object.

Requirements:

- no NumPy-array-only type contract for numerical state;
- create intercept columns on the selected backend;
- construct weighted design/residual arrays with backend-native operations;
- preserve dtype and concrete CUDA device;
- do not infer intercept-penalty policy by copying/scanning a GPU design on CPU when fit metadata already identifies the intercept;
- preserve existing weighted Ridge normalization and the direct/CV alpha scale described in Section 2.2;
- formula/Patsy parsing may remain CPU-side, but after model-matrix transfer numerical fitting/inference must not silently return to NumPy.

## 7. Backend-native linear algebra

For each maintained backend:

- compute `X'X`, Ridge penalty additions, and bread matrices natively;
- reuse existing backend abstraction and rank-failure helpers;
- preserve concrete Torch device placement;
- catch only recognized rank/linear-algebra failures for pseudoinverse recovery;
- keep OOM, device, dtype, shape, programming, and contract errors fatal;
- do not introduce an independent rank policy unless parity evidence demonstrates that the existing policy is insufficient.

Rank-deficient behavior remains a separate regression surface from full-rank behavior.

## 8. Covariance migration

Maintain the current Gaussian covariance surface:

- `nonrobust`;
- `hc0`;
- `hc1`;
- `hc2`;
- `hc3`;
- `hac`.

A common semantic dispatcher is allowed, but code unification is not itself a goal.

Before deleting or bypassing covariance/HAC helpers in `LinearRegression`, determine whether they are active and whether they own numerical or performance semantics, including mixed-precision HAC selection.

Rules:

- do not remove an active helper solely because a shared helper exists;
- remove duplication only after call-graph and golden-regression evidence proves it redundant;
- do not change HC corrections or HAC lag/kernel definitions merely to simplify dispatch;
- keep full numerical arrays on the selected backend.

## 9. Distribution inference

Replace estimator-level direct `scipy.stats.t` / `scipy.stats.norm` numerical inference with the maintained inference-distribution infrastructure.

Required behavior:

- inference backend matches the executed numerical backend;
- concrete Torch CUDA device is preserved;
- normal/Student-t p-values and critical values use maintained shared distribution logic;
- no estimator-specific SciPy fallback for explicit CuPy/Torch execution.

Existing internal LUT/cache construction inside the shared distribution subsystem is not automatically estimator fallback. It remains acceptable only when user design/residual/statistic arrays are not moved to CPU and the actual numerical inference result stays on the requested backend/device.

If the shared distribution layer moves the actual statistic vector to CPU for a required Gaussian case, treat that as a blocking inference-layer defect rather than hiding it in #127.

## 10. Multi-target inference

Correctness and locality come before vectorization.

An initial per-target loop is acceptable if:

- every target remains on the selected backend;
- there is no target-by-target GPU-to-CPU-to-GPU round trip;
- output identities and ordering remain unchanged.

Vectorize only when numerical equivalence is demonstrated and temporary GPU memory remains bounded.

## 11. Provenance contract

Tests and physical evidence must distinguish:

- requested backend;
- executed fit backend;
- executed numerical-inference backend;
- concrete execution device;
- reporting representation.

Prefer additive/internal provenance or existing inference metadata instead of changing public result shapes.

Input array type alone is not proof of executed backend.

## 12. Hosted test matrix

### 12.1 Golden NumPy regression

Freeze pre-migration ordinary-regime results for:

- nonrobust;
- HC0-HC3;
- HAC;
- weighted/unweighted;
- intercept/no-intercept;
- Ridge/L2;
- scalar target;
- multi-target;
- full rank;
- rank deficient.

### 12.2 Three-backend inference matrix

For NumPy, CuPy, and Torch validate:

- covariance;
- BSE;
- t/z statistic;
- p-value;
- confidence interval;
- degrees of freedom;
- inference-result fields;
- actual execution provenance.

Use Torch CPU hosted coverage for as much Torch implementation as possible; final Torch CUDA evidence remains physical.

### 12.3 Precision matrix

Explicitly cover:

- small residual df;
- Student-t df=1 and df=2;
- central probabilities;
- extreme but float64-representable tails;
- normal-tail cases;
- confidence-interval critical values.

Use the strongest available reference and maintain or improve repository precision.

### 12.4 External alignment

Use:

- analytic closed forms where available;
- pre-migration NumPy/SciPy behavior as migration oracle for ordinary cases;
- statsmodels OLS/WLS for nonrobust and HC/HAC inference where definitions align;
- analytic Ridge covariance identities plus coefficient alignment for Ridge/L2 cases.

Record intercept convention, weighting, covariance type, HAC lag definition, df convention, Ridge objective/penalty scaling, and tolerance.

### 12.5 Formula regression

For formula-facing consumers verify:

- array/formula numerical agreement;
- Patsy missing-row alignment;
- intercept semantics;
- feature names and summary ordering;
- CPU-side formula parsing does not imply CPU numerical inference after backend transfer.

Do not add formula support to a currently non-formula-facing consumer such as `RidgeCV` as part of #127.

### 12.6 CV final-refit regression

`RidgeCV` is a confirmed final-refit consumer because it selects alpha and constructs a final `Ridge(..., compute_inference=...)` estimator. Add representative coverage proving:

- folds and candidate alpha semantics remain unchanged;
- scoring/selection and selected alpha remain unchanged;
- final refit selects the intended backend;
- final-refit inference uses that backend;
- final coefficients/intercept remain unchanged within tolerance;
- failed final refit remains transactional;
- no new CV capability or tuning behavior is introduced.

If Phase 1 discovers another maintained Gaussian/L2 CV final-refit consumer, add it to the capability table and choose representative matrix coverage rather than assuming RidgeCV is exhaustive.

## 13. No-host-transfer and failure tests

For explicit GPU execution prove that:

- `_to_numpy()` or equivalent full-array conversion cannot occur before the designated reporting boundary;
- estimator-level SciPy numerical inference is not invoked;
- unsupported backend operations fail closed;
- rank recovery does not catch unrelated exceptions;
- concrete Torch device is preserved;
- failed inference does not leave a misleading partially populated inference result.

Behavioral/instrumented tests are primary evidence. Source-text guards may supplement them but must not be the only proof.

## 14. Physical GPU validator

Add a dedicated maintained validator, for example:

`dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py`

Finalize and review the validator contract before canonical physical evidence is collected.

Required artifact fields:

- schema version;
- exact `git_sha`;
- clean-worktree proof;
- explicit `--expected-sha`;
- explicit `--validation-tier`;
- requested backend;
- executed fit backend;
- executed inference backend;
- concrete CUDA device;
- Python/CuPy/Torch versions;
- GPU model;
- covariance case identity;
- maximum covariance/BSE/statistic/p-value/CI errors;
- rank/full-rank disposition;
- weighted/multi-target disposition;
- no-silent-fallback result;
- host-transfer/provenance evidence;
- overall status.

Physical cases must cover both CuPy CUDA and Torch CUDA and include nonrobust, representative HC, HAC, Ridge/L2, weighted, rank-deficient, multi-target, small-df/tail inference, and at least one `RidgeCV` final-refit inference case.

### Validator freeze rule

Canonical evidence is tied to both numerical implementation and the validator acceptance contract.

If the validator, acceptance logic, provenance checks, or case matrix changes after a physical run, rerun affected canonical evidence on an exact clean candidate head.

Do not classify historical artifacts generated by an older validator contract as proof of a newer contract.

## 15. Performance boundary

Do not promise GPU speedup as part of #127.

Measure enough to establish:

- no full-array device-to-host inference transfer;
- no pathological per-target synchronization loop;
- no new dense `n x n` materialization;
- no obvious performance regression introduced solely by the migration.

User-facing speed/crossover claims require separate synchronized benchmark artifacts.

## 16. Documentation

Update EN first, CN second where applicable:

- linear-model inference/device documentation;
- Ridge/penalized Gaussian documentation when its execution behavior changes;
- device-and-memory documentation if it currently implies CPU inference;
- root/EN/CN changelogs.

Document this boundary explicitly:

> Numerical covariance and reference-distribution inference are backend-native; the established public result/reporting layer may take one final NumPy snapshot after numerical inference completes.

Do not claim that every inference implementation in the repository is backend-native.

## 17. Implementation and review sequence

### Phase 1 — inventory and golden freeze

- resolve exact consumer matrix and finalize the capability table;
- resolve direct and indirect CV consumers;
- freeze NumPy/statistical behavior;
- freeze reporting/API shapes;
- record active/dead duplicate helpers;
- record exact Ridge/L2 objective and penalty mapping used by each touched direct/CV consumer.

### Phase 2 — numerical state

- preserve design/residual/parameter/weight arrays on backend;
- introduce explicit reporting boundary;
- add backend/device provenance.

### Phase 3 — covariance and linear algebra

- backend-native bread/inverse/pinv;
- backend-native HC/HAC execution;
- preserve Ridge/intercept/rank semantics and objective scaling.

### Phase 4 — distribution inference

- route normal/t inference through maintained distribution backend;
- add small-df/extreme-tail gates;
- preserve concrete Torch device.

### Phase 5 — consumers

- migrate `LinearRegression`;
- migrate bounded penalized Gaussian/L2 consumer graph;
- cover `RidgeCV` final refit and any additional confirmed CV consumers;
- preserve formula behavior without expanding formula surface.

### Phase 6 — hosted validation

- run three-backend matrix;
- run analytic/statsmodels alignment;
- run direct/CV objective-scale regression;
- run no-host-transfer/failure tests;
- run maintained hosted CI.

### Phase 7 — implementation review/fix

Run `.claude/skills/code-review.md` in auto-fix mode and repeat until:

- CRITICAL = 0;
- HIGH = 0;
- relevant actionable MEDIUM = 0, or a remaining MEDIUM is explicitly bounded as a follow-up and does not conceal a completion gate.

### Phase 8 — validator freeze

Complete and review the physical validator and negative controls before remote canonical execution.

### Phase 9 — physical acceptance

Run exact clean-head CuPy and Torch CUDA acceptance and commit canonical evidence.

### Phase 10 — post-physical re-review

If only evidence/docs were added, prove numerical source and validator acceptance logic did not change.

If production code or validator acceptance logic changed, rerun affected physical evidence.

### Phase 11 — docs and completion

Synchronize documentation/changelogs and rerun the complete hosted gate set.

## 18. Completion criteria

#127 is `COMPLETE` only when:

- exact consumer graph and final capability table are documented;
- numerical Gaussian inference stays on the selected NumPy/CuPy/Torch backend;
- explicit GPU execution has no silent NumPy/SciPy fallback;
- final reporting conversion is the only allowed full-array NumPy snapshot;
- nonrobust/HC0-HC3/HAC semantics pass;
- Ridge/intercept, weighted, and objective/penalty scaling semantics pass;
- scalar/multi-target and rank-deficient paths pass;
- small-df/extreme-tail precision passes;
- formula behavior remains compatible without accidental surface expansion;
- `RidgeCV` and any additional confirmed CV final-refit inference consumers remain statistically/transactionally unchanged except for backend-local inference execution;
- actual fit/inference backend is machine-auditable;
- hosted tests and external alignment pass;
- exact clean-head CuPy and Torch CUDA acceptance passes;
- validator and canonical evidence correspond to the same acceptance contract;
- EN/CN documentation is synchronized;
- final review has no unresolved CRITICAL/HIGH/relevant actionable MEDIUM findings.

If local work is complete and only physical GPU or remote external evidence remains, use `PARTIAL_REMOTE_PENDING`. Missing local correctness, backend, formula, precision, CV, or provenance evidence is not eligible for that status.

## 19. Explicit non-goals

- no repository-wide inference rewrite;
- no removal of every SciPy import from `linear_model`;
- no new model family;
- no new covariance definition;
- no solver redesign;
- no sparse-input work;
- no penalized-multinomial work;
- no public result-container redesign;
- no new CV selection/path behavior;
- no new formula support for non-formula-facing consumers;
- no mandatory multi-target vectorization;
- no GPU speedup claim without separate synchronized evidence.

## 20. Successor sequencing

After #127:

1. #105 — systematic linear/GLM inference benchmark and canonical dashboard evidence;
2. #108 — complete canonical Panel estimator/covariance evidence using fresh or source/validator-contract-matched provenance rather than automatic reuse of the disputed 0.2.5 final-release artifact chain;
3. feature lanes may proceed independently when they do not compete for the same backend/inference surface:
   - #94 -> #95 for survival;
   - #96 -> #98 for multinomial;
   - #97 as the shared sparse-array foundation;
4. #114, #117, and #118 remain bounded performance/provenance follow-ups unless new evidence raises their severity.
