# statgpu Roadmap

> Canonical development roadmap  
> Last verified release: **0.2.4**  
> Last verified commit: `0aeeb95b60e3e274053b8f1b6427ae50c8eec015`  
> Last verified: **2026-08-06**

## 1. Purpose and authority

This document defines current development priority and sequencing. It is not a public support matrix and does not override repository development gates.

- Hard development and completion gates come first from the applicable `.claude` workflow/skill and then from `dev/AGENTS.md`.
- For implemented public methods and backend support, use validated implementation/tests together with `docs/en/guides/implemented-methods.md` and the linked model pages.
- For executable scope, use open GitHub issues and pull requests.
- Module-specific plans under `dev/plans/` provide design and literature context but may contain historical checklists.

Roadmap priorities and issue scope may narrow work, but they may not weaken or override the hard development gates. When a module plan conflicts with current public documentation or tests, update the stale plan rather than reimplementing an already delivered feature.

## 2. Baseline after 0.2.4

Version 0.2.4 established a stable correctness baseline for:

- public estimator validation and sklearn cloning;
- transactional refits and cross-validation behavior;
- solver/penalty compatibility and narrow numerical fallbacks;
- analytic-weight semantics;
- NumPy/CuPy/Torch finite-input handling;
- binary LogisticRegression and GLM correctness;
- CoxPH/CoxPHCV core contracts;
- release packaging and production installation.

The next cycle should convert that correctness baseline into maintainable product evidence and complete selected statistical workflows. It should not immediately expand into many unrelated zero-percent modules.

## 3. Prioritization principles

Work is ranked by the following criteria:

1. **Correctness and contract risk:** fix ambiguous or incomplete public behavior before adding breadth.
2. **Workflow completeness:** finish a partially implemented statistical workflow before starting a new module family.
3. **Shared infrastructure leverage:** prefer work that reduces duplication or enables several later features.
4. **Evidence quality:** implementation, external alignment, physical-GPU validation, benchmark provenance, and documentation must move together.
5. **Controlled scope:** avoid PRs that combine framework refactors, multiple new model families, and broad performance work.
6. **Tunable capability closure:** do not expose a direct-fit penalty while leaving its CV path merely planned.

## 4. Current priority queue

### P0 — Roadmap and integration control

#### P0.1 Reconcile planning documents with 0.2.4

Deliverables:

- establish this file as the canonical priority source;
- keep `TO_DO.md` synchronized as the mandatory compact gate checklist and queue;
- classify older module plans as active references, historical research plans, or archive material;
- create GitHub issues for every active work package;
- require future roadmap changes to cite an implementation, test, release, or issue.

#### P0.2 Synchronize benchmark dashboard PR #76 with current `master`

PR #76 is the only active product branch at the 0.2.4 baseline, but it was built from an older base and is not currently mergeable.

The synchronization change must be isolated from new benchmark families:

- merge or rebase current `master` into the dashboard branch;
- resolve test, workflow, documentation, package-layout, and generated-asset conflicts;
- regenerate the deterministic three-file data bundle and deployment assets;
- rerun Python, TypeScript, build, staleness, and Playwright gates;
- preserve source hashes, canonical identities, and no-fabrication rules.

### P1 — Benchmark evidence and dashboard readiness

#### P1.1 Add a canonical cross-validation benchmark source

The dashboard implements the CV presentation contract but has no current canonical CV source.

Initial matrix:

- `RidgeCV`;
- `LassoCV`;
- `ElasticNetCV`;
- `LogisticRegressionCV`;
- `PenalizedGLM_CV`;
- `CoxPHCV`.

Required dimensions include backend, folds, candidate-grid size, path/warm-start configuration, CV time, final-refit time, selected parameter, score, convergence/failure diagnostics, timing scope, synchronization policy, and peak memory where available.

#### P1.2 Complete dashboard product QA

Before PR #76 is proposed for integration into `master`:

- test the production build from the nested documentation path;
- complete Chrome/Chromium, Firefox, and WebKit/Safari smoke coverage;
- verify filter cascades, chart/table consistency, empty states, and source metadata;
- verify keyboard navigation, visible focus, control labels, and an accessible table path;
- integrate the user guide into documentation navigation;
- keep generated data and deployment assets deterministic and current.

URL-persisted state, mobile redesign, virtualization, and bundle partitioning remain deferred until supported by measured product need.

### P1 — Panel workflow completion

Panel data has substantial estimator coverage but lacks several standard econometric diagnostics and shared infrastructure.

Implement in three bounded changes:

1. **Shared panel base and covariance registry**
   - consolidate validation, fitted-state handling, summary construction, and covariance dispatch;
   - preserve all current numerical behavior with golden regression tests.
2. **Specification tests and fit statistics**
   - Hausman FE-vs-RE test;
   - pooling F-test;
   - Breusch-Pagan LM test;
   - within, between, overall, and adjusted R-squared;
   - model F-statistic;
   - shared structured test-result object.
3. **Extended covariance support**
   - robust covariance for RandomEffects;
   - HC0/HC2/HC3 where statistically defined;
   - Driscoll-Kraay covariance;
   - explicit one-way/two-way cluster and bandwidth/kernel contracts.

External alignment should use `linearmodels`, R `plm`, and R/Python sandwich implementations with explicitly matched formulas, effects, covariance definitions, and degrees-of-freedom corrections.

Panel IV, high-dimensional fixed-effect absorption, DID/event-study, and dynamic-panel GMM are blocked on this shared foundation.

### P2 — Survival Phase 2

Cox Phase 1 is implemented. The next survival work should complete foundational analysis and prediction before advanced latent-event structures.

#### P2.1 Nonparametric survival estimators

Implement Kaplan-Meier and Nelson-Aalen with:

- right censoring;
- backend-consistent input validation;
- Greenwood or corresponding variance;
- confidence intervals and median survival where defined;
- stratified/grouped output;
- explicit left-truncation follow-up scope;
- alignment with R `survival` and `lifelines`.

#### P2.2 Parametric AFT models

Initial distributions:

- Weibull;
- log-normal;
- log-logistic.

Required contracts:

- censored likelihood and parameterization documented explicitly;
- NumPy, CuPy, and Torch paths;
- model-based covariance and summary output;
- survival, hazard, cumulative-hazard, and quantile prediction;
- formula support;
- alignment with R `survreg` and `lifelines`, including scale/sign mappings.

Frailty, Fine-Gray competing risks, multi-state models, joint models, and survival forests remain deferred until these foundations are complete.

### P2 — Linear-model API parity and sparse infrastructure

#### P2.3 Unpenalized multinomial logistic regression

Issue #96 defines the base multinomial/softmax contract and implements only the unpenalized estimator.

The work must fix:

- identifiability convention;
- coefficient, covariance, and probability shapes;
- class and sample weighting;
- unpenalized likelihood and information criteria;
- unpenalized solver support and convergence diagnostics;
- model-based inference;
- formula semantics;
- sklearn compatibility;
- NumPy, CuPy, and Torch backend behavior.

The Phase-1 implementation includes fit, decision function, probability prediction, hard prediction, likelihood diagnostics, and model-based inference. It must not expose L2 or any other penalty, regularization parameter, or penalized solver. Because the capability is non-tunable, no multinomial CV surface is introduced in #96.

#### P2.4 Complete penalized multinomial suite

Issue #98 begins only after #96 is merged and its public contract is stable.

Penalized multinomial support should be implemented as one coherent capability package rather than exposing L2 first and leaving the remainder fragmented. The declared minimum matrix is:

- L2;
- L1;
- ElasticNet;
- SCAD;
- MCP.

Adaptive and group penalties may be included when their initialization and multiclass grouping conventions are mathematically fixed. If excluded, the design review must record the reason, stable unsupported behavior, tests, documentation, explicit approval, and follow-up.

For every supported penalty, the same work package must close:

- direct-fit objective, scaling, intercept policy, solver dispatch, warm starts, convergence, and KKT/proximal/LLA checks;
- alpha/lambda/C and mixing-parameter path/grid behavior;
- deterministic folds, scoring, selection, tie breaking, and no-leakage tests;
- backend-preserving final refit and supported final-refit inference;
- NumPy/CuPy/Torch parity and physical-GPU validation;
- external alignment and machine-readable benchmark evidence where performance is claimed;
- EN/CN documentation and changelog synchronization.

The issue may use a bounded internal PR sequence, but no partial public capability should be advertised as complete, and #98 must not close after only L2 or only direct-fit support.

#### P2.5 Sparse backend contract

Define a shared sparse-input policy before adding estimator-specific support:

- SciPy CSR/CSC;
- CuPy sparse;
- Torch sparse CSR where viable;
- supported operations and solver matrix;
- no silent densification;
- memory-budget and failure tests;
- explicit unsupported combinations.

This work is a prerequisite for high-dimensional fixed effects, mixed models, sparse multinomial follow-up, and several large-scale algorithms.

### P3 — Feature-driven technical debt

Refactor only when a bounded feature or correctness task provides regression coverage.

Current candidates:

- split candidate generation, fold execution, selection, and final refit in `_penalized_cv.py`;
- split long FISTA/FISTA-BB solver functions by state update, line search, stopping, and diagnostics;
- unify repeated backend array-copy and scalar-extraction helpers;
- reduce duplicated CPU/CuPy/Torch fit paths where one backend-generic implementation preserves device semantics;
- unify duplicated IRLS coordinate-descent implementations only after objective and stopping contracts are frozen.

Do not open a single repository-wide “unify all backends and solvers” PR.

### P4 — Deferred module expansion

The following remain valid long-term directions but are not in the immediate queue:

- mixed-effects models and GEE;
- meta-analysis;
- changepoint detection;
- multivariate methods;
- copulas;
- multiple imputation;
- nonlinear least squares;
- advanced ANOVA/repeated-measures workflows;
- advanced robust covariance;
- tensor/adaptive/shape-constrained GAM;
- kernel SVM and broad unsupervised expansion.

A deferred module can be promoted only with a concrete user need, a scoped design, three-backend feasibility, external baselines, and a clear maintenance owner.

## 5. Definition of done

A statistical feature is complete only when all applicable items pass:

- applicable `.claude` and `dev/AGENTS.md` hard gates are satisfied;
- public API and failure behavior are documented;
- NumPy, CuPy, and Torch execution paths exist, or an explicitly approved exception is recorded;
- explicit device requests do not silently fall back;
- every tunable direct-fit capability has its CV path, selection, and final refit completed in the same declared work package;
- strict inference is implemented or the estimator is explicitly estimation-only;
- formula semantics are tested where the API supports formulas;
- external comparisons use aligned objective normalization, penalties, solvers, ties, tolerances, and feature sets;
- CPU unit/regression/compatibility tests pass;
- physical-GPU validation covers maintained CuPy and Torch paths;
- performance claims use synchronized, provenance-bearing artifacts;
- English and Chinese user documentation and changelog claims remain consistent;
- no stale fitted state, hidden fallback, or untracked diagnostic script is introduced.

## 6. Issue hygiene

Each active roadmap package must have one primary GitHub issue. Split implementation into child or follow-up issues only when this does not create a partially advertised public capability or violate direct-fit/CV closure.

Every issue must include:

- context and user impact;
- scope and non-goals;
- public API decisions;
- statistical definitions and parameterization;
- backend/device behavior;
- direct-fit/CV status for tunable capabilities;
- inference and formula implications;
- external baseline matrix;
- test and physical-GPU gates;
- documentation and benchmark outputs;
- dependencies;
- acceptance criteria.

Close issues using evidence from merged commits, CI, external comparisons, and physical-GPU runs. Do not close an issue solely because a class or function name exists.
