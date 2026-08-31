# statgpu Roadmap

> Canonical development roadmap  
> Planning baseline release: **0.2.5**  
> Planning baseline commit: `84f8bc7e17f66466b3a325cbb007b6cb41843821`  
> Reconciled: **2026-08-28**

## 1. Purpose and authority

This document defines current development priority and sequencing. It is not a public support matrix and does not override repository development gates.

- Hard development and completion gates come first from the applicable `.claude` workflow/skill and then from `dev/AGENTS.md`.
- For implemented public methods and backend support, use validated implementation/tests together with `docs/en/guides/implemented-methods.md` and linked maintained model pages.
- For executable scope, use open GitHub issues and pull requests.
- Module-specific plans under `dev/plans/` provide design and literature context but may contain historical checklists.

Roadmap priorities and issue scope may narrow work, but they may not weaken hard development gates. When a module plan conflicts with validated implementation or current issue state, update the stale plan rather than reimplementing already delivered behavior.

## 2. Baseline after 0.2.5

Version 0.2.5 contains the Panel Tier-1 implementation line delivered through the release cycle, including shared panel infrastructure, diagnostics/fit statistics, and expanded covariance/inference support. PR #126 is merged and is the Stage-C implementation source incorporated into the 0.2.5 baseline.

The benchmark dashboard synchronization, canonical CV source, audited benchmark-source catalog, and production QA work tracked by #90, #91, #92, and #100 are also complete. They are no longer active execution items.

### Known 0.2.5 evidence caveat

The published 0.2.5 release must not be described as having a fully closed final-validator evidence chain. The release audit found that the physical artifacts under `results/pr126_release_697de113/` were generated before the maintained physical-runner contract was subsequently changed in PR #128. Those artifacts remain historical evidence for the numerical source they identify, but under `RELEASING.md` they do not prove the later validator acceptance contract.

This provenance gap does **not** by itself demonstrate a defect in the released numerical implementation. It does mean planning, #93 closure/evidence reconciliation, and future benchmark work must distinguish implementation evidence from final release-validator provenance and must not promote the historical artifacts as proof of a newer acceptance contract.

The next implementation cycle should prioritize correctness and execution-contract debt exposed by the 0.2.5 work before opening another broad model family. In particular, #127 identifies a maintained Gaussian linear-model inference path that still forces numerical state through NumPy/SciPy even when fitting occurs on CuPy/Torch.

## 3. Prioritization principles

Work is ranked by:

1. **Correctness and contract risk:** fix ambiguous, fallback-prone, or incomplete public behavior before adding breadth.
2. **Workflow completeness:** close released or nearly completed work with evidence and issue hygiene before treating stale roadmap text as new implementation scope.
3. **Shared infrastructure leverage:** prefer work that stabilizes reusable backend/inference contracts for several later estimators and benchmark packages.
4. **Evidence quality:** implementation, external alignment, physical-GPU validation, validator provenance, and documentation move together.
5. **Controlled scope:** avoid PRs that combine repository-wide refactors, multiple new model families, and broad performance work.
6. **Tunable capability closure:** do not expose a direct-fit penalty while leaving its CV selection/final-refit path merely planned.

## 4. Current priority queue

### P0 — post-0.2.5 planning and issue reconciliation

#### P0.1 Reconcile planning state with 0.2.5

Keep `ROADMAP.md`, `TO_DO.md`, `ISSUES.md`, and `dev/plans/README.md` synchronized to the 0.2.5 baseline.

Required maintenance:

- remove completed #90/#91/#92/#100 from active execution queues;
- audit #93 against merged Stage A/B/C implementation and its original acceptance evidence;
- keep the separate PR #128 release-validator provenance caveat visible during that audit;
- do not reopen delivered Panel numerical scope merely because #93 remains open;
- do not treat historical release artifacts produced under an older runner contract as proof of the final PR #128 validator contract;
- classify any remaining #93 work as issue/evidence reconciliation unless a concrete missing production acceptance criterion is demonstrated;
- keep future roadmap changes tied to issues, merged implementation, tests, release evidence, or maintained plans.

### P1 — Gaussian inference backend-native execution

#### P1.1 Issue #127 — migrate legacy Gaussian inference

This is the highest-priority implementation package after the 0.2.5 rebaseline.

The maintained Gaussian inference stack still contains CPU-forcing boundaries in shared and consumer-specific code. The implementation must follow [`gaussian_inference_backend_native_plan.md`](gaussian_inference_backend_native_plan.md).

Core requirements:

- inventory the exact Gaussian consumer/data-lifecycle graph before edits;
- keep covariance, BSE, statistics, p-values, and CI computation on the selected NumPy/CuPy/Torch backend;
- preserve existing nonrobust, HC0-HC3, HAC, Ridge/L2, weighted, rank-deficient, scalar/multi-target, formula, and sklearn-facing semantics;
- allow a final NumPy reporting snapshot only after numerical inference completes;
- reuse the maintained inference-distribution backend instead of estimator-specific SciPy fallback;
- prove actual fit and inference backend/device provenance in tests and physical evidence;
- freeze the physical validator contract before canonical GPU evidence is collected;
- rerun physical evidence if validator acceptance logic changes afterward.

### P1 — inference and Panel evidence follow-up

#### P1.2 Issue #105 — systematic linear/GLM inference evidence

Sequence #105 after #127 so canonical benchmark/validation evidence measures the repaired inference execution contract rather than preserving the known legacy CPU-forcing path.

Required coverage includes coefficient, covariance, BSE, statistic, p-value, CI, likelihood/information criteria where applicable, explicit backend identity, strict failure behavior, and external alignment.

#### P1.3 Issue #108 — Panel canonical benchmark coverage

The Panel Tier-1 implementation is released; the remaining benchmark work should therefore be treated as evidence breadth, not as a reason to repeat #93 implementation.

Extend canonical Panel coverage across maintained estimators and covariance variants with synchronized backend timing, exact method/covariance identities, external alignment, and machine-readable provenance. Do not reuse the disputed final-release validator artifacts as current canonical acceptance evidence unless a source/validator identity audit proves the required contract.

### P2 — survival foundations

#### P2.1 Issue #94 — Kaplan-Meier and Nelson-Aalen

Implement foundational nonparametric survival estimators with:

- right censoring;
- explicit grouped/stratified risk sets;
- Greenwood/corresponding variance and confidence intervals;
- median/quantile behavior where defined;
- NumPy/CuPy/Torch backend consistency;
- external alignment with R `survival`, lifelines, and statsmodels where definitions align.

#### P2.2 Issue #95 — initial AFT family

Sequence after #94 unless an explicitly isolated parallel implementation lane is available.

Initial maintained distributions:

- Weibull;
- log-normal;
- log-logistic.

The package must define parameterization, inference, formula semantics, prediction functions, backend behavior, and mappings to R `survreg` and lifelines.

### P2 — multinomial and sparse foundations

#### P2.3 Issue #96 — unpenalized multinomial logistic regression

Stabilize the unpenalized multinomial/softmax public contract first: identifiability, shapes, class/weight semantics, likelihood, model-based inference, formula behavior, sklearn compatibility, and three-backend execution.

No regularization parameter or penalized solver belongs in #96; it is non-tunable and introduces no multinomial CV surface.

#### P2.4 Issue #98 — complete penalized multinomial suite

Begin only after #96 is merged and stable.

Minimum declared matrix:

- L2;
- L1;
- ElasticNet;
- SCAD;
- MCP.

For every supported tunable penalty, direct fit, path/grid generation, deterministic CV, selection, final refit, supported inference, three-backend behavior, external alignment, physical-GPU validation, and EN/CN documentation must close in the same declared capability package.

#### P2.5 Issue #97 — shared sparse backend contract

Define the shared sparse representation/operation policy before estimator-specific sparse expansion:

- SciPy CSR/CSC;
- CuPy sparse;
- Torch sparse CSR where viable;
- no silent densification;
- explicit operation/solver compatibility matrix;
- device/memory failure behavior;
- representative end-to-end estimator coverage.

This remains the prerequisite for HDFE, mixed models, sparse multinomial follow-up, and broader sparse estimator support.

### P3 — benchmark breadth and bounded hardening

The remaining benchmark coverage issues #101-#104, #106, #107, and #109 are valid evidence packages but should not displace active correctness work unless a new benchmark exposes a correctness defect.

Bounded hardening issues #114, #117, and #118 remain non-blocking unless new measurements raise their severity:

- #114 — dashboard bundle/DOM optimization;
- #117 — mixed-precision benchmark dtype provenance;
- #118 — CV GPU path-buffer memory measurement/bounds.

### P3 — feature-driven technical debt

Refactor only when a bounded correctness or feature task provides regression coverage.

Current candidates:

- split `_penalized_cv.py` by candidate generation, fold execution, selection, and final refit when existing benchmark/regression evidence makes the behavior boundary explicit;
- split long FISTA-family solver functions into bounded numerical components without changing objective/stopping contracts;
- unify repeated backend array-copy/scalar-extraction helpers when device semantics are frozen;
- reduce duplicated CPU/CuPy/Torch paths only where execution provenance and failure behavior remain explicit.

Do not open a repository-wide “unify all backends/solvers/inference” PR.

### P4 — deferred module expansion

The following remain long-term directions rather than immediate priorities:

- Panel IV/high-dimensional fixed effects/DID/event study/dynamic-panel GMM;
- frailty/Fine-Gray/multi-state survival;
- mixed-effects models and GEE;
- meta-analysis;
- changepoint detection;
- multivariate methods;
- copulas;
- multiple imputation;
- nonlinear least squares;
- advanced ANOVA/repeated-measures workflows;
- tensor/adaptive/shape-constrained GAM;
- kernel SVM and broad unsupervised expansion.

A deferred item may be promoted only through a scoped issue with a public contract, three-backend feasibility, external baselines, validation plan, and clear maintenance ownership.

## 5. Definition of done

A statistical feature is complete only when all applicable items pass:

- applicable `.claude` and `dev/AGENTS.md` hard gates;
- documented public API and failure behavior;
- NumPy, CuPy, and Torch execution paths, or an explicitly approved exception;
- no silent fallback for explicit device requests;
- direct-fit/CV closure for tunable public capabilities;
- strict inference or explicit tested estimation-only behavior;
- formula semantics where the API is formula-facing;
- external comparisons with aligned objective/penalty/solver/tie/tolerance conventions;
- CPU unit/regression/compatibility gates;
- maintained physical-GPU validation for active CuPy/Torch paths;
- validator/evidence provenance tied to the accepted implementation contract;
- synchronized benchmark evidence for performance claims;
- consistent EN/CN documentation and changelog claims;
- no stale fitted state, hidden fallback, or untracked diagnostic script.

## 6. Issue hygiene

Each active roadmap package must have one primary GitHub issue. Split implementation only when doing so does not create a partially advertised capability or violate direct-fit/CV closure.

Every active issue should define:

- context and user impact;
- scope and explicit non-goals;
- public API decisions;
- statistical definitions/parameterization;
- backend/device behavior;
- direct-fit/CV status for tunable capabilities;
- inference and formula implications;
- external baseline matrix;
- hosted and physical-GPU gates;
- documentation/benchmark outputs;
- dependencies and acceptance criteria.

Close issues using merged implementation, CI, external comparisons where applicable, physical-GPU evidence, and synchronized documentation. An issue must not remain a roadmap implementation blocker solely because its checkbox list or old plan text was not synchronized after release.
