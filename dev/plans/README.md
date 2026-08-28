# statgpu Development Plans

This directory contains the project roadmap, execution backlog, active implementation plans, and historical design notes.

## Authority by responsibility

There is no single global precedence order across documents with different responsibilities. Use the authority that matches the question being answered.

### Hard development and completion gates

1. Applicable `.claude/workflows/` and `.claude/skills/` protocol.
2. `dev/AGENTS.md`.
3. The mandatory checklist in [`TO_DO.md`](TO_DO.md), which summarizes but does not weaken the two sources above.

Roadmap priorities, issue scope, and module plans may narrow a task, but they may not weaken or override these hard gates. Any approved exception must follow the explicit approval/deferral contract in the applicable workflow and `dev/AGENTS.md`.

### Current public capability

Use validated implementation/tests together with `docs/en/guides/implemented-methods.md` and linked maintained model pages. When a capability claim conflicts with validated behavior, correct stale documentation rather than treating the claim as implementation evidence.

### Current priority and sequencing

Use [`ROADMAP.md`](ROADMAP.md). It selects what should be worked on next; it does not redefine development-completion requirements.

### Executable scope and dependencies

Use open GitHub issues and active pull requests, summarized in [`ISSUES.md`](ISSUES.md). Issues may split or narrow roadmap packages but may not declare completion below repository hard gates.

### Research and historical context

Module plans in this directory provide design, literature, and historical context. Their checklists are not a reliable current capability/priority inventory unless the document states a recent verified release/commit.

## Verified baseline

- Last verified release: **statgpu 0.2.5**
- Last verified commit: `84f8bc7e17f66466b3a325cbb007b6cb41843821`
- Verification/rebaseline date: **2026-08-28**
- The 0.2.5 release is the current `master` baseline for new planning.
- Benchmark/dashboard synchronization, canonical CV source, audited source catalog, and production QA tracked by #90/#91/#92/#100 are completed work rather than active queue items.
- Panel Tier-1 implementation delivered through the 0.2.5 line must be reconciled against #93 issue state rather than reimplemented from stale planning text.

The release baseline does not imply that every historical plan item is complete. It establishes the code/documentation snapshot from which new work must branch.

## Document status

| Document | Status | How to use it |
|---|---|---|
| `ROADMAP.md` | Canonical priority source | Current priorities, sequencing, dependencies, and roadmap-level definition of done. |
| `ISSUES.md` | Canonical navigation | Maps roadmap work packages to executable GitHub issues and dependency order. GitHub issue state remains authoritative for execution. |
| `TO_DO.md` | Mandatory summary checklist | Compact hard-gate checklist plus active queue; subordinate to `.claude` and `dev/AGENTS.md`, not a weaker alternative. |
| `gaussian_inference_backend_native_plan.md` | Active implementation plan | Issue #127: backend-native Gaussian linear-model numerical inference, consumer inventory, provenance, precision, validator, and review/fix contract. |
| `panel_framework_proposal.md` | Delivered-design/reference document | Panel architecture/design context. Validate against 0.2.5 implementation and #93 evidence; do not treat old unchecked work as current scope. |
| `panel_stage_c_plan.md` | Delivered phase reference | Stage-C covariance design and acceptance context for the 0.2.5 Panel line; useful for provenance/review, not a new implementation queue. |
| `plan_survival.md` | Active module reference | Cox Phase 1 status and Survival Phase 2+ design context; executable scope is #94/#95. |
| `cran_r_package_mapping.md` | Comparative reference | Method-family gap map. Individual rows may lag current implementation. |
| `plan_anova.md` | Historical research plan | Early implementation-status header is stale; use implemented-methods/model docs for current ANOVA support. |
| `plan_covariance.md` | Historical research plan | Early implementation-status header is stale; current covariance estimators are documented elsewhere. |
| `plan_krr.md` | Historical research plan | Old Nystroem/KernelPCA/kernel checklist may be stale. |
| `plan_spline.md` | Historical research plan | Old spline/cyclic/thin-plate checklist may be stale. |
| `plan_unsupervised.md` | Historical phase record | Benchmark/algorithm history, not current priority queue. |
| `plan.md` | Historical delta | Superseded by `ROADMAP.md`. |
| `archive/` | Archive | Completed or superseded planning material. |

## Current planning focus

The immediate implementation plan is #127 / [`gaussian_inference_backend_native_plan.md`](gaussian_inference_backend_native_plan.md).

Its purpose is to close a correctness/backend-locality gap exposed by the 0.2.5 work before expanding another broad model family. It must inventory the real Gaussian inference consumer graph, preserve statistical behavior, keep numerical inference on the selected backend, reuse shared distribution infrastructure, prove execution provenance, and freeze the physical validator before canonical GPU evidence is collected.

After #127, #105 should establish systematic linear/GLM inference evidence against the repaired contract. #108 is the evidence successor for the released Panel capability. Feature lanes #94 -> #95, #96 -> #98, and #97 remain valid subsequent work when they do not compete for the same backend/inference surface.

## Planning rules

A roadmap item becomes executable only after it has a GitHub issue defining:

- user/developer problem;
- scope and explicit non-goals;
- public API and failure behavior;
- NumPy/CuPy/Torch backend contract;
- direct-fit/CV closure for tunable capabilities;
- inference/formula implications where applicable;
- external baselines and normalization/alignment settings;
- unit/regression/compatibility and physical-GPU validation;
- validator/evidence provenance where remote acceptance is active;
- documentation/benchmark deliverables;
- dependencies and completion criteria.

Do not mark a module complete using only implementation count or passing CPU smoke tests. Completion is contract-based, evidence-based, and subject to hard workflow gates.

Conversely, do not treat stale unchecked planning text as proof that released numerical work is absent. Reconcile plans/issues against merged implementation and acceptance evidence before reopening production scope.
