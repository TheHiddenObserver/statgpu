# statgpu Development Plans

This directory contains the project roadmap, execution backlog, and historical design notes.

## Authority by responsibility

There is no single global precedence order across documents with different responsibilities. Use the authority that matches the question being answered.

### Hard development and completion gates

1. Applicable `.claude/workflows/` and `.claude/skills/` protocol.
2. `dev/AGENTS.md`.
3. The mandatory checklist in [`TO_DO.md`](TO_DO.md), which summarizes but does not weaken the two sources above.

Roadmap priorities, issue scope, and module plans may narrow a task, but they may not weaken or override these hard gates. Any approved exception must follow the explicit approval and deferral contract in the applicable workflow and `dev/AGENTS.md`.

### Current public capability

Use the validated implementation and tests together with `docs/en/guides/implemented-methods.md` and the linked maintained model pages. When a capability claim conflicts with validated behavior, correct the stale documentation rather than treating the claim as implementation evidence.

### Current priority and sequencing

Use [`ROADMAP.md`](ROADMAP.md). It selects what should be worked on next; it does not redefine development-completion requirements.

### Executable scope and dependencies

Use open GitHub issues and active pull requests, summarized in [`ISSUES.md`](ISSUES.md). Issues may split or narrow roadmap packages, but may not declare work complete below the repository hard gates.

### Research and historical context

Module plans in this directory provide design, literature, and historical context. Their checklists are not a reliable current capability or priority inventory unless the document states a recent verification release and commit.

## Verified baseline

- Last verified release: **statgpu 0.2.4**
- Last verified commit: `0aeeb95b60e3e274053b8f1b6427ae50c8eec015`
- Verification date: **2026-08-06**
- Release workflow, PyPI wheel/sdist publication, clean production installation, and representative model smoke tests passed.

The release baseline does not imply that every historical plan item is complete. It establishes the code and documentation snapshot from which future work must branch.

## Document status

| Document | Status | How to use it |
|---|---|---|
| `ROADMAP.md` | Canonical priority source | Current priorities, sequencing, dependencies, and roadmap-level definition of done. |
| `ISSUES.md` | Canonical navigation | Maps roadmap work packages to executable GitHub issues and dependency order. GitHub issue state remains authoritative for execution. |
| `TO_DO.md` | Mandatory summary checklist | Compact hard-gate checklist plus active queue; subordinate to `.claude` and `dev/AGENTS.md`, not a weaker alternative. |
| `panel_framework_proposal.md` | Active design reference | Shared panel architecture and Tier-1 diagnostics proposal. Validate details against current code before implementation. |
| `plan_survival.md` | Active module reference | Cox Phase 1 status and Survival Phase 2+ scope; last materially updated 2026-07-12. |
| `cran_r_package_mapping.md` | Comparative reference | Method-family gap map. Some individual rows may lag current implementation. |
| `plan_anova.md` | Historical research plan | Its early implementation-status header is stale; use implemented-methods/model docs for current ANOVA support. |
| `plan_covariance.md` | Historical research plan | Its early implementation-status header is stale; current covariance estimators are documented elsewhere. |
| `plan_krr.md` | Historical research plan | Nystroem, KernelPCA, and chi-square kernel status in the old checklist is stale. |
| `plan_spline.md` | Historical research plan | SplineTransformer, cyclic, and thin-plate status in the old checklist is stale. |
| `plan_unsupervised.md` | Historical phase record | Useful for benchmark and algorithm history, not the current priority queue. |
| `plan.md` | Historical delta | Superseded by `ROADMAP.md`. |
| `archive/` | Archive | Completed or superseded planning material. |

## Planning rules

A roadmap item becomes executable only after it has a GitHub issue that defines:

- user or developer problem;
- scope and explicit non-goals;
- public API and failure behavior;
- NumPy, CuPy, and Torch backend contract;
- direct-fit/CV closure for every tunable capability;
- inference and formula implications where applicable;
- external baselines and normalization/alignment settings;
- unit, regression, compatibility, and physical-GPU validation;
- documentation and benchmark deliverables;
- dependencies and completion criteria.

Do not mark a module complete using only an implementation count or a passing CPU smoke test. Completion is contract-based, evidence-based, and subject to the hard workflow gates.
