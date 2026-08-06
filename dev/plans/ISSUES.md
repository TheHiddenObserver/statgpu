# statgpu Roadmap Issue Index

> Last synchronized: **2026-08-06**  
> Roadmap PR: **#89**  
> Baseline release: **0.2.4**

This file maps the canonical roadmap to executable GitHub issues. GitHub issue state is authoritative for execution; repository hard development gates remain authoritative for completion.

## Active issues

| Priority | Issue | Work package | Dependencies |
|---|---:|---|---|
| P0 | #90 | Synchronize benchmark dashboard PR #76 with the 0.2.4 `master` baseline | PR #89 planning reference |
| P1 | #91 | Add a canonical cross-validation benchmark source and dashboard coverage | #90 |
| P1 | #92 | Complete dashboard production QA, cross-browser smoke, accessibility, and documentation integration | #90; preferably #91 before final QA |
| P1 | #93 | Complete Panel Tier-1 shared framework, diagnostics, fit statistics, and covariance support | Independent of dashboard lane |
| P2 | #94 | Implement Kaplan-Meier and Nelson-Aalen estimators | Independent; shares survival result design with future work |
| P2 | #95 | Implement initial Weibull, log-normal, and log-logistic AFT family | May proceed independently; sequence after #94 unless resources justify parallel work |
| P2 | #96 | Design and implement unpenalized multinomial logistic regression Phase 1 | Non-tunable base contract; prerequisite for #98 |
| P2 | #98 | Implement the complete penalized multinomial suite with direct-fit/CV closure | #96; sparse input remains blocked on #97 |
| P2 | #97 | Define the shared sparse-array/backend contract with no silent densification | Prerequisite for HDFE, mixed models, sparse multinomial follow-up, and broad sparse estimator support |

## Recommended sequencing

### Product and benchmark lane

```text
#90 → #91 → #92 → propose PR #76 for master integration
```

Do not add new benchmark families during #90. Do not perform final dashboard QA on a stale or unsynchronized branch.

### Statistical workflow lane

```text
#93
#94 → #95
#96 → #98
#97 → future sparse estimator issues
```

#96 is strictly unpenalized and non-tunable. #98 owns the complete penalized multinomial matrix and may not close after only L2 or only direct-fit support. Every tunable penalty exposed by #98 must ship with CV selection and final refit in the same work package.

These issues may proceed in parallel only when they do not compete for the same backend, inference, solver, or review surface.

## Issue maintenance rules

- Keep one primary issue per statistical or product contract.
- Split an issue only when doing so does not produce a partially advertised capability or violate direct-fit/CV closure.
- Add explicit links when an issue blocks or is blocked by another issue.
- Update `ROADMAP.md`, `TO_DO.md`, and this file when priorities change.
- Roadmap and issue scope may narrow work but may not weaken `.claude` or `dev/AGENTS.md` hard gates.
- Close issues only with merged implementation evidence, required CI, external alignment where applicable, physical-GPU validation, and synchronized documentation.
- Do not close an issue solely because a class, function, parser, or frontend control exists.
