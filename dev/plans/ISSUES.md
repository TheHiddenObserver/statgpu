# statgpu Roadmap Issue Index

> Last synchronized: **2026-08-28**  
> Baseline release: **0.2.5**  
> Baseline commit: `84f8bc7e17f66466b3a325cbb007b6cb41843821`

This file maps the canonical roadmap to executable GitHub issues. GitHub issue state is authoritative for execution; repository hard development gates remain authoritative for completion.

Completed benchmark/dashboard setup issues #90, #91, #92, and #100 are intentionally absent from the active table.

## Active issues

| Priority | Issue | Work package | Dependencies / sequencing |
|---|---:|---|---|
| P0 | #93 | Panel Tier-1 issue/evidence closure audit | Stage A/B/C implementation delivered through the 0.2.5 line; distinguish original implementation evidence from the separate PR #128 final-validator provenance caveat |
| P1 | #127 | Migrate legacy Gaussian linear-model inference to backend-native execution | Follow-up from #126; use `gaussian_inference_backend_native_plan.md` |
| P1 | #105 | Systematic linear/GLM inference benchmark and validation coverage | Sequence after #127 so canonical evidence measures the repaired inference contract |
| P1 | #108 | Extend canonical Panel estimator/covariance coverage | Evidence successor to released Panel Tier-1 capability; use fresh or contract-matched provenance rather than automatic reuse of disputed final-release validator artifacts |
| P2 | #94 | Implement Kaplan-Meier and Nelson-Aalen estimators | Independent feature lane |
| P2 | #95 | Implement initial Weibull/log-normal/log-logistic AFT family | Prefer after #94 unless isolated resources justify parallel work |
| P2 | #96 | Implement unpenalized multinomial logistic regression Phase 1 | Non-tunable base contract; prerequisite for #98 |
| P2 | #98 | Implement complete penalized multinomial suite with direct-fit/CV closure | Requires stable #96; sparse-input expansion remains blocked on #97 |
| P2 | #97 | Define shared sparse-array/backend contract with no silent densification | Prerequisite for HDFE, mixed models, sparse multinomial follow-up, and broader sparse support |
| P3 | #101 | Canonical Distribution benchmark coverage | Benchmark breadth |
| P3 | #102 | Robust Huber/Bisquare/Fair backend benchmark matrix | Benchmark breadth |
| P3 | #103 | Feature Selection/Knockoff benchmark coverage | Benchmark breadth |
| P3 | #104 | Ordered/ANOVA synchronized crossover benchmark coverage | Benchmark breadth |
| P3 | #106 | Covariance/Nonparametric canonical benchmark matrices | Benchmark breadth |
| P3 | #107 | Penalized CoxPH/CoxPHCV canonical benchmark coverage | Benchmark breadth |
| P3 | #109 | Multiple-testing/resampling/unsupervised operation coverage | Benchmark breadth |
| P3 | #114 | Dashboard bundle and collapsed chart-table optimization | Measurement-first, non-blocking hardening |
| P3 | #117 | Clarify benchmark input/working dtype provenance | LOW-severity provenance hardening |
| P3 | #118 | Measure and bound GPU CV path-buffer memory | LOW-severity performance hardening |

## Recommended sequencing

### Correctness and inference lane

```text
post-0.2.5 rebaseline
        |
        +--> #93 closure/evidence audit
        |
        +--> #127 --> #105
```

#127 owns the implementation repair. #105 should not establish a new canonical inference baseline before #127 stabilizes the maintained backend-native execution contract.

#93 is not a prerequisite for #127 implementation unless its audit demonstrates a concrete missing production acceptance criterion. Stale unchecked boxes alone are not such evidence.

The #93 audit must keep two evidence questions separate:

1. whether Stage A/B/C production implementation satisfied its original acceptance contract; and
2. whether the later PR #128 release artifacts prove the final maintained physical-validator contract.

The known answer to the second question is currently no: `results/pr126_release_697de113/` predates the final PR #128 runner-contract changes. That provenance limitation must not be silently converted into either a claim that the Panel numerical implementation is defective or a claim that the final validator contract was physically rerun.

### Panel evidence lane

```text
released Panel Tier-1 / v0.2.5 --> #108
```

#108 expands canonical evidence for already maintained Panel estimators/covariances. It must not duplicate #93 implementation work and must use fresh or source/validator-contract-matched evidence when promoting canonical acceptance claims.

### Survival lane

```text
#94 --> #95
```

### Multinomial and sparse lanes

```text
#96 --> #98
#97 --> future sparse estimator issues
```

#96 is strictly unpenalized/non-tunable. #98 owns the complete declared penalized matrix and may not close after only L2 or direct-fit support. Every tunable penalty exposed by #98 must ship with CV selection and final refit in the same declared capability package.

#97 defines shared sparse semantics; estimator-specific sparse adapters should not precede that contract.

### Benchmark breadth and hardening

#101-#104, #106, #107, and #109 are valid evidence packages but are subordinate to active correctness work unless they uncover a correctness defect.

#114, #117, and #118 remain bounded non-blocking hardening under their issue-specific measurement/provenance contracts unless new evidence raises severity.

## Issue maintenance rules

- Keep one primary issue per statistical/product contract.
- Split an issue only when doing so does not create a partially advertised capability or violate direct-fit/CV closure.
- Add explicit dependency links when one issue genuinely blocks another.
- Update `ROADMAP.md`, `TO_DO.md`, and this file when priorities or release baselines change.
- Roadmap/issue scope may narrow work but may not weaken `.claude` or `dev/AGENTS.md` hard gates.
- Close issues only with merged implementation evidence, required CI, external alignment where applicable, physical-GPU evidence where active, and synchronized documentation.
- Do not keep a released capability in the active implementation queue solely because historical checkboxes or plan text were not updated after merge/release.
- Do not promote historical evidence generated under an older validator contract as proof of a newer acceptance contract.
- Do not close an issue solely because a class/function/parser/frontend control exists.
