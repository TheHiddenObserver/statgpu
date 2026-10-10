# PR168 third complete documentation/runtime-contract review

## Exact target and change boundary

Original reviewed head: `c6bd229b62aff7c82cec92df9f2b57667e810ece`.
Comparison base: `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.

The review starts from a clean source export verified against immutable Git blobs: all 527 initially materialized files matched, including all 102 original full-PR paths. The whole 1,433-file remote inventory was retained for unrelated-file preservation. Root `CLAUDE.md`, `dev/AGENTS.md`, `dev/DOCUMENTATION_STYLE.md`, and all three `.claude/skills/code-review` documents were read. The previous review's clean verdict was not reused.

Classification: documentation/public-contract reconciliation. Active axes include public/runtime API, formulas, CV/inference/resampling, backend/output ownership, fitted-state behavior, Chinese idiom, learner usability, and evidence placement. Directly linked kernel/spline, multiple-testing and generic-inference pages were included when their claims contradicted the reviewed user journeys. This does not introduce new numerical capabilities.

All production executable ASTs remain identical to the comparison base after removing only initial module/class/function docstrings. The edits consist of documentation, public docstrings and regression tests. No numerical algorithm, dependency, merge, draft/ready status or default branch change is included.

## Documentation corrections

- Rewrote the directly linked multiple-testing learner/API pages in both languages: individual decisions versus a global null, FWER/FDR definitions, all five adjustment formulas, dependence assumptions, Fisher/Stouffer calibration and Cauchy tail-approximation limits. Corrected Fisher weight rejection, alias positional order, legacy `fdr_hochberg` semantics, shapes/backend selection and endpoint/tail limits. Moved unverifiable-current timing claims into a clearly labeled historical developer record.
- Made single-class Logistic evaluation failure explicit: `include_curves=False` still calculates ROC AUC. Pointed readers to threshold-only metrics, with executable coverage. Clarified custom-CV split validation and post-selection public-provenance limits in both linear references.
- Added a complete knockoff CPU learner workflow and explained input-lifetime hazards in seeded Lasso statistic/tuning caches. A new selector is not sufficient isolation. A fresh process is the general safe isolation; retained immutable float64 NumPy copies are a narrower tested workflow.
- Added MiniBatchNMF representative-initialization/restart guidance for features absent from the first streaming batch. Explained DBSCAN's independent GPU float32/expanded-distance translation risk and a training-derived float64 centering precaution. Corrected GMM covariance assumptions and remaining Chinese prose fragments.
- Corrected the kernel-ridge RKHS penalty, Nystroem decomposition/shape/backend details, complete kernel/spline APIs, natural-spline dimensions and the explicit NumPy `xp=None` default. Removed the false suggestion that GAM supplies spline inference. Described unavailable weighted KRR, nonfinite CV selection and unreliable cyclic boundary constraints honestly.
- Added safe numeric resampling-label validation, explained absolute-statistic two-sided permutation comparisons, corrected “inverse precision” to inverse covariance/Gram, and made the generic post-selection example self-contained. Distinguished correct result-owned post-selection method metadata from absent shared public provenance fields.

## Independently reproduced implementation follow-ups

All 94 open/closed issues through #209 were read before filing, with targeted searches and explicit distinctions from related reports. Each candidate was executed again at the coordinating review level before registration. Exact erroneous uninitialized-memory values are illustrative, not regression expectations.

- [#210](https://github.com/TheHiddenObserver/statgpu/issues/210): NaN multiple-testing significance levels return apparently valid all-false decisions.
- [#211](https://github.com/TheHiddenObserver/statgpu/issues/211): seeded knockoff Lasso caches reuse stale statistics after input values change; both native and sklearn implementations reproduced, with a cold-process baseline.
- [#212](https://github.com/TheHiddenObserver/statgpu/issues/212): MiniBatchNMF initial zero dictionary columns cannot recover when later streaming data become positive.
- [#213](https://github.com/TheHiddenObserver/statgpu/issues/213): independently implemented DBSCAN GPU distance paths lose radius neighborhoods under large offsets; distinct from shared-helper #205.
- [#214](https://github.com/TheHiddenObserver/statgpu/issues/214): missing resampling group labels omit rows and leave uninitialized batch entries/indices, allowing invalid permutation p-values.
- [#215](https://github.com/TheHiddenObserver/statgpu/issues/215): the specialized post-selection OLS path omits shared public method/target provenance, despite correct result-owned method identity.
- [#216](https://github.com/TheHiddenObserver/statgpu/issues/216): cyclic cubic spline basis fails analytic periodic value/derivative constraints.
- [#217](https://github.com/TheHiddenObserver/statgpu/issues/217): KernelRidge silently ignores supplied sample weights.
- [#218](https://github.com/TheHiddenObserver/statgpu/issues/218): KernelRidgeCV can choose a nonfinite zero-alpha candidate over a finite alternative.
- [#219](https://github.com/TheHiddenObserver/statgpu/issues/219): KernelRidge and KernelRidgeCV omit constructor gamma for chi-squared kernels, unlike other kernel consumers.
- [#220](https://github.com/TheHiddenObserver/statgpu/issues/220): the independent kernel-method RBF distance implementation loses translation invariance and corrupts downstream predictions.
- [#221](https://github.com/TheHiddenObserver/statgpu/issues/221): UMAP exact search uses a finite diagonal sentinel, allowing self-neighbors and missing graph edges at large finite scales.
- [#222](https://github.com/TheHiddenObserver/statgpu/issues/222): the non-NumPy chi-squared kernel replaces small positive denominators with an absolute floor, changing valid similarities across backends.

These issues request separate bounded production work. Documentation cautions and safer preparation do not repair the underlying algorithms. Missing post-selection public fields are an API consistency issue; no incorrect numerical method execution is alleged there. DBSCAN evidence exercises production routines with NumPy/Torch CPU and explicitly does not claim a physical GPU fit.

## Additional independent whole-candidate corrections

The fresh whole-candidate pass corrected inherited prediction help that confused the selected prediction backend with NumPy coefficient storage, and removed a misleading fixed-X/model-X “performance versus precision” explanation. It completed the reader-facing kernel constructor/method/fitted-result inventory and corrected the repeated-knot B-spline recurrence convention. New kernel cautions include a tested `kernel_params` gamma workaround, consistent float64 centering for RBF geometry, and the small-feature chi-squared backend discrepancy. Natural-spline descriptions consistently qualify the numerical approximation and constraint rank; KernelPCA/Nystroem parameter changes explicitly require refitting retained state. The fixed-penalty GLM inference guide distinguishes the HC0 formula from the conditional HC1 finite-sample multiplier. UMAP guidance now distinguishes additive-offset cancellation from a separate finite self-distance sentinel; its tested common-scale preparation preserves the Euclidean metric up to a uniform factor.

The four additional issue reproductions were executed independently by the domain reviewer, the complete-candidate reviewer and the coordinating review. Live open/closed searches were checked before filing #219–#222. None of these corrections changes production executable code.

## Tests and evidence

The new ordinary tests cover all bilingual snippets, p-value analytic/external baselines, callable aliases, label validation and alignment, current-data cache isolation, streaming restart, centered geometry, all GMM covariance structures, spline dimensions, RKHS identities, positive-grid finite checks, and one-class threshold metrics. Existing complete API, Chinese-language and all twelve unsupervised model example tests are rerun.

New strict expected-failure cases assert intended correct behavior for three post-selection-provenance consumers, the three initial kernel/spline implementation gaps, chi-squared constructor routing across both aliases and direct/CV consumers, RBF translation invariance, small positive chi-squared denominators on Torch CPU, and UMAP distinct-neighbor selection. They are not correctness passes. The prior two F-sampling and nine optional Torch small-shape failures remain separate. Tests do not require an arbitrary wrong runtime result to persist.

Local evidence uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and optional Torch 2.14.1+cpu. Fresh bytecode directories are used for final runs so source filenames identify this candidate rather than a copied export. No physical CuPy/Torch CUDA, R or performance validation is performed. Skips and CPU tests cannot establish those capabilities.

The completed candidate undergoes a separate full-range independent documentation review. Final complete-path fingerprint, byte-preservation inventory, local counts and terminal hosted CI for the published head are recorded in the PR description after verification; this source record does not pre-claim later results. Earlier CI and review records remain historical after the head changes.
