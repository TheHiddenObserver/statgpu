# Changelog

All notable changes to statgpu are documented here, organized by release and date.

## Unreleased — maintenance hardening

- Fixed Issue #45 by routing statgpu-owned Torch compilation through a
  centralized policy that avoids CUDA Graph lifecycle hazards for iterative
  solvers; compile decisions are observable, and only the known lifecycle
  failure falls back to eager execution. Performance comparison with
  `reduce-overhead` remains explicitly deferred.
- Addressed Issue #81 with backend-native finite-value validation at public
  estimator boundaries without full GPU-array transfers.
- Aligned formula sample weights after missing-row filtering across linear,
  GLM, and penalized estimators; retained Torch/CuPy weights on device; and
  corrected Gaussian GLM FISTA to use weighted centering and the intended
  weighted squared-loss intercept.
- Unified analytic-weight GLM semantics across IRLS ridge scaling, line search,
  pseudo-loglikelihood, AIC/BIC, dispersion, and sandwich inference; centralized
  active GLM Torch compilation; narrowed singular-system fallbacks; and added
  backend-native response-domain validation for every supported GLM family,
  including penalized estimators and cross-validation entrypoints; scalar
  GLMs now normalize single-column responses and reject multicolumn or
  length-mismatched responses before solver/fold dispatch.
- Addressed Issue #82 by preserving exact raw constructor arguments for
  legacy scikit-learn clone identity while retaining normalized runtime
  attributes and `set_params` bookkeeping.
- Addressed Issue #83 by making maintained `test_*.py` files visible to git,
  documenting the manual GPU diagnostic boundary, and adding maintained
  regression coverage.
## 0.2.3 — 2026-08-04

### Added
- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed-entry and `(start, stop]` counting-process data; shared-coefficient stratification; subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for Cox objectives, gradients, information matrices, and baseline estimation, including backend-native dynamic programming for Exact ties.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods, delayed entry, start-stop rows, strata, and subject-grouped folds.

### Changed
- Hardened Cox inference, numerical stability, formula NA alignment, singular-information handling, CV cache identity, fold eligibility, selected-penalty refitting, and failed-fit state resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed the unidentified intercept; corrected Cox-specific warm starts; and made Torch Efron value, gradient, and Hessian paths native.
- Standardized public Group Lasso and Adaptive Group Lasso behavior through the generic loss-gradient and exact group-proximal path across supported backends.
- Made requested CoxPHCV two-stage and successive-halving controls execute one explicit exhaustive full-precision candidate pass, avoiding repeated complete-grid fitting while preserving deterministic selection semantics.
- Made one-shot `CoxPHCV.cv_splits` iterators reusable across repeated fit, scikit-learn clone, parameter reconstruction, and pickle.

### Validation
- Hosted workflow #960 passed on the final reviewed head `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU, and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported 1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published at https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4. It records schema 3, 134/134 passing checks, zero child and nested return codes, empty gate-failure arrays, clean source state before and after execution, and SHA-256 `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- Added release-package validation that checks version consistency, builds the pure-Python wheel and sdist, runs `twine check`, validates artifact contents, clean-installs the sdist on Ubuntu, and clean-installs the same wheel on Ubuntu, Windows, and macOS.

### Packaging and release publication
- Bumped the package version to `0.2.3` in `pyproject.toml` and `statgpu/__init__.py`.
- The official wheel remains a universal `py3-none-any` artifact built with `STATGPU_NO_EXT=1`; optional Cython sources remain available in the sdist.
- Added the authoritative GitHub Release document at `.github/releases/v0.2.3.md`, a release-note completeness gate, and tag automation that publishes that file as the GitHub Release body after the PyPI job succeeds.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
