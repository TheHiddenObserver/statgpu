# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## 2026-07-11

### PR #79 — Full repository review and hardening

- Completed an iterative repository-wide review covering correctness, backend routing,
  statistical/API contracts, readability, maintainability, extensibility, performance
  risks, test quality, and compliance with `dev/AGENTS.md`.
- Fixed backend/device validation, sklearn-style estimator parameters, Torch inference
  routing, UMAP fuzzy-union and random-state semantics, NNDescent correctness, adaptive
  L1 and knockoff runtime errors, CV input contracts, KMeans/UMAP edge cases, Ridge
  penalty scaling, and Cox Efron observed-information orientation.
- Hardened tests so optional Torch/CuPy dependencies skip explicitly instead of failing
  collection or swallowing unexpected errors; moved the remote GPU runner outside the
  pytest test tree.
- Added focused review regression suites and permanent Python 3.9–3.12, full CPU,
  compilation, static-contract, and complete test-collection CI gates.
- Added `dev/reviews/pr79_full_repository_review.md` with accepted fixes, deferred
  architectural debt, and the physical-GPU validation plan.
- Validation status: `PARTIAL_REMOTE_PENDING`; CPU and contract gates pass, while
  physical CuPy/Torch CUDA numerical, memory, and performance validation remains required.

## 2026-07-08

### v0.2.1 — Packaging / PyPI release hygiene

- **Version bump** 0.2.0 → 0.2.1 (`pyproject.toml`, `statgpu/__init__.py`).
- **Pure-Python wheel policy**: the PyPI release workflow now sets `STATGPU_NO_EXT=1`,
  so the published wheel is tagged `py3-none-any` and installs on every OS / Python
  version. Previously `python -m build` compiled the optional Cython extensions during
  `bdist_wheel`, producing a platform-locked wheel (e.g. `cp311-linux_x86_64`) that
  served almost no one and forced everyone else onto the sdist.
- **setup.py**: added the `STATGPU_NO_EXT` switch — when set to `1`, `ext_modules` is
  empty (forces a pure-Python build). The Cython extensions remain optional CPU
  accelerators with pure-Python fallbacks; users who want the C speedups build them
  from the sdist, which still ships the `.pyx`/`.pxd` sources via `MANIFEST.in`.
- **publish.yml**: added `twine check dist/*` before upload.

### PR #74 — Ordered Newton-Raphson + Analytical Hessian Inference + Unified Sandwich Engine
- Ordered Logit/Probit: L-BFGS replaced with Newton-Raphson + trust-region (3-backend)
- Ordered inference: analytical Hessian, SE/z/p/CI, loglikelihood/aic/bic (CPU+GPU)
- Sandwich engine: m_estimation_inference, fisher_information, penalty curvature API
- Penalized inference: sandwich (L2/EN), oracle active-set (SCAD/MCP)
- QuantileRegression standalone class with kernel+bootstrap inference
- 28 bug fixes across 4 code review rounds; scipy→get_distribution; GPU guards
- Docs: ordered.md rewrite, v0.2.1 coverage matrix, solver-algorithms/quantile/robust
- Validated: R ordinal::clm, three-backend GPU (CuPy+Torch), 226 CI tests