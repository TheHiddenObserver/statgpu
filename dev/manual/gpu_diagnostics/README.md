# Manual GPU diagnostics

This directory is for intentionally ad-hoc GPU reproducers and
hardware-specific exploratory scripts. Files here are not collected by the
maintained pytest gate.

Maintained regression coverage belongs under `dev/tests/test_*.py` and must:

- expose discoverable pytest functions or classes;
- avoid substantial work at module import time;
- use explicit CuPy/Torch/CUDA availability checks and deterministic skip
  reasons;
- run from a clean checkout without ignored local fixtures;
- assert the current public inference and device contracts.

## Issue #83 legacy-script triage

The files reported in Issue #83 were ignored or unversioned local diagnostics,
so the original source is not available from a clean Git checkout. They must not
be recreated under `dev/tests/` as import-time scripts. The disposition of each
reported filename is recorded below.

| Historical filename | Classification | Disposition / maintained replacement |
|---|---|---|
| `test_coxph_3backends.py` | Unversioned diagnostic with an unclassified historical CuPy runtime failure | Retired as a test asset. Current Cox backend semantics are covered by `test_cox_phase1_completion.py`, `test_survival_risk_sets.py`, and the maintained physical-GPU suite. Because the original reproducer is unavailable, no claim is made that its exact failure was pre-existing or fixed; a recovered reproducer must be run on the Issue #83 base and current head before classification. |
| `test_irls_gpu.py` | Test-harness defect: undeclared `loss_name` fixture | Not promoted. Maintained IRLS/loss backend tests provide explicit parametrization and skip conditions. |
| `test_lasso_cv_torch_quick.py` | Import-time script; zero pytest tests | Retired. Lasso/CV contracts remain in maintained Lasso, CV, backend, and physical-GPU tests. |
| `test_ridge_cv_torch_backend.py` | Import-time script; zero pytest tests | Retired. RidgeCV and cross-backend behavior remain in maintained Ridge/CV regression tests. |
| `test_torch_comprehensive.py` | Broad import-time diagnostic; zero pytest tests | Replaced by focused, discoverable backend and model regression tests. New hardware-specific exploration belongs in this directory. |
| `test_lassocv_inference_simple.py` | Obsolete expectation against an older inference API | Retired. Current strict inference and unsupported-combination behavior is asserted by maintained Lasso inference tests. |

The Issue #45 Torch-CUDA lifecycle regression now has a discoverable physical
GPU test in `dev/tests/test_maintenance_024_025.py`. It skips deterministically
unless PyTorch is at least 2.1, CUDA is available, and the device has compute
capability 7 or newer.

## Adding a new diagnostic

A manual script should state its environment, expected command, and whether it
is exploratory or a minimized reproducer. Once a behavior becomes a supported
contract or regression, move the smallest deterministic assertion into
`dev/tests/` and leave the manual script only when it remains useful for
hardware investigation.


## Torch compile performance note

The maintenance release prioritizes correctness by defaulting iterative kernels
to Torch `default` compile mode. No claim is made that this matches the
steady-state latency of `reduce-overhead`; representative Lasso, ElasticNet,
nonconvex, adaptive, and group-penalty benchmarks remain an optimization task.
Users may opt into `reduce-overhead` explicitly, and construction/runtime
fallback decisions remain available through `get_torch_compile_diagnostics()`.
