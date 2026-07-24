# Contributing to statgpu

Thank you for helping improve `statgpu`. Contributions are welcome in the form of bug reports, documentation, tests, benchmarks, statistical validation, performance work, and new methods.

This guide describes the public contribution workflow. For a deeper technical map of the repository and its backend invariants, see [`dev/AGENTS.md`](dev/AGENTS.md).

## Before starting

For a small bug fix or documentation correction, a pull request can be opened directly. For a new estimator, public API change, solver, penalty, inference method, or large refactor, open an issue first so that the statistical contract, backend coverage, and validation plan can be agreed before implementation.

A useful issue should include:

- the affected class, function, or module;
- a minimal reproducible example for bugs;
- expected and actual behavior;
- backend and device information (`cpu`, `cuda`, or `torch`);
- package versions and hardware details when the issue is GPU- or performance-related;
- references or external implementations when proposing a statistical method.

Do not include credentials, private datasets, API tokens, or remote-server configuration.

## Development setup

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/TheHiddenObserver/statgpu.git
cd statgpu
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,validation,formula]"
```

For a CuPy development environment, choose exactly one CUDA-major extra:

```bash
# CUDA 11.x
python -m pip install -e ".[dev,validation,formula,gpu11]"

# CUDA 12.x
python -m pip install -e ".[dev,validation,formula,gpu12]"
```

For the PyTorch backend:

```bash
python -m pip install -e ".[dev,validation,formula,torch]"
```

GPU contributors are responsible for using CuPy and PyTorch builds compatible with their installed CUDA driver/runtime.

## Project structure

The main public implementation lives under `statgpu/`:

- `statgpu/backends/`: NumPy, CuPy, and Torch backend abstractions;
- `statgpu/linear_model/`: regression, GLM, penalized, robust, quantile, and CV estimators;
- `statgpu/survival/`: Cox proportional-hazards estimators;
- `statgpu/panel/`: panel-data estimators and covariance routines;
- `statgpu/inference/`: distributions, resampling, multiple testing, and inference helpers;
- `statgpu/nonparametric/`, `statgpu/semiparametric/`, and `statgpu/unsupervised/`: additional method families;
- `dev/tests/`: primary test tree;
- `dev/benchmarks/`: reproducible performance and numerical-validation scripts;
- `docs/en/` and `docs/cn/`: English and Chinese documentation.

## Core implementation requirements

### Three-backend behavior

Changes to statistical or numerical methods should support NumPy, CuPy, and Torch unless the pull request is explicitly scoped otherwise and documents the limitation.

- `device="cpu"` must use NumPy.
- `device="cuda"` must use CuPy and must not silently fall back to CPU.
- `device="torch"` must use Torch CUDA and must not silently fall back to CPU or Torch CPU.
- `device="auto"` is the only mode allowed to select an available backend automatically.

Preserve input dtype/device where the public contract requires it. Avoid full-array GPU-to-CPU transfers in core numerical paths; scalar transfers for control flow or unsupported scalar distribution functions should be explicit and limited.

### Statistical contracts

A contribution should preserve or clearly define:

- estimator API (`fit`, `predict`, `score`, fitted attributes, cloning, and parameter semantics);
- objective and penalty normalization;
- convergence and stopping criteria;
- covariance, standard errors, test statistics, p-values, confidence intervals, likelihood, AIC/BIC, or other inference fields exposed by the estimator;
- rank-deficient and degenerate-data behavior;
- formula intercept, categorical, interaction, transformation, missing-data, and sample-alignment behavior when formula interfaces are involved.

If an estimator exposes inference, new backend paths should provide equivalent inference or fail explicitly with a documented error. Silent approximate inference is not acceptable.

### GPU memory and timing

Estimators that retain GPU caches should follow the repository's `gpu_memory_cleanup` pattern described in [`dev/AGENTS.md`](dev/AGENTS.md).

GPU benchmarks must synchronize before starting and after finishing a measured region:

```python
cp.cuda.Stream.null.synchronize()
torch.cuda.synchronize()
```

Report the hardware, data dimensions, dtype, warmup count, measured repetitions, and whether timing covers end-to-end fitting or only a kernel/solver region.

## Testing

Run focused tests first. Examples:

```bash
python -m pytest dev/tests/test_linear.py -q
python -m pytest dev/tests/test_cox.py -q
python -m pytest dev/tests/test_panel_formula.py -q
python -m pytest dev/tests/test_penalties_and_exports.py -q
```

Run the complete CPU test tree before requesting review when practical:

```bash
python -m pytest dev/tests -q --tb=short
```

Basic static checks:

```bash
python -m compileall -q statgpu dev/validation dev/benchmarks
```

For GPU changes, run relevant tests on physical hardware for both CuPy and Torch. Tests that require real GPU execution should use the repository's physical-GPU guard where applicable:

```bash
STATGPU_REQUIRE_PHYSICAL_GPU=1 \
python -m pytest <relevant-gpu-test-files> -q -rs --tb=short
```

If physical GPU tests cannot be run locally, state that clearly in the pull request. Do not present skipped GPU tests as validation evidence.

## Numerical and external validation

For statistical changes, include comparisons against a suitable independent reference whenever possible:

- scikit-learn for estimator and prediction behavior;
- statsmodels for regression and inference;
- established R packages for methods whose reference implementation is primarily in R;
- finite-difference derivatives, objective/KKT checks, or independent formulas for optimizer internals.

Use explicitly aligned features, observations, tie handling, solvers, tolerances, and regularization scaling. A difference caused by incompatible objective normalization should be resolved by an equivalent parameter mapping, not by changing the library's documented objective solely to match another package.

## Documentation and changelog

User-visible changes must update the relevant documentation. The default order is English first, followed by the corresponding Chinese documentation.

Depending on scope, update:

- `README.md` for project-level capabilities and entry points;
- `docs/en/` and `docs/cn/` for detailed user documentation;
- `CHANGELOG.md` for a concise pull-request-level summary;
- `docs/en/changelog.md` and `docs/cn/changelog.md` for release-level details.

Performance claims must include hardware, workload dimensions, numerical-accuracy evidence, and an auditable benchmark/result path.

## Pull request workflow

1. Create a focused branch from the current target branch.
2. Keep the change set narrow enough to review and validate.
3. Add or update tests before requesting review.
4. Update documentation and changelog entries when behavior is user-visible.
5. Describe the root cause, implementation, compatibility impact, and validation in the pull request.
6. List every relevant command that was run and distinguish passed, skipped, and not-run checks.
7. Address review feedback with targeted commits and rerun affected tests.

A pull request that changes a statistical method is generally not complete until implementation, three-backend behavior, tests, numerical validation, and documentation agree.

## Pull request checklist

- [ ] The public API and statistical behavior are documented.
- [ ] NumPy, CuPy, and Torch behavior is implemented or an explicit limitation is justified.
- [ ] Explicit GPU devices do not silently fall back to CPU.
- [ ] Focused tests pass.
- [ ] The full CPU suite was run, or the reason it was not run is stated.
- [ ] Relevant physical-GPU tests were run, or are clearly marked as not run.
- [ ] External/numerical validation is included for statistical changes.
- [ ] GPU timings include synchronization and complete benchmark metadata.
- [ ] README/docs/changelog files are synchronized where needed.
- [ ] No credentials, private data, generated caches, or unrelated result files are committed.

## Licensing

By submitting a contribution, you agree that it may be distributed under the repository's [Apache License 2.0](LICENSE).
