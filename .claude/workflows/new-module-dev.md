# New Module Development Workflow

> Reusable workflow for adding new statistical modules to statgpu.
> Invoke with: `/new-module-dev <module_description>`

## Overview

This workflow implements the full lifecycle of a new statgpu module:
1. Design (API, backend strategy, external comparison targets)
2. Implement (three-backend: numpy/cupy/torch)
3. Test (unit tests + external framework comparison)
4. Review (bug hunt, readability, performance)
5. Document (model docs, changelog)
6. Benchmark (remote server, 3-backend timing)
7. Release (version bump, PyPI)

## Prerequisites

- Remote GPU server access configured in memory (hz-4.matpool.com)
- `scipy`, `sklearn`, `statsmodels` available for external comparison
- `cupy`, `torch` available on remote server for GPU testing

## Phase 1: Design

Before writing code, clarify:

1. **What statistical method?** (e.g., "Welch ANOVA", "GraphicalLasso")
2. **What's the sklearn/R equivalent?** (for API design and comparison)
3. **What's the formula?** (reference paper or textbook)
4. **Which backends?** (numpy always; cupy/torch if GPU benefits expected)
5. **What parameters?** (match sklearn's API where possible)

Check existing code for reusable components:
- `statgpu/backends/_utils.py` — `xp_*` helpers, `_to_numpy`, `_to_float_scalar`
- `statgpu/inference/_distributions_backend.py` — `get_distribution()` for p-values
- `statgpu/_base.py` — `BaseEstimator` for class-based models
- Module-specific utils (e.g., `panel/_utils.py`, `covariance/_empirical.py`)

## Phase 2: Implement

### File structure
```
statgpu/<module>/
├── __init__.py          # Export new symbols
├── _new_feature.py      # New implementation
└── _existing.py         # Existing code (may need minor updates)
```

### Three-backend pattern

```python
from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy

# For function-based APIs:
def my_function(*args, backend="auto", dtype=None):
    resolved = _resolve_backend(backend, *args)
    xp = _get_xp(resolved)
    # ... use xp.* for all array operations ...
    result = _to_float_scalar(some_scalar)  # sync to CPU at end

# For class-based APIs:
class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        # ... use xp.* operations ...
        self._fitted = True
        return self
```

### Device consistency rules
- GPU input → GPU output (don't auto-convert to numpy)
- Use `hasattr(x, 'is_cuda')` for torch detection (not `hasattr(x, 'device')`)
- Use `xp_asarray` (not `xp.asarray`) for device-aware conversion
- Use `xp_zeros`, `xp_eye`, `xp_asarray` from `backends/_utils.py`

### Update exports
- `statgpu/<module>/__init__.py` — add new symbols
- `statgpu/__init__.py` — add top-level exports

## Phase 3: Test

### Test file: `dev/tests/test_<module>.py`

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestNewFeature:
    def test_basic(self):
        """Smoke test: does it run and return correct shape?"""
        ...

    def test_vs_external(self):
        """Precision comparison with sklearn/scipy/statsmodels."""
        from sklearn.xxx import Xxx as SkXxx
        ours = OurXxx(params).fit(X)
        sk = SkXxx(params).fit(X)
        assert_allclose(ours.result_, sk.result_, rtol=1e-6)

    def test_three_backend_consistency(self):
        """numpy/cupy/torch produce same results."""
        r_np = our_func(X, backend="numpy")
        r_cp = our_func(cp.asarray(X), backend="cupy")
        r_t = our_func(torch.from_numpy(X).cuda().double(), backend="torch")
        assert_allclose(r_np, _to_np(r_cp), rtol=1e-10)
        assert_allclose(r_np, _to_np(r_t), rtol=1e-10)

    def test_edge_cases(self):
        """Empty input, single observation, singular matrix, etc."""
        ...

    def test_error_handling(self):
        """Invalid parameters raise appropriate errors."""
        ...
```

### Run tests
```bash
pytest dev/tests/test_<module>.py -v
```

## Phase 4: Review

Run the review loop until clean:

1. **Correctness**: Are formulas right? Edge cases handled?
2. **Backend consistency**: Same backend in/out? No silent conversions?
3. **API contract**: Methods match docstrings? Error messages clear?
4. **Performance**: Unnecessary allocations? Missed vectorization?
5. **Readability**: Naming, comments, code structure

Fix issues found, re-run tests, repeat until CLEAN.

## Phase 5: Document

### Model docs: `docs/en/models/<module>.md`
Follow existing format (Overview, Path, Objective Function, Parameters, Examples, FAQ, References).

### Changelog (3 files):
- `CHANGELOG.md` — PR-level summary (1-5 lines)
- `docs/en/changelog.md` — English detailed (Added/Optimized/Validation)
- `docs/cn/changelog.md` — Chinese detailed

## Phase 6: Benchmark (with R comparison)

### Script: `dev/tests/bench_<module>.py`

```python
# 3-backend comparison at multiple scales
for n in [5000, 20000, 50000]:
    # numpy
    _, t_np = timeit(lambda: our_func(X, backend="numpy"))
    # cupy
    X_c = cp.asarray(X)
    _, t_cp = timeit(lambda: our_func(X_c, backend="cupy"))
    # torch
    X_t = torch.from_numpy(X).cuda().double()
    _, t_t = timeit(lambda: our_func(X_t, backend="torch"))
    # sklearn/scipy
    _, t_sk = timeit(lambda: sk_func(X))
    print(f"n={n}: numpy={t_np:.0f}ms cupy={t_cp:.0f}ms torch={t_t:.0f}ms sklearn={t_sk:.0f}ms")

# R comparison (via rpy2)
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    r_result = ro.r('aov(y ~ factor(group))')
    # ... extract F-statistic and p-value ...
except ImportError:
    print("rpy2 not available, skipping R comparison")
```

### Remote execution
```bash
# Upload and run on remote server
scp -P 28838 dev/tests/bench_<module>.py root@hz-4.matpool.com:/root/statgpu/dev/tests/
ssh -p 28838 root@hz-4.matpool.com "cd /root/statgpu && conda activate myconda && python dev/tests/bench_<module>.py"
```

## Phase 7: Commit and PR (NO AUTO-MERGE)

```bash
# 1. Create feature branch
git checkout -b feature/<module>

# 2. Commit with conventional commit message
git add -A && git commit -m "feat(<module>): <description>"

# 3. Push and create PR
git push -u origin feature/<module>
gh pr create --title "feat: <module>" --body "..."

# DO NOT merge - wait for user approval
```

**IMPORTANT**: Never merge PRs or push to master without explicit user approval.

## Phase 8: Release (only after user approval)

```bash
# 1. Merge PR (after user approval)
git checkout master && git merge feature/<module>

# 2. Bump version
# Edit statgpu/__init__.py and pyproject.toml

# 3. Tag and push
git tag v<version> && git push origin master --tags

# 4. Build and upload to PyPI
python -m build
python -m twine upload dist/statgpu-<version>*
```

## Checklist

- [ ] Three-backend implementation (numpy/cupy/torch)
- [ ] Unit tests with external framework comparison
- [ ] Three-backend consistency tests
- [ ] Edge case and error handling tests
- [ ] Code review (bugs, readability, performance)
- [ ] Model documentation (docs/en/models/)
- [ ] Three changelog files updated
- [ ] Benchmark script with 3-backend timing
- [ ] Remote benchmark executed
- [ ] Version bumped, tagged, uploaded to PyPI
