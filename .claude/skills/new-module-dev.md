---
name: new-module-dev
description: Develop a new statgpu module with three-backend support, tests, docs, and benchmark
---

# New Module Development Skill

When the user invokes this skill with a module description, execute the full development workflow.

## Input

The user provides:
- Module name and statistical method
- Target location in the codebase (e.g., `statgpu/anova/`, `statgpu/panel/`)
- Reference implementation (sklearn class name, R function, or paper)

## Workflow

### Step 1: Understand existing code

Read the target module's existing files, `AGENTS.md`, and identify:
- Existing patterns (backend dispatch, BaseEstimator usage, test structure)
- Reusable components (inference distributions, xp_* helpers, PanelSummary, etc.)
- External comparison targets (sklearn, scipy, statsmodels)

### Step 2: Implement

For each new file:
1. Follow the three-backend pattern from `.claude/workflows/new-module-dev.md`
2. Use `xp_*` helpers from `backends/_utils.py` for device-aware operations
3. Use `get_distribution()` from `inference/_distributions_backend` for p-values
4. Inherit from `BaseEstimator` for class-based models
5. Update `__init__.py` exports

### Step 3: Test

Create `dev/tests/test_<module>.py` with:
- Basic smoke tests
- External framework comparison (assert_allclose with sklearn/scipy)
- Three-backend consistency tests
- Edge case tests
- Error handling tests

Run: `pytest dev/tests/test_<module>.py -v`

### Step 4: Review loop

Do a code review focusing on:
- Correctness (formulas, edge cases)
- Backend consistency (GPU in → GPU out)
- Performance (unnecessary allocations, missed vectorization)
- Readability (naming, docstrings)

Fix issues, re-run tests, repeat until clean.

### Step 5: Document

- Update `docs/en/models/<module>.md`
- Update `CHANGELOG.md`, `docs/en/changelog.md`, `docs/cn/changelog.md`

### Step 6: Benchmark

Create `dev/tests/bench_<module>.py` with three-backend timing at n=5000/20000/50000.
Upload to remote server and execute.

### Step 7: Commit and PR

```bash
git checkout -b feature/<module>
git add -A && git commit -m "feat(<module>): <description>"
git push -u origin feature/<module>
gh pr create --title "feat: <module>" --body "..."
```

## Key patterns to follow

### Function-based API (like f_oneway)
```python
def my_func(*args, backend="auto", dtype=None):
    resolved = _resolve_backend(backend, *args)
    xp = _get_xp(resolved)
    # ... computation with xp.* ...
    return result
```

### Class-based API (like EmpiricalCovariance)
```python
class MyEstimator(BaseEstimator):
    def __init__(self, params, device=Device.AUTO):
        super().__init__(device=device)
        self.params = params

    def fit(self, X, y=None):
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        # ... computation ...
        self._fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        # ... prediction ...
```

### External comparison test
```python
def test_vs_sklearn():
    from sklearn.xxx import Xxx as SkXxx
    X = np.random.randn(100, 10)
    ours = OurXxx(params).fit(X)
    sk = SkXxx(params).fit(X)
    assert_allclose(ours.result_, sk.result_, rtol=1e-6)
```

### Three-backend test
```python
def test_three_backends():
    X = np.random.randn(100, 10)
    r_np = our_func(X, backend="numpy")
    try:
        import cupy as cp
        r_cp = our_func(cp.asarray(X), backend="cupy")
        assert_allclose(r_np, _to_numpy(r_cp), rtol=1e-10)
    except ImportError:
        pass
    try:
        import torch
        r_t = our_func(torch.from_numpy(X).cuda().double(), backend="torch")
        assert_allclose(r_np, _to_numpy(r_t), rtol=1e-10)
    except (ImportError, RuntimeError):
        pass
```
