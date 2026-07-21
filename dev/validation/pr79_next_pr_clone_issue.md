# PR79 Follow-up: sklearn Clone Compatibility Refactor

## Status

Tracked as strict xfail in `dev/tests/test_second_full_review.py`:

```python
@pytest.mark.xfail(
    condition=_SKLEARN_LT_13,
    reason="Pre-existing sklearn<=1.2 clone incompatibility",
    strict=True,
)
def test_all_default_public_estimators_clone(): ...
```

## Root Cause

sklearn 1.0–1.2 uses legacy clone protocol: `get_params(deep=False)` → deepcopy →
`__init__(**params)` → identity check (constructor's output must be `is` the input).

26 estimators canonicalize or copy public parameters in `__init__`:

| Pattern | Example | Affected Params |
|---------|---------|-----------------|
| `str(x).lower()` | `self.cov_type = str(cov_type).lower()` | cov_type, solver, criterion, direction |
| `bool(x)` | `self.verbose = bool(verbose)` | verbose |
| `dict(x)` | `self.model_kwargs = dict(model_kwargs)` | model_kwargs, options |
| `list(x)` | `self.alphas = list(alphas)` | alphas |
| Validation returns new string | `self.kernel = validate_kernel(kernel)` | kernel, method, metric, link |

sklearn 1.3+ supports `__sklearn_clone__` which bypasses this check.

## Base SHA Verification

Both `a4879fb4` (base) and `e30cec67` (head) fail with identical error:
`AssertionError: assert ['LogisticReg...ov_type', ...] == []`

Confirmed pre-existing, not a PR79 regression.

## Recommended Fix (next PR)

### 1. Shared `normalize_choice` helper

```python
def normalize_choice(value, *, name, choices, aliases=None):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if aliases is not None:
        normalized = aliases.get(normalized, normalized)
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return normalized
```

### 2. Pattern for each estimator

```python
class SomeEstimator:
    def __init__(self, ..., cov_type="nonrobust", solver="auto", ...):
        # Public params: preserve exactly
        self.cov_type = cov_type
        self.solver = solver

        # Private normalized: for internal use
        self._cov_type = normalize_choice(cov_type, name="cov_type",
                                          choices={"nonrobust", "robust", "clustered", "hac"})
        self._solver = normalize_choice(solver, name="solver",
                                        choices={"auto", "fista", "fista_bb", "newton", "irls", "exact"})

    def _validate_params(self):
        self._cov_type = normalize_choice(self.cov_type, ...)
        self._solver = normalize_choice(self.solver, ...)

    def set_params(self, **params):
        super().set_params(**params)
        self._validate_params()
        return self
```

All internal logic uses `self._cov_type`, `self._solver`, etc.
`get_params()` returns the original public values.

### 3. Affected estimators (26)

LogisticRegression, Lasso, Ridge, ElasticNet, SCAD, MCP, AdaptiveLasso,
QuantileRegression, RobustRegression, PenalizedGLM variants,
GraphicalLasso, GraphicalLassoCV, KernelRidge, KernelRidgeCV,
Nystroem, SplineTransformer, CoxPH, PooledOLS, FamaMacBeth,
RandomEffects, FixedEffects, FirstDifferenceOLS, BetweenOLS,
StepwiseSelector (fixed in this PR), and others.

## When sklearn >= 1.3

The `strict=True` xfail will cause XPASS (test fail) when sklearn is upgraded,
forcing removal of the xfail marker. If `__sklearn_clone__` is correct,
the test should pass naturally on sklearn >= 1.3.
