---
name: code-review
description: Comprehensive code review of statgpu modules - bugs, readability, maintainability, extensibility, performance
---

# Code Review Skill

When the user invokes this skill, perform a thorough code review of the specified files across 5 dimensions.

## Input

The user provides:
- File paths or module name to review
- Focus areas (optional): bugs, readability, maintainability, extensibility, performance

## Review checklist

### 1. Correctness (Bug check)
- [ ] Formulas match reference (paper, sklearn, R)
- [ ] Edge cases handled (empty input, single observation, singular matrix)
- [ ] Degrees of freedom correct
- [ ] Numerical stability (division by zero, overflow, underflow)
- [ ] Backend consistency: GPU input -> GPU output, no silent numpy conversion
- [ ] `hasattr(x, 'is_cuda')` for torch detection (not `hasattr(x, 'device')`)
- [ ] API contract: methods match docstrings, `fit()` sets `_fitted = True`
- [ ] Error handling: clear error messages, no silent failures

### 2. Readability
- [ ] No magic numbers (use named constants)
- [ ] Docstrings complete (Parameters, Returns, Raises, Examples)
- [ ] Comments explain "why", not "what"
- [ ] Consistent naming conventions (functions/variables/classes follow project conventions)
- [ ] Clear code structure (reasonable function length, single responsibility)
- [ ] Consistent indentation and formatting

### 3. Maintainability
- [ ] No code duplication (extract shared functions/base classes)
- [ ] Changing one place does not require changing multiple places (DRY principle)
- [ ] Error handling does not rely on implicit behavior
- [ ] Hardcoded values extracted as constants or parameters
- [ ] Clear dependencies (no circular imports, no unnecessary dependencies)
- [ ] Backward compatibility: existing APIs not broken

### 4. Extensibility
- [ ] New parameters do not affect existing calls (default values compatible)
- [ ] Appropriate abstraction level (base classes/interfaces reusable)
- [ ] Registry/factory patterns for easy extension (e.g., kernel registry, covariance registry)
- [ ] Hook/callback points well designed (e.g., `_transform_data`, `_estimate`)
- [ ] Configuration can be externalized (no hardcoded devices, thresholds, etc.)

### 5. Performance
- [ ] No unnecessary n x n temporary arrays
- [ ] In-place operations to reduce memory allocation
- [ ] Python loops vectorized (use numpy/cupy operations instead)
- [ ] Float32 for large matrices to reduce memory bandwidth
- [ ] Chunked computation to avoid OOM
- [ ] Caching to avoid recomputation (e.g., KPCA caching training kernel statistics)
- [ ] Small matrices avoid GPU kernel launch overhead

## Output format

For each file, report:
- **CLEAN** if no issues found
- **severity | dimension | file:line | description | fix** for each issue

Severity levels:
- **CRITICAL**: Wrong results, crashes, data loss
- **HIGH**: Silent incorrect behavior, API violations
- **MEDIUM**: Performance issues, code quality, missing edge cases
- **LOW**: Style, documentation, minor improvements

Dimensions:
- `BUG` -- correctness issue
- `READ` -- readability issue
- `MAINT` -- maintainability issue
- `EXT` -- extensibility issue
- `PERF` -- performance issue

## Review loop

After the initial review:
1. Fix all CRITICAL and HIGH issues
2. Run tests: `pytest dev/tests/ -q`
3. Re-review the fixed files
4. Repeat until all files are CLEAN

## Example output

```
CRITICAL | BUG  | _graphical_lasso.py:146 | Precision off-diagonal not divided by Schur complement | theta_j[not_j] = -beta / c
HIGH     | MAINT | _shrinkage.py:230     | OAS denominator doesn't match sklearn | Revert to (n+1)
MEDIUM   | PERF  | _kpca.py:184          | Training kernel recomputed every transform() | Cache during fit()
LOW      | READ  | _welch.py:122         | df_within rounded to int, loses precision | Change to float
```
