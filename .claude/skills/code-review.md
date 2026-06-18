---
name: code-review
description: Comprehensive code review of statgpu modules - find and fix bugs, readability, performance issues
---

# Code Review Skill

When the user invokes this skill, perform a thorough code review of the specified files.

## Input

The user provides:
- File paths or module name to review
- Focus areas (optional): bugs, readability, performance, backend consistency

## Review checklist

For each file, check:

### Correctness
- [ ] Formulas match reference (paper, sklearn, R)
- [ ] Edge cases handled (empty input, single obs, singular matrix)
- [ ] Degrees of freedom correct
- [ ] Numerical stability (division by zero, overflow, underflow)

### Backend consistency
- [ ] Same backend in → same backend out
- [ ] No silent numpy conversions for GPU inputs
- [ ] `hasattr(x, 'is_cuda')` for torch detection (not `hasattr(x, 'device')`)
- [ ] `xp_asarray` used (not `xp.asarray`) for device-aware conversion
- [ ] `xp_zeros`, `xp_eye` used for device-aware allocation

### API contract
- [ ] Methods match docstrings
- [ ] Error messages are clear and helpful
- [ ] `get_params`/`set_params` work correctly
- [ ] `fit()` sets `_fitted = True`

### Performance
- [ ] No unnecessary n×n temporary arrays
- [ ] In-place operations where possible
- [ ] Python loops vectorized where possible
- [ ] Float32 used for large matrices when precision allows

### Readability
- [ ] No magic numbers (use named constants)
- [ ] Docstrings complete (Parameters, Returns, Raises)
- [ ] Comments explain "why", not "what"
- [ ] Consistent naming conventions

## Output format

For each file, report:
- **CLEAN** if no issues found
- **severity | file:line | description | fix** for each issue

Severity levels:
- **CRITICAL**: Wrong results, crashes, data loss
- **HIGH**: Silent incorrect behavior, API violations
- **MEDIUM**: Performance issues, code quality, missing edge cases
- **LOW**: Style, documentation, minor improvements

## Review loop

After the initial review:
1. Fix all CRITICAL and HIGH issues
2. Run tests: `pytest dev/tests/ -q`
3. Re-review the fixed files
4. Repeat until all files are CLEAN
