"""Characterize penalized Cox CV validation and compatibility ownership."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest

from statgpu.linear_model import PenalizedCoxPHModel, PenalizedGLM_CV
from statgpu.linear_model.penalized import _penalized_cox_cv as cv_impl
from statgpu.linear_model.penalized import _penalized_cox_public_contract as compat


@pytest.mark.parametrize("penalty", ["l1", "l2", "elasticnet", "scad", "mcp", "none"])
@pytest.mark.parametrize("grid", [[0.3, 0.1], (0.2, 0.2), np.array([1, 2]), np.array([0.1, 0.4], dtype=np.float32)])
def test_strict_alpha_grid_preserves_order_dtype_and_input(grid, penalty):
    before = np.asarray(grid).copy()
    result = cv_impl._validate_alpha_grid(grid, penalty)
    np.testing.assert_array_equal(result, np.asarray(grid, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(grid), before)
    assert result.dtype == np.float64
    assert inspect.signature(cv_impl._validate_alpha_grid) == inspect.signature(compat._validate_alpha_grid_strict)


@pytest.mark.parametrize("penalty", ["l1", "scad"])
@pytest.mark.parametrize("grid, message", [
    ([True, -1.0], "not booleans"),
    (["0.1", -1.0], "not strings or bytes"),
    ([b"0.1", -1.0], "not strings or bytes"),
    ([0.1 + 0j, -1.0], "real numeric"),
    (np.array([True, False]), "not booleans"),
    (np.array([1 + 0j]), "real numeric"),
    ([], "non-empty one-dimensional"),
    ([[0.1]], "non-empty one-dimensional"),
    (0.1, "non-empty one-dimensional"),
    ([np.nan, -1.0], "finite non-negative"),
    ([np.inf, 0.1], "finite non-negative"),
    ([-1.0, 0.0], "finite non-negative"),
])
def test_strict_alpha_grid_error_precedence(grid, message, penalty):
    with pytest.raises(ValueError, match=message):
        cv_impl._validate_alpha_grid(grid, penalty)


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_nonconvex_zero_rejection_is_after_numeric_validation(penalty):
    with pytest.raises(ValueError, match="strictly positive"):
        cv_impl._validate_alpha_grid([0.0, 0.1], penalty)
    np.testing.assert_array_equal(cv_impl._validate_alpha_grid([0.0, 0.1], "l2"), [0.0, 0.1])


def test_original_validator_keeps_unchecked_coercion_link():
    original = compat._ORIGINAL_VALIDATE_ALPHA_GRID
    assert cv_impl._validate_alpha_grid.__wrapped__ is original
    np.testing.assert_array_equal(original([True, "0.2"], "l1"), [1.0, 0.2])
    with pytest.raises(ValueError, match="booleans"):
        cv_impl._validate_alpha_grid([True, "0.2"], "l1")


def test_historical_validator_callable_pickle_restores_canonical_entry():
    assert pickle.loads(pickle.dumps(compat._validate_alpha_grid_strict)) is cv_impl._validate_alpha_grid
    assert pickle.loads(b"cstatgpu.linear_model.penalized._penalized_cox_public_contract\n_validate_alpha_grid_strict\n.") is cv_impl._validate_alpha_grid


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_penalized_cox_fit_refit_pickle_and_cv_consumer(dtype):
    rng = np.random.default_rng(261)
    X = rng.normal(size=(18, 2)).astype(dtype)
    y = np.column_stack([np.linspace(1, 18, 18), np.tile([1., 0., 1.], 6)]).astype(dtype)
    model = PenalizedCoxPHModel(alpha=0.2, device="cpu", max_iter=100)
    assert model.fit(X, y) is model
    before = model.predict(X)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X), before)
    assert model.fit(X, y) is model
    np.testing.assert_allclose(model.predict(X), before, rtol=1e-12, atol=1e-12)
    cv = PenalizedGLM_CV(loss="cox_ph", penalty="l2", alpha_grid=[0.2, 0.1], cv=2,
                         random_state=0, device="cpu", max_iter=100)
    assert cv.fit(X, y) is cv
    assert isinstance(cv.estimator_, PenalizedCoxPHModel)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(cv)).predict(X), cv.predict(X))


def test_canonical_owners_do_not_require_compatibility_module():
    import subprocess
    import sys
    code = """
import sys
from statgpu.linear_model import PenalizedCoxPHModel, PenalizedGLM_CV
from statgpu.linear_model.penalized import _penalized_cox_cv as owner
assert 'statgpu.linear_model.penalized._penalized_cox_public_contract' not in sys.modules
try:
    owner._validate_alpha_grid([True, 0.1], 'l1')
except ValueError as exc:
    assert 'booleans' in str(exc)
else:
    raise AssertionError('strict validation was not owned by canonical helper')
assert 'estimation-only' in PenalizedCoxPHModel.__doc__
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_compatibility_reload_has_no_owner_mutations():
    import importlib
    before = cv_impl._validate_alpha_grid
    document = PenalizedCoxPHModel.__doc__
    for _ in range(3):
        importlib.reload(compat)
        assert cv_impl._validate_alpha_grid is before
        assert compat._validate_alpha_grid_strict is before
        assert PenalizedCoxPHModel.__doc__ is document


def test_class_source_owns_the_current_runtime_docstring():
    import ast
    from pathlib import Path
    source = Path(inspect.getsourcefile(PenalizedCoxPHModel)).read_text()
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "PenalizedCoxPHModel")
    assert ast.get_docstring(definition, clean=False) == PenalizedCoxPHModel.__doc__
