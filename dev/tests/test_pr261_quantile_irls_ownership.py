"""Characterize the direct IRLS boundary before changing its source owner."""

import importlib
import inspect
import pickle
from pathlib import Path
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.losses import _quantile_irls_validation_contract as legacy
from statgpu.penalties import L2Penalty
from statgpu.solvers._convergence import ConvergenceWarning


def _data(dtype=np.float64):
    X = np.asarray([[1, 1], [2, 1], [4, 1], [7, 1]], dtype=dtype)
    y = np.asarray([2, 0, 5, 3], dtype=dtype)
    return X, y


def test_irls_public_signature_pickle_and_unwrapped_kernel():
    public = QuantileLoss.irls
    kernel = public.__wrapped__
    assert inspect.signature(public) == inspect.signature(kernel)
    assert str(inspect.signature(public)) == (
        "(self, X, y, penalty=None, max_iter=100, tol=1e-06, "
        "init_coef=None, eps=1e-08, sample_weight=None, fit_intercept=False)"
    )
    assert public.__name__ == "irls"
    assert public.__qualname__ == "QuantileLoss.irls"
    assert public.__module__ == "statgpu.losses._quantile"
    assert public._statgpu_original is kernel
    assert "IRLS (Iteratively Reweighted Least Squares)" in inspect.getdoc(public)
    assert pickle.loads(pickle.dumps(public)) is public
    loss = pickle.loads(pickle.dumps(QuantileLoss(quantile=0.35)))
    bound = pickle.loads(pickle.dumps(loss.irls))
    assert bound.__func__ is public
    assert bound.__self__.quantile == loss.quantile


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("penalized", [False, True])
def test_irls_matches_unwrapped_kernel_without_mutating_inputs(dtype, weighted, penalized):
    X, y = _data(dtype)
    init = np.array([0.1, 0.2], dtype=np.float64)
    original = [a.copy() for a in (X, y, init)]
    kwargs = dict(
        penalty=L2Penalty(alpha=0.03) if penalized else None,
        max_iter=np.int64(3), tol=np.float64(1e-12),
        init_coef=init, eps=np.float64(1e-8),
        sample_weight=np.array([0, 1, 2, 3]) if weighted else None,
        fit_intercept=np.bool_(True),
    )
    loss = QuantileLoss(quantile=0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        actual, iterations = loss.irls(X, y, **kwargs)
        expected, expected_iterations = QuantileLoss.irls.__wrapped__(loss, X, y, **kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    assert iterations == expected_iterations == 3
    for value, before in zip((X, y, init), original):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fit_intercept": 1, "max_iter": 0}, "fit_intercept must be boolean"),
        ({"max_iter": 0, "tol": 0}, "max_iter must be a positive integer"),
        ({"tol": 0, "eps": 0}, "tol must be a finite positive number"),
        ({"eps": 0, "X": []}, "eps must be a finite positive number"),
        ({"X": [], "init_coef": []}, "X must be two-dimensional for Quantile IRLS"),
        ({"init_coef": [], "penalty": types.SimpleNamespace(name="l2", alpha=-1)},
         "init_coef must have length n_features"),
        ({"penalty": types.SimpleNamespace(name="l2", alpha=-1), "sample_weight": []},
         "L2 penalty alpha must be a finite non-negative real number"),
        ({"penalty": types.SimpleNamespace(name="l1"), "sample_weight": []},
         "sample_weight must have length n_samples"),
    ],
)
def test_irls_preserves_validation_precedence(kwargs, message):
    X, y = _data()
    arguments = {"X": X, "y": y, **kwargs}
    with pytest.raises(ValueError) as exc:
        QuantileLoss().irls(**arguments)
    assert str(exc.value) == message


def test_irls_budget_warning_points_to_the_external_call():
    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        call_line = inspect.currentframe().f_lineno + 1
        QuantileLoss(quantile=0.35).irls(X, y, max_iter=1, tol=1e-20)
    assert len(caught) == 1
    assert caught[0].category is ConvergenceWarning
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line
    assert str(caught[0].message) == (
        "QuantileLoss.irls() did not converge within 1 iterations; returning the final iterate."
    )


def test_irls_legacy_reload_preserves_function_and_kernel_identity():
    before = QuantileLoss.irls
    kernel = before.__wrapped__
    assert legacy.install_quantile_irls_validation_contract() is None
    importlib.reload(legacy)
    assert QuantileLoss.irls is before
    assert QuantileLoss.irls.__wrapped__ is kernel
    assert getattr(before, legacy._MARKER)
    assert legacy._validate_quantile_xy_shapes(*_data()) == 4


def test_legacy_installer_does_not_replace_unmarked_methods(monkeypatch):
    def replacement(self, X, y):
        return X, y

    monkeypatch.setattr(QuantileLoss, "irls", replacement)
    assert legacy.install_quantile_irls_validation_contract() is None
    importlib.reload(legacy)
    assert QuantileLoss.irls is replacement


@pytest.mark.parametrize("first", [
    "statgpu.losses._quantile_irls_validation_contract",
    "statgpu.losses._quantile", "statgpu.glm_core", "statgpu.solvers",
])
def test_irls_fresh_import_orders(first):
    code = f"""
import importlib
importlib.import_module({first!r})
from statgpu.losses import QuantileLoss
from statgpu.losses._quantile import QuantileLoss as canonical
from statgpu.losses._quantile_irls_validation_contract import install_quantile_irls_validation_contract
before = QuantileLoss.irls
assert QuantileLoss is canonical
assert install_quantile_irls_validation_contract() is None
assert QuantileLoss.irls is before
try:
    QuantileLoss().irls([[1]], [1], max_iter=False)
except ValueError as exc:
    assert str(exc) == 'max_iter must be a positive integer'
else:
    raise AssertionError('validation was not installed')
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
