"""Accepted-iterate stopping for quadratic FISTA, without backend substitution.

The explicitly named simulated_cupy case only exercises GPU control flow on
NumPy arrays. It is not physical-GPU evidence. CUDA cases require actual device
arrays, while torch_cpu covers the same tensor operations without CUDA.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.glm_core import SquaredErrorLoss
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    SelectivePenalty,
)
from statgpu.penalties import L2Penalty
from statgpu.solvers import fista_solver
from statgpu.solvers._convergence import ConvergenceWarning


@pytest.fixture(params=["numpy", "simulated_cupy", "torch_cpu", "cupy_cuda", "torch_cuda"])
def quadratic_backend(request, monkeypatch):
    kind = request.param
    if kind == "simulated_cupy":
        import statgpu.solvers._fista as module

        # Isolate the control-flow regression from optional accelerator
        # dependencies. All numerical operations deliberately remain NumPy.
        monkeypatch.setattr(module, "_resolve_backend", lambda *args: "cupy")
        monkeypatch.setattr(module, "_get_xp", lambda backend: np)
        monkeypatch.setattr(
            module, "_as_backend_vector",
            lambda values, backend, ref: np.asarray(values, dtype=ref.dtype),
        )
        monkeypatch.setattr(
            module, "_zeros",
            lambda n, backend, ref_tensor=None: np.zeros(n, dtype=ref_tensor.dtype),
        )
        monkeypatch.setattr(
            module, "_sync_scalars",
            lambda *values, backend: tuple(float(value) for value in values),
        )
        original_proximal = SelectivePenalty.proximal
        monkeypatch.setattr(
            SelectivePenalty, "proximal",
            lambda self, w, step, backend=None: original_proximal(
                self, w, step, backend="numpy"
            ),
        )
        return SimpleNamespace(kind=kind, backend="cupy", asarray=np.asarray)
    if kind == "numpy":
        return SimpleNamespace(kind=kind, backend="numpy", asarray=np.asarray)
    if kind == "cupy_cuda":
        request.getfixturevalue("cupy_available")
        import cupy as cp

        return SimpleNamespace(kind=kind, backend="cupy", asarray=cp.asarray)
    if kind == "torch_cuda":
        request.getfixturevalue("torch_cuda_available")
    torch = pytest.importorskip("torch")
    device = "cuda" if kind == "torch_cuda" else "cpu"
    return SimpleNamespace(
        kind=kind, backend="torch",
        asarray=lambda value: torch.as_tensor(value, dtype=torch.float64, device=device),
    )


def _assert_backend(coef, design, context):
    assert type(coef) is type(design)
    assert coef.dtype == design.dtype
    if context.kind == "simulated_cupy":
        # A simulated control-flow test must never be represented as CUDA.
        assert isinstance(coef, np.ndarray)
        assert _resolve_backend("auto", coef) == "numpy"
        return
    assert _resolve_backend("auto", coef) == context.backend
    if context.backend == "torch":
        assert coef.device == design.device
        assert coef.is_cuda == (context.kind == "torch_cuda")
    elif context.backend == "cupy":
        assert coef.device.id == design.device.id


def _problem(weighted, intercept):
    rng = np.random.default_rng(52168037)
    X = rng.normal(size=(48, 3))
    y = X @ np.array([0.8, -0.6, 0.4]) + 1.7 + rng.normal(scale=0.05, size=48)
    if intercept:
        X = np.column_stack((X, np.ones(len(X))))
    weights = np.linspace(0.3, 1.8, len(X)) if weighted else None
    return X, y, weights


def _penalty(name, n_features, intercept, backend):
    # Public penalty='none' is resolved to L2Penalty(alpha=0) by the estimator.
    base = L2Penalty(alpha=0.2 if name == "l2" else 0.0)
    if not intercept:
        return base
    selective = SelectivePenalty()
    selective.configure(base, p=n_features - 1, backend=backend)
    return selective


def _closed_form(X, y, weights, penalty, intercept):
    w = np.ones(len(X)) if weights is None else weights
    regularizer = np.eye(X.shape[1]) * (0.2 if penalty == "l2" else 0.0)
    if intercept:
        regularizer[-1, -1] = 0.0
    gram = X.T @ (w[:, None] * X) / w.sum()
    rhs = X.T @ (w * y) / w.sum()
    return np.linalg.solve(gram + regularizer, rhs), gram, rhs, regularizer


@pytest.mark.parametrize("penalty", ["l2", "none"])
@pytest.mark.parametrize("weighted,intercept", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("cv_mode", [False, True])
def test_smooth_quadratic_converges_to_weighted_closed_form(
    quadratic_backend, penalty, weighted, intercept, cv_mode
):
    context = quadratic_backend
    X, y, weights = _problem(weighted, intercept)
    expected, gram, rhs, regularizer = _closed_form(X, y, weights, penalty, intercept)
    design = context.asarray(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_solver(
            SquaredErrorLoss(), _penalty(penalty, X.shape[1], intercept, context.backend),
            design, context.asarray(y), sample_weight=weights,
            max_iter=3000, tol=1e-12, cv_mode=cv_mode,
        )
    _assert_backend(coef, design, context)
    actual = np.asarray(_to_numpy(coef))
    assert 1 <= n_iter < 3000
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)
    assert np.linalg.norm((gram + regularizer) @ actual - rhs, ord=np.inf) < 3e-6
    # The objective difference is exact for this aligned convex quadratic.
    error = actual - expected
    assert 0.5 * error @ (gram + regularizer) @ error < 1e-11


@pytest.mark.parametrize("penalty", ["l2", "none"])
def test_orthogonal_quadratic_stops_in_two_accepted_iterations(quadratic_backend, penalty):
    context = quadratic_backend
    X = np.array([[1., 1., 1.], [1., -1., 1.], [-1., 1., 1.], [-1., -1., 1.]])
    y = X @ np.array([0.5, -0.75, 0.25])
    expected, _, _, _ = _closed_form(X, y, None, penalty, intercept=True)
    design = context.asarray(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_solver(
            SquaredErrorLoss(), _penalty(penalty, X.shape[1], True, context.backend),
            design, context.asarray(y), max_iter=3000, tol=1e-9,
        )
    _assert_backend(coef, design, context)
    assert n_iter == 2
    np.testing.assert_allclose(_to_numpy(coef), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("penalty", ["l2", "none"])
def test_last_iteration_stationary_point_is_not_reported_as_exhaustion(
    quadratic_backend, penalty
):
    context = quadratic_backend
    X, y, weights = _problem(weighted=True, intercept=True)
    expected, _, _, _ = _closed_form(X, y, weights, penalty, intercept=True)
    design = context.asarray(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, n_iter = fista_solver(
            SquaredErrorLoss(), _penalty(penalty, X.shape[1], True, context.backend),
            design, context.asarray(y), sample_weight=weights,
            init_coef=context.asarray(expected), max_iter=1, tol=1e-12,
        )
    _assert_backend(coef, design, context)
    np.testing.assert_allclose(_to_numpy(coef), expected, rtol=0, atol=1e-14)
    assert n_iter == 1


@pytest.mark.parametrize("penalty", ["l2", "none"])
@pytest.mark.parametrize("max_iter", [0, 1])
def test_genuine_budget_exhaustion_still_warns(quadratic_backend, penalty, max_iter):
    context = quadratic_backend
    X, y, weights = _problem(weighted=True, intercept=True)
    design = context.asarray(X)
    with pytest.warns(ConvergenceWarning, match=f"did not converge within {max_iter} iterations") as caught:
        coef, n_iter = fista_solver(
            SquaredErrorLoss(), _penalty(penalty, X.shape[1], True, context.backend),
            design, context.asarray(y), sample_weight=weights,
            max_iter=max_iter, tol=1e-30,
        )
    assert len(caught) == 1
    assert n_iter == max_iter
    _assert_backend(coef, design, context)
    expected, gram, rhs, regularizer = _closed_form(X, y, weights, penalty, True)
    actual = np.asarray(_to_numpy(coef))
    assert np.linalg.norm((gram + regularizer) @ actual - rhs, ord=np.inf) > 1e-3
    assert np.linalg.norm(actual - expected) > 1e-3


def test_quadratic_backtracking_returns_only_accepted_iterates(quadratic_backend):
    context = quadratic_backend

    class RecordingPenalty(L2Penalty):
        def __init__(self):
            super().__init__(alpha=0)
            self.proposals = []

        def proximal(self, w, step, backend="numpy"):
            result = super().proximal(w, step, backend=backend)
            self.proposals.append(np.asarray(_to_numpy(result)).copy())
            return result

    X = np.diag(np.sqrt(2 * np.array([1.0, 100.0])))
    y = X @ np.ones(2)
    design = context.asarray(X)
    penalty = RecordingPenalty()
    with pytest.warns(ConvergenceWarning, match="did not converge within 1 iterations"):
        coef, n_iter = fista_solver(
            SquaredErrorLoss(), penalty, design, context.asarray(y),
            lipschitz_L=1, max_iter=1, tol=1e-30,
        )
    _assert_backend(coef, design, context)
    actual = np.asarray(_to_numpy(coef))
    assert len(penalty.proposals) > 1, "fixture must reject an oversized trial"
    assert n_iter == 1
    assert not np.array_equal(actual, penalty.proposals[0])
    np.testing.assert_array_equal(actual, penalty.proposals[-1])
    gradient = -X.T @ y / len(X)
    step = actual[0] / -gradient[0]
    q_initial = 0.5 * np.mean(y ** 2)
    q_actual = 0.5 * np.mean((y - X @ actual) ** 2)
    bound = q_initial + gradient @ actual + 0.5 / step * (actual @ actual)
    assert q_actual <= bound + 1e-12


def test_quadratic_line_search_failure_does_not_become_convergence(quadratic_backend):
    context = quadratic_backend

    class RejectEveryTrial(SquaredErrorLoss):
        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            return coef.sum() * 0.0, coef * 0.0 + 1.0

        def value(self, X, y, coef, sample_weight=None):
            return super().value(X, y, coef, sample_weight=sample_weight) + 1e30

    X = context.asarray(np.eye(2))
    initial = context.asarray(np.zeros(2))
    with pytest.warns(ConvergenceWarning, match="line search failed") as caught:
        coef, n_iter = fista_solver(
            RejectEveryTrial(), L2Penalty(alpha=0.1), X, context.asarray(np.ones(2)),
            init_coef=initial, lipschitz_L=1, max_iter=3, tol=1e10,
        )
    assert len(caught) == 1
    assert n_iter == 1
    _assert_backend(coef, X, context)
    np.testing.assert_array_equal(_to_numpy(coef), np.zeros(2))


@pytest.mark.parametrize("device,backend", [("cpu", "numpy"), ("cuda", "cupy"), ("torch", "torch")])
@pytest.mark.parametrize("penalty", ["l2", "none"])
@pytest.mark.parametrize("weighted,intercept", [(False, False), (True, True)])
def test_public_quadratic_fista_reports_actual_solver_and_backend(
    request, monkeypatch, device, backend, penalty, weighted, intercept
):
    if device == "cuda":
        request.getfixturevalue("cupy_available")
    elif device == "torch":
        request.getfixturevalue("torch_cuda_available")
    import statgpu.solvers as solvers

    calls = []
    original_fista = solvers.fista_solver

    def recording_fista(loss, regularizer, design, response, **kwargs):
        actual_backend = _resolve_backend("auto", design)
        assert actual_backend == backend
        if backend == "torch":
            assert design.is_cuda
        result = original_fista(loss, regularizer, design, response, **kwargs)
        context = SimpleNamespace(
            kind={"cpu": "numpy", "cuda": "cupy_cuda", "torch": "torch_cuda"}[device],
            backend=backend,
        )
        _assert_backend(result[0], design, context)
        calls.append(actual_backend)
        return result

    monkeypatch.setattr(solvers, "fista_solver", recording_fista)
    X, y, weights = _problem(weighted, intercept=False)
    expected_design = np.column_stack((X, np.ones(len(X)))) if intercept else X
    expected, _, _, _ = _closed_form(expected_design, y, weights, penalty, intercept)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty=penalty, alpha=0.2,
            solver="fista", device=device, fit_intercept=intercept,
            max_iter=3000, tol=1e-12,
        ).fit(X, y, sample_weight=weights)
    if backend == "numpy":
        # Gaussian CPU has its own maintained FISTA loop. Do not force it
        # through the generic accelerator solver to make parity look simpler.
        assert not calls
    else:
        assert calls and set(calls) == {backend}
    assert model._selected_solver == "fista"
    assert model._selected_backend_name == backend
    if device != "cpu":
        assert str(model._selected_backend_device).startswith("cuda")
    actual = np.asarray(model.coef_)
    if intercept:
        actual = np.r_[actual, model.intercept_]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6)
    assert 1 <= model.n_iter_ < 3000


def test_cuda_array_guard_rejects_numpy_substitution():
    values = np.zeros(2)
    with pytest.raises(AssertionError):
        _assert_backend(values, values, SimpleNamespace(kind="cupy_cuda", backend="cupy"))


def test_cuda_array_guard_rejects_torch_cpu_substitution():
    torch = pytest.importorskip("torch")
    values = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(AssertionError):
        _assert_backend(values, values, SimpleNamespace(kind="torch_cuda", backend="torch"))
