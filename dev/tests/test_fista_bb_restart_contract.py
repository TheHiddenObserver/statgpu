"""Semantic regression tests for the gradient-evaluation point in BB restart.

The independent recurrence deliberately includes the historical wrong-point
variant as a negative control. GPU cases require real CUDA; never emulate them.
"""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

from statgpu.backends import _to_numpy
from statgpu.glm_core import (
    SquaredErrorLoss, LogisticLoss, NegativeBinomialLoss,
    InverseGaussianLoss, PoissonLoss, GammaLoss, TweedieLoss,
)
from statgpu.penalties import L1Penalty, L2Penalty
from statgpu.solvers import fista_bb_solver
from statgpu.solvers._convergence import ConvergenceWarning


def _solve(loss, penalty, X, y, **kwargs):
    # Fixed-budget trajectory tests intentionally do not stop early.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return fista_bb_solver(loss, penalty, X, y, **kwargs)


@pytest.fixture(params=['numpy', 'cupy', 'torch'])
def backend_arrays(request):
    if request.param == 'numpy':
        return np.asarray
    if request.param == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CuPy CUDA runtime is unavailable')
        if not count:
            pytest.skip('CuPy requires a physical CUDA device')
        return cp.asarray
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Torch requires a physical CUDA device')
    return lambda a: torch.as_tensor(a, device='cuda')


def _reference(h, target, steps, alpha=0, restart=True, wrong_point=False,
               cap=None, burn_in=0):
    x = np.zeros_like(target)
    y = x.copy()
    t = 1.
    trajectory, dots, resets = [], [], []
    for k in range(steps):
        trajectory.append(y.copy())
        previous = x.copy()
        z = y - h * (y - target)
        x = np.sign(z) * np.maximum(np.abs(z) - alpha, 0)
        if k < burn_in:
            y, t = x.copy(), 1.
            continue
        if cap is not None:
            y, t = x + cap * (x - previous), 1.
            continue
        t_next = (1 + math.sqrt(1 + 4*t*t))/2
        next_y = x + (t-1)/t_next * (x-previous)
        dot = (next_y-x if wrong_point else y-x) @ (x-previous)
        dots.append(dot)
        if restart and k > 0 and dot > 0:
            next_y, t_next = x.copy(), 1.
            resets.append(k)
        y, t = next_y, t_next
    return x, np.array(trajectory), np.array(dots), resets


class _RecordingSquared(SquaredErrorLoss):
    def __init__(self):
        self.evaluations = []

    def gradient(self, X, y, coef, sample_weight=None):
        self.evaluations.append(np.asarray(_to_numpy(coef)).copy())
        return super().gradient(X, y, coef, sample_weight=sample_weight)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('cv_mode', [False, True])
@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('alpha', [0., .001])
def test_quadratic_trajectory_matches_generalized_gradient_reference(
        backend_arrays, dtype, cv_mode, restart, alpha):
    h = np.array([1., .01])
    target = np.array([1., -1.])
    X = backend_arrays(np.diag(np.sqrt(2*h)).astype(dtype))
    y = backend_arrays((np.diag(np.sqrt(2*h)) @ target).astype(dtype))
    loss = _RecordingSquared()
    coef, n_iter = _solve(loss, L1Penalty(alpha=alpha), X, y,
                         max_iter=100, tol=0, lipschitz_L=1,
                         cv_mode=cv_mode, use_restart=restart)
    assert type(coef) is type(X)
    assert coef.dtype == X.dtype
    if hasattr(X, "device"):
        assert coef.device == X.device
    expected, trajectory, dots, resets = _reference(h, target, 100, alpha, restart)
    atol = 2e-6 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(_to_numpy(coef), expected, rtol=0, atol=atol)
    # Initialization evaluates one gradient before the 100 loop evaluations.
    np.testing.assert_allclose(loss.evaluations[1:], trajectory, rtol=0, atol=atol)
    assert n_iter == 100
    assert np.any(dots < 0) and np.any(dots > 0)
    if restart:
        wrong, wrong_trajectory, _, wrong_resets = _reference(
            h, target, 100, alpha, wrong_point=True)
        optimum = np.sign(target)*np.maximum(abs(target)-alpha/h, 0)
        assert np.linalg.norm(expected-optimum) < 1e-5
        assert np.linalg.norm(wrong-optimum) > .3
        assert len(resets) == 2 and len(wrong_resets) == 50
        assert not np.allclose(trajectory, wrong_trajectory)


@pytest.mark.parametrize('h', [[1., 1.], [1., .5]])
def test_zero_step_and_well_conditioned_optimum(h):
    h = np.array(h)
    target = np.array([1., -1.])
    X = np.diag(np.sqrt(2*h))
    expected, trajectory, _, _ = _reference(h, target, 12)
    loss = _RecordingSquared()
    coef, _ = _solve(loss, L2Penalty(alpha=0), X, X@target,
                     max_iter=12, tol=0, lipschitz_L=1)
    np.testing.assert_allclose(coef, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(loss.evaluations[1:], trajectory, rtol=0, atol=2e-15)
    zero_loss = _RecordingSquared()
    coef, n = _solve(zero_loss, L2Penalty(alpha=0), X, np.zeros(2),
                     max_iter=12, tol=1e-12, lipschitz_L=1)
    np.testing.assert_array_equal(coef, np.zeros(2))
    assert n == 1


@pytest.mark.parametrize('flags', [
    {'_skip_momentum': True}, {'_momentum_disabled': True},
    {'_skip_momentum': True, '_momentum_beta_cap': .5},
    {'_momentum_disabled': True, '_momentum_beta_cap': .5},
    {'_skip_momentum': True, '_poisson_like': True, '_momentum_beta_cap': .5},
    {'_momentum_disabled': True, '_poisson_like': True, '_momentum_beta_cap': .5},
    {'_momentum_beta_cap': .2},
    {'_gamma_like': True}, {'_tweedie': True},
])
def test_disabled_cap_and_burn_in_trajectories(flags):
    h = np.array([1., .01]); target = np.array([1., -1.])
    X = np.diag(np.sqrt(2*h)); loss = _RecordingSquared()
    for name, value in flags.items():
        setattr(loss, name, value)
    steps = 120
    disabled = flags.get('_skip_momentum') or flags.get('_momentum_disabled')
    burn_in = steps+1 if disabled else (100 if '_tweedie' in flags else
                                       30 if '_gamma_like' in flags else 0)
    cap = flags.get('_momentum_beta_cap',
                    .2 if '_gamma_like' in flags or '_tweedie' in flags else None)
    expected, trajectory, _, _ = _reference(h, target, steps, cap=cap, burn_in=burn_in)
    coef, _ = _solve(loss, L2Penalty(alpha=0), X, X@target,
                     max_iter=steps, tol=0, lipschitz_L=1)
    np.testing.assert_allclose(coef, expected, rtol=0, atol=2e-13)
    np.testing.assert_allclose(loss.evaluations[1:], trajectory, rtol=0, atol=2e-13)


@pytest.mark.parametrize('loss_cls', [InverseGaussianLoss, PoissonLoss, GammaLoss, TweedieLoss])
def test_actual_protected_families_are_restart_toggle_invariant(loss_cls):
    X = np.column_stack([np.ones(30), np.linspace(-.5, .5, 30)])
    y = np.exp(X @ np.array([.2, .3]))
    loss = loss_cls()
    kwargs = dict(max_iter=1000, tol=1e-11)
    enabled, n1 = _solve(loss, L2Penalty(alpha=0), X, y, use_restart=True, **kwargs)
    disabled, n2 = _solve(loss, L2Penalty(alpha=0), X, y, use_restart=False, **kwargs)
    np.testing.assert_array_equal(enabled, disabled)
    assert n1 == n2
    np.testing.assert_allclose(enabled, [.2, .3], rtol=0, atol=2e-6)
    assert np.linalg.norm(loss.gradient(X, y, enabled), ord=np.inf) < 1e-7


@pytest.mark.parametrize('loss_cls', [LogisticLoss, NegativeBinomialLoss])
def test_weighted_nonlinear_sparse_objective_matches_independent_baseline(loss_cls):
    rng = np.random.default_rng(6813)
    X = np.column_stack([np.ones(80), rng.normal(size=(80, 2))])
    eta = X @ np.array([.3, .7, -.5])
    y = (rng.random(80) < 1/(1+np.exp(-eta))).astype(float) if loss_cls is LogisticLoss else rng.poisson(np.exp(eta)).astype(float)
    weights = np.linspace(.5, 1.5, len(y))
    loss = loss_cls(); alpha = .01
    def objective(b):
        return loss.value(X, y, b, sample_weight=weights)+alpha*np.abs(b).sum()
    # Epigraph-free smooth reference valid once all components are nonzero;
    # verify that premise and the L1 KKT residual independently below.
    reference = minimize(objective, np.array([.2, .5, -.4]), method='Powell',
                         options={'xtol': 1e-12, 'ftol': 1e-12, 'maxiter': 5000})
    assert reference.success and np.min(np.abs(reference.x)) > .01
    coef, _ = _solve(loss, L1Penalty(alpha=alpha), X, y,
                     sample_weight=weights, max_iter=3000, tol=1e-11)
    assert objective(coef) <= objective(reference.x)+1e-9
    np.testing.assert_allclose(coef, reference.x, rtol=0, atol=2e-6)
    kkt = loss.gradient(X, y, coef, sample_weight=weights)+alpha*np.sign(coef)
    assert np.linalg.norm(kkt, ord=np.inf) < 1e-7


def test_restart_uses_accepted_backtracked_iterate(monkeypatch):
    import statgpu.solvers._fista_bb as bb

    class NonquadraticDispatch(_RecordingSquared):
        # Exercise BB's safeguarded backtracking with an exactly known loss.
        _is_quadratic = False
        _lipschitz_uses_y = True

    class RecordingPenalty(L2Penalty):
        def __init__(self):
            super().__init__(alpha=0)
            self.proposals = []

        def proximal(self, w, step, backend='numpy'):
            result = super().proximal(w, step, backend=backend)
            self.proposals.append(result.copy())
            return result

    loss = NonquadraticDispatch(); penalty = RecordingPenalty()
    real_update, real_dot = bb._nesterov_update, bb._dot_dev
    pending = []; seen = []; accepted = []
    def record_update(coef, old, t):
        # Nonquadratic BB just evaluated grad_new at accepted coef; the
        # preceding gradient is the evaluation point for this proximal step.
        evaluation_point = loss.evaluations[-2]
        pending.append((evaluation_point-coef, coef-old))
        accepted.append(coef.copy())
        return real_update(coef, old, t)
    def check_dot(a, b):
        if pending:
            expected_a, expected_b = pending.pop()
            # Iteration zero skips restart; discard its pending observation
            # below before any subsequent dot reductions occur.
            np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-13)
            np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-13)
            seen.append(float(a@b))
        return real_dot(a, b)
    def update_after_first(coef, old, t):
        result = record_update(coef, old, t)
        if len(accepted) == 1:
            pending.clear()
        return result
    monkeypatch.setattr(bb, '_nesterov_update', update_after_first)
    monkeypatch.setattr(bb, '_dot_dev', check_dot)
    X = np.diag(np.sqrt(2*np.array([1., 100.])))
    _solve(loss, penalty, X, X@np.ones(2), max_iter=8, tol=0,
           lipschitz_L=1, bb_burn_in=20)
    assert len(penalty.proposals) > len(accepted), 'fixture must backtrack'
    assert not np.array_equal(penalty.proposals[0], accepted[0])
    assert seen, 'fixture must execute the actual restart comparison'


def test_inverse_gaussian_never_calls_nesterov(monkeypatch):
    import statgpu.solvers._fista_bb as bb
    def forbidden(*args, **kwargs):
        pytest.fail('Inverse Gaussian explicitly disables momentum')
    monkeypatch.setattr(bb, '_nesterov_update', forbidden)
    X = np.column_stack([np.ones(20), np.linspace(-.5, .5, 20)])
    _solve(InverseGaussianLoss(), L2Penalty(alpha=0), X,
           np.exp(X@np.array([.2, .3])), max_iter=40, tol=0)


def test_logistic_smooth_dispatch_still_delegates_to_fista(monkeypatch):
    import statgpu.solvers._fista_bb as bb
    calls = []
    def delegated(*args, **kwargs):
        calls.append((args, kwargs))
        return np.array([.25]), 17
    monkeypatch.setattr(bb, 'fista_solver', delegated)
    coef, n = _solve(LogisticLoss(), L2Penalty(alpha=.1),
                     np.ones((4, 1)), np.array([0., 0., 1., 1.]))
    np.testing.assert_array_equal(coef, [.25])
    assert n == 17 and len(calls) == 1
