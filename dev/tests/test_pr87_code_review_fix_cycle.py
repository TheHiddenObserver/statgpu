from __future__ import annotations

import numpy as np
import pytest


def test_binomial_probit_irls_matches_direct_bernoulli_optimum():
    scipy = pytest.importorskip("scipy")
    from scipy.optimize import minimize
    from scipy.special import ndtr

    from statgpu.glm_core._family import Binomial, ProbitLink
    from statgpu.glm_core._irls import IRLSSolver

    X = np.column_stack([np.ones(10), np.linspace(-1.8, 1.8, 10)])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=float)
    family = Binomial(link=ProbitLink())
    params, _ = IRLSSolver(family, max_iter=100, tol=1e-9).fit(
        X, y, backend="numpy"
    )

    def objective(beta):
        mu = np.clip(ndtr(X @ beta), 1e-10, 1 - 1e-10)
        return float(np.sum(-y * np.log(mu) - (1 - y) * np.log(1 - mu)))

    reference = minimize(objective, np.zeros(X.shape[1]), method="BFGS")
    assert reference.success
    np.testing.assert_allclose(params, reference.x, rtol=2e-4, atol=2e-4)
    assert objective(params) <= objective(np.zeros(X.shape[1]))


def test_direct_logistic_rejects_soft_and_out_of_range_labels():
    from statgpu.linear_model import LogisticRegression

    X = np.arange(12, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match="binary y"):
        LogisticRegression(device="cpu").fit(
            X, np.array([0.0, 1.0, 0.5, 0.0, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match="binary y"):
        LogisticRegression(device="cpu").fit(
            X, np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0])
        )


def test_irls_numpy_warm_start_is_normalized_to_torch_backend():
    torch = pytest.importorskip("torch")
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    params, _ = IRLSSolver(Gaussian(), max_iter=5, tol=1e-12).fit(
        X, y, init_coef=np.zeros(2), backend="torch"
    )
    assert torch.is_tensor(params)
    assert params.device == X.device
    assert params.dtype == X.dtype
    np.testing.assert_allclose(params.detach().cpu().numpy(), [0.0, 1.0], atol=1e-10)


def test_irls_rejects_wrong_length_warm_start():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="init_coef"):
        IRLSSolver(Gaussian()).fit(
            np.ones((4, 2)), np.arange(4.0), init_coef=np.zeros(3)
        )


@pytest.mark.parametrize(
    "exc",
    [
        IndexError("index 5 is out of bounds for axis 0 with size 5"),
        TypeError("bad call signature"),
        AttributeError("missing coefficient state"),
        KeyError("scores"),
        AssertionError("unexpected path state"),
    ],
)
def test_cv_candidate_programming_errors_are_never_converted_to_nan(exc):
    from statgpu.linear_model.penalized._penalized_cv import (
        _raise_unless_recoverable_cv_candidate_failure,
    )

    with pytest.raises(type(exc)):
        _raise_unless_recoverable_cv_candidate_failure(exc)


def test_cv_candidate_numeric_failure_remains_explicitly_recoverable():
    from statgpu.linear_model.penalized._penalized_cv import (
        _raise_unless_recoverable_cv_candidate_failure,
    )

    _raise_unless_recoverable_cv_candidate_failure(
        np.linalg.LinAlgError("singular matrix")
    )
    _raise_unless_recoverable_cv_candidate_failure(
        FloatingPointError("non-finite iterate")
    )


def test_integer_weight_sum_uses_float64_accumulator():
    from statgpu.glm_core._validation import _safe_weight_sum

    weights = np.full(5, 2**62, dtype=np.int64)
    expected = float(5 * (2**62))
    assert _safe_weight_sum(weights) == pytest.approx(expected, rel=1e-15)


def test_reduce_overhead_repeated_cuda_calls_are_correct_or_visible_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a physical Torch CUDA backend")
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("torch.compile requires CUDA capability >= 7")

    from statgpu.backends._torch_compile import compile_torch

    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "reduce-overhead")

    def update(x):
        return x.square() + 1.0

    guarded = compile_torch(update, workload="iterative")
    x = torch.linspace(-2.0, 2.0, 128, device="cuda", dtype=torch.float64)
    for _ in range(8):
        result = guarded(x)
        torch.cuda.synchronize()
        assert torch.allclose(result, update(x))
    assert guarded.__statgpu_compile_status__ in {
        "compiled", "runtime-fallback"
    }


def test_integral_glm_weights_are_promoted_before_downstream_normalization():
    from statgpu.glm_core._validation import validate_glm_sample_weight

    raw = np.full(5, 2**62, dtype=np.int64)
    validated = validate_glm_sample_weight(raw, raw.size)
    assert validated.dtype == np.float64
    assert np.isfinite(validated.sum())
    assert validated.sum() == pytest.approx(float(5 * 2**62), rel=1e-15)


def test_integral_solver_weights_are_promoted_before_uniform_checks():
    from statgpu.solvers._utils import _validated_sample_weight

    raw = np.full(5, 2**62, dtype=np.int64)
    backend, _, validated = _validated_sample_weight(raw, raw.size)
    assert backend == "numpy"
    assert validated.dtype == np.float64
    assert np.isfinite(validated.sum())


def test_weighted_glm_objective_with_integral_weights_stays_finite():
    from statgpu.glm_core._logistic import LogisticLoss
    from statgpu.glm_core._validation import validate_glm_sample_weight

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    coef = np.array([-1.0, 0.5])
    weights = validate_glm_sample_weight(
        np.full(4, 2**62, dtype=np.int64), 4
    )
    value, gradient = LogisticLoss().fused_value_and_gradient(
        X, y, coef, sample_weight=weights
    )
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()


def test_torch_tensor_warm_start_does_not_use_copy_constructor_warning():
    import warnings

    torch = pytest.importorskip("torch")
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    init = torch.zeros(2, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        params, _ = IRLSSolver(Gaussian(), max_iter=5, tol=1e-12).fit(
            X, y, init_coef=init, backend="torch"
        )
    assert not any("copy construct from a tensor" in str(w.message) for w in captured)
    assert params.dtype == torch.float64


def _fitted_logistic_fixture():
    from statgpu.linear_model import LogisticRegression

    X = np.array(
        [[-2.0], [-1.0], [-0.25], [0.25], [1.0], [2.0]], dtype=float
    )
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    model = LogisticRegression(
        device="cpu", max_iter=100, compute_inference=False
    ).fit(X, y)
    return model, X, y


def _assert_logistic_state_cleared(model):
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.n_iter_ is None
    assert model._params is None
    assert model._X_design is None
    assert model._y is None
    assert model._accuracy is None
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((1, 1)))


def test_logistic_invalid_binary_refit_clears_stale_state():
    model, X, _ = _fitted_logistic_fixture()
    with pytest.raises(ValueError, match="binary y"):
        model.fit(X, np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))
    _assert_logistic_state_cleared(model)


def test_logistic_nonfinite_refit_clears_stale_state_before_shared_guard():
    model, X, y = _fitted_logistic_fixture()
    bad_y = y.copy()
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, bad_y)
    _assert_logistic_state_cleared(model)


@pytest.mark.parametrize(
    "bad_X, message",
    [
        (np.asarray(1.0), "two-dimensional design matrix"),
        (np.arange(6.0), "two-dimensional design matrix"),
        (np.empty((0, 1)), "at least one observation"),
    ],
)
def test_logistic_design_boundary_has_public_error_and_clears_state(
    bad_X, message
):
    model, _, _ = _fitted_logistic_fixture()
    with pytest.raises(ValueError, match=message):
        model.fit(bad_X, np.empty(0))
    _assert_logistic_state_cleared(model)


def test_base_fit_guard_prefers_general_transaction_hook():
    from statgpu._base import BaseEstimator

    class TransactionalEstimator(BaseEstimator):
        def __init__(self):
            super().__init__(device="cpu")
            self.reset_calls = 0
            self.body_calls = 0

        def _reset_fit_state(self):
            self.reset_calls += 1
            self._fitted = False

        def fit(self, X, y=None):
            self.body_calls += 1
            self._fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    model = TransactionalEstimator()
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[np.nan]]), np.array([0.0]))
    assert model.reset_calls == 1
    assert model.body_calls == 0
    assert model._fitted is False


def test_shared_irls_convergence_on_last_iteration_emits_no_false_warning():
    import warnings

    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver
    from statgpu.solvers import ConvergenceWarning

    X = np.column_stack([np.ones(5), np.linspace(-1.0, 1.0, 5)])
    y = 0.5 + 2.0 * X[:, 1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params, n_iter = IRLSSolver(Gaussian(), max_iter=1, tol=1e-12).fit(
            X, y, backend="numpy"
        )
    assert n_iter == 1
    assert not any(isinstance(w.message, ConvergenceWarning) for w in caught)
    np.testing.assert_allclose(params, [0.5, 2.0], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"tol": np.nan}, "tol"),
        ({"ridge_alpha": -1.0}, "ridge_alpha"),
        ({"ridge_penalize_intercept": 1}, "ridge_penalize_intercept"),
        ({"backend": "mystery"}, "backend"),
    ],
)
def test_shared_irls_rejects_invalid_controls(kwargs, message):
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    solver_kwargs = {k: v for k, v in kwargs.items() if k in {"max_iter", "tol"}}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in solver_kwargs}
    with pytest.raises(ValueError, match=message):
        IRLSSolver(Gaussian(), **solver_kwargs).fit(
            np.ones((4, 1)), np.arange(4.0), **fit_kwargs
        )


def test_shared_irls_rejects_bad_penalty_matrix_shape():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="penalty_matrix"):
        IRLSSolver(Gaussian()).fit(
            np.ones((4, 2)), np.arange(4.0), penalty_matrix=np.eye(3)
        )


def test_logistic_nonconvergence_is_visible_and_state_is_published():
    from statgpu.linear_model import LogisticRegression
    from statgpu.solvers import ConvergenceWarning

    X = np.linspace(-2.0, 2.0, 20)[:, None]
    y = (X[:, 0] > 0).astype(float)
    model = LogisticRegression(
        C=1.0, max_iter=1, tol=1e-14, device="cpu",
        compute_inference=False,
    )
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model.fit(X, y)
    assert model._fitted is True
    assert model.converged_ is False
    assert model.n_iter_ == 1


def test_logistic_zero_C_legacy_unregularized_path_stays_finite():
    from statgpu.linear_model import LogisticRegression

    rng = np.random.default_rng(20260806)
    X = rng.normal(size=(80, 2))
    probability = 1.0 / (1.0 + np.exp(-(0.2 + X @ np.array([0.7, -0.4]))))
    y = (rng.random(80) < probability).astype(float)
    model = LogisticRegression(
        C=0.0, max_iter=100, device="cpu", compute_inference=False
    ).fit(X, y)
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_)


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("fit_intercept", "False", "fit_intercept"),
        ("C", -1.0, "C"),
        ("C", np.inf, "C"),
        ("max_iter", 0, "max_iter"),
        ("tol", 0.0, "tol"),
        ("compute_inference", "False", "compute_inference"),
        ("gpu_memory_cleanup", "False", "gpu_memory_cleanup"),
        ("cov_type", "invalid", "cov_type"),
        ("hac_maxlags", 1.5, "hac_maxlags"),
    ],
)
def test_logistic_invalid_mutated_control_clears_stale_state(
    name, value, message
):
    model, X, y = _fitted_logistic_fixture()
    setattr(model, name, value)
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)
    _assert_logistic_state_cleared(model)


def test_logistic_direct_control_mutation_is_used_by_refit():
    from statgpu.linear_model import LogisticRegression

    rng = np.random.default_rng(20260807)
    X = rng.normal(size=(120, 2))
    p = 1.0 / (1.0 + np.exp(-(0.3 + X @ np.array([0.8, -0.5]))))
    y = (rng.random(120) < p).astype(float)
    model = LogisticRegression(
        C=1.0, max_iter=100, fit_intercept=True, device="cpu",
        compute_inference=False,
    ).fit(X, y)
    model.fit_intercept = False
    model.C = 0.0
    model.max_iter = 200
    model.tol = 1e-8
    model.fit(X, y)
    assert model._fit_intercept is False
    assert model._C == 0.0
    assert model._max_iter == 200
    assert model._tol == pytest.approx(1e-8)
    assert model.intercept_ == 0.0


def _cv_evaluation_owner(loss_name):
    from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV

    owner = object.__new__(PenalizedGLM_CV)
    owner.loss = loss_name
    return owner


class _CVScoreModel:
    fit_intercept = True
    intercept_ = 0.25
    coef_ = np.array([0.5])

    def predict(self, X):
        return self.intercept_ + np.asarray(X) @ self.coef_


def test_cv_primary_scoring_programming_error_is_not_silently_retried(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Loss:
        def value(self, *args, **kwargs):
            raise AssertionError("generic scorer must not run")

    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError('bad scoring signature')),
    )
    with pytest.raises(TypeError, match='bad scoring signature'):
        _cv_evaluation_owner('poisson')._evaluate_single(
            _CVScoreModel(), np.ones((3, 1)), np.ones(3), loss_fn=Loss()
        )


def test_cv_recoverable_primary_scoring_fallback_is_visible(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    calls = {'generic': 0}

    class Loss:
        def value(self, X, y, coef, sample_weight=None):
            calls['generic'] += 1
            return 2.75

    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError('optimized scorer unavailable')
        ),
    )
    with pytest.warns(RuntimeWarning, match='generic loss interface'):
        value = _cv_evaluation_owner('poisson')._evaluate_single(
            _CVScoreModel(), np.ones((3, 1)), np.ones(3), loss_fn=Loss()
        )
    assert value == pytest.approx(2.75)
    assert calls == {'generic': 1}


def test_cv_generic_scoring_programming_error_is_not_converted_to_mse(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Loss:
        def value(self, *args, **kwargs):
            raise TypeError('generic scorer bug')

    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            NotImplementedError('optimized scorer unavailable')
        ),
    )
    with pytest.warns(RuntimeWarning, match='generic loss interface'):
        with pytest.raises(TypeError, match='generic scorer bug'):
            _cv_evaluation_owner('squared_error')._evaluate_single(
                _CVScoreModel(), np.ones((3, 1)), np.ones(3), loss_fn=Loss()
            )


def test_cv_squared_error_numeric_failure_uses_visible_equivalent_mse(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Loss:
        def value(self, *args, **kwargs):
            raise FloatingPointError('generic non-finite score')

    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FloatingPointError('optimized non-finite score')
        ),
    )
    X = np.arange(3.0)[:, None]
    y = np.array([0.0, 1.0, 2.0])
    with pytest.warns(RuntimeWarning) as caught:
        value = _cv_evaluation_owner('squared_error')._evaluate_single(
            _CVScoreModel(), X, y, loss_fn=Loss()
        )
    assert len(caught) == 2
    expected = np.mean((y - _CVScoreModel().predict(X)) ** 2)
    assert value == pytest.approx(expected)


def test_irls_cupy_rank_failure_uses_lstsq_and_oom_propagates(monkeypatch):
    import sys
    import types
    from statgpu.glm_core._irls import _solve

    fake = types.ModuleType('cupy')
    calls = {'lstsq': 0}

    class Linalg:
        @staticmethod
        def solve(A, b):
            raise RuntimeError('singular matrix')

        @staticmethod
        def lstsq(A, b):
            calls['lstsq'] += 1
            return (np.array([3.0]), None, None, None)

    fake.linalg = Linalg()
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    result = _solve(np.eye(1), np.ones(1), backend='cupy')
    np.testing.assert_allclose(result, [3.0])
    assert calls == {'lstsq': 1}

    def oom(A, b):
        raise RuntimeError('CUDA out of memory')

    fake.linalg.solve = oom
    with pytest.raises(RuntimeError, match='out of memory'):
        _solve(np.eye(1), np.ones(1), backend='cupy')
    assert calls == {'lstsq': 1}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cov_type": 1}, "cov_type"),
        ({"hac_maxlags": 1.5}, "hac_maxlags"),
        ({"hac_maxlags": True}, "hac_maxlags"),
        ({"gpu_memory_cleanup": "False"}, "gpu_memory_cleanup"),
    ],
)
def test_logistic_constructor_rejects_silently_coerced_types(kwargs, message):
    from statgpu.linear_model import LogisticRegression

    with pytest.raises(ValueError, match=message):
        LogisticRegression(**kwargs)


def test_irls_penalty_matrix_matches_quadratic_contract():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = np.column_stack([np.ones(6), np.linspace(-1.0, 1.0, 6)])
    y = 0.4 + 1.2 * X[:, 1]
    penalty = np.diag([0.0, 0.5])
    params, _ = IRLSSolver(Gaussian(), max_iter=10, tol=1e-12).fit(
        X, y, penalty_matrix=penalty
    )
    expected = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    np.testing.assert_allclose(params, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "penalty, message",
    [
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.diag([0.0, -1.0]), "positive semidefinite"),
        (np.array([[0.0, np.nan], [np.nan, 1.0]]), "finite"),
        (np.array([[0.0, 1.0j], [-1.0j, 1.0]]), "real numeric"),
    ],
)
def test_irls_rejects_invalid_quadratic_penalty_matrix(penalty, message):
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match=message):
        IRLSSolver(Gaussian()).fit(
            np.ones((5, 2)), np.arange(5.0), penalty_matrix=penalty
        )


def _patch_sparse_cv_loss(monkeypatch, loss):
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin

    monkeypatch.setattr(
        fit_mixin, '_resolve_loss_name', lambda *args, **kwargs: loss
    )


def test_sparse_cv_lipschitz_programming_value_error_propagates(monkeypatch):
    from statgpu.linear_model.penalized._penalized_cv import _glm_sparse_cv_path

    class Loss:
        _lipschitz_at_init = False

        def lipschitz(self, *args, **kwargs):
            raise ValueError('programming shape bug')

    _patch_sparse_cv_loss(monkeypatch, Loss())
    with pytest.raises(ValueError, match='programming shape bug'):
        _glm_sparse_cv_path(
            'logistic', np.ones((6, 1)), np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1]), 'l1', 1.0, 5, 1e-4, 'cpu'
        )


def test_sparse_cv_recoverable_lipschitz_fallback_is_visible(monkeypatch):
    import statgpu.solvers as solvers
    from statgpu.linear_model.penalized._penalized_cv import _glm_sparse_cv_path

    class Loss:
        _lipschitz_at_init = False

        def lipschitz(self, *args, **kwargs):
            raise NotImplementedError('no closed-form hint')

    def fake_solver(loss, penalty, X, y, **kwargs):
        assert 'lipschitz_L' not in kwargs
        return np.zeros(X.shape[1]), 1

    _patch_sparse_cv_loss(monkeypatch, Loss())
    monkeypatch.setattr(solvers, 'fista_solver', fake_solver)
    with pytest.warns(RuntimeWarning, match='solver will estimate'):
        result = _glm_sparse_cv_path(
            'logistic', np.ones((6, 1)), np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1]), 'l1', 1.0, 5, 1e-4, 'cpu', return_path=True
        )
    assert result['n_iter'].tolist() == [1]
    assert result['coef'].shape == (1, 1)


def test_sparse_cv_invalid_lipschitz_value_fallback_is_visible(monkeypatch):
    import statgpu.solvers as solvers
    from statgpu.linear_model.penalized._penalized_cv import _glm_sparse_cv_path

    class Loss:
        _lipschitz_at_init = False

        def lipschitz(self, *args, **kwargs):
            return np.nan

    def fake_solver(loss, penalty, X, y, **kwargs):
        assert 'lipschitz_L' not in kwargs
        return np.zeros(X.shape[1]), 1

    _patch_sparse_cv_loss(monkeypatch, Loss())
    monkeypatch.setattr(solvers, 'fista_solver', fake_solver)
    with pytest.warns(RuntimeWarning, match='non-finite or non-positive'):
        result = _glm_sparse_cv_path(
            'logistic', np.ones((6, 1)), np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1]), 'l1', 1.0, 5, 1e-4, 'cpu', return_path=True
        )
    assert result['n_iter'].tolist() == [1]


def test_logistic_failed_refit_clears_gpu_accuracy_shadow():
    model, X, _ = _fitted_logistic_fixture()
    model._accuracy = 0.875
    with pytest.raises(ValueError, match="binary y"):
        model.fit(X, np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))
    assert model._accuracy is None


def test_logistic_cpu_likelihood_diagnostics_do_not_require_inference():
    from statgpu.linear_model import LogisticRegression

    X = np.array([[-1.5], [-0.5], [0.25], [1.0], [1.75]], dtype=float)
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 1.5, 4.0])
    model = LogisticRegression(
        C=2.0, max_iter=200, tol=1e-10, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    eta = model.intercept_ + X @ model.coef_
    probability = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1.0 - 1e-15)
    expected = np.sum(
        weights * (y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
    )
    y_mean = np.average(y, weights=weights)
    expected_null = np.sum(
        weights * (y * np.log(y_mean) + (1.0 - y) * np.log(1.0 - y_mean))
    )
    assert model.loglikelihood == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert model.loglikelihood_null == pytest.approx(expected_null, rel=1e-12, abs=1e-12)
    assert np.isfinite(model.aic)
    assert np.isfinite(model.bic)
    assert np.isfinite(model.pseudo_rsquared)
    assert model._bse is None


def test_cv_valueerror_retry_is_visible_but_typeerror_stays_fatal(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Model:
        coef_ = np.array([0.0])
        intercept_ = 0.0
        fit_intercept = True
        def predict(self, X):
            return np.zeros(len(X))

    class Loss:
        def value(self, *args, **kwargs):
            return 1.25

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"
    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('registered evaluator unavailable')),
    )
    with pytest.warns(RuntimeWarning, match='generic loss interface'):
        assert owner._evaluate_single(
            Model(), np.ones((2, 1)), np.ones(2), loss_fn=Loss()
        ) == pytest.approx(1.25)
    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError('programming bug')),
    )
    with pytest.raises(TypeError, match='programming bug'):
        owner._evaluate_single(
            Model(), np.ones((2, 1)), np.ones(2), loss_fn=Loss()
        )
