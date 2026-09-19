"""Fresh-review regressions for public low-level Quantile boundaries."""

from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pytest

from statgpu import solvers
from statgpu.solvers import _fista as _fista_mod
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty, L1Penalty, L2Penalty, MCPPenalty, SCADPenalty
from statgpu.glm_core._squared import SquaredErrorLoss
import statgpu.losses._quantile_irls_validation_contract as _irls_contract
import statgpu.solvers._quantile_proximal_public_contract as _prox_contract
from statgpu.solvers import _proximal_irls_quantile as _prox_kernel


def _data(seed=16701):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 3))
    y = 0.2 + X @ np.array([0.5, -0.25, 0.1])
    y = y + rng.laplace(scale=0.08, size=X.shape[0])
    return X, y


def test_quantile_fused_value_stays_on_torch_backend():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    X = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]],
        dtype=torch.float64,
    )
    y = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64)
    coef = torch.tensor([0.1, -0.2], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

    value, grad = loss.fused_value_and_gradient(
        X,
        y,
        coef,
        sample_weight=weights,
    )

    assert torch.is_tensor(value)
    assert value.ndim == 0
    assert value.device == X.device
    assert value.dtype == X.dtype
    assert torch.is_tensor(grad)
    assert grad.device == X.device
    assert grad.dtype == X.dtype

    eta = X @ coef
    residual = y - eta
    per_sample = torch.where(
        residual >= 0.0,
        0.3 * residual,
        (0.3 - 1.0) * residual,
    )
    expected_value = torch.sum(weights * per_sample) / torch.sum(weights)
    expected_grad = X.T @ (
        weights
        * torch.where(
            residual < 0.0,
            torch.tensor(0.7, dtype=X.dtype),
            torch.tensor(-0.3, dtype=X.dtype),
        )
    ) / torch.sum(weights)

    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(grad, expected_grad)


def test_quantile_response_validation_preserves_torch_backend():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    response = torch.tensor(
        [[-0.2], [0.1], [0.4]],
        dtype=torch.float64,
    )

    validated = loss.validate_response(response)

    assert torch.is_tensor(validated)
    assert validated.device == response.device
    assert validated.dtype == response.dtype
    assert tuple(validated.shape) == (3,)
    torch.testing.assert_close(validated, response.reshape(-1))


def test_async_quantile_fista_cv_mode_can_converge_on_torch_cpu():
    torch = pytest.importorskip("torch")

    X = torch.zeros((8, 1), dtype=torch.float64)
    y = torch.zeros(8, dtype=torch.float64)
    loss = QuantileLoss(quantile=0.5)
    penalty = L1Penalty(alpha=0.1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        coef, n_iter = solvers.fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=5,
            tol=1e-10,
            cv_mode=True,
        )

    assert n_iter == 2
    assert torch.is_tensor(coef)
    torch.testing.assert_close(coef, torch.zeros_like(coef))
    assert not any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )


def test_weighted_fista_uses_spectral_gram_scale_not_max_diagonal():
    gram = np.asarray(
        [[1.0, 0.99], [0.99, 1.0]],
        dtype=np.float64,
    )

    observed = _fista_mod._weighted_gram_lipschitz(gram)
    max_diagonal = float(np.max(np.diag(gram)))
    exact = float(np.linalg.eigvalsh(gram)[-1])

    assert observed == pytest.approx(exact, rel=2e-2, abs=1e-8)
    assert observed > max_diagonal * 1.9


def test_quantile_lipschitz_reflects_in_place_design_mutation():
    X = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=np.float64,
    )
    coef = np.zeros(X.shape[1], dtype=np.float64)
    loss = QuantileLoss(quantile=0.3)

    before = float(loss.lipschitz(X, coef))
    X *= 5.0
    after = float(loss.lipschitz(X, coef))

    assert after > before * 10.0


def test_quantile_lipschitz_is_declared_static_within_one_fista_solve():
    assert QuantileLoss._lipschitz_static is True


def _invalid_weights(n):
    negative = np.ones(n, dtype=np.float64)
    negative[2] = -0.25
    nonfinite = np.ones(n, dtype=np.float64)
    nonfinite[3] = np.nan
    return [
        pytest.param(np.ones(n - 1), id="wrong-length"),
        pytest.param(negative, id="negative"),
        pytest.param(nonfinite, id="nonfinite"),
        pytest.param(np.zeros(n), id="zero-sum"),
    ]


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_direct_quantile_irls_rejects_invalid_xy_shapes_before_numerics(
    X_transform, y_transform, message
):
    X, y = _data(seed=16713)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(X_transform(X), y_transform(y), max_iter=3)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_direct_quantile_irls_rejects_nonfinite_xy_before_numerics(target, message):
    X, y = _data(seed=16718)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)
    with pytest.raises(ValueError, match=message):
        loss.irls(X_bad, y_bad, max_iter=3)


@pytest.mark.parametrize(
    ("solver_name", "penalty"),
    [
        ("fista_solver", L2Penalty(alpha=0.04)),
        ("lbfgs_solver", None),
    ],
)
@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_public_quantile_fista_lbfgs_reject_nonfinite_xy_before_loss_work(
    monkeypatch, solver_name, penalty, target, message
):
    X, y = _data(seed=16719)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match=message):
        getattr(solvers, solver_name)(
            loss,
            penalty,
            X_bad,
            y_bad,
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("solver_name", "penalty"),
    [
        ("fista_solver", L2Penalty(alpha=0.04)),
        ("lbfgs_solver", None),
    ],
)
@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_public_quantile_fista_lbfgs_reject_invalid_xy_before_loss_work(
    monkeypatch, solver_name, penalty, X_transform, y_transform, message
):
    X, y = _data(seed=16715)
    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    solver_fn = getattr(solvers, solver_name)

    with pytest.raises(ValueError, match=message):
        solver_fn(
            loss,
            penalty,
            X_transform(X),
            y_transform(y),
            max_iter=3,
        )


@pytest.mark.parametrize(
    "solver_name",
    ["newton_solver", "proximal_newton_solver", "lbfgs_b_solver"],
)
def test_public_smooth_second_order_solvers_reject_quantile_before_loss_work(
    monkeypatch, solver_name
):
    X, y = _data(seed=16717)
    loss = QuantileLoss(quantile=0.3)
    penalty = L2Penalty(alpha=0.04)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported Quantile solver must fail before loss work")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        getattr(solvers, solver_name)(loss, penalty, X, y, max_iter=3)


def test_public_quantile_cd_solver_is_fail_closed_compatibility_symbol(monkeypatch):
    X, y = _data(seed=16716)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.04)

    def forbidden(*args, **kwargs):
        raise AssertionError("retired Quantile CD kernel must not execute")

    import statgpu.solvers._quantile_solver_guard as guard_mod

    monkeypatch.setattr(guard_mod, "_quantile_cd_solver", forbidden)
    with pytest.raises(NotImplementedError, match="compatibility symbol"):
        solvers.quantile_cd_solver(
            loss,
            penalty,
            X,
            y,
            sample_weight=np.linspace(0.5, 1.5, X.shape[0]),
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_public_proximal_quantile_rejects_nonfinite_xy_before_path_work(
    monkeypatch, target, message
):
    X, y = _data(seed=16720)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )
    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X_bad,
            y_bad,
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


def test_direct_quantile_irls_budget_exhaustion_is_observable():
    from statgpu.solvers._convergence import ConvergenceWarning

    X, y = _data(seed=16721)
    loss = QuantileLoss(quantile=0.3)

    with pytest.warns(ConvergenceWarning, match="did not converge within 1 iterations"):
        coef, n_iter = loss.irls(
            X,
            y,
            max_iter=1,
            tol=1e-16,
        )

    assert n_iter == 1
    assert np.all(np.isfinite(np.asarray(coef)))


@pytest.mark.parametrize(
    ("init_coef", "message"),
    [
        (np.zeros((2, 1), dtype=np.float64), "init_coef must be one-dimensional"),
        (np.zeros(1, dtype=np.float64), "init_coef must have length n_features"),
        (np.asarray([0.0, np.nan, 0.0]), "init_coef must contain finite values"),
        (
            np.asarray([0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            "init_coef must contain real numeric values",
        ),
    ],
)
def test_direct_quantile_irls_rejects_invalid_init_coef(init_coef, message):
    X, y = _data(seed=16722)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(
            X,
            y,
            init_coef=init_coef,
            max_iter=3,
        )


@pytest.mark.parametrize("alpha", [np.nan, -0.1, True, "0.1"])
def test_direct_quantile_irls_rejects_invalid_mutated_l2_alpha(alpha):
    X, y = _data(seed=16723)
    loss = QuantileLoss(quantile=0.3)
    penalty = L2Penalty(alpha=0.1)
    penalty.alpha = alpha

    with pytest.raises(ValueError, match="L2 penalty alpha must be"):
        loss.irls(
            X,
            y,
            penalty=penalty,
            max_iter=3,
        )


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_direct_quantile_irls_rejects_invalid_weights_before_numerics(sample_weight):
    X, y = _data()
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match="sample_weight"):
        loss.irls(X, y, sample_weight=sample_weight, max_iter=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"max_iter": True}, "max_iter must be a positive integer"),
        ({"tol": 0.0}, "tol must be a finite positive number"),
        ({"tol": True}, "tol must be a finite positive number"),
        ({"tol": "1e-6"}, "tol must be a finite positive number"),
        ({"eps": 0.0}, "eps must be a finite positive number"),
        ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
    ],
)
def test_direct_quantile_irls_rejects_invalid_stopping_controls(kwargs, message):
    X, y = _data(seed=16705)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(X, y, **kwargs)


def test_direct_quantile_irls_validation_preserves_array_like_design_input():
    X, y = _data(seed=16704)
    loss = QuantileLoss(quantile=0.3)

    coef, n_iter = loss.irls(
        X.tolist(),
        y.tolist(),
        sample_weight=np.ones(X.shape[0]),
        max_iter=2,
    )

    assert np.asarray(coef).shape == (X.shape[1],)
    assert np.all(np.isfinite(np.asarray(coef)))
    assert 1 <= n_iter <= 2


@pytest.mark.parametrize(
    ("loss_factory", "penalty_factory", "message"),
    [
        (
            lambda: SquaredErrorLoss(),
            lambda: SCADPenalty(alpha=0.05),
            "requires QuantileLoss",
        ),
        (
            lambda: QuantileLoss(quantile=0.3),
            lambda: L2Penalty(alpha=0.05),
            "requires scalar SCAD or MCP",
        ),
        (
            lambda: QuantileLoss(quantile=0.3),
            lambda: GroupSCADPenalty(
                alpha=0.05,
                a=3.7,
                groups=[[0, 1], [2]],
            ),
            "requires scalar SCAD or MCP",
        ),
    ],
)
def test_public_proximal_quantile_solver_rejects_wrong_objective_before_path_work(
    monkeypatch, loss_factory, penalty_factory, message
):
    X, y = _data(seed=16711)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss_factory(),
            penalty_factory(),
            X,
            y,
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_public_proximal_quantile_rejects_invalid_xy_shapes_before_path_work(
    monkeypatch, X_transform, y_transform, message
):
    X, y = _data(seed=16714)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X_transform(X),
            y_transform(y),
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("factory", "attr", "bad_value", "message"),
    [
        (
            lambda: SCADPenalty(alpha=0.05, a=3.7),
            "a",
            np.nan,
            "SCAD penalty a must be",
        ),
        (
            lambda: SCADPenalty(alpha=0.05, a=3.7),
            "a",
            2.0,
            "SCAD penalty a must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.05, gamma=3.0),
            "gamma",
            np.nan,
            "MCP penalty gamma must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.05, gamma=3.0),
            "gamma",
            1.0,
            "MCP penalty gamma must be",
        ),
    ],
)
def test_public_proximal_quantile_rejects_mutated_penalty_shape_before_path_work(
    monkeypatch, factory, attr, bad_value, message
):
    X, y = _data(seed=16724)
    loss = QuantileLoss(quantile=0.3)
    penalty = factory()
    setattr(penalty, attr, bad_value)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("invalid penalty shape must fail before path work")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )
    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_public_proximal_quantile_solver_rejects_invalid_weights(sample_weight):
    X, y = _data(seed=16702)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    for solver_fn in (
        solvers.proximal_irls_quantile_solver,
        _prox_kernel.proximal_irls_quantile_solver,
    ):
        with pytest.raises(ValueError, match="sample_weight"):
            solver_fn(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.08, 0.05]),
                max_iter=3,
                sample_weight=sample_weight,
            )


@pytest.mark.parametrize(
    ("alpha_path", "message"),
    [
        ([], "alpha_path must be a non-empty"),
        (np.array([[0.08, 0.05]]), "one-dimensional"),
        ([0.08, 0.0], "finite positive"),
        ([0.08, np.nan], "finite positive"),
        ([0.08, True], "finite positive"),
        ([0.05, 0.08], "non-increasing"),
    ],
)
def test_public_proximal_quantile_rejects_invalid_alpha_path_before_path_work(
    monkeypatch, alpha_path, message
):
    X, y = _data(seed=16712)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=alpha_path,
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": True}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": [2, 0]}, "max_iter sequence must contain only positive integers"),
        ({"max_iter": [2]}, "max_iter sequence must have one positive integer per alpha_path step"),
        ({"max_lla_per_step": 0}, "max_lla_per_step must be a positive integer"),
        ({"max_lla_per_step": True}, "max_lla_per_step must be a positive integer"),
        ({"tol": 0.0}, "tol must be a finite positive number"),
        ({"tol": "1e-6"}, "tol must be a finite positive number"),
        ({"lla_tol": False}, "lla_tol must be a finite positive number"),
        ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
    ],
)
def test_public_proximal_quantile_rejects_invalid_stopping_controls(kwargs, message):
    X, y = _data(seed=16706)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.array([0.08, 0.05]),
            **kwargs,
        )


def test_public_proximal_quantile_flat_target_remains_valid_when_lla_weights_stay_zero(
    monkeypatch,
):
    X, y = _data(seed=16709)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    monkeypatch.setattr(
        _prox_kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.zeros(
            p, dtype=coef.dtype
        ),
    )

    def converged_flat_irls(
        X_arg,
        y_arg,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        point = np.full(X_arg.shape[1], 2.0, dtype=np.float64)
        return point, 1

    monkeypatch.setattr(loss, "irls", converged_flat_irls)

    token = _prox_kernel._STRICT_CV_TARGET.set(True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.05]),
                max_lla_per_step=1,
                max_iter=2,
                tol=1e-12,
                lla_tol=1e-12,
                fit_intercept=False,
            )
    finally:
        _prox_kernel._STRICT_CV_TARGET.reset(token)

    assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert n_iter == 1
    np.testing.assert_array_equal(coef, np.full(X.shape[1], 2.0))
    assert intercept == 0.0


def test_public_proximal_quantile_active_target_exhaustion_warns(monkeypatch):
    X, y = _data(seed=16707)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    from statgpu.solvers import _proximal_irls_quantile as kernel

    monkeypatch.setattr(
        kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.ones(p, dtype=coef.dtype),
    )

    def drift(X_arg, X_sq_arg, y_arg, w_arg, beta_arg, thresh_arg, p_arg, eps_arg, xp, backend):
        return beta_arg + 0.1

    monkeypatch.setattr(kernel, "_parallel_majorization_step", drift)

    with pytest.warns(ConvergenceWarning, match="target reached max_iter=2"):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.array([0.05]),
            max_lla_per_step=1,
            max_iter=2,
            tol=1e-12,
            lla_tol=1.0,
        )


def test_public_proximal_quantile_strict_target_exhaustion_raises(monkeypatch):
    X, y = _data(seed=16708)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    from statgpu.solvers import _proximal_irls_quantile as kernel

    monkeypatch.setattr(
        kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.ones(p, dtype=coef.dtype),
    )
    monkeypatch.setattr(
        kernel,
        "_parallel_majorization_step",
        lambda X_arg, X_sq_arg, y_arg, w_arg, beta_arg, thresh_arg, p_arg, eps_arg, xp, backend: beta_arg + 0.1,
    )

    token = kernel._STRICT_CV_TARGET.set(True)
    try:
        with pytest.raises(FloatingPointError, match="target reached max_iter=2"):
            solvers.proximal_irls_quantile_solver(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.05]),
                max_lla_per_step=1,
                max_iter=2,
                tol=1e-12,
                lla_tol=1.0,
            )
    finally:
        kernel._STRICT_CV_TARGET.reset(token)


def test_public_proximal_quantile_none_budget_has_defined_default():
    X, y = _data(seed=16703)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=np.array([0.05]),
        max_lla_per_step=1,
        sample_weight=np.ones(X.shape[0]),
    )

    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert 0 <= n_iter <= 100


def test_public_proximal_quantile_wrapper_preserves_introspection_and_alias():
    public = solvers.proximal_irls_quantile_solver
    historical = _prox_kernel.proximal_irls_quantile_solver
    kernel = getattr(public, "__wrapped__", None)

    assert historical is public
    assert kernel is not None
    assert inspect.signature(public) == inspect.signature(kernel)
    doc = " ".join((inspect.getdoc(public) or "").split())
    assert "None`` uses 100 iterations per step" in doc
    assert "Non-empty one-dimensional continuation path" in doc
    assert "finite positive values in non-increasing order" in doc


def test_quantile_irls_validation_installer_is_idempotent_under_reload():
    before = QuantileLoss.irls
    before_wrapped = getattr(before, "__wrapped__", None)

    _irls_contract.install_quantile_irls_validation_contract()
    assert QuantileLoss.irls is before

    importlib.reload(_irls_contract)
    after = QuantileLoss.irls
    assert after is before
    assert getattr(after, _irls_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped


def test_quantile_proximal_contract_is_idempotent_under_reload():
    before = solvers.proximal_irls_quantile_solver
    before_wrapped = getattr(before, "__wrapped__", None)

    assert _prox_contract.install_quantile_proximal_public_contract() is before
    importlib.reload(_prox_contract)
    after = _prox_kernel.proximal_irls_quantile_solver

    assert after is before
    assert getattr(after, _prox_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped
