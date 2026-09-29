"""Post-merge review regressions for unsupported explicit Quantile solvers."""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from statgpu import glm_core
from statgpu.linear_model.penalized import (
    PenalizedGLM_CV,
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)
import statgpu.linear_model.penalized._quantile_unsupported_solver_guard_contract as _guard_contract
import statgpu.solvers._quantile_solver_guard as _solver_guard
from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty, L2Penalty
from statgpu import solvers
from statgpu.solvers import _admm as _admm_mod
from statgpu.solvers import _fista_bb as _fista_bb_mod
from statgpu.solvers import _fista as _fista_mod
from statgpu.solvers import _lbfgs as _lbfgs_mod
from statgpu.solvers import _quantile_cd as _quantile_cd_mod


# Each row below is a public estimator/CV request that the new guard itself
# closes. ADMM needs both an L2/no-penalty-branch representative and a sparse
# representative because the pre-existing Quantile validator did not reject
# ADMM on either branch, while L2/no-penalty FISTA-BB was already rejected by
# the older Quantile contract.
_NEW_UNSUPPORTED_ESTIMATOR_CASES = [
    ("admm", "l1"),
    ("admm", "l2"),
    ("fista_bb", "l1"),
]

_LOW_LEVEL_GUARD_CASES = [
    (solvers.admm_solver, glm_core.admm_solver, _admm_mod.admm_solver),
    (
        solvers.fista_bb_solver,
        glm_core.fista_bb_solver,
        _fista_bb_mod.fista_bb_solver,
    ),
]

_LOW_LEVEL_PENALTIES = [
    pytest.param(lambda: L1Penalty(0.04), id="l1"),
    pytest.param(lambda: L2Penalty(0.04), id="l2"),
]


def _data(seed=16491):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 3))
    y = 0.15 + X @ np.array([0.55, -0.3, 0.12])
    y = y + rng.exponential(scale=0.18, size=X.shape[0]) - 0.18
    return X, y


def _direct_quantile_model(estimator_kind, *, solver_name, penalty):
    common = dict(
        penalty=penalty,
        alpha=0.04,
        solver=solver_name,
        device="cpu",
    )
    if estimator_kind == "generic":
        return PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.2},
            **common,
        )
    return PenalizedQuantileRegression(quantile=0.2, **common)


@pytest.mark.parametrize("estimator_kind", ["generic", "typed"])
@pytest.mark.parametrize("solver_name,penalty", _NEW_UNSUPPORTED_ESTIMATOR_CASES)
def test_direct_quantile_new_unsupported_solver_fails_before_backend_work(
    monkeypatch, estimator_kind, solver_name, penalty
):
    X, y = _data()
    model = _direct_quantile_model(
        estimator_kind,
        solver_name=solver_name,
        penalty=penalty,
    )

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        model.fit(X, y)


@pytest.mark.parametrize("solver_name,penalty", _NEW_UNSUPPORTED_ESTIMATOR_CASES)
@pytest.mark.parametrize("cv_strategy", ["strict", "two_stage"])
def test_cv_quantile_new_unsupported_solver_fails_before_alpha_grid(
    monkeypatch, solver_name, penalty, cv_strategy
):
    X, y = _data(seed=16492)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty=penalty,
        cv=2,
        solver=solver_name,
        cv_strategy=cv_strategy,
        acknowledge_approx=(cv_strategy == "two_stage"),
        device="cpu",
    )

    def forbidden_grid(*args, **kwargs):
        raise AssertionError("alpha-grid numerical work must not start")

    monkeypatch.setattr(model, "_generate_alpha_grid", forbidden_grid)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        model.fit(X, y)


@pytest.mark.parametrize("estimator_kind", ["generic", "typed"])
def test_direct_public_solver_replacement_to_unsupported_admm_fails_before_backend(
    monkeypatch, estimator_kind
):
    X, y = _data(seed=16497)
    model = _direct_quantile_model(
        estimator_kind,
        solver_name="auto",
        penalty="l1",
    )
    model.solver = "admm"

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        model.fit(X, y)

    assert model._solver == "admm"


@pytest.mark.parametrize("estimator_kind", ["generic", "typed"])
def test_estimator_quantile_lbfgs_rejection_has_truthful_reason(
    monkeypatch, estimator_kind
):
    X, y = _data(seed=16494)
    model = _direct_quantile_model(
        estimator_kind,
        solver_name="lbfgs",
        penalty="l2",
    )

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(
        ValueError,
        match="not supported for Quantile estimators or Quantile CV.*smooth loss gradient",
    ):
        model.fit(X, y)


def test_cv_quantile_lbfgs_rejection_has_truthful_reason(monkeypatch):
    X, y = _data(seed=16495)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="l2",
        cv=2,
        solver="lbfgs",
        device="cpu",
    )

    def forbidden_grid(*args, **kwargs):
        raise AssertionError("alpha-grid numerical work must not start")

    monkeypatch.setattr(model, "_generate_alpha_grid", forbidden_grid)
    with pytest.raises(
        ValueError,
        match="not supported for Quantile estimators or Quantile CV.*smooth loss gradient",
    ):
        model.fit(X, y)


@pytest.mark.parametrize("penalty_factory", _LOW_LEVEL_PENALTIES)
@pytest.mark.parametrize("solver_fn,glm_alias,internal_fn", _LOW_LEVEL_GUARD_CASES)
def test_public_low_level_new_guard_rejects_quantile_before_loss_work(
    monkeypatch, penalty_factory, solver_fn, glm_alias, internal_fn
):
    X, y = _data(seed=16493)
    loss = QuantileLoss(quantile=0.2)
    penalty = penalty_factory()

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "gradient", forbidden)
    monkeypatch.setattr(loss, "fused_value_and_gradient", forbidden)

    for public_fn in (solver_fn, glm_alias):
        with pytest.raises(ValueError, match="does not support Quantile loss"):
            public_fn(loss, penalty, X, y, max_iter=5)

    # ``functools.wraps`` keeps public introspection compatible with the
    # underlying generic solver while changing only the Quantile support row.
    assert inspect.signature(solver_fn) == inspect.signature(internal_fn)
    assert inspect.signature(glm_alias) == inspect.signature(internal_fn)


def test_glm_core_solver_aliases_preserve_guard_and_existing_lbfgs_export():
    assert glm_core.admm_solver is solvers.admm_solver
    assert glm_core.fista_bb_solver is solvers.fista_bb_solver
    assert glm_core.lbfgs_solver is solvers.lbfgs_solver
    assert solvers.fista_solver is _fista_mod.fista_solver
    assert solvers.lbfgs_solver is _lbfgs_mod.lbfgs_solver
    assert solvers.quantile_cd_solver is _quantile_cd_mod.quantile_cd_solver


def test_low_level_quantile_solver_guard_reload_is_idempotent_and_alias_safe():
    guarded_names = (
        "fista_solver",
        "fista_bb_solver",
        "newton_solver",
        "proximal_newton_solver",
        "lbfgs_solver",
        "lbfgs_b_solver",
        "admm_solver",
        "quantile_cd_solver",
    )
    before = {name: getattr(solvers, name) for name in guarded_names}
    originals = {
        name: getattr(function, "_statgpu_original", None)
        for name, function in before.items()
    }

    importlib.reload(_solver_guard)

    for name, function in before.items():
        assert getattr(solvers, name) is function
        assert getattr(function, "_statgpu_original", None) is originals[name]

    assert _fista_mod.fista_solver is before["fista_solver"]
    assert _lbfgs_mod.lbfgs_solver is before["lbfgs_solver"]
    assert _quantile_cd_mod.quantile_cd_solver is before["quantile_cd_solver"]
    assert glm_core.fista_solver is before["fista_solver"]
    assert glm_core.fista_bb_solver is before["fista_bb_solver"]
    assert glm_core.newton_solver is before["newton_solver"]
    assert glm_core.lbfgs_solver is before["lbfgs_solver"]
    assert glm_core.admm_solver is before["admm_solver"]


@pytest.mark.parametrize("reconstruction", ["set_params", "clone"])
def test_quantile_guard_survives_standard_estimator_reconstruction(
    monkeypatch, reconstruction
):
    X, y = _data(seed=16496)
    if reconstruction == "set_params":
        model = PenalizedQuantileRegression(
            quantile=0.2,
            penalty="l1",
            alpha=0.04,
            solver="auto",
            device="cpu",
        )
        assert model.set_params(solver="admm") is model
    else:
        sklearn_base = pytest.importorskip("sklearn.base")
        model = sklearn_base.clone(
            PenalizedQuantileRegression(
                quantile=0.2,
                penalty="l1",
                alpha=0.04,
                solver="admm",
                device="cpu",
            )
        )

    assert model.solver == "admm"
    assert model._solver == "admm"

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        model.fit(X, y)


def test_estimator_guard_installer_is_import_order_safe_and_idempotent():
    before = _guard_contract._quantile_contract._validate_quantile_solver_request
    before_signature = inspect.signature(before)
    before_wrapped = getattr(before, "__wrapped__", None)
    estimator_before = (
        _guard_contract._quantile_contract.PenalizedGeneralizedLinearModel
        ._validate_solver_penalty
    )
    estimator_signature = inspect.signature(estimator_before)
    estimator_wrapped = getattr(estimator_before, "__wrapped__", None)

    _guard_contract.install_quantile_unsupported_solver_guard_contract()
    assert (
        _guard_contract._quantile_contract._validate_quantile_solver_request
        is before
    )
    assert (
        _guard_contract._quantile_contract.PenalizedGeneralizedLinearModel
        ._validate_solver_penalty
        is estimator_before
    )

    # Re-executing the contract module models a repeated/import-order install.
    # The markers live on the installed wrappers, so reload must not stack a
    # second wrapper or change public validator/estimator introspection.
    importlib.reload(_guard_contract)
    after = _guard_contract._quantile_contract._validate_quantile_solver_request
    estimator_after = (
        _guard_contract._quantile_contract.PenalizedGeneralizedLinearModel
        ._validate_solver_penalty
    )
    assert after is before
    assert getattr(after, _guard_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped
    assert inspect.signature(after) == before_signature
    assert estimator_after is estimator_before
    assert getattr(estimator_after, _guard_contract._ESTIMATOR_MARKER, False)
    assert getattr(estimator_after, "__wrapped__", None) is estimator_wrapped
    assert inspect.signature(estimator_after) == estimator_signature
