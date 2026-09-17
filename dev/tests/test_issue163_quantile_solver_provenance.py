"""Issue #163 regressions for truthful penalized Quantile solver identity."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


def _data(seed=16301, n=96, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.75, -0.3], dtype=np.float64)[:p]
    y = (0.25 + X @ beta + rng.laplace(scale=0.18, size=n)).astype(np.float64)
    return X, y


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": 0.5},
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
        lambda: PenalizedQuantileRegression(
            quantile=0.5,
            penalty="l2",
            alpha=0.03,
            solver="auto",
            device="cpu",
            max_iter=300,
            tol=1e-8,
        ),
    ],
)
def test_l2_quantile_auto_reports_and_executes_irls(factory):
    X, y = _data()
    model = factory().fit(X, y)

    assert model._selected_solver == "irls"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


def test_explicit_irls_l2_matches_auto_quantile_fit():
    X, y = _data(seed=16302)
    common = dict(
        quantile=0.4,
        penalty="l2",
        alpha=0.02,
        device="cpu",
        max_iter=400,
        tol=1e-9,
    )
    auto = PenalizedQuantileRegression(solver="auto", **common).fit(X, y)
    explicit = PenalizedQuantileRegression(solver="irls", **common).fit(X, y)

    assert auto._selected_solver == explicit._selected_solver == "irls"
    np.testing.assert_allclose(auto.coef_, explicit.coef_, rtol=0.0, atol=0.0)
    assert auto.intercept_ == pytest.approx(explicit.intercept_, rel=0.0, abs=0.0)


def test_explicit_fista_l2_quantile_executes_true_fista(monkeypatch):
    from statgpu.losses import QuantileLoss

    X, y = _data(seed=16303)

    def forbidden_irls(*args, **kwargs):
        raise AssertionError("explicit solver='fista' must not execute Quantile IRLS")

    monkeypatch.setattr(QuantileLoss, "irls", forbidden_irls)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista",
        device="cpu",
        max_iter=1200,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    assert model.n_iter_ >= 1


def test_explicit_fista_bb_l2_quantile_fails_before_backend_fit(monkeypatch):
    X, y = _data(seed=16303)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l2",
        alpha=0.03,
        solver="fista_bb",
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="not a maintained L2/no-penalty Quantile route"):
        model.fit(X, y)


def test_sparse_quantile_auto_remains_fista_family():
    X, y = _data(seed=16304)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="l1",
        alpha=0.03,
        solver="auto",
        device="cpu",
        max_iter=500,
        tol=1e-7,
    ).fit(X, y)

    assert model._selected_solver == "fista"
    assert np.all(np.isfinite(model.coef_))


def test_nonconvex_quantile_auto_reports_dedicated_proximal_irls_cd():
    X, y = _data(seed=16307, n=64)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver="auto",
        device="cpu",
        max_iter=160,
        tol=1e-6,
    ).fit(X, y)

    assert model._selected_solver == "proximal_irls_cd"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)


@pytest.mark.parametrize("solver", ["fista", "fista_bb", "admm"])
def test_explicit_solver_does_not_silently_replace_quantile_scad(
    monkeypatch, solver
):
    X, y = _data(seed=16308, n=48)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="scad",
        alpha=0.02,
        solver=solver,
        device="cpu",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("backend fit must not run for rejected explicit solver")

    monkeypatch.setattr(model, "_fit_cpu", forbidden)
    with pytest.raises(ValueError, match="dedicated Proximal IRLS-CD"):
        model.fit(X, y)


def test_quantile_cv_l2_uses_irls_for_candidates_and_final_refit():
    X, y = _data(seed=16305, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l2",
        alpha_grid=np.array([0.04, 0.02], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=250,
        tol=1e-8,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "irls"
    assert cv.estimator_._selected_solver == "irls"
    assert cv.alpha_ in {0.04, 0.02}
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_sparse_path_and_final_refit_remain_fista():
    X, y = _data(seed=16306, n=72)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="l1",
        alpha_grid=np.array([0.04], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-7,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "fista"
    assert cv.estimator_._selected_solver == "fista"
    assert cv.alpha_ == pytest.approx(0.04)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_cv_scad_reports_dedicated_solver_for_candidates_and_refit():
    X, y = _data(seed=16309, n=60)
    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="scad",
        alpha_grid=np.array([0.025], dtype=np.float64),
        cv=2,
        random_state=163,
        solver="auto",
        device="cpu",
        max_iter=140,
        tol=1e-6,
    ).fit(X, y)

    assert cv._solver_for_cv("cpu", X=X) == "proximal_irls_cd"
    assert cv.estimator_._selected_solver == "proximal_irls_cd"
    assert cv.alpha_ == pytest.approx(0.025)
    assert np.all(np.isfinite(cv.coef_))


def test_quantile_formula_route_preserves_auto_irls_identity_and_numerics():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")

    X, y = _data(seed=16310, n=84)
    df = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
    common = dict(
        quantile=0.45,
        penalty="l2",
        alpha=0.025,
        solver="auto",
        device="cpu",
        max_iter=350,
        tol=1e-9,
    )

    array_model = PenalizedQuantileRegression(**common).fit(X, y)
    formula_model = PenalizedQuantileRegression(**common).fit(
        formula="y ~ x1 + x2", data=df
    )

    assert array_model._selected_solver == "irls"
    assert formula_model._selected_solver == "irls"
    assert array_model._selected_backend_name == formula_model._selected_backend_name == "numpy"
    np.testing.assert_allclose(
        formula_model.coef_, array_model.coef_, rtol=0.0, atol=2e-10
    )
    assert formula_model.intercept_ == pytest.approx(
        array_model.intercept_, rel=0.0, abs=2e-10
    )


def test_quantile_solver_contract_installer_is_idempotent_and_signature_safe():
    from statgpu.linear_model.penalized import _fit_mixin
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    before_policy = _fit_mixin._preferred_penalized_glm_solver
    before_validate = PenalizedGeneralizedLinearModel._validate_solver_penalty
    policy_signature = inspect.signature(before_policy)
    validate_signature = inspect.signature(before_validate)

    contract.install_quantile_solver_contract()

    assert _fit_mixin._preferred_penalized_glm_solver is before_policy
    assert PenalizedGeneralizedLinearModel._validate_solver_penalty is before_validate
    assert inspect.signature(before_policy) == policy_signature
    assert inspect.signature(before_validate) == validate_signature
    assert hasattr(before_policy, "__wrapped__")
    assert hasattr(before_validate, "__wrapped__")


def test_quantile_solver_installer_is_import_order_safe_in_fresh_interpreter():
    import json
    import subprocess
    import sys

    code = r'''
import json
from statgpu.linear_model.penalized import _fit_mixin
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
payload = {
    "l2_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l2", backend_name="numpy"
    ),
    "sparse_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "l1", backend_name="numpy"
    ),
    "nonconvex_auto": _fit_mixin._preferred_penalized_glm_solver(
        "quantile", "scad", backend_name="numpy"
    ),
    "policy_wrapped": hasattr(
        _fit_mixin._preferred_penalized_glm_solver, "__wrapped__"
    ),
    "validate_wrapped": hasattr(
        PenalizedGeneralizedLinearModel._validate_solver_penalty, "__wrapped__"
    ),
}
print(json.dumps(payload, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {
        "l2_auto": "irls",
        "nonconvex_auto": "proximal_irls_cd",
        "policy_wrapped": True,
        "sparse_auto": "fista",
        "validate_wrapped": True,
    }
