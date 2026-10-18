"""Unpenalized, family-valid oracle refits (issue #227).

CPU acceptance uses independently written score/Fisher formulas and SciPy's
root solver. GPU cases preserve the declared CPU-only boundary; skips are not
physical-device validation.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import optimize, special, stats
from sklearn.base import clone

from statgpu.linear_model import (
    LogisticRegression, PenalizedGLM_CV, PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression, PenalizedLogisticRegression,
    PenalizedPoissonRegression, PenalizedGammaRegression,
    PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression,
    PenalizedTweedieRegression,
)


FAMILIES = [
    ("squared_error", {}, PenalizedLinearRegression, {}),
    ("logistic", {}, PenalizedLogisticRegression, {}),
    ("poisson", {}, PenalizedPoissonRegression, {}),
    ("gamma", {"link": "log"}, PenalizedGammaRegression, {"link": "log"}),
    ("gamma", {"link": "inverse_power"}, PenalizedGammaRegression,
     {"link": "inverse_power"}),
    ("inverse_gaussian", {}, PenalizedInverseGaussianRegression, {}),
    ("negative_binomial", {"alpha": .25}, PenalizedNegativeBinomialRegression,
     {"alpha_nb": .25}),
    ("tweedie", {"power": 1.2}, PenalizedTweedieRegression, {"power": 1.2}),
]


def _data(loss, settings, *, intercept=True):
    rng = np.random.default_rng(227)
    x = rng.uniform(.2, 1.2, (240, 2))
    eta = (0.6 if intercept else 0.) + x @ np.array([.6, .3])
    mu = np.exp(eta)
    if loss == "squared_error":
        y = eta + rng.normal(scale=.3, size=len(x))
    elif loss == "logistic":
        y = rng.binomial(1, special.expit(eta))
    elif loss == "poisson":
        y = rng.poisson(mu)
    elif loss == "negative_binomial":
        r = 1 / settings["alpha"]
        y = rng.negative_binomial(r, r / (r + mu))
    elif loss == "inverse_gaussian":
        y = rng.wald(mu, 5)
    elif loss == "gamma" and settings["link"] == "inverse_power":
        y = rng.gamma(8, 1 / (8 * eta))
    else:
        y = rng.gamma(8, mu / 8)
    weights = rng.uniform(.4, 1.8, len(x))
    weights[::17] = 0
    # An exactly zero middle column makes selected-column ordering observable.
    return np.insert(x, 1, 0, axis=1), y, weights


def _moments(loss, settings, eta, y):
    """Independent score, observed curvature, Fisher weight, mean, variance."""
    if loss == "squared_error":
        one = np.ones(len(y))
        return eta - y, one, one, eta, one
    if loss == "logistic":
        mu = special.expit(eta)
        v = mu * (1 - mu)
        return mu - y, v, v, mu, v
    if loss == "gamma" and settings["link"] == "inverse_power":
        mu = 1 / eta
        return y - mu, mu ** 2, mu ** 2, mu, mu ** 2
    mu = np.exp(eta)
    if loss == "poisson":
        return mu - y, mu, mu, mu, mu
    if loss == "gamma":
        return 1 - y / mu, y / mu, np.ones(len(y)), mu, mu ** 2
    if loss == "inverse_gaussian":
        return (mu - y) / mu ** 2, (2*y - mu) / mu ** 2, 1/mu, mu, mu ** 3
    if loss == "negative_binomial":
        a = settings["alpha"]
        return ((mu-y)/(1+a*mu), mu*(1+a*y)/(1+a*mu)**2,
                mu/(1+a*mu), mu, mu+a*mu**2)
    power = settings["power"]
    return (mu**(1-power)*(mu-y),
            (2-power)*mu**(2-power)+(power-1)*y*mu**(1-power),
            mu**(2-power), mu, mu**power)


def _reference(loss, settings, design, y, weights=None, cov_type="nonrobust"):
    weights = np.ones(len(y)) if weights is None else weights
    # Positive initial eta also puts inverse-link Gamma inside its domain.
    initial = np.full(design.shape[1], .3)
    def score(params):
        psi = _moments(loss, settings, design @ params, y)[0]
        return design.T @ (weights * psi)
    def jac(params):
        curvature = _moments(loss, settings, design @ params, y)[1]
        return design.T @ ((weights * curvature)[:, None] * design)
    if loss == "squared_error":
        params = np.linalg.solve(design.T @ (weights[:, None]*design),
                                 design.T @ (weights*y))
    else:
        solved = optimize.root(score, initial, jac=jac, tol=1e-11)
        assert solved.success, solved.message
        params = solved.x
    np.testing.assert_allclose(score(params), 0, atol=1e-8)
    psi, curvature, fisher, mu, variance = _moments(loss, settings, design @ params, y)
    if cov_type == "nonrobust":
        dispersion = 1.
        if loss in ("squared_error", "gamma", "inverse_gaussian", "tweedie"):
            dispersion = np.sum(weights * (y-mu)**2 / variance) / (len(y)-design.shape[1])
        cov = dispersion * np.linalg.inv(design.T @ ((weights*fisher)[:, None] * design))
    else:
        bread = np.linalg.inv(design.T @ ((weights*curvature)[:, None]*design))
        meat = design.T @ (((weights*psi)**2)[:, None]*design)
        cov = bread @ meat @ bread
        if cov_type == "hc1":
            cov *= len(y)/(len(y)-design.shape[1])
    return params, np.sqrt(np.diag(cov))


def _assert_result(model, loss, settings, X, y, weights=None, cov_type="nonrobust"):
    result = model._inference_result
    active = np.asarray(result.metadata["active_set"], dtype=bool)
    np.testing.assert_array_equal(active, [True, False, True])
    offset = int(model._effective_intercept)
    indices = np.flatnonzero(active) + offset
    design = X[:, active]
    if offset:
        design = np.column_stack([np.ones(len(y)), design])
        indices = np.r_[0, indices]
    expected, se = _reference(loss, settings, design, y, weights, cov_type)
    np.testing.assert_allclose(result.params[indices], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.bse[indices], se, rtol=1e-6, atol=1e-7)
    z = expected/se
    np.testing.assert_allclose(result.statistic[indices], z, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.pvalues[indices], 2*stats.norm.sf(abs(z)),
                               rtol=2e-5, atol=1e-10)
    expected_ci = expected[:, None] + se[:, None]*stats.norm.ppf([.025, .975])
    np.testing.assert_allclose(result.conf_int[indices], expected_ci, rtol=1e-6, atol=1e-7)
    assert np.isnan(result.bse[1+offset])
    assert np.isnan(result.conf_int[1+offset]).all()
    assert result.params[1+offset] == model.coef_[1]
    assert result.method == model.inference_requested_method_ == model.inference_resolved_method_ == "oracle"
    assert model.inference_target_ == "active_set_refit_coefficient"
    assert result.distribution == "normal"
    assert result.cov_type == cov_type
    assert result.metadata["refit_penalty"] == "none"
    assert result.metadata["selection_adjusted"] is False
    assert result.metadata["numerical_backend"] == "numpy"
    assert result.metadata["numerical_device"] == "cpu"
    assert result.metadata["numerical_dtype"] == "float64"
    assert result.params.dtype == np.float64
    np.testing.assert_array_equal(model._params, result.params)
    np.testing.assert_array_equal(model._bse, result.bse)
    assert len(result.feature_names) == X.shape[1]+offset


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES)
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("typed_entry", [False, True])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
def test_all_declared_families_match_independent_weighted_oracle(
        loss, settings, typed, typed_settings, penalty, typed_entry, cov_type):
    X, y, weights = _data(loss, settings)
    kwargs = dict(penalty=penalty, alpha=.0001, solver="fista", device="cpu",
                  max_iter=10000, tol=1e-10, compute_inference=True,
                  inference_method="oracle", cov_type=cov_type)
    model = (typed(**kwargs, **typed_settings) if typed_entry else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    model.fit(X, y, sample_weight=weights)
    _assert_result(model, loss, settings, X, y, weights, cov_type)
    assert model._inference_result.metadata["refit_family_parameters"] == settings


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logistic_reproduction_rejects_default_penalized_child(penalty, cov_type, dtype):
    rng = np.random.default_rng(48)
    X = rng.normal(size=(200, 2)).astype(dtype)
    y = rng.binomial(1, special.expit(.6 + X @ np.array([2., -1.])))
    model = PenalizedLogisticRegression(
        penalty=penalty, alpha=.01, solver="fista", device="cpu", cov_type=cov_type,
        compute_inference=True, inference_method="oracle", max_iter=10000, tol=1e-10,
    ).fit(X, y)
    design = np.column_stack([np.ones(len(y)), X])
    params, se = _reference("logistic", {}, design, y, cov_type=cov_type)
    np.testing.assert_allclose(model._params, params, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model._bse, se, rtol=1e-6, atol=1e-7)
    bad = LogisticRegression(C=1, device="cpu", max_iter=10000, tol=1e-10).fit(X, y)
    assert np.max(np.abs(model._params-np.r_[bad.intercept_, bad.coef_])) > .2
    np.testing.assert_allclose(design.T@(special.expit(design@model._params)-y), 0, atol=1e-7)


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES)
def test_no_intercept_preserves_support_and_weights(loss, settings, typed, typed_settings):
    X, y, weights = _data(loss, settings, intercept=False)
    model = typed(**typed_settings, penalty="mcp", alpha=.0001, fit_intercept=False,
                  solver="fista", device="cpu", max_iter=10000, tol=1e-10,
                  compute_inference=True, inference_method="oracle").fit(X, y, sample_weight=weights)
    _assert_result(model, loss, settings, X, y, weights)


@pytest.mark.parametrize("cls,typed_key,loss_key,value,override", [
    (PenalizedNegativeBinomialRegression, "alpha_nb", "alpha", .25, .6),
    (PenalizedTweedieRegression, "power", "power", 1.2, 1.7),
])
def test_typed_family_settings_are_clone_safe_and_explicit_loss_kwargs_win(
        cls, typed_key, loss_key, value, override):
    loss = "negative_binomial" if loss_key == "alpha" else "tweedie"
    for loss_kwargs, effective in [(None, value), ({loss_key: override}, override)]:
        model = cls(**{typed_key: value}, loss_kwargs=loss_kwargs, penalty="scad",
                    alpha=.0001, solver="fista", device="cpu", max_iter=10000,
                    tol=1e-10, compute_inference=True, inference_method="oracle")
        copied = clone(model)
        assert copied.get_params()[typed_key] == value
        assert copied.get_params()["loss_kwargs"] == loss_kwargs
        X, y, weights = _data(loss, {loss_key: effective})
        copied.fit(X, y, sample_weight=weights)
        assert getattr(copied._loss, loss_key) == effective
        assert copied.get_params()["loss_kwargs"] == loss_kwargs
        assert copied._inference_result.metadata["refit_family_parameters"] == {loss_key: effective}
        _assert_result(copied, loss, {loss_key: effective}, X, y, weights)
        # Direct replacement must affect the next resolution, not stale kwargs.
        copied.set_params(**{typed_key: value, "loss_kwargs": None})
        copied.fit(X, y, sample_weight=weights)
        assert getattr(copied._loss, loss_key) == value


@pytest.mark.parametrize("loss,settings", [("logistic", {}),
    ("negative_binomial", {"alpha": .25}), ("tweedie", {"power": 1.2}),
    ("inverse_gaussian", {})])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
def test_cv_final_refit_preserves_family_and_reports_selection_conditioning(loss, settings, cov_type):
    X, y, weights = _data(loss, settings)
    model = PenalizedGLM_CV(loss=loss, loss_kwargs=settings, penalty="scad",
        alpha_grid=[.0001, .0002], cv=2, solver="fista", device="cpu",
        max_iter=10000, tol=1e-10, compute_inference=True, inference_method="oracle",
        cov_type=cov_type)
    model.fit(X, y, sample_weight=weights)
    _assert_result(model.estimator_, loss, settings, X, y, weights, cov_type)
    assert model._inference_result is model.estimator_._inference_result
    assert model.penalty_conditioning_ == "cv_selected_penalty"
    assert model.penalty_selection_adjusted_ is False
    assert model._inference_result.metadata["selected_alpha"] == model.alpha_


def test_child_constructor_uses_public_wrapped_signature_and_pins_cpu(monkeypatch):
    original = LogisticRegression.fit
    seen = []
    def record(self, X, y, **kwargs):
        seen.append(self.get_params())
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(LogisticRegression, "fit", record)
    assert "C" in inspect.signature(LogisticRegression).parameters
    X, y, weights = _data("logistic", {})
    model = PenalizedLogisticRegression(penalty="scad", alpha=.0001, device="cpu",
        solver="fista", max_iter=12345, tol=1e-10, compute_inference=True,
        inference_method="oracle").fit(X, y, sample_weight=weights)
    assert len(seen) == 1
    assert seen[0]["device"] == "cpu"
    assert seen[0]["C"] == 0
    assert seen[0]["solver"] == "irls"
    assert seen[0]["compute_inference"] is False
    assert seen[0]["max_iter"] == 12345
    assert seen[0]["tol"] == 1e-10
    _assert_result(model, "logistic", {}, X, y, weights)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES)
def test_empty_support_refits_intercept_when_present(loss, settings, typed, typed_settings, fit_intercept):
    X, y, weights = _data(loss, settings)
    # A zero design forces empty support without relying on a penalty threshold.
    X[:] = 0
    model = typed(**typed_settings, penalty="scad", alpha=.1, fit_intercept=fit_intercept,
        solver="fista", device="cpu", max_iter=10000, tol=1e-10,
        compute_inference=True, inference_method="oracle")
    if not fit_intercept and loss == "gamma" and settings["link"] == "inverse_power":
        # eta=0 cannot define a finite mean for the inverse Gamma link.
        with pytest.raises(ValueError, match="positive inverse-link predictor"):
            model.fit(X, y, sample_weight=weights)
        return
    model.fit(X, y, sample_weight=weights)
    result = model._inference_result
    assert result.metadata["n_active"] == 0
    assert result.metadata["active_set"] == [False, False, False]
    if fit_intercept:
        expected, se = _reference(loss, settings, np.ones((len(y), 1)), y, weights)
        np.testing.assert_allclose(result.params[:1], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(result.bse[:1], se, rtol=1e-6, atol=1e-7)
        assert np.isnan(result.bse[1:]).all()
    else:
        assert np.isnan(result.bse).all()


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_helper_rejects_gpu_provenance_before_host_conversion(backend, monkeypatch):
    from statgpu.linear_model.penalized import _inference_mixin
    model = PenalizedLogisticRegression(device="cpu")
    model._selected_backend_name = backend
    def forbid_host(*args, **kwargs):
        raise AssertionError("GPU oracle attempted a host transfer")
    monkeypatch.setattr(_inference_mixin, "_to_numpy", forbid_host)
    with pytest.raises(NotImplementedError, match="not silently moved to CPU"):
        model._compute_oracle_inference(np.ones((5, 2)), np.ones(5))
    _assert_failed_oracle_fit(model, np.ones((5, 2)))
    assert model._selected_backend_name is None


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_physical_gpu_oracle_request_keeps_existing_fail_closed_contract(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip("No physical CUDA device")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("No physical Torch CUDA device")
    X, y, _ = _data("logistic", {})
    model = PenalizedLogisticRegression(device=device, penalty="scad", alpha=.0001,
        solver="fista", compute_inference=True, inference_method="oracle")
    with pytest.raises(NotImplementedError, match="CPU active-set refit"):
        model.fit(X, y)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("loss,settings,default_settings", [
    ("gamma", {"link": "inverse_power"}, {"link": "log"}),
    ("negative_binomial", {"alpha": .25}, {"alpha": 1.}),
    ("tweedie", {"power": 1.2}, {"power": 1.5}),
])
def test_family_fixtures_distinguish_default_reset_and_dropped_weights(loss, settings, default_settings):
    X, y, weights = _data(loss, settings)
    design = np.column_stack([np.ones(len(y)), X[:, [0, 2]]])
    correct, correct_se = _reference(loss, settings, design, y, weights)
    wrong_family, wrong_se = _reference(loss, default_settings, design, y, weights)
    wrong_weights, _ = _reference(loss, settings, design, y)
    assert max(np.max(np.abs(correct-wrong_family)),
               np.max(np.abs(correct_se-wrong_se))) > 1e-3
    assert np.max(np.abs(correct-wrong_weights)) > 1e-3


def test_formula_support_names_and_missing_row_weight_alignment():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, y, weights = _data("logistic", {})
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 2], "zero": X[:, 1], "y": y})
    frame.loc[[4, 9, 15], "x1"] = np.nan
    model = PenalizedLogisticRegression(penalty="mcp", alpha=.0001, solver="fista",
        device="cpu", max_iter=10000, tol=1e-10, compute_inference=True,
        inference_method="oracle").fit(formula="y ~ x1 + zero + x2", data=frame,
                                        sample_weight=weights)
    retained = frame.x1.notna().to_numpy()
    _assert_result(model, "logistic", {}, X[retained], y[retained], weights[retained])
    assert model._inference_result.feature_names == ["(Intercept)", "x1", "zero", "x2"]


def test_logistic_child_nonconvergence_is_not_published_as_oracle(monkeypatch):
    def fail(self, *args, **kwargs):
        self.converged_ = False
        return self
    monkeypatch.setattr(LogisticRegression, "fit", fail)
    X, y, _ = _data("logistic", {})
    model = PenalizedLogisticRegression(penalty="scad", alpha=.0001, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle")
    with pytest.raises(RuntimeError, match="oracle refit did not converge"):
        model.fit(X, y)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_oracle_reporting_does_not_replace_prediction_coefficients(penalty):
    X, y, weights = _data("logistic", {})
    kwargs = dict(penalty=penalty, alpha=.03, solver="fista", device="cpu",
                  max_iter=10000, tol=1e-10)
    selection = PenalizedLogisticRegression(**kwargs).fit(X, y, sample_weight=weights)
    oracle = PenalizedLogisticRegression(**kwargs, compute_inference=True,
        inference_method="oracle").fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(oracle.coef_, selection.coef_)
    assert oracle.intercept_ == selection.intercept_
    np.testing.assert_array_equal(oracle.predict_proba(X), selection.predict_proba(X))


def test_newton_child_convergence_warning_aborts_inference(monkeypatch):
    import warnings
    from statgpu.linear_model import GammaRegression
    from statgpu.solvers._convergence import ConvergenceWarning
    def fail(self, *args, **kwargs):
        warnings.warn("Synthetic child convergence failure", ConvergenceWarning)
        raise AssertionError("The convergence warning was ignored")
    monkeypatch.setattr(GammaRegression, "fit", fail)
    X, y, _ = _data("gamma", {"link": "log"})
    model = PenalizedGammaRegression(penalty="scad", alpha=.0001, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle")
    with pytest.raises(RuntimeError, match="oracle refit did not converge"):
        model.fit(X, y)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("cls,loss,typed_key,loss_key,value,override", [
    (PenalizedNegativeBinomialRegression, "negative_binomial", "alpha_nb", "alpha", .3, .65),
    (PenalizedTweedieRegression, "tweedie", "power", "power", 1.2, 1.7),
])
@pytest.mark.parametrize("entry", ["typed", "explicit", "generic"])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("compute_inference", [False, True])
def test_fitted_family_predict_and_score_preserve_effective_parameters(
        cls, loss, typed_key, loss_key, value, override, entry, penalty, compute_inference):
    effective = override if entry == "explicit" else value
    settings = {loss_key: effective}
    X, y, weights = _data(loss, settings)
    kwargs = dict(penalty=penalty, alpha=.001, device="cpu", solver="fista",
                  max_iter=10000, tol=1e-9, compute_inference=compute_inference,
                  inference_method="oracle")
    if entry == "generic":
        model = PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs)
    else:
        model = cls(**{typed_key: value},
                    loss_kwargs=settings if entry == "explicit" else None, **kwargs)
    model.fit(X, y, sample_weight=weights)
    expected = np.exp(model.intercept_ + X @ model.coef_)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model.predict(X, return_cpu=False), expected,
                               rtol=1e-13, atol=1e-13)
    for evaluation_weights in (None, weights[::-1]):
        w = np.ones(len(y)) if evaluation_weights is None else evaluation_weights
        mean = np.sum(w*y)/np.sum(w)
        expected_score = 1-np.sum(w*(y-expected)**2)/np.sum(w*(y-mean)**2)
        np.testing.assert_allclose(model.score(X, y, sample_weight=evaluation_weights),
                                   expected_score, rtol=1e-13, atol=1e-13)
    assert getattr(model._loss, loss_key) == effective
    assert getattr(model._family_for_loss(), loss_key) == effective
    if entry == "typed":
        assert model.get_params()["loss_kwargs"] is None
    if compute_inference:
        assert model._inference_result.metadata["refit_family_parameters"] == settings


@pytest.mark.parametrize("cls,typed_key,loss_key,value", [
    (PenalizedNegativeBinomialRegression, "alpha_nb", "alpha", .3),
    (PenalizedTweedieRegression, "power", "power", 1.2),
])
def test_prediction_family_uses_fitted_loss_without_evaluating_constructor_fallback(
        cls, typed_key, loss_key, value):
    loss = "negative_binomial" if loss_key == "alpha" else "tweedie"
    X, y, _ = _data(loss, {loss_key: value})
    model = cls(**{typed_key: value}, alpha=.001, device="cpu", solver="fista").fit(X, y)
    class UnreadableFallback(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Prediction read constructor fallback despite fitted loss")
    model.loss_kwargs = UnreadableFallback({loss_key: 999})
    model._family_cache = None
    np.testing.assert_allclose(model.predict(X), np.exp(model.intercept_+X@model.coef_),
                               rtol=1e-13, atol=1e-13)
    assert getattr(model._family_for_loss(), loss_key) == value


@pytest.mark.parametrize("typed_link,loss_kwargs,effective", [
    ("inverse_power", None, "inverse_power"),
    ("log", {"link": "inverse_power"}, "inverse_power"),
    ("inverse_power", {"link": "log"}, "log"),
    ("log", None, "log"),
])
@pytest.mark.parametrize("typed_entry", [False, True])
def test_gamma_prediction_and_score_preserve_fitted_link(typed_link, loss_kwargs, effective, typed_entry):
    X, y, weights = _data("gamma", {"link": effective})
    kwargs = dict(penalty="scad", alpha=.001, device="cpu", solver="fista",
                  max_iter=10000, tol=1e-9, compute_inference=True,
                  inference_method="oracle")
    model = (PenalizedGammaRegression(link=typed_link, loss_kwargs=loss_kwargs, **kwargs)
             if typed_entry else PenalizedGeneralizedLinearModel(
                 loss="gamma", loss_kwargs={"link": effective}, **kwargs))
    model.fit(X, y, sample_weight=weights)
    eta = model.intercept_+X@model.coef_
    if effective == "inverse_power":
        assert np.all((eta > 1e-4) & (eta < 1e3))
        expected = 1/eta
        assert np.max(np.abs(expected-np.exp(eta))) > .5
    else:
        expected = np.exp(eta)
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-13, atol=1e-13)
    mean = np.average(y, weights=weights)
    expected_score = 1-np.sum(weights*(y-expected)**2)/np.sum(weights*(y-mean)**2)
    np.testing.assert_allclose(model.score(X, y, sample_weight=weights), expected_score,
                               rtol=1e-13, atol=1e-13)
    assert model._loss.link == effective
    assert model._family_for_loss().link.name == effective
    assert model._inference_result.metadata["refit_family_parameters"] == {"link": effective}
    if typed_entry:
        assert model.get_params()["link"] == typed_link
        assert model.get_params()["loss_kwargs"] == loss_kwargs


def test_gamma_prediction_family_uses_resolved_loss_and_refreshes_after_refit():
    X, y, _ = _data("gamma", {"link": "inverse_power"})
    model = PenalizedGammaRegression(link="inverse_power", penalty="scad", alpha=.001,
        device="cpu", solver="fista", max_iter=10000, tol=1e-9).fit(X, y)
    assert model._family_for_loss().link.name == "inverse_power"
    model.link = "log"
    model.loss_kwargs = {"link": "log"}
    model._family_cache = None
    np.testing.assert_allclose(model.predict(X), 1/(model.intercept_+X@model.coef_),
                               rtol=1e-13, atol=1e-13)
    assert model._family_for_loss().link.name == "inverse_power"
    model.fit(X, y)
    assert model._family_for_loss().link.name == "log"
    np.testing.assert_allclose(model.predict(X), np.exp(model.intercept_+X@model.coef_),
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("cls,typed_key,loss_key,value,replacement", [
    (PenalizedNegativeBinomialRegression, "alpha_nb", "alpha", .3, .65),
    (PenalizedTweedieRegression, "power", "power", 1.2, 1.7),
])
def test_prediction_family_cache_refreshes_after_family_refit(
        cls, typed_key, loss_key, value, replacement):
    loss = "negative_binomial" if loss_key == "alpha" else "tweedie"
    X, y, _ = _data(loss, {loss_key: value})
    model = cls(**{typed_key: value}, alpha=.001, device="cpu", solver="fista").fit(X, y)
    assert getattr(model._family_for_loss(), loss_key) == value
    model.set_params(**{typed_key: replacement})
    model.fit(X, y)
    assert getattr(model._family_for_loss(), loss_key) == replacement
    np.testing.assert_allclose(model.predict(X), np.exp(model.intercept_+X@model.coef_),
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("device", ["cuda", "torch"])
@pytest.mark.parametrize("cls,loss,settings,typed_settings", [
    (PenalizedGammaRegression, "gamma", {"link": "inverse_power"}, {"link": "inverse_power"}),
    (PenalizedNegativeBinomialRegression, "negative_binomial", {"alpha": .3}, {"alpha_nb": .3}),
    (PenalizedTweedieRegression, "tweedie", {"power": 1.2}, {"power": 1.2}),
])
def test_physical_gpu_family_predictions_remain_on_fitted_backend(
        device, cls, loss, settings, typed_settings):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip("No physical CUDA device")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("No physical Torch CUDA device")
    X, y, _ = _data(loss, settings)
    model = cls(**typed_settings, device=device, penalty="scad", alpha=.001,
                solver="fista", max_iter=10000, tol=1e-9).fit(X, y)
    native = model.predict(X, return_cpu=False)
    if device == "cuda":
        assert isinstance(native, cp.ndarray)
        assert f"cuda:{native.device.id}" == model._selected_backend_device
        actual = cp.asnumpy(native)
    else:
        assert isinstance(native, torch.Tensor) and native.is_cuda
        assert str(native.device) == model._selected_backend_device
        actual = native.detach().cpu().numpy()
    eta = model.intercept_+X@model.coef_
    expected = 1/np.clip(eta, 1e-4, 1e3) if loss == "gamma" else np.exp(eta)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[5:7])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_robust_bread_rejects_fisher_substitution(
        loss, settings, typed, typed_settings, cov_type, weighted):
    X, y, weights = _data(loss, settings)
    sample_weight = weights if weighted else None
    model = typed(**typed_settings, penalty="mcp", alpha=.0001, solver="fista",
        device="cpu", max_iter=10000, tol=1e-10, compute_inference=True,
        inference_method="oracle", cov_type=cov_type).fit(X, y, sample_weight=sample_weight)
    _assert_result(model, loss, settings, X, y, sample_weight, cov_type)
    design = np.column_stack([np.ones(len(y)), X[:, [0, 2]]])
    params, expected_se = _reference(loss, settings, design, y, sample_weight, cov_type)
    psi, _, fisher, _, _ = _moments(loss, settings, design @ params, y)
    w = weights if weighted else np.ones(len(y))
    fisher_bread = np.linalg.inv(design.T @ ((w*fisher)[:, None]*design))
    meat = design.T @ (((w*psi)**2)[:, None]*design)
    wrong_cov = fisher_bread @ meat @ fisher_bread
    if cov_type == "hc1":
        wrong_cov *= len(y)/(len(y)-design.shape[1])
    wrong_se = np.sqrt(np.diag(wrong_cov))
    assert np.max(np.abs(expected_se-wrong_se)/expected_se) > 1e-3
    # The fit still uses its original Fisher solver curvature. The observed
    # bread belongs only to oracle robust inference, not the shared loss.
    np.testing.assert_allclose(model._loss.hessian(design, y, params, sample_weight),
        design.T @ ((w*fisher)[:, None]*design)/w.sum(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[5:7])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_observed_bread_matches_finite_difference_of_fitted_score(
        loss, settings, typed, typed_settings, weighted):
    from statgpu.linear_model.penalized._inference_mixin import _OracleObservedHessianLoss
    X, y, weights = _data(loss, settings)
    design = np.column_stack([np.ones(len(y)), X[:, [0, 2]]])
    model = typed(**typed_settings)
    resolved = model._resolve_loss()
    adapter = _OracleObservedHessianLoss(resolved)
    params = np.array([.5, .3, .2])
    sample_weight = weights if weighted else None
    eye = np.eye(len(params))
    step = 1e-6
    numerical = np.column_stack([
        (resolved.gradient(design, y, params+step*e, sample_weight=sample_weight)
         - resolved.gradient(design, y, params-step*e, sample_weight=sample_weight))/(2*step)
        for e in eye
    ])
    np.testing.assert_allclose(adapter.hessian(design, y, params, sample_weight),
                               numerical, rtol=2e-8, atol=2e-10)
    assert adapter.gradient.__self__ is resolved
    assert adapter.score_outer.__self__ is resolved
    assert adapter.fisher_information.__self__ is resolved
    assert model._resolve_loss().__class__ is resolved.__class__


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[5:7])
def test_robust_oracle_weight_scaling_and_hc1_observation_count(
        loss, settings, typed, typed_settings):
    X, y, weights = _data(loss, settings)
    kwargs = dict(**typed_settings, penalty="scad", alpha=.0001, solver="fista",
                  device="cpu", max_iter=10000, tol=1e-10, compute_inference=True,
                  inference_method="oracle")
    hc0 = typed(**kwargs, cov_type="hc0").fit(X, y, sample_weight=weights)
    scaled = typed(**kwargs, cov_type="hc0").fit(X, y, sample_weight=7*weights)
    hc1 = typed(**kwargs, cov_type="hc1").fit(X, y, sample_weight=weights)
    indices = [0, 1, 3]
    np.testing.assert_allclose(scaled._params, hc0._params, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(scaled._bse[indices], hc0._bse[indices], rtol=1e-6, atol=1e-7)
    correction = np.sqrt(len(y)/(len(y)-len(indices)))
    np.testing.assert_allclose(hc1._bse[indices], correction*hc0._bse[indices],
                               rtol=1e-13, atol=1e-13)
    positive_count = np.count_nonzero(weights)
    wrong_correction = np.sqrt(positive_count/(positive_count-len(indices)))
    assert abs(correction-wrong_correction) > 1e-4


@pytest.mark.parametrize("loss,cls,typed_key,loss_key,value,override", [
    ("negative_binomial", PenalizedNegativeBinomialRegression, "alpha_nb", "alpha", .2, .7),
    ("tweedie", PenalizedTweedieRegression, "power", "power", 1.2, 1.8),
])
@pytest.mark.parametrize("explicit_override", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_adaptive_initialization_preserves_resolved_typed_family(
        loss, cls, typed_key, loss_key, value, override, explicit_override, weighted, monkeypatch):
    from statgpu.linear_model.penalized import _fit_mixin
    rng = np.random.default_rng(333)
    X = rng.normal(size=(300, 4))
    mu = np.exp(.2+X @ np.array([.6, -.3, .1, 0.]))
    y = rng.negative_binomial(2, 2/(2+mu))
    weights = rng.uniform(.3, 2., len(y)) if weighted else None
    effective = override if explicit_override else value
    settings = {loss_key: effective}
    observed = []
    original = _fit_mixin._irls_ridge_init
    def capture(*args, **kwargs):
        observed.append(dict(kwargs["loss_kwargs"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(_fit_mixin, "_irls_ridge_init", capture)
    kwargs = dict(penalty="adaptive_l1", alpha=.03, solver="fista", device="cpu",
                  max_iter=10000, tol=1e-10, compute_inference=False)
    model = clone(cls(**{typed_key: value},
        loss_kwargs=settings if explicit_override else None, **kwargs))
    model.fit(X, y, sample_weight=weights)
    generic = PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs)
    generic.fit(X, y, sample_weight=weights)
    assert observed == [settings, settings]
    assert model.get_params()["loss_kwargs"] == (settings if explicit_override else None)
    np.testing.assert_allclose(model.coef_, generic.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.intercept_, generic.intercept_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.predict(X), generic.predict(X), rtol=1e-12, atol=1e-12)
    if not explicit_override and not weighted:
        default_family = PenalizedGeneralizedLinearModel(loss=loss, **kwargs).fit(X, y)
        assert np.max(np.abs(model.coef_-default_family.coef_)) > 1e-4


_BOUNDARY_RESPONSES = [
    ("poisson", PenalizedPoissonRegression, 0.),
    ("negative_binomial", PenalizedNegativeBinomialRegression, 0.),
    ("tweedie", PenalizedTweedieRegression, 0.),
    ("logistic", PenalizedLogisticRegression, 0.),
    ("logistic", PenalizedLogisticRegression, 1.),
]


def _assert_failed_oracle_fit(model, X):
    assert model._fitted is False
    for name in ("coef_", "intercept_", "_params", "_inference_result", "_bse",
                 "_tvalues", "_zvalues", "_pvalues", "_conf_int",
                 "inference_target_", "inference_resolved_method_",
                 "_selected_solver", "_selected_backend_name", "_selected_backend_device",
                 "_native_fit_coef", "_native_fit_intercept", "_design_info",
                 "_feature_names", "_family_cache"):
        assert getattr(model, name, None) is None, name
    with pytest.raises(RuntimeError, match="not.*fitted|has not been fitted"):
        model.predict(X)


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("entry", ["typed", "generic"])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_boundary_response_default_solver_fit_rejects_nonfinite_target(
        loss, typed, boundary, entry, penalty, cov_type, weighted):
    # Genuine public fits with default convergence controls. Positive responses
    # (or the other logistic class) exclusively in zero-weight rows do not
    # change the target. No fitted-state injection or fake convergence signal.
    X = np.zeros((50, 2))
    y = np.full(len(X), boundary)
    weights = None
    if weighted:
        weights = np.linspace(.2, 2., len(X))
        weights[::5] = 0
        y[::5] = 1. - boundary
    kwargs = dict(penalty=penalty, alpha=.1, solver="fista", device="cpu",
                  compute_inference=True, inference_method="oracle", cov_type=cov_type)
    model = typed(**kwargs) if entry == "typed" else PenalizedGeneralizedLinearModel(
        loss=loss, **kwargs)
    with pytest.raises(ValueError, match="oracle refit has no finite optimum"):
        model.fit(X, y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_boundary_failed_refit_clears_old_results_and_can_recover(
        loss, typed, boundary, weighted):
    X = np.zeros((60, 2))
    good_y = np.tile([0., 1.] if loss == "logistic" else [0., 1., 2.],
                     30 if loss == "logistic" else 20)
    weights = np.linspace(.4, 1.6, len(X)) if weighted else None
    model = typed(penalty="mcp", alpha=.1, solver="fista", device="cpu",
                  compute_inference=True, inference_method="oracle")
    model.fit(X, good_y, sample_weight=weights)
    old_params = model._params.copy()
    old_bse = model._bse.copy()
    with pytest.raises(ValueError, match="no finite optimum"):
        model.fit(X, np.full(len(X), boundary), sample_weight=weights)
    _assert_failed_oracle_fit(model, X)
    assert model.inference_method == "oracle"
    model.fit(X, good_y, sample_weight=weights)
    np.testing.assert_array_equal(model._params, old_params)
    np.testing.assert_array_equal(model._bse, old_bse)
    assert model._inference_result.method == "oracle"


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("design", ["empty", "constant", "varying_positive", "quasi"])
def test_oracle_boundary_no_intercept_distinguishes_fixed_and_unbounded_targets(
        loss, typed, boundary, design):
    x = {"empty": np.zeros(60), "constant": np.ones(60),
         "varying_positive": np.tile([.5, 2.], 30),
         "quasi": np.tile([0., 1.], 30)}[design]
    X = np.column_stack([x, np.zeros(len(x))])
    model = typed(penalty="mcp", alpha=.001, solver="fista", device="cpu",
                  fit_intercept=False, compute_inference=True, inference_method="oracle")
    if design == "empty":
        model.fit(X, np.full(len(X), boundary))
        assert model._inference_result.metadata["n_active"] == 0
        assert np.isnan(model._bse).all()
        assert np.isnan(model._pvalues).all()
        assert np.isnan(model._conf_int).all()
    else:
        with pytest.raises(ValueError, match="no finite optimum"):
            model.fit(X, np.full(len(X), boundary))
        _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_boundary_mixed_sign_no_intercept_preserves_finite_refit(
        loss, typed, boundary, columns, weighted):
    X = (np.tile([[-1.], [2.]], (30, 1)) if columns == 1
         else np.tile([[1., 0.], [0., 1.], [-2., -3.]], (20, 1)))
    y = np.full(len(X), boundary)
    weights = np.linspace(.8, 1.2, len(X)) if weighted else None
    model = typed(penalty="mcp", alpha=.000001, solver="fista", device="cpu",
                  fit_intercept=False, max_iter=10000, tol=1e-8,
                  compute_inference=True, inference_method="oracle").fit(
                      X, y, sample_weight=weights)
    assert model._inference_result.metadata["n_active"] == columns
    settings = ({"power": 1.5} if loss == "tweedie" else
                {"alpha": 1.} if loss == "negative_binomial" else {})
    params, se = _reference(loss, settings, X, y, weights)
    np.testing.assert_allclose(model._params, params, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model._bse, se, rtol=1e-6, atol=1e-7)
    assert np.isfinite(model._conf_int).all()


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_boundary_multicolumn_recession_uses_only_effective_design_rows(
        loss, typed, boundary, weighted):
    # Each column has both signs, but d=(1, 1) gives margins (0, 1).
    # Rescale rows/columns without changing the existence of that direction.
    X = np.tile([[.25, -.25], [-4., 8.]], (25, 1)) * [.5, 2.]
    weights = None
    if weighted:
        X = np.vstack([X, -X[:2]])
        weights = np.r_[np.ones(50), 0., 0.]
    y = np.full(len(X), boundary)
    kwargs = dict(penalty="mcp", alpha=.000001, solver="fista", device="cpu",
                  fit_intercept=False)
    selection = typed(**kwargs).fit(X, y, sample_weight=weights)
    assert np.count_nonzero(np.abs(selection.coef_) > 1e-10) == 2
    model = typed(**kwargs, compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError, match="no finite optimum"):
        model.fit(X, y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
def test_oracle_boundary_nearly_antipodal_recession_is_not_treated_as_lp_infeasibility(
        loss, typed, boundary):
    X = np.vstack([np.tile([[1., 1.]], (20, 1)),
                   np.tile([[-1., -1.+1e-10]], (40, 1))])
    kwargs = dict(penalty="mcp", alpha=.000001, solver="fista", device="cpu",
                  fit_intercept=False)
    selection = typed(**kwargs).fit(X, np.full(len(X), boundary))
    assert np.count_nonzero(np.abs(selection.coef_) > 1e-10) == 2
    model = typed(**kwargs, compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError, match="no finite optimum"):
        model.fit(X, np.full(len(X), boundary))
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("design", ["near_finite", "rank_deficient"])
def test_oracle_boundary_numerical_uncertainty_is_not_called_nonexistent_optimum(design):
    X = (np.tile([[1., 1.], [-1., -1.+1e-10], [0., -1.]], (20, 1))
         if design == "near_finite" else np.tile([[-1., -1.], [2., 2.]], (30, 1)))
    model = PenalizedTweedieRegression(penalty="mcp", alpha=.000001, solver="fista",
        device="cpu", fit_intercept=False, compute_inference=True, inference_method="oracle")
    with pytest.raises(RuntimeError, match="could not determine whether"):
        model.fit(X, np.zeros(len(X)))
    _assert_failed_oracle_fit(model, X)


def test_oracle_boundary_every_positive_weight_participates_without_thresholding():
    X = np.tile([[-1.], [2.]], (30, 1))
    weights = np.tile([1e-8, 1.], 30)
    model = PenalizedPoissonRegression(penalty="mcp", alpha=.000001, solver="fista",
        device="cpu", fit_intercept=False, max_iter=10000, tol=1e-12,
        compute_inference=True, inference_method="oracle").fit(
            X, np.zeros(len(X)), sample_weight=weights)
    # w_minus exp(-b) = 2 w_plus exp(2b), a finite target even for tiny w_minus.
    expected = np.log(1e-8/2)/3
    np.testing.assert_allclose(model._params, [expected], rtol=1e-6, atol=1e-7)
    assert np.isfinite(model._conf_int).all()


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_boundary_cv_final_refit_rejection_is_transactional(
        loss, typed, boundary, penalty, weighted):
    X = np.zeros((60, 2))
    good_y = np.tile([0., 1.] if loss == "logistic" else [0., 1., 2.],
                     30 if loss == "logistic" else 20)
    weights = np.ones(len(X)) if weighted else None
    bad_y = np.full(len(X), boundary)
    if weighted:
        weights[::5] = 0
        bad_y[::5] = 1. - boundary
    model = PenalizedGLM_CV(loss=loss, penalty=penalty, alpha_grid=[.1, .2], cv=2,
        solver="fista", device="cpu", compute_inference=True, inference_method="oracle")
    model.fit(X, good_y, sample_weight=weights)
    assert model._inference_result is model.estimator_._inference_result
    with pytest.raises(ValueError, match="no finite optimum"):
        model.fit(X, bad_y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)
    for name in ("estimator_", "alpha_", "cv_results_", "best_score_"):
        assert getattr(model, name) is None
    model.fit(X, good_y, sample_weight=weights)
    assert model._inference_result is model.estimator_._inference_result
    assert model.penalty_conditioning_ == "cv_selected_penalty"


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[:1] + FAMILIES[2:])
@pytest.mark.parametrize("constant", [False, True])
def test_oracle_boundary_guard_preserves_finite_intercept_only_family_targets(
        loss, settings, typed, typed_settings, constant):
    X = np.zeros((60, 2))
    y = np.full(60, 2.) if constant else np.tile([.5, 1., 2.], 20)
    weights = np.linspace(.5, 1.5, len(X))
    model = typed(**typed_settings, penalty="mcp", alpha=.1, solver="fista",
        device="cpu", max_iter=10000, tol=1e-8, compute_inference=True, inference_method="oracle").fit(
            X, y, sample_weight=weights)
    mean = np.average(y, weights=weights)
    expected = (mean if loss == "squared_error" else 1/mean
                if loss == "gamma" and settings["link"] == "inverse_power" else np.log(mean))
    np.testing.assert_allclose(model._params[0], expected, rtol=2e-4, atol=2e-5)
    assert np.isfinite(model._bse[0])
    assert np.isnan(model._bse[1:]).all()


@pytest.mark.parametrize("value", [0., 3.])
def test_oracle_boundary_constant_gaussian_has_finite_target(value):
    X = np.zeros((50, 2))
    model = PenalizedLinearRegression(penalty="mcp", alpha=.1, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle").fit(
            X, np.full(len(X), value))
    np.testing.assert_allclose(model._params[0], value)
    np.testing.assert_allclose(model._bse[0], 0, atol=1e-14)
    np.testing.assert_allclose(model._conf_int[0], [value, value])


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[3:6])
def test_oracle_boundary_positive_response_domains_reject_zero_before_inference(
        loss, settings, typed, typed_settings):
    X = np.zeros((50, 2))
    model = typed(**typed_settings, penalty="mcp", alpha=.1, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(X, np.zeros(len(X)))
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("power", [0., 1., 2., 3.])
def test_oracle_boundary_tweedie_guard_does_not_expand_supported_power_domain(power):
    model = PenalizedTweedieRegression(power=power, penalty="mcp", alpha=.1,
        solver="fista", device="cpu", compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError):
        model.fit(np.zeros((50, 2)), np.zeros(50))
    assert model._inference_result is None


@pytest.mark.parametrize("loss,typed,boundary", _BOUNDARY_RESPONSES)
@pytest.mark.parametrize("intercept", [False, True])
def test_oracle_boundary_formula_uses_effective_intercept_and_retained_weights(
        loss, typed, boundary, intercept):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X = np.zeros(60) if intercept else np.ones(60)
    y = np.full(60, boundary)
    weights = np.ones(60)
    weights[::7] = 0
    y[::7] = 1. - boundary
    # Drop a row of the opposite response, independently of the zero weights.
    X[1] = np.nan
    y[1] = 1. - boundary
    frame = pd.DataFrame({"x": X, "y": y})
    model = typed(penalty="mcp", alpha=.001, solver="fista", device="cpu",
                  compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError, match="no finite optimum"):
        model.fit(formula="y ~ x" if intercept else "y ~ 0 + x",
                  data=frame, sample_weight=weights)
    _assert_failed_oracle_fit(model, np.zeros((3, 1)))
    assert model._design_info is None
    assert model._feature_names is None


_UNKNOWN_SCALE_FAMILIES = FAMILIES[:1] + FAMILIES[3:6] + FAMILIES[7:]


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
@pytest.mark.parametrize("entry", ["typed", "generic"])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("weight_scale", [None, 1e-6, 1., 1e6])
def test_oracle_residual_df_rejects_unknown_scale_with_one_effective_observation(
        loss, settings, typed, typed_settings, entry, penalty, weight_scale):
    # Analytic weights are not replication counts. Neither large weights nor
    # zero-weight padding can create residual information for one intercept.
    X, y, weights = np.zeros((1, 3)), np.array([2.]), None
    if weight_scale is not None:
        X, y = np.zeros((4, 3)), np.array([2., 3., 5., 7.])
        weights = weight_scale * np.array([1., 0., 0., 0.])
    kwargs = dict(penalty=penalty, alpha=.1, solver="fista", device="cpu",
                  compute_inference=True, inference_method="oracle")
    model = (typed(**typed_settings, **kwargs) if entry == "typed" else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(X, y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)
    if hasattr(model, "summary"):
        with pytest.raises(RuntimeError):
            model.summary()
    # The suggested estimation-only alternative remains a real public fit.
    model.set_params(compute_inference=False)
    model.fit(X, y, sample_weight=weights)
    assert np.isfinite(model.predict(X)).all()
    assert model._inference_result is None


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_oracle_residual_df_counts_selected_slopes_and_effective_intercept(
        loss, settings, typed, typed_settings, fit_intercept, penalty):
    X = np.array([[1.], [2.]]) if fit_intercept else np.ones((1, 1))
    y = np.array([2., 4.]) if fit_intercept else np.array([2.])
    kwargs = dict(penalty=penalty, alpha=.0001, solver="fista", device="cpu",
                  fit_intercept=fit_intercept)
    selection = typed(**typed_settings, **kwargs).fit(X, y)
    assert np.count_nonzero(np.abs(selection.coef_) > 1e-10) == 1
    model = typed(**typed_settings, **kwargs, compute_inference=True,
                  inference_method="oracle")
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(X, y)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_oracle_residual_df_does_not_rescue_unidentifiable_coefficients_by_rank(fit_intercept):
    # Two selected but collinear slopes have rank one. Using rank in the
    # availability gate would wrongly rescue the two reported coefficients.
    X = np.array([[1., 1.], [2., 2.]])
    y = np.array([2., 4.])
    kwargs = dict(penalty="scad", alpha=.0001, solver="fista", device="cpu",
                  fit_intercept=fit_intercept)
    selection = PenalizedLinearRegression(**kwargs).fit(X, y)
    assert np.count_nonzero(np.abs(selection.coef_) > 1e-10) == 2
    assert np.linalg.matrix_rank(X) == 1
    # With an intercept, k=3>n; without it, k=n=2.
    model = PenalizedLinearRegression(**kwargs, compute_inference=True,
                                     inference_method="oracle")
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(X, y)
    _assert_failed_oracle_fit(model, X)


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
@pytest.mark.parametrize("entry", ["typed", "generic"])
@pytest.mark.parametrize("weight_scale", [None, 1e-6, 1., 1e6])
def test_oracle_residual_df_preserves_positive_df_weighted_inference(
        loss, settings, typed, typed_settings, entry, weight_scale, capsys):
    X, y, weights = np.zeros((2, 7)), np.array([1., 3.]), None
    if weight_scale is not None:
        X, y = np.zeros((3, 7)), np.array([1., 3., 7.])
        weights = weight_scale * np.array([1., 3., 0.])
    kwargs = dict(penalty="mcp", alpha=.1, solver="fista", device="cpu",
                  max_iter=10000, tol=1e-10,
                  compute_inference=True, inference_method="oracle")
    model = (typed(**typed_settings, **kwargs) if entry == "typed" else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    model.fit(X, y, sample_weight=weights)
    result = model._inference_result
    # Inactive columns do not consume refit df, and small sums of analytic
    # weights do not reduce observation count. Preserve the established
    # raw-row n-k scale convention once enough positive-weight rows exist.
    expected, se = _reference(loss, settings, np.ones((len(y), 1)), y, weights)
    np.testing.assert_allclose(result.params[:1], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.bse[:1], se, rtol=1e-6, atol=1e-7)
    z = expected / se
    np.testing.assert_allclose(result.statistic[:1], z, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.pvalues[:1], 2*stats.norm.sf(abs(z)),
                               rtol=2e-5, atol=1e-10)
    np.testing.assert_allclose(result.conf_int[:1],
        expected[:, None] + se[:, None]*stats.norm.ppf([.025, .975]),
        rtol=1e-6, atol=1e-7)
    assert se[0] > 0
    assert np.isnan(result.bse[1:]).all()
    assert result.metadata["n_active"] == 0
    if hasattr(model, "summary"):
        model.summary()
        output = capsys.readouterr().out
        assert "(Intercept)" in output
        assert f"{se[0]:.4f}" in output


@pytest.mark.parametrize("loss,settings,typed,typed_settings", [FAMILIES[2], FAMILIES[6]])
@pytest.mark.parametrize("entry", ["typed", "generic"])
@pytest.mark.parametrize("weight_scale", [None, 1e-6, 1e6])
def test_oracle_residual_df_preserves_fixed_dispersion_at_saturation(
        loss, settings, typed, typed_settings, entry, weight_scale):
    X, y, weights = np.zeros((1, 2)), np.array([2.]), None
    if weight_scale is not None:
        X, y = np.zeros((3, 2)), np.array([2., 4., 6.])
        weights = weight_scale * np.array([1., 0., 0.])
    kwargs = dict(penalty="mcp", alpha=.1, solver="fista", device="cpu",
                  max_iter=10000, tol=1e-10,
                  compute_inference=True, inference_method="oracle")
    model = (typed(**typed_settings, **kwargs) if entry == "typed" else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    model.fit(X, y, sample_weight=weights)
    # Fixed-dispersion Fisher information: w*mu (Poisson),
    # w*mu/(1+alpha*mu) (NB); residual df is not needed.
    weight = 1. if weights is None else weights.sum()
    variance_factor = 1. if loss == "poisson" else 1.+2*settings["alpha"]
    se = np.sqrt(variance_factor/(2*weight))
    np.testing.assert_allclose(model._params[0], np.log(2), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-6, atol=1e-7)
    assert model._inference_result.metadata["dispersion"] == 1.


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_oracle_residual_df_preserves_existing_robust_saturated_paths(
        loss, settings, typed, typed_settings, cov_type):
    model = typed(**typed_settings, penalty="scad", alpha=.1, solver="fista",
        device="cpu", max_iter=10000, tol=1e-10, cov_type=cov_type,
        compute_inference=True, inference_method="oracle").fit(
            np.zeros((1, 2)), np.array([2.]))
    # HC0 is an uncorrected empirical sandwich, which is zero at this exact
    # fit. HC1's existing n<=k convention is outside this dispersion repair.
    assert model._inference_result.cov_type == cov_type
    np.testing.assert_allclose(model._bse[0], 0, atol=1e-10)
    assert np.isnan(model._bse[1:]).all()


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
def test_oracle_residual_df_preserves_empty_no_intercept_target(
        loss, settings, typed, typed_settings):
    model = typed(**typed_settings, penalty="mcp", alpha=.1, fit_intercept=False,
        solver="fista", device="cpu", compute_inference=True,
        inference_method="oracle")
    X, y = np.zeros((1, 2)), np.array([2.])
    if settings.get("link") == "inverse_power":
        with pytest.raises(ValueError, match="positive inverse-link predictor"):
            model.fit(X, y)
        _assert_failed_oracle_fit(model, X)
        return
    model.fit(X, y)
    assert model._inference_result.metadata["n_active"] == 0
    assert np.isnan(model._bse).all()


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
@pytest.mark.parametrize("intercept", [False, True])
def test_oracle_residual_df_formula_retained_rows_and_effective_intercept(
        loss, settings, typed, typed_settings, intercept):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    frame = pd.DataFrame({"x": [0., np.nan, 0.] if intercept else [1., np.nan, 2.],
                          "y": [2., 3., 4.]})
    weights = np.array([1., 1., 0.])
    # Deliberately keep the constructor default True for the no-intercept
    # formula: only the actual formula matrix determines refit columns.
    model = typed(**typed_settings, penalty="mcp", alpha=.0001, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle")
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(formula="y ~ x" if intercept else "y ~ 0 + x",
                  data=frame, sample_weight=weights)
    _assert_failed_oracle_fit(model, np.zeros((1, 1)))
    assert model._design_info is None
    assert model._feature_names is None


@pytest.mark.parametrize("loss,settings,typed,typed_settings", _UNKNOWN_SCALE_FAMILIES)
def test_oracle_residual_df_cv_final_refit_rejection_and_recovery(
        loss, settings, typed, typed_settings):
    X, y = np.array([[1.], [2.], [3.]]), np.array([2., 4., 5.])
    model = PenalizedGLM_CV(loss=loss, loss_kwargs=settings, penalty="scad",
        alpha_grid=[.0001, .0002], cv=2, solver="fista", device="cpu",
        compute_inference=True, inference_method="oracle")
    model.fit(X, y)
    assert model._inference_result is model.estimator_._inference_result
    assert np.isfinite(model._bse).all()
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(X[:2], y[:2])
    _assert_failed_oracle_fit(model, X)
    for name in ("estimator_", "alpha_", "cv_results_", "best_score_"):
        assert getattr(model, name) is None
    model.fit(X, y)
    assert model._inference_result is model.estimator_._inference_result
    assert model.penalty_conditioning_ == "cv_selected_penalty"


def test_oracle_residual_df_summary_cannot_reuse_previous_valid_inference(capsys):
    X, y = np.zeros((2, 3)), np.array([1., 3.])
    model = PenalizedLinearRegression(penalty="scad", alpha=.1, solver="fista",
        device="cpu", compute_inference=True, inference_method="oracle").fit(X, y)
    model.summary()
    assert "(Intercept)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.fit(X[:1], y[:1])
    _assert_failed_oracle_fit(model, X)
    with pytest.raises(RuntimeError, match="not.*fitted|has not been fitted"):
        model.summary()
    assert capsys.readouterr().out == ""
    model.fit(X, y)
    model.summary()
    assert "(Intercept)" in capsys.readouterr().out
    np.testing.assert_allclose(model._bse[0], 1., rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc2", "hc3", "hac"])
@pytest.mark.parametrize("design", ["active", "intercept_only"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_oracle_covariance_failure_clears_complete_typed_and_generic_fit(
        cls, penalty, cov_type, design, previous_fit):
    # Intercept-only oracle refitting newly reaches covariance validation.
    # The typed Gaussian fit must not leave usable coefficients after it raises.
    x = np.linspace(.2, 1.2, 18)
    X = x[:, None] if design == "active" else np.zeros((len(x), 2))
    y = 2. * x + .1 * np.sin(5. * x)
    weights = np.linspace(.4, 1.6, len(x))
    weights[::7] = 0.
    model = cls(penalty=penalty, alpha=.0001, solver="fista", device="cpu",
        fit_intercept=design == "intercept_only" and not previous_fit, compute_inference=True,
        inference_method="oracle", cov_type=cov_type)
    if previous_fit:
        # An empty no-intercept target has no covariance to compute. Start with
        # that successful public fit, then request a nonempty refit without
        # reconstructing the estimator or injecting any fitted state.
        model.fit(np.zeros_like(X), y, sample_weight=weights)
        assert model._fitted
        assert np.isfinite(model.predict(X)).all()
        assert model._inference_result is not None
    message = "HAC requires lag selection" if cov_type == "hac" else "HC2/HC3 requires leverage"
    with pytest.raises(NotImplementedError, match=message):
        if previous_fit and design == "intercept_only":
            pd = pytest.importorskip("pandas")
            pytest.importorskip("patsy")
            model.fit(formula="y ~ 1", data=pd.DataFrame({"y": y}),
                      sample_weight=weights)
        else:
            model.fit(X, y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)
    assert model.inference_method == "oracle"
    assert model.cov_type == cov_type
    if hasattr(model, "summary"):
        with pytest.raises(RuntimeError, match="not.*fitted|has not been fitted"):
            model.summary()
    model.set_params(cov_type="nonrobust", fit_intercept=design == "intercept_only")
    model.fit(X, y, sample_weight=weights)
    assert model._fitted
    assert model._inference_result.method == "oracle"
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
def test_oracle_empty_no_intercept_preserves_undefined_uncertainty(cls, penalty, cov_type):
    # With no refitted parameter there is no covariance calculation to reject.
    X, y = np.zeros((6, 2)), np.arange(1., 7.)
    model = cls(penalty=penalty, alpha=.1, solver="fista", device="cpu",
        fit_intercept=False, compute_inference=True, inference_method="oracle",
        cov_type=cov_type).fit(X, y)
    assert model._fitted
    np.testing.assert_array_equal(model.predict(X), 0.)
    np.testing.assert_array_equal(model._inference_result.params, 0.)
    assert np.isnan(model._inference_result.bse).all()
    assert model._inference_result.metadata["n_active"] == 0
    assert "refit_model" not in model._inference_result.metadata


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc2", "hc3", "hac"])
def test_oracle_transaction_does_not_change_estimation_only_fit(cls, penalty, cov_type):
    X, y = np.zeros((6, 2)), np.arange(1., 7.)
    model = cls(penalty=penalty, alpha=.1, solver="fista", device="cpu",
        compute_inference=False, inference_method="oracle", cov_type=cov_type).fit(X, y)
    assert model._fitted
    np.testing.assert_allclose(model.predict(X), y.mean())
    assert model._inference_result is None


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_oracle_rank_failure_clears_state_and_recovers(cls, penalty, previous_fit):
    x = np.linspace(.2, 1.2, 30)
    good_X = np.column_stack([x, np.sin(4. * x)])
    bad_X = np.column_stack([x, x])
    y = 1. + 2. * x + .1 * np.cos(3. * x)
    model = cls(penalty=penalty, alpha=.0001, solver="fista", device="cpu",
        compute_inference=True, inference_method="oracle", tol=1e-8)
    if previous_fit:
        model.fit(good_X, y)
        assert model._inference_result is not None
        model.predict(good_X)
    with pytest.raises(np.linalg.LinAlgError, match="Singular Hessian"):
        model.fit(bad_X, y)
    _assert_failed_oracle_fit(model, bad_X)
    model.fit(good_X, y)
    assert model._fitted
    assert model._inference_result.method == "oracle"
    assert np.isfinite(model.predict(good_X)).all()


@pytest.mark.parametrize("entry", ["typed", "generic"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_oracle_public_child_nonconvergence_is_transactional(entry, previous_fit):
    X, y, weights = _data("logistic", {})
    kwargs = dict(penalty="scad", alpha=.0001, solver="fista", device="cpu",
        max_iter=1, tol=1e-12, compute_inference=True,
        inference_method="oracle")
    model = (PenalizedLogisticRegression(**kwargs) if entry == "typed" else
             PenalizedGeneralizedLinearModel(loss="logistic", **kwargs))
    if previous_fit:
        # The balanced intercept-only MLE starts at zero and converges within
        # this same one-iteration budget; the nontrivial refit below cannot.
        model.fit(np.zeros_like(X), np.tile([0., 1.], len(X) // 2))
        assert model._fitted
        assert model._inference_result is not None
    with pytest.raises(RuntimeError, match="oracle refit did not converge"):
        model.fit(X, y, sample_weight=weights)
    _assert_failed_oracle_fit(model, X)
    model.set_params(max_iter=10000)
    model.fit(X, y, sample_weight=weights)
    assert model._fitted
    assert model._inference_result.method == "oracle"


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("weight_scale", [1e-170, 1., 1e160])
def test_oracle_intercept_robust_extreme_weights_match_exact_gaussian_se(
        cls, penalty, cov_type, weight_scale):
    X, y = np.zeros((60, 2)), np.tile([1., 2., 4.], 20)
    weights = np.tile([.4, 1., 1.6], 20)
    weights[::7] = 0.
    mean = np.dot(weights, y) / weights.sum()
    # Compute the scale-free closed form before forming extreme weights.
    se = np.sqrt(np.sum((weights * (y - mean)) ** 2)) / weights.sum()
    if cov_type == "hc1":
        se *= np.sqrt(len(y) / (len(y) - 1))
    model = cls(penalty=penalty, alpha=.1, solver="fista", device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=weights * weight_scale)
    np.testing.assert_allclose(model._params[0], mean, rtol=1e-13)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-13)
    np.testing.assert_allclose(model._conf_int[0],
        mean + se * stats.norm.ppf([.025, .975]), rtol=1e-13)
    assert model._bse[0] > 0
    assert np.isfinite(model._conf_int[0]).all()
    assert np.isnan(model._bse[1:]).all()


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES)
@pytest.mark.parametrize("typed_entry", [False, True])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("intercept_only", [False, True])
def test_oracle_robust_weight_scale_matches_independent_family_score_bread(
        loss, settings, typed, typed_settings, typed_entry, penalty,
        cov_type, intercept_only):
    X, y, weights = _data(loss, settings)
    if intercept_only:
        X = np.zeros_like(X)
    # The Gaussian selection solver has separate extreme-scale limits; do not
    # turn this inference-only test into a change to its selected support.
    scales = ([1e-6, 1., 1e6] if loss == "squared_error" and not intercept_only
              else [1e-170, 1., 1e160])
    for scale in scales:
        kwargs = dict(penalty=penalty, alpha=.0001, solver="fista", device="cpu",
            max_iter=10000, tol=1e-10, compute_inference=True,
            inference_method="oracle", cov_type=cov_type)
        model = (typed(**kwargs, **typed_settings) if typed_entry else
                 PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
        model.fit(X, y, sample_weight=weights * scale)
        if not intercept_only:
            _assert_result(model, loss, settings, X, y, weights, cov_type)
        else:
            params, se = _reference(loss, settings, np.ones((len(y), 1)),
                                    y, weights, cov_type)
            np.testing.assert_allclose(model._params[:1], params, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(model._bse[:1], se, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(model._pvalues[:1],
                2 * stats.norm.sf(np.abs(params / se)), rtol=1e-5, atol=1e-10)
            np.testing.assert_allclose(model._conf_int[:1],
                params[:, None] + se[:, None] * stats.norm.ppf([.025, .975]),
                rtol=1e-6, atol=1e-7)
            assert model._inference_result.metadata["n_active"] == 0
            assert np.isnan(model._bse[1:]).all()
        assert np.isfinite(model._bse[0])


@pytest.mark.parametrize("weight_scale", [1e-170, 1., 1e160])
def test_oracle_robust_scaling_keeps_original_child_fit_weights(monkeypatch, weight_scale):
    original = LogisticRegression.fit
    seen = []

    def record(self, X, y, **kwargs):
        seen.append(np.array(kwargs["sample_weight"], copy=True))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", record)
    X, y = np.zeros((60, 1)), np.tile([0., 0., 1.], 20)
    weights = np.linspace(.4, 1.6, len(y)) * weight_scale
    weights[::7] = 0.
    PenalizedLogisticRegression(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type="hc0").fit(
            X, y, sample_weight=weights)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], weights)


@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_oracle_robust_weight_scaling_preserves_valid_zero_uncertainty(cov_type):
    X, y = np.zeros((20, 1)), np.full(20, 3.)
    model = PenalizedLinearRegression(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=np.ones(len(y)))
    np.testing.assert_allclose(model._params[0], 3., atol=1e-14)
    np.testing.assert_allclose(model._bse[0], 0., atol=1e-14)
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("weight_scale", [1e-40, 1., 1e40])
def test_oracle_robust_normalization_retains_small_weight_large_score(
        cls, penalty, cov_type, weight_scale):
    # The last row has a tiny analytic weight but supplies 3/4 of the meat.
    # w / max(w) drops it entirely and silently halves the standard error.
    X = np.zeros((4, 1))
    y = np.array([0., 0., 0., 1e200])
    weights = np.array([1e180, 1e180, 1e180, 1e-150]) * weight_scale
    model = cls(penalty=penalty, alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=weights)
    mean = 1e-130 / 3
    se = mean * np.sqrt(4 / 3)
    if cov_type == "hc1":
        se *= np.sqrt(4 / 3)
    np.testing.assert_allclose(model._params[0], mean, rtol=1e-13, atol=0)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-13, atol=0)
    np.testing.assert_allclose(model._conf_int[0],
        mean + se * stats.norm.ppf([.025, .975]), rtol=1e-13, atol=0)
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES[:1] + FAMILIES[2:])
@pytest.mark.parametrize("typed_entry", [False, True])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_oracle_robust_large_relative_weight_range_preserves_family_score(
        loss, settings, typed, typed_settings, typed_entry, penalty, cov_type):
    X = np.zeros((4, 1))
    y = np.array([1., 1., 1., 1e200])
    weights = np.array([1e180, 1e180, 1e180, 1e-150])
    kwargs = dict(penalty=penalty, alpha=.1, device="cpu", max_iter=10000,
        tol=1e-10, compute_inference=True, inference_method="oracle", cov_type=cov_type)
    model = (typed(**kwargs, **typed_settings) if typed_entry else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    model.fit(X, y, sample_weight=weights)
    # At mu=1 all supported links have |d mu/d eta|=1. The last weighted
    # residual is 1e50; division by sum(w)=3e180 leaves this finite SE.
    se = 1e-130 / 3
    if cov_type == "hc1":
        se *= np.sqrt(4 / 3)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-12, atol=0)
    assert model._inference_result.metadata["refit_family_parameters"] == settings
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_oracle_robust_normalization_preserves_existing_subnormal_weight(cov_type):
    X = np.zeros((4, 1))
    tiny = np.nextafter(0., 1.)
    y = np.array([0., 0., 0., 1e308])
    weights = np.array([1., 1., 1., tiny])
    mean = (tiny * y[-1]) / 3
    se = mean * np.sqrt(4 / 3)
    if cov_type == "hc1":
        se *= np.sqrt(4 / 3)
    model = PenalizedLinearRegression(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=weights)
    np.testing.assert_allclose(model._params[0], mean, rtol=1e-13, atol=0)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-13, atol=0)


@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("weight_scale", [1e60, 1e100, 1e140])
def test_oracle_robust_normalization_preserves_small_design_score_meat(cov_type, weight_scale):
    # Isolate inference on a fixed legitimate support: selection itself has a
    # separate absolute-tolerance limit for designs this small. Unit weights
    # would underflow the score squares even though slope covariance is finite.
    model = PenalizedLinearRegression(penalty="mcp", alpha=.001, device="cpu",
        fit_intercept=False, compute_inference=False,
        inference_method="oracle", cov_type=cov_type).fit(
            np.ones((12, 1)), np.tile([.9, 1.1], 6))
    X = np.ones((12, 1)) * 1e-85
    y = X[:, 0] + np.tile([-1., 1.], 6) * 1e-100
    model._compute_oracle_inference(X, y, np.full(len(y), weight_scale))
    # Closed form in slope units avoids tiny products in the reference itself.
    errors = (y - X[:, 0] * model._params[0]) / X[:, 0]
    se = np.linalg.norm(errors) / len(y)
    if cov_type == "hc1":
        se *= np.sqrt(len(y) / (len(y) - 1))
    assert se > 0
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-12, atol=0)


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_oracle_nonfinite_active_covariance_is_transactional(cls, cov_type, previous_fit):
    X = np.zeros((4, 1))
    model = cls(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type)
    if previous_fit:
        model.fit(X, np.array([1., 2., 3., 4.]))
    # The target exists but its covariance (~1e400) cannot be represented in
    # float64. Do not publish inf standard errors or a partial fitted model.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="finite covariance"):
            model.fit(X, np.array([-1e200, 1e200, -1e200, 1e200]))
    assert not model._fitted
    assert model.coef_ is model._params is model._inference_result is None
    model.fit(X, np.array([1., 2., 3., 4.]))
    assert model._fitted
    assert np.isfinite(model._bse[0])


@pytest.mark.parametrize("loss,settings,typed,typed_settings", FAMILIES)
def test_oracle_scaling_curvature_matches_existing_bread_including_clipping(
        loss, settings, typed, typed_settings):
    from statgpu.linear_model.penalized._inference_mixin import (
        _oracle_observed_curvature, _OracleObservedHessianLoss,
    )
    from statgpu.glm_core import get_glm_loss
    resolved = get_glm_loss(loss, **settings)
    bread_loss = (_OracleObservedHessianLoss(resolved)
                  if loss in ("inverse_gaussian", "negative_binomial") else resolved)
    eta = np.array([-1000., -30., -.1, .1, 1., 30., 1000.])
    y = np.array([.1, 1., .3, 2., 1.3, 3., 1e-250])
    actual = _oracle_observed_curvature(bread_loss, eta[:, None], y, np.ones(1))
    expected = np.array([bread_loss.hessian(np.ones((1, 1)), np.array([yi]),
        np.array([ei]))[0, 0] for ei, yi in zip(eta, y)])
    np.testing.assert_array_equal(actual, expected)
    # Independent observed-score derivatives inside the unclipped domain.
    eta = np.array([.2, .7, 1.1])
    y = np.array([.4, 2., 1.3])
    independent = _moments(loss, settings, eta, y)[1]
    np.testing.assert_allclose(_oracle_observed_curvature(
        bread_loss, eta[:, None], y, np.ones(1)), independent, rtol=1e-14)


@pytest.mark.parametrize("design_exponent", [-140, -100, -85, -40, 0, 40, 85, 100, 140])
@pytest.mark.parametrize("weight_exponent", [-300, -200, -100, 0, 100, 200, 300])
def test_oracle_binary_scaling_preserves_finite_gaussian_covariance_both_directions(
        design_exponent, weight_exponent):
    from statgpu.glm_core import get_glm_loss
    from statgpu.inference._sandwich import m_estimation_inference
    from statgpu.linear_model.penalized._inference_mixin import _oracle_robust_covariance_weights
    loss = get_glm_loss("squared_error")
    X = np.full((12, 1), 10. ** design_exponent)
    y = X[:, 0] * np.tile([.5, 1.5], 6)
    weights = np.full(12, 10. ** weight_exponent)
    normalized = _oracle_robust_covariance_weights(loss, X, y, np.ones(1), weights)
    assert np.all(normalized > 0)
    assert np.all(np.isfinite(normalized))
    result = m_estimation_inference(loss, X, y, np.ones(1), cov_type="hc0",
                                   sample_weight=normalized)
    # Slope residuals are +/- .5, so HC0 variance is 12*.5**2 / 12**2.
    np.testing.assert_allclose(result["bse"], np.sqrt(1 / 48), rtol=1e-12, atol=0)


def test_oracle_binary_scaling_preserves_small_weighted_curvature_products():
    from statgpu.glm_core import get_glm_loss
    from statgpu.inference._sandwich import m_estimation_inference
    from statgpu.linear_model.penalized._inference_mixin import _oracle_robust_covariance_weights
    loss = get_glm_loss("gamma", link="log")
    X = np.array([[1e120], [0.]])
    y = np.array([1e-250, 1.])
    weights = np.array([1e-50, 1e30])
    params = np.zeros(1)
    normalized = _oracle_robust_covariance_weights(loss, X, y, params, weights)
    # Raw w*h=1e-300 is representable, while max-weight normalization makes
    # it zero. X.T@(X*w*h)/sum(w)=1e-90 must survive the common rescaling.
    np.testing.assert_allclose(loss.hessian(X, y, params, sample_weight=normalized),
                               [[1e-90]], rtol=1e-14, atol=0)
    result = m_estimation_inference(loss, X, y, params, cov_type="hc0",
                                   sample_weight=normalized)
    np.testing.assert_allclose(result["bse"], [1e130], rtol=1e-14, atol=0)


@pytest.mark.parametrize("weight", [1e-300, 1., 1e300])
def test_oracle_binary_scaling_keeps_exact_zero_scores_legal(weight):
    X, y = np.zeros((4, 1)), np.full(4, 3.)
    model = PenalizedLinearRegression(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type="hc0").fit(
            X, y, sample_weight=np.full(4, weight))
    np.testing.assert_allclose(model._params[0], 3., atol=1e-14)
    np.testing.assert_allclose(model._bse[0], 0., atol=1e-14)


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_oracle_scaling_conflict_retains_finite_raw_range_edge(cls, penalty, cov_type):
    X = np.zeros((3, 1))
    y = np.array([-9e153, 9e153, 0.])
    weights = np.array([1., 1., np.nextafter(0., 1.)])
    model = cls(penalty=penalty, alpha=.1, device="cpu", compute_inference=True,
        inference_method="oracle", cov_type=cov_type).fit(X, y, sample_weight=weights)
    # sum((w*y)**2)=1.62e308 is finite. Exponent-only upper bounds exceed
    # float64, but the tiny third weight prevents downward binary scaling.
    se = 9e153 / np.sqrt(2)
    if cov_type == "hc1":
        se *= np.sqrt(3 / 2)
    # Least-squares cancellation can leave an intercept at roundoff scale.
    assert abs(model._params[0]) <= 4 * np.finfo(float).eps * 9e153
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-14, atol=0)
    np.testing.assert_allclose(model._conf_int[0], se * stats.norm.ppf([.025, .975]),
                               rtol=1e-14, atol=0)
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("response_scale,weight", [(1e154, 1.), (1e-308, 5e307)])
def test_oracle_scaling_conflict_rejects_raw_range_loss_transactionally(
        cov_type, response_scale, weight):
    X = np.zeros((3, 1))
    model = PenalizedLinearRegression(penalty="mcp", alpha=.1, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type)
    model.fit(X, np.array([1., 2., 3.]))
    weights = np.array([weight, weight, np.nextafter(0., 1.)])
    # Respectively: the score-square sum overflows despite finite covariance,
    # or the true positive covariance (~5e-617) is below float64 range.
    # A finite real answer alone does not certify the required operations.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="stable weight scaling"):
            model.fit(X, np.array([-response_scale, response_scale, 0.]),
                      sample_weight=weights)
    assert not model._fitted
    assert model.coef_ is model._params is model._inference_result is None
    model.fit(X, np.array([1., 2., 3.]))
    assert model._fitted
    assert np.isfinite(model._bse[0])


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("tail_design", [0., 1e-100])
def test_oracle_scaling_conflict_allows_negligible_raw_products(
        cls, penalty, cov_type, tail_design):
    X = np.array([1., 1., 1., 1., tail_design])[:, None]
    y = np.array([1e153, 11e153, 1e153, 11e153, 1e-320])
    weights = np.array([1., 1., 1., 1., np.nextafter(0., 1.)])
    model = cls(penalty=penalty, alpha=.1, solver="fista", fit_intercept=False,
        device="cpu", compute_inference=True, inference_method="oracle",
        cov_type=cov_type).fit(X, y, sample_weight=weights)
    # The last row is exactly irrelevant at X=0. At tiny nonzero X its
    # vanished score/bread products are below ordinary aggregate roundoff.
    se = 2.5e153 * (np.sqrt(5 / 4) if cov_type == "hc1" else 1)
    np.testing.assert_allclose(model._params[0], 6e153, rtol=1e-14, atol=0)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-14, atol=0)


def test_oracle_scaling_conflict_rejects_whole_nonzero_score_column_loss():
    from statgpu.glm_core import get_glm_loss
    from statgpu.linear_model.penalized._inference_mixin import _oracle_robust_covariance_weights
    X = np.array([[1., 0.], [1., 0.], [0., 1e-100], [0., 1e-100]])
    y = np.array([-9e153, 9e153, .5e-100, 1.5e-100])
    # The second column has a genuine HC0 variance of 1/8, but raw score
    # squares vanish. Scaling upward to retain them overflows the first.
    with pytest.raises(RuntimeError, match="stable weight scaling"):
        _oracle_robust_covariance_weights(get_glm_loss("squared_error"),
                                         X, y, np.ones(2), np.ones(4))


@pytest.mark.parametrize("lost_factor", ["gradient", "curvature"])
def test_oracle_raw_certificate_rejects_material_lost_products(lost_factor):
    from statgpu.linear_model.penalized._inference_mixin import (
        _oracle_raw_covariance_products_are_representable,
    )
    X = np.array([[1.], [1e300]])
    weights = np.array([1., np.nextafter(0., 1.)])
    if lost_factor == "gradient":
        # w*g vanishes before a huge X could rescue a material score.
        gradient = np.array([1e-150, 1e-100])
        curvature = np.array([1., 0.])
    else:
        # w*h vanishes before X**2 could rescue all the bread.
        gradient = np.zeros(2)
        curvature = np.array([0., 1e-100])
    assert not _oracle_raw_covariance_products_are_representable(
        X, gradient, curvature, weights)


def test_oracle_scaling_conflict_keeps_exact_zero_score_columns_legal():
    from statgpu.glm_core import get_glm_loss
    from statgpu.inference._sandwich import m_estimation_inference
    from statgpu.linear_model.penalized._inference_mixin import _oracle_robust_covariance_weights
    loss = get_glm_loss("squared_error")
    X = np.full((3, 1), 9e153)
    y = X[:, 0].copy()
    weights = np.array([1., 1., np.nextafter(0., 1.)])
    normalized = _oracle_robust_covariance_weights(loss, X, y, np.ones(1), weights)
    np.testing.assert_array_equal(normalized, weights)
    result = m_estimation_inference(loss, X, y, np.ones(1), cov_type="hc0",
                                   sample_weight=normalized)
    np.testing.assert_array_equal(result["cov_params"], np.zeros((1, 1)))
    np.testing.assert_array_equal(result["bse"], np.zeros(1))


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("weight_mode", ["unweighted", "uniform", "nonuniform"])
@pytest.mark.parametrize("previous_fit", [False, True])
def test_oracle_entire_covariance_underflow_is_transactional(
        cls, penalty, cov_type, weight_mode, previous_fit):
    X = np.zeros((3, 1))
    weights = {"unweighted": None, "uniform": np.ones(3),
               "nonuniform": np.array([1., 2., 1.])}[weight_mode]
    model = cls(penalty=penalty, alpha=.1, device="cpu", compute_inference=True,
        inference_method="oracle", cov_type=cov_type)
    if previous_fit:
        model.fit(X, np.array([1., 2., 3.]), sample_weight=weights)
    # Uniform HC0 covariance is 2e-400/9: positive but below float64 range.
    # Its exact SE is representable; publishing zero instead is not valid.
    with pytest.raises(RuntimeError, match="represent nonzero (score|residual) variation"):
        model.fit(X, np.array([-1e-200, 1e-200, 0.]), sample_weight=weights)
    assert not model._fitted
    assert model.coef_ is model._params is model._inference_result is None
    model.fit(X, np.array([1., 2., 3.]), sample_weight=weights)
    assert model._fitted
    assert np.isfinite(model._bse[0]) and model._bse[0] > 0
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("control", ["intercept_unweighted", "intercept_weighted",
                                   "zero_design_unweighted", "zero_design_weighted",
                                   "zero_weight"])
def test_oracle_entire_zero_covariance_remains_legal_for_zero_scores(
        cls, penalty, cov_type, control):
    intercept = control.startswith("intercept")
    if intercept:
        X, y = np.zeros((3, 1)), np.zeros(3)
        weights = np.ones(3) if control.endswith("_weighted") else None
    else:
        X = np.array([[1., 0.], [0., 0.]])
        y = np.array([1., 1e-200])
        weights = np.ones(2) if control.endswith("_weighted") else None
        if control == "zero_weight":
            X[1, 0] = 1.
            weights = np.array([1., 0.])
    model = cls(penalty=penalty, alpha=.1, device="cpu", fit_intercept=intercept,
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=weights)
    # Nonzero gradients confined to zero-design or zero-weight rows do not
    # form nonzero scores. Inactive uncertainty remains undefined.
    np.testing.assert_array_equal(model._bse[:1], np.zeros(1))
    assert np.isnan(model._bse[1:]).all()


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
def test_oracle_partially_zero_covariance_remains_legal(cls, penalty, cov_type, weighted):
    X = np.array([[1., 0.], [0., 1.], [0., 1.]])
    y = np.array([1., 0., 2.])
    model = cls(penalty=penalty, alpha=.1, fit_intercept=False, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type).fit(
            X, y, sample_weight=np.ones(3) if weighted else None)
    # The first coefficient fits its single observation exactly; only the
    # second coefficient has score variation. Do not reject a zero diagonal.
    se = np.array([0., np.sqrt(.5 if cov_type == "hc0" else 1.5)])
    np.testing.assert_allclose(model._bse, se, rtol=1e-14, atol=0)


@pytest.mark.parametrize("loss,settings,typed,typed_settings",
                         [FAMILIES[i] for i in (0, 3, 4, 5, 7)])
@pytest.mark.parametrize("typed_entry", [False, True])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("case", ["moderate", "exact_fit", "common_subnormal",
                                "zero_tail", "tiny_tail", "twice_tiny_tail", "four_tiny_tail"])
def test_oracle_unknown_scale_covariance_range_matrix(
        loss, settings, typed, typed_settings, typed_entry, penalty, cov_type, case):
    tiny = np.nextafter(0., 1.)
    if case in ("moderate", "exact_fit", "common_subnormal"):
        y = np.tile([.8, 1., 1.2], 2)
        weights = np.full(6, tiny) if case == "common_subnormal" else None
        if case == "exact_fit":
            y[:] = 0. if loss == "squared_error" else 1.
    else:
        y = np.array([0., 0., 1.]) if loss == "squared_error" else np.array([1., 1., 2.])
        multiple = {"zero_tail": 0, "tiny_tail": 1,
                    "twice_tiny_tail": 2, "four_tiny_tail": 4}[case]
        weights = np.array([1., 1., multiple * tiny])
    X = np.zeros((len(y), 1))
    kwargs = dict(penalty=penalty, alpha=.1, device="cpu", max_iter=10000,
                  tol=1e-10, compute_inference=True, inference_method="oracle", cov_type=cov_type)
    model = (typed(**kwargs, **typed_settings) if typed_entry else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    fails = (case in ("tiny_tail", "twice_tiny_tail")
             or (case == "common_subnormal" and cov_type == "nonrobust")
             or (case == "four_tiny_tail" and cov_type != "nonrobust"))
    if fails:
        with pytest.raises(RuntimeError):
            model.fit(X, y, sample_weight=weights)
        _assert_failed_oracle_fit(model, X)
        model.fit(np.zeros((6, 1)), np.tile([.8, 1., 1.2], 2))
        assert model._fitted and model._bse[0] > 0
        return
    model.fit(X, y, sample_weight=weights)
    if case in ("exact_fit", "zero_tail"):
        se = 0.
    elif case == "four_tiny_tail":
        # Positive dispersion and representable subnormal covariance stay legal.
        se = np.sqrt(tiny)
        assert model._inference_result.metadata["dispersion"] == 2 * tiny
    else:
        # At mean1, all these links have |d mu/d eta|=1 and V(mu)=1.
        se = np.sqrt(.16 / 30) if cov_type == "nonrobust" else np.sqrt(.16) / 6
        if cov_type == "hc1":
            se *= np.sqrt(6 / 5)
    np.testing.assert_allclose(model._bse[0], se, rtol=1e-12, atol=0)
    assert np.isnan(model._bse[1])


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("amplitude", [1., 1e-200])
def test_oracle_zero_design_residuals_affect_dispersion_but_not_scores(
        cls, penalty, cov_type, weighted, amplitude):
    X = np.array([[1.], [0.], [0.]])
    y = np.array([1., amplitude, -amplitude])
    weights = np.ones(3) if weighted else None
    model = cls(penalty=penalty, alpha=.1, fit_intercept=False, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type=cov_type)
    if cov_type == "nonrobust" and amplitude < 1:
        with pytest.raises(RuntimeError, match="nonzero residual variation"):
            model.fit(X, y, sample_weight=weights)
        _assert_failed_oracle_fit(model, X)
    else:
        model.fit(X, y, sample_weight=weights)
        np.testing.assert_array_equal(model._bse, [amplitude if cov_type == "nonrobust" else 0.])


@pytest.mark.parametrize("cls", [PenalizedLinearRegression, PenalizedGeneralizedLinearModel])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
def test_oracle_nonrobust_positive_dispersion_rejects_partial_zero_variance(cls, penalty):
    X = np.array([[2., 0.], [0., 1.], [0., 0.]])
    y = np.array([2., 1., 1.])
    model = cls(penalty=penalty, alpha=.1, fit_intercept=False, device="cpu",
        compute_inference=True, inference_method="oracle", cov_type="nonrobust")
    model.fit(X, y)
    # Dispersion=minsubnormal, inverse unnormalized information=diag(1/4,1).
    # Raw covariance becomes diag(0,minsubnormal): a positive-dispersion zero
    # variance is unavailable, unlike a legitimate partial-zero HC covariance.
    with pytest.raises(RuntimeError, match="positive model-based variances"):
        model.fit(X, y, sample_weight=np.array([1., 1., np.nextafter(0., 1.)]))
    _assert_failed_oracle_fit(model, X)
    model.fit(X, y)
    np.testing.assert_allclose(model._bse, [.5, 1.], rtol=1e-14, atol=0)


@pytest.mark.parametrize("loss,settings,typed,typed_settings",
                         [FAMILIES[i] for i in (1, 2, 6)])
@pytest.mark.parametrize("typed_entry", [False, True])
@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
@pytest.mark.parametrize("zero_weight_tail", [False, True])
def test_oracle_fixed_scale_keeps_positive_model_based_and_legal_robust_variances(
        loss, settings, typed, typed_settings, typed_entry, penalty, cov_type, zero_weight_tail):
    y = np.tile([0., 1.], 3) if loss == "logistic" else np.ones(6)
    weights = None
    if zero_weight_tail:
        y = np.append(y, 1. if loss == "logistic" else 3.)
        weights = np.append(np.ones(6), 0.)
    kwargs = dict(penalty=penalty, alpha=.1, device="cpu", max_iter=10000,
                  tol=1e-10, compute_inference=True, inference_method="oracle", cov_type=cov_type)
    model = (typed(**kwargs, **typed_settings) if typed_entry else
             PenalizedGeneralizedLinearModel(loss=loss, loss_kwargs=settings, **kwargs))
    model.fit(np.zeros((len(y), 1)), y, sample_weight=weights)
    if cov_type == "nonrobust":
        variance = (4. if loss == "logistic" else 1.25 if loss == "negative_binomial" else 1.) / 6
        assert model._inference_result.metadata["dispersion"] == 1.
    else:
        variance = 4 / 6 if loss == "logistic" else 0.
        if cov_type == "hc1":
            variance *= len(y) / (len(y) - 1)
    np.testing.assert_allclose(model._bse[0], np.sqrt(variance), rtol=1e-13, atol=0)
    assert np.isnan(model._bse[1])
