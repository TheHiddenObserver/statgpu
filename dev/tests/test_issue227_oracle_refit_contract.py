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
    assert model._inference_result is None


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
    assert model._inference_result is None


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
    assert model._inference_result is None


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
                 "inference_target_", "inference_resolved_method_"):
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
