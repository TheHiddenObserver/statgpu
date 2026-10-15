"""Issue #131: objective, compatibility and backend contracts for logistic solvers.

The independent reference minimizes the *summed* weighted Bernoulli NLL plus
||coef||**2 / (2*C).  C=0 removes the ridge term and the intercept is never
penalized.  GPU cases require actual CUDA; their skips are not GPU evidence.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from statgpu import LogisticRegression, LogisticRegressionCV
from statgpu.solvers._convergence import ConvergenceWarning


@pytest.fixture
def logistic_problem():
    rng = np.random.default_rng(131)
    X = rng.normal(size=(96, 4))
    y = rng.binomial(1, expit(0.45 + X @ [0.7, -0.9, 0.35, 0.15])).astype(float)
    weight = rng.uniform(0.15, 2.5, len(y))
    weight[::11] = 0.0
    return X, y, weight


def _objective_functions(X, y, weight, C, fit_intercept):
    design = np.column_stack([np.ones(len(y)), X]) if fit_intercept else X
    weight = np.ones(len(y)) if weight is None else np.asarray(weight)
    ridge = np.full(design.shape[1], 1.0 / C if C > 0 else 0.0)
    if fit_intercept:
        ridge[0] = 0.0

    def value(params):
        eta = design @ params
        return np.dot(weight, np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.dot(
            ridge, params**2
        )

    def gradient(params):
        return design.T @ (weight * (expit(design @ params) - y)) + ridge * params

    def hessian(params):
        p = expit(design @ params)
        return design.T @ ((weight * p * (1.0 - p))[:, None] * design) + np.diag(ridge)

    return value, gradient, hessian


def _reference_fit(X, y, weight, C, fit_intercept):
    value, gradient, hessian = _objective_functions(X, y, weight, C, fit_intercept)
    result = minimize(
        value,
        np.zeros(X.shape[1] + int(fit_intercept)),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    # Trust-exact can report roundoff at an otherwise stationary solution.
    assert np.linalg.norm(gradient(result.x)) < 2e-7
    return result.x


def _parameters(model):
    return np.r_[model.intercept_, model.coef_] if model.fit_intercept else model.coef_


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


@pytest.mark.parametrize("C", [0.0, 0.2, 3.0])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("weight_kind", ["none", "uniform", "nonuniform_zero"])
def test_lbfgs_matches_independent_summed_objective(
    logistic_problem, C, fit_intercept, weight_kind
):
    X, y, weight = logistic_problem
    if weight_kind == "none":
        weight = None
    elif weight_kind == "uniform":
        weight = np.full(len(y), 4.0)
    model = LogisticRegression(
        solver="lbfgs",
        C=C,
        fit_intercept=fit_intercept,
        device="cpu",
        max_iter=500,
        tol=1e-9,
        compute_inference=False,
    ).fit(X, y, sample_weight=weight)
    expected = _reference_fit(X, y, weight, C, fit_intercept)
    value, gradient, _ = _objective_functions(X, y, weight, C, fit_intercept)
    params = _parameters(model)
    np.testing.assert_allclose(params, expected, rtol=2e-6, atol=2e-7)
    assert abs(value(params) - value(expected)) < 2e-9
    assert np.linalg.norm(gradient(params)) < 2e-5
    assert model.solver_ == "lbfgs"
    assert model.converged_ is True
    assert 1 <= model.n_iter_ <= model.max_iter
    expected_probability = expit(X @ model.coef_ + model.intercept_)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expected_probability)
    np.testing.assert_array_equal(model.predict(X), expected_probability >= 0.5)
    assert model.score(X, y) == np.mean((expected_probability >= 0.5) == y)
    if not fit_intercept:
        assert model.intercept_ == 0.0


@pytest.mark.parametrize("weight_kind", ["uniform", "nonuniform_zero"])
def test_absolute_weight_scale_changes_ridge_strength(logistic_problem, weight_kind):
    X, y, weight = logistic_problem
    if weight_kind == "uniform":
        weight = np.ones(len(y))
    controls = {
        "solver": "lbfgs",
        "device": "cpu",
        "compute_inference": False,
        "max_iter": 500,
        "tol": 1e-9,
    }
    original = LogisticRegression(C=0.3, **controls).fit(X, y, sample_weight=weight)
    scaled = LogisticRegression(C=0.3, **controls).fit(X, y, sample_weight=7 * weight)
    equivalent = LogisticRegression(C=2.1, **controls).fit(X, y, sample_weight=weight)
    np.testing.assert_allclose(
        _parameters(scaled), _parameters(equivalent), rtol=2e-6, atol=2e-7
    )
    assert np.linalg.norm(original.coef_ - scaled.coef_) > 0.01


def test_zero_weights_equal_omitting_rows(logistic_problem):
    X, y, weight = logistic_problem
    controls = {
        "solver": "lbfgs",
        "C": 0.8,
        "device": "cpu",
        "compute_inference": False,
        "max_iter": 500,
        "tol": 1e-9,
    }
    weighted = LogisticRegression(**controls).fit(X, y, sample_weight=weight)
    keep = weight > 0
    subset = LogisticRegression(**controls).fit(
        X[keep], y[keep], sample_weight=weight[keep]
    )
    np.testing.assert_allclose(
        _parameters(weighted), _parameters(subset), rtol=1e-6, atol=2e-7
    )


def test_intercept_remains_unpenalized_with_strong_ridge():
    X = np.zeros((40, 2))
    y = np.r_[np.zeros(10), np.ones(30)]
    model = LogisticRegression(
        solver="lbfgs", C=1e-5, device="cpu", max_iter=200, tol=1e-9
    ).fit(X, y)
    assert model.intercept_ == pytest.approx(np.log(3.0), abs=2e-7)
    np.testing.assert_array_equal(model.coef_, np.zeros(2))


@pytest.mark.parametrize("C", [0.0, 0.7])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
def test_lbfgs_preserves_inference_contract(logistic_problem, C, cov_type):
    X, y, weight = logistic_problem
    controls = {
        "C": C,
        "device": "cpu",
        "max_iter": 500,
        "tol": 1e-9,
        "cov_type": cov_type,
        "hac_maxlags": 2,
    }
    irls = LogisticRegression(solver="irls", **controls).fit(X, y, sample_weight=weight)
    lbfgs = LogisticRegression(solver="lbfgs", **controls).fit(
        X, y, sample_weight=weight
    )
    for name in ("_params", "_bse", "_zvalues", "_pvalues", "_conf_int"):
        np.testing.assert_allclose(
            getattr(lbfgs, name), getattr(irls, name), rtol=2e-5, atol=2e-7
        )
    for name in (
        "loglikelihood",
        "loglikelihood_null",
        "aic",
        "bic",
        "pseudo_rsquared",
    ):
        assert getattr(lbfgs, name) == pytest.approx(
            getattr(irls, name), rel=2e-7, abs=2e-7
        )
    if cov_type == "nonrobust":
        _, _, hessian = _objective_functions(X, y, weight, C, True)
        expected_bse = np.sqrt(np.diag(np.linalg.inv(hessian(_parameters(lbfgs)))))
        np.testing.assert_allclose(lbfgs._bse, expected_bse, rtol=2e-7, atol=2e-8)


def test_default_auto_and_irls_are_bitwise_identical(logistic_problem):
    X, y, weight = logistic_problem
    controls = {"C": 0.7, "device": "cpu", "max_iter": 150, "tol": 1e-9}
    models = [
        LogisticRegression(**controls),
        LogisticRegression(solver="auto", **controls),
        LogisticRegression(solver="irls", **controls),
    ]
    for model in models:
        model.fit(X, y, sample_weight=weight)
        assert model.solver_ == "irls"
    for other in models[1:]:
        for name in ("_params", "coef_", "_bse", "_zvalues", "_pvalues", "_conf_int"):
            np.testing.assert_array_equal(
                getattr(other, name), getattr(models[0], name)
            )
        assert other.n_iter_ == models[0].n_iter_
        assert other.loglikelihood == models[0].loglikelihood
        np.testing.assert_array_equal(
            other.predict_proba(X), models[0].predict_proba(X)
        )


@pytest.mark.parametrize("solver", ["auto", "irls", "lbfgs"])
def test_summary_identifies_resolved_solver(logistic_problem, capsys, solver):
    X, y, _ = logistic_problem
    model = LogisticRegression(solver=solver, device="cpu", max_iter=300, tol=1e-8).fit(
        X, y
    )
    model.summary()
    summary = capsys.readouterr().out.lower()
    assert "solver" in summary
    assert model.solver_ in summary


@pytest.mark.parametrize("estimator_cls", [LogisticRegression, LogisticRegressionCV])
def test_solver_is_append_only_and_cloneable(estimator_cls):
    clone = pytest.importorskip("sklearn.base").clone
    signature = inspect.signature(estimator_cls)
    names = list(signature.parameters)
    assert names[-1] == "solver"
    assert signature.parameters["solver"].default == "auto"
    model = estimator_cls(solver="lbfgs", device="cpu")
    assert model.get_params()["solver"] == "lbfgs"
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned is not model
    assert not cloned.__sklearn_is_fitted__()
    assert model.set_params(solver="irls") is model
    assert model.get_params()["solver"] == "irls"


@pytest.mark.parametrize("estimator_cls", [LogisticRegression, LogisticRegressionCV])
@pytest.mark.parametrize("invalid", [None, 1, True, "newton", "", ["irls"]])
def test_invalid_solver_is_rejected_at_construction(estimator_cls, invalid):
    with pytest.raises(ValueError, match="solver"):
        estimator_cls(solver=invalid, device="cpu")


@pytest.mark.parametrize("estimator_cls", [LogisticRegression, LogisticRegressionCV])
def test_invalid_set_params_preserves_existing_fit(logistic_problem, estimator_cls):
    X, y, _ = logistic_problem
    extra = (
        {"Cs": [0.2, 1.0], "cv": 2, "random_state": 4}
        if estimator_cls is LogisticRegressionCV
        else {}
    )
    model = estimator_cls(solver="lbfgs", device="cpu", max_iter=300, **extra).fit(X, y)
    prediction = model.predict_proba(X).copy()
    with pytest.raises(ValueError, match="solver"):
        model.set_params(solver="not-a-solver")
    assert model.solver == "lbfgs"
    assert model.__sklearn_is_fitted__()
    np.testing.assert_array_equal(model.predict_proba(X), prediction)
    model.set_params(solver="irls")
    assert not model.__sklearn_is_fitted__()
    assert model.coef_ is None


@pytest.mark.parametrize("estimator_cls", [LogisticRegression, LogisticRegressionCV])
def test_mutated_invalid_solver_clears_failed_fit_state(
    logistic_problem, estimator_cls
):
    X, y, _ = logistic_problem
    extra = {"Cs": [0.2, 1.0], "cv": 2} if estimator_cls is LogisticRegressionCV else {}
    model = estimator_cls(solver="lbfgs", device="cpu", max_iter=300, **extra).fit(X, y)
    model.solver = "invalid"
    with pytest.raises(ValueError, match="solver"):
        model.fit(X, y)
    assert not model.__sklearn_is_fitted__()
    assert model.coef_ is None
    assert getattr(model, "solver_", None) is None
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_lbfgs_max_iter_has_one_solver_specific_warning(logistic_problem):
    X, y, weight = logistic_problem
    model = LogisticRegression(
        solver="lbfgs", device="cpu", max_iter=1, tol=1e-12, compute_inference=False
    )
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        model.fit(X, y, sample_weight=weight)
    convergence = [w for w in observed if issubclass(w.category, ConvergenceWarning)]
    assert len(convergence) == 1
    assert "lbfgs" in str(convergence[0].message).lower()
    assert "irls" not in str(convergence[0].message).lower()
    assert model.converged_ is False
    assert model.n_iter_ == 1
    assert np.isfinite(model.predict_proba(X)).all()


def test_lbfgs_inference_can_be_disabled(logistic_problem):
    X, y, _ = logistic_problem
    model = LogisticRegression(
        solver="lbfgs", device="cpu", compute_inference=False, max_iter=300, tol=1e-8
    ).fit(X, y)
    assert model._bse is None
    assert model._pvalues is None
    assert model._conf_int is None
    assert np.isfinite(model.loglikelihood)
    with pytest.raises(RuntimeError, match="compute_inference=False"):
        model.summary()


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_lbfgs_requires_physical_backend_for_explicit_gpu(
    logistic_problem, monkeypatch, device, has_cupy, has_torch_cuda
):
    available = has_cupy if device == "cuda" else has_torch_cuda
    if available:
        pytest.skip("Backend is available; native success is tested separately")
    X, y, _ = logistic_problem

    def forbid_cpu(*args, **kwargs):
        pytest.fail("An explicit GPU request must never fall back to CPU")

    monkeypatch.setattr(LogisticRegression, "_fit_cpu", forbid_cpu)
    with pytest.raises((RuntimeError, ImportError)):
        LogisticRegression(solver="lbfgs", device=device).fit(X, y)


@pytest.mark.parametrize("device", ["cuda", "torch"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_lbfgs_physical_gpu_matches_weighted_reference(
    logistic_problem, monkeypatch, request, device, fit_intercept
):
    # Deliberately use the strict fixture so STATGPU_REQUIRE_PHYSICAL_GPU=1 fails
    # instead of silently converting missing-hardware coverage into skips.
    request.getfixturevalue(
        "cupy_available" if device == "cuda" else "torch_cuda_available"
    )
    X, y, weight = logistic_problem
    if device == "cuda":
        import cupy as cp

        native = cp.asarray
        is_native = lambda value: isinstance(value, cp.ndarray)
    else:
        import torch

        native = lambda value: torch.as_tensor(
            value, dtype=torch.float64, device="cuda"
        )
        is_native = lambda value: isinstance(value, torch.Tensor) and value.is_cuda

    def forbid_cpu(*args, **kwargs):
        pytest.fail("Explicit GPU L-BFGS must remain on the selected backend")

    monkeypatch.setattr(LogisticRegression, "_fit_cpu", forbid_cpu)
    model = LogisticRegression(
        solver="lbfgs",
        C=0.7,
        fit_intercept=fit_intercept,
        device=device,
        max_iter=500,
        tol=1e-9,
    ).fit(native(X), native(y), sample_weight=native(weight))
    expected = _reference_fit(X, y, weight, 0.7, fit_intercept)
    np.testing.assert_allclose(_parameters(model), expected, rtol=2e-5, atol=2e-6)
    probabilities = model.predict_proba(native(X))
    assert is_native(probabilities)
    np.testing.assert_allclose(
        _numpy(probabilities)[:, 1],
        expit(X @ model.coef_ + model.intercept_),
        rtol=2e-6,
        atol=2e-7,
    )
    assert model.solver_ == "lbfgs"
    assert model.converged_ is True
    assert np.isfinite(model._bse).all()


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [0.0, 0.4])
@pytest.mark.parametrize("normalizer", [1.0, 13.0])
def test_adapter_value_and_gradient_match_analytic_and_finite_difference(
    logistic_problem, fit_intercept, C, normalizer
):
    from statgpu.linear_model.wrappers._logistic_solver import _LogisticObjective

    X, y, weight = logistic_problem
    design = np.column_stack([np.ones(len(y)), X]) if fit_intercept else X
    params = np.linspace(-0.6, 0.7, design.shape[1])
    objective = _LogisticObjective(weight, C, fit_intercept, normalizer=normalizer)
    value, gradient = objective.fused_value_and_gradient(design, y, params)
    expected_value, expected_gradient, _ = _objective_functions(
        X, y, weight, C, fit_intercept
    )
    assert value == pytest.approx(expected_value(params) / normalizer, rel=2e-14)
    np.testing.assert_allclose(
        gradient, expected_gradient(params) / normalizer, rtol=2e-14, atol=2e-14
    )
    step = 1e-5
    finite_difference = np.asarray(
        [
            (
                objective.fused_value_and_gradient(
                    design, y, params + step * direction
                )[0]
                - objective.fused_value_and_gradient(
                    design, y, params - step * direction
                )[0]
            )
            / (2 * step)
            for direction in np.eye(len(params))
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8, atol=2e-9)


def test_large_uniform_weight_mass_does_not_break_line_search(logistic_problem):
    X, y, _ = logistic_problem
    controls = {
        "solver": "lbfgs",
        "device": "cpu",
        "compute_inference": False,
        "max_iter": 500,
        "tol": 1e-10,
    }
    weighted = LogisticRegression(C=0.5, **controls).fit(
        X, y, sample_weight=np.full(len(y), 1e8)
    )
    equivalent = LogisticRegression(C=5e7, **controls).fit(X, y)
    expected = _reference_fit(X, y, None, 5e7, True)
    np.testing.assert_allclose(_parameters(weighted), expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(
        _parameters(weighted), _parameters(equivalent), rtol=3e-6, atol=3e-7
    )
    assert weighted.converged_ is True


def test_line_search_failure_is_never_published_as_convergence(
    logistic_problem, monkeypatch
):
    from statgpu.linear_model.wrappers import _logistic_solver as adapter

    X, y, _ = logistic_problem

    def failed_solver(loss, penalty, design, response, **kwargs):
        warnings.warn(
            "lbfgs_solver: line search failed to find a descent step", RuntimeWarning
        )
        return np.zeros(design.shape[1]), 2

    monkeypatch.setattr(adapter, "lbfgs_solver", failed_solver)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        model = LogisticRegression(
            solver="lbfgs", device="cpu", max_iter=30, compute_inference=False
        ).fit(X, y)
    assert model.converged_ is False
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "line search failed" in str(w.message)
        for w in observed
    )
    assert sum(issubclass(w.category, ConvergenceWarning) for w in observed) == 1


def test_stationary_point_on_last_allowed_iteration_is_converged():
    X = np.zeros((20, 2))
    y = np.tile([0.0, 1.0], 10)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        model = LogisticRegression(
            solver="lbfgs", C=0.7, device="cpu", max_iter=1, compute_inference=False
        ).fit(X, y)
    assert model.converged_ is True
    assert model.n_iter_ == 1
    assert not any(issubclass(w.category, ConvergenceWarning) for w in observed)


def test_nonfinite_lbfgs_result_clears_previous_fit(logistic_problem, monkeypatch):
    from statgpu.linear_model.wrappers import _logistic_solver as adapter

    X, y, _ = logistic_problem
    model = LogisticRegression(solver="lbfgs", device="cpu", max_iter=300).fit(X, y)

    def nonfinite_solver(loss, penalty, design, response, **kwargs):
        return np.full(design.shape[1], np.nan), 2

    monkeypatch.setattr(adapter, "lbfgs_solver", nonfinite_solver)
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.fit(X, y)
    assert not model.__sklearn_is_fitted__()
    for name in ("coef_", "intercept_", "solver_", "converged_", "_bse", "_params"):
        assert getattr(model, name) is None


@pytest.fixture
def cv_splits(logistic_problem):
    n = len(logistic_problem[1])
    indices = np.arange(n)
    return [
        (indices[indices % 3 != fold], indices[indices % 3 == fold])
        for fold in range(3)
    ]


def test_cv_lbfgs_path_selection_and_refit_match_independent_reference(
    logistic_problem, cv_splits
):
    X, y, weight = logistic_problem
    Cs = np.asarray([0.08, 0.7, 6.0])
    expected_path = np.empty((len(Cs), len(cv_splits)))
    for fold, (train, validation) in enumerate(cv_splits):
        for c_idx, C in enumerate(Cs):
            params = _reference_fit(X[train], y[train], weight[train], C, True)
            eta = params[0] + X[validation] @ params[1:]
            expected_path[c_idx, fold] = np.average(
                np.logaddexp(0.0, eta) - y[validation] * eta,
                weights=weight[validation],
            )
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=3,
        cv_splits=cv_splits,
        solver="lbfgs",
        device="cpu",
        max_iter=500,
        tol=1e-9,
    ).fit(X, y, sample_weight=weight)
    expected_C = Cs[np.argmin(expected_path.mean(axis=1))]
    np.testing.assert_allclose(
        model.cv_results_["loss_path"], expected_path, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        model.mean_loss_, expected_path.mean(axis=1), rtol=2e-6, atol=2e-7
    )
    assert model.C_ == expected_C
    assert model.best_score_ == pytest.approx(
        -expected_path.mean(axis=1).min(), abs=2e-7
    )
    assert model.solver_ == model.estimator_.solver_ == "lbfgs"
    assert model.estimator_.get_params()["solver"] == "lbfgs"
    expected_refit = _reference_fit(X, y, weight, expected_C, True)
    np.testing.assert_allclose(
        _parameters(model.estimator_), expected_refit, rtol=2e-6, atol=2e-7
    )
    assert model.estimator_._bse is not None
    np.testing.assert_array_equal(
        model.predict_proba(X), model.estimator_.predict_proba(X)
    )


@pytest.mark.parametrize("solver", ["auto", "irls", "lbfgs"])
def test_cv_propagates_solver_to_every_candidate_and_only_refit_inference(
    logistic_problem, cv_splits, monkeypatch, solver
):
    X, y, weight = logistic_problem
    calls = []
    original_fit = LogisticRegression.fit

    def tracked_fit(self, X, y, sample_weight=None):
        calls.append(
            (
                self.solver,
                self.compute_inference,
                self._device.value,
                len(y),
                np.asarray(sample_weight).copy(),
            )
        )
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", tracked_fit)
    model = LogisticRegressionCV(
        Cs=[0.2, 2.0],
        cv=3,
        cv_splits=cv_splits,
        solver=solver,
        device="cpu",
        max_iter=300,
    ).fit(X, y, sample_weight=weight)
    assert len(calls) == 2 * len(cv_splits) + 1
    assert all(call[0] == solver and call[2] == "cpu" for call in calls)
    assert all(call[1] is False and call[3] == 64 for call in calls[:-1])
    assert calls[-1][1] is True and calls[-1][3] == len(y)
    np.testing.assert_array_equal(calls[-1][4], weight)
    assert model.solver_ == ("irls" if solver == "auto" else solver)


def test_cv_auto_and_irls_have_identical_paths_and_refits(logistic_problem, cv_splits):
    X, y, weight = logistic_problem
    controls = {
        "Cs": [0.2, 2.0],
        "cv": 3,
        "cv_splits": cv_splits,
        "device": "cpu",
        "max_iter": 150,
        "tol": 1e-9,
    }
    default = LogisticRegressionCV(**controls).fit(X, y, sample_weight=weight)
    explicit = LogisticRegressionCV(solver="irls", **controls).fit(
        X, y, sample_weight=weight
    )
    np.testing.assert_array_equal(
        default.cv_results_["loss_path"], explicit.cv_results_["loss_path"]
    )
    np.testing.assert_array_equal(default.coef_, explicit.coef_)
    assert default.intercept_ == explicit.intercept_
    assert default.C_ == explicit.C_
    assert default.solver_ == explicit.solver_ == "irls"


@pytest.mark.parametrize("failure_stage", ["candidate", "refit"])
def test_cv_solver_failure_does_not_publish_partial_or_stale_state(
    logistic_problem, cv_splits, monkeypatch, failure_stage
):
    X, y, weight = logistic_problem
    model = LogisticRegressionCV(
        Cs=[0.2, 2.0],
        cv=3,
        cv_splits=cv_splits,
        solver="lbfgs",
        device="cpu",
        max_iter=300,
    ).fit(X, y)
    original_fit = LogisticRegression.fit

    def failing_fit(self, X, y, sample_weight=None):
        is_refit = len(y) == len(logistic_problem[1])
        if is_refit == (failure_stage == "refit"):
            raise RuntimeError("deliberate solver failure")
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", failing_fit)
    with pytest.raises(RuntimeError, match="deliberate solver failure"):
        model.fit(X, y, sample_weight=weight)
    assert not model.__sklearn_is_fitted__()
    for name in (
        "C_",
        "Cs_",
        "cv_results_",
        "mean_loss_",
        "best_score_",
        "coef_",
        "intercept_",
        "n_iter_",
        "estimator_",
        "cv_selected_device_",
        "solver_",
    ):
        assert getattr(model, name) is None


def test_explicit_cv_cache_is_solver_aware(logistic_problem, cv_splits, monkeypatch):
    from statgpu.linear_model.cv import _logistic_cv as cv_module

    X, y, weight = logistic_problem
    # Isolate this regression from other tests and from callers' process caches.
    from collections import OrderedDict

    monkeypatch.setattr(cv_module, "_LOGISTIC_CV_C_CACHE", OrderedDict())
    calls = []
    original_fit = LogisticRegression.fit

    def tracked_fit(self, *args, **kwargs):
        calls.append(self.solver)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", tracked_fit)
    controls = {
        "Cs": [0.2, 2.0],
        "cv_folds": 3,
        "cv_splits": cv_splits,
        "sample_weight": weight,
        "device": "cpu",
        "max_iter": 300,
        "return_details": True,
        "cache_key": ("issue-131-solver",),
    }
    first = cv_module._select_logistic_c_cv(X, y, solver="irls", **controls)
    assert calls == ["irls"] * 6
    cached = cv_module._select_logistic_c_cv(X, y, solver="irls", **controls)
    assert len(calls) == 6
    np.testing.assert_array_equal(first["loss_path"], cached["loss_path"])
    cv_module._select_logistic_c_cv(X, y, solver="lbfgs", **controls)
    assert calls == ["irls"] * 6 + ["lbfgs"] * 6
    cv_module._select_logistic_c_cv(X, y, solver="lbfgs", **controls)
    assert len(calls) == 12


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
@pytest.mark.parametrize("solver", ["auto", "irls", "lbfgs"])
def test_gpu_cv_dispatch_selects_serial_lbfgs_and_preserves_batched_irls(
    logistic_problem, cv_splits, monkeypatch, backend_name, solver
):
    """Deterministic routing test using NumPy doubles, not GPU execution evidence."""
    from statgpu.backends import NumpyBackend
    from statgpu.linear_model.cv import _logistic_cv as cv_module

    X, y, weight = logistic_problem
    device = "cuda" if backend_name == "cupy" else "torch"
    backend = NumpyBackend()
    monkeypatch.setattr(
        cv_module,
        "resolve_cv_backend",
        lambda requested, X: (device, backend_name, backend, True, False, False),
    )
    calls = []

    class Candidate:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            assert getattr(kwargs["device"], "value", kwargs["device"]) == device
            assert kwargs["solver"] == solver
            assert kwargs["compute_inference"] is False

        def fit(self, X, y, sample_weight=None):
            assert X.dtype == np.float64
            assert sample_weight is not None
            return self

        def predict_proba(self, X):
            p = expit(X[:, 0])
            return np.column_stack([1 - p, p])

    def batched(X_batch, y_batch, n_train, Cs, backend, **kwargs):
        calls.append("batched")
        return np.zeros((len(Cs), len(n_train), X_batch.shape[2])), np.zeros(
            (len(Cs), len(n_train))
        )

    monkeypatch.setattr(cv_module, "LogisticRegression", Candidate)
    monkeypatch.setattr(cv_module, "_solve_logistic_path_gpu_from_batch", batched)
    result = cv_module._select_logistic_c_cv(
        X,
        y,
        sample_weight=weight,
        Cs=[0.2, 2.0],
        cv_folds=3,
        cv_splits=cv_splits,
        solver=solver,
        device=device,
        gpu_cv_mixed_precision=True,
        return_details=True,
    )
    assert np.isfinite(result["loss_path"]).all()
    if solver == "lbfgs":
        assert len(calls) == 6
        assert all(isinstance(call, dict) for call in calls)
    else:
        assert calls == ["batched"]


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_cv_lbfgs_physical_gpu_candidate_and_final_refit_route(
    logistic_problem, cv_splits, request, monkeypatch, device
):
    request.getfixturevalue(
        "cupy_available" if device == "cuda" else "torch_cuda_available"
    )
    X, y, weight = logistic_problem
    if device == "cuda":
        import cupy as cp

        native = cp.asarray
        is_native = lambda value: isinstance(value, cp.ndarray)
    else:
        import torch

        native = lambda value: torch.as_tensor(
            value, dtype=torch.float64, device="cuda"
        )
        is_native = lambda value: isinstance(value, torch.Tensor) and value.is_cuda
    controls = {
        "Cs": [0.2, 2.0],
        "cv": 3,
        "cv_splits": cv_splits,
        "solver": "lbfgs",
        "max_iter": 500,
        "tol": 1e-9,
    }
    cpu = LogisticRegressionCV(device="cpu", **controls).fit(X, y, sample_weight=weight)
    original_fit = LogisticRegression.fit
    calls = []

    def tracked_fit(self, X, y, sample_weight=None):
        assert self._device.value == device
        assert self.solver == "lbfgs"
        assert is_native(X) and is_native(y) and is_native(sample_weight)
        calls.append((len(y), self.compute_inference))
        return original_fit(self, X, y, sample_weight=sample_weight)

    def forbid_cpu(*args, **kwargs):
        pytest.fail("Explicit GPU CV candidates and refit cannot fall back to CPU")

    monkeypatch.setattr(LogisticRegression, "fit", tracked_fit)
    monkeypatch.setattr(LogisticRegression, "_fit_cpu", forbid_cpu)
    gpu = LogisticRegressionCV(device=device, **controls).fit(
        native(X), native(y), sample_weight=native(weight)
    )
    assert calls == [(64, False)] * 6 + [(96, True)]
    assert getattr(gpu.cv_selected_device_, "value", gpu.cv_selected_device_) == device
    assert gpu.solver_ == gpu.estimator_.solver_ == "lbfgs"
    assert gpu.C_ == cpu.C_
    np.testing.assert_allclose(
        gpu.cv_results_["loss_path"], cpu.cv_results_["loss_path"], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        gpu.estimator_._bse, cpu.estimator_._bse, rtol=2e-5, atol=2e-6
    )
    assert is_native(gpu.predict_proba(native(X)))


@pytest.fixture
def torch_cpu_logistic_backend(monkeypatch):
    """Exercise native Torch math on CPU; never present this as CUDA evidence."""
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv import _logistic_cv as cv_module
    from statgpu.linear_model.wrappers import _logistic as logistic_module

    backend = TorchBackend(device="cpu")
    monkeypatch.setattr(
        LogisticRegression, "_get_backend", lambda self, backend="auto": torch_backend
    )
    torch_backend = backend
    monkeypatch.setattr(
        LogisticRegression,
        "_to_array",
        lambda self, X, *args, **kwargs: backend.asarray(X),
    )
    monkeypatch.setattr(logistic_module, "_get_torch_device_str", lambda: "cpu")
    monkeypatch.setattr(
        cv_module,
        "resolve_cv_backend",
        lambda device, X: ("torch", "torch", backend, True, False, True),
    )

    def forbid_cpu(*args, **kwargs):
        pytest.fail("The Torch numerical test must not use NumPy fitting")

    monkeypatch.setattr(LogisticRegression, "_fit_cpu", forbid_cpu)
    return torch, backend


@pytest.mark.parametrize("C", [0.0, 0.7])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"])
def test_torch_cpu_lbfgs_native_math_inference_and_prediction(
    logistic_problem, monkeypatch, request, C, cov_type
):
    X, y, weight = logistic_problem
    controls = {
        "solver": "lbfgs",
        "C": C,
        "cov_type": cov_type,
        "hac_maxlags": 2,
        "max_iter": 500,
        "tol": 1e-9,
    }
    cpu = LogisticRegression(device="cpu", **controls).fit(X, y, sample_weight=weight)
    torch, backend = request.getfixturevalue("torch_cpu_logistic_backend")
    from statgpu.linear_model.wrappers import _logistic_solver as adapter

    shared_solver = adapter.lbfgs_solver
    calls = []

    def native_solver(loss, penalty, design, response, **kwargs):
        assert isinstance(design, torch.Tensor) and design.device.type == "cpu"
        assert isinstance(response, torch.Tensor)
        assert design.dtype == response.dtype == torch.float64
        value, gradient = loss.fused_value_and_gradient(
            design, response, torch.zeros(design.shape[1], dtype=torch.float64)
        )
        assert isinstance(value, torch.Tensor) and isinstance(gradient, torch.Tensor)
        params, n_iter = shared_solver(loss, penalty, design, response, **kwargs)
        assert isinstance(params, torch.Tensor) and params.device.type == "cpu"
        calls.append(n_iter)
        return params, n_iter

    monkeypatch.setattr(adapter, "lbfgs_solver", native_solver)
    model = LogisticRegression(device="torch", **controls).fit(
        backend.asarray(X), backend.asarray(y), sample_weight=backend.asarray(weight)
    )
    assert len(calls) == 1
    assert model.solver_ == "lbfgs" and model.converged_ is True
    for name in ("_params", "_bse", "_zvalues", "_pvalues", "_conf_int"):
        np.testing.assert_allclose(
            getattr(model, name), getattr(cpu, name), rtol=2e-5, atol=2e-7
        )
    for name in ("loglikelihood", "loglikelihood_null", "aic", "bic"):
        assert getattr(model, name) == pytest.approx(
            getattr(cpu, name), rel=2e-7, abs=2e-7
        )
    probabilities = model.predict_proba(backend.asarray(X))
    assert (
        isinstance(probabilities, torch.Tensor) and probabilities.device.type == "cpu"
    )
    np.testing.assert_allclose(
        probabilities.numpy(), cpu.predict_proba(X), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_torch_cpu_cv_lbfgs_uses_native_candidates_and_refit(
    logistic_problem, cv_splits, monkeypatch, request, mixed_precision
):
    from statgpu.linear_model.cv import _logistic_cv as cv_module

    X, y, weight = logistic_problem
    controls = {
        "Cs": [0.2, 2.0],
        "cv": 3,
        "cv_splits": cv_splits,
        "solver": "lbfgs",
        "gpu_cv_mixed_precision": mixed_precision,
        "max_iter": 500,
        "tol": 1e-9,
    }
    cpu = LogisticRegressionCV(device="cpu", **controls).fit(X, y, sample_weight=weight)
    torch, backend = request.getfixturevalue("torch_cpu_logistic_backend")
    original_fit = LogisticRegression.fit
    calls = []

    def native_fit(self, X, y, sample_weight=None):
        assert self._device.value == "torch" and self.solver == "lbfgs"
        assert isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
        assert isinstance(sample_weight, torch.Tensor)
        assert X.dtype == y.dtype == sample_weight.dtype == torch.float64
        calls.append((len(y), self.compute_inference))
        return original_fit(self, X, y, sample_weight=sample_weight)

    def forbid_batched(*args, **kwargs):
        pytest.fail("L-BFGS CV must not route through the batched IRLS solver")

    monkeypatch.setattr(LogisticRegression, "fit", native_fit)
    monkeypatch.setattr(
        cv_module, "_solve_logistic_path_gpu_from_batch", forbid_batched
    )
    model = LogisticRegressionCV(device="torch", **controls).fit(
        backend.asarray(X), backend.asarray(y), sample_weight=backend.asarray(weight)
    )
    assert calls == [(64, False)] * 6 + [(96, True)]
    assert model.solver_ == model.estimator_.solver_ == "lbfgs"
    assert (
        getattr(model.cv_selected_device_, "value", model.cv_selected_device_)
        == "torch"
    )
    assert model.C_ == cpu.C_
    np.testing.assert_allclose(
        model.cv_results_["loss_path"],
        cpu.cv_results_["loss_path"],
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(model.coef_, cpu.coef_, rtol=2e-5, atol=2e-7)
    assert isinstance(model.predict_proba(backend.asarray(X)), torch.Tensor)


def test_torch_cpu_cv_failure_is_explicit_and_has_no_fallback(
    logistic_problem, cv_splits, monkeypatch, torch_cpu_logistic_backend
):
    _, backend = torch_cpu_logistic_backend
    X, y, weight = logistic_problem
    model = LogisticRegressionCV(
        Cs=[0.2, 2.0],
        cv=3,
        cv_splits=cv_splits,
        solver="lbfgs",
        device="torch",
        max_iter=300,
    )
    model.fit(
        backend.asarray(X), backend.asarray(y), sample_weight=backend.asarray(weight)
    )

    def candidate_failure(*args, **kwargs):
        raise ArithmeticError("deliberate native candidate failure")

    monkeypatch.setattr(LogisticRegression, "fit", candidate_failure)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled") as caught:
        model.fit(
            backend.asarray(X),
            backend.asarray(y),
            sample_weight=backend.asarray(weight),
        )
    assert isinstance(caught.value.__cause__, ArithmeticError)
    assert model.coef_ is None and model.estimator_ is None and model.solver_ is None
    assert not model.__sklearn_is_fitted__()


def test_formula_helper_design_and_weights_preserve_lbfgs_array_semantics(
    logistic_problem,
):
    """LogisticRegression accepts arrays; exercise its supported parser-helper route."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.core.formula import FormulaParser, align_formula_sample_weight

    X, y, weight = logistic_problem
    frame = pd.DataFrame(
        {
            "y": y,
            "x": X[:, 0],
            "z": np.exp(0.3 * X[:, 1]),
            "group": np.resize(["a", "b", "c"], len(y)),
        }
    )
    frame.loc[[4, 19], "x"] = np.nan
    weight = weight.copy()
    weight[[4, 19]] = np.nan  # Dropped formula rows must not leak into the fit.
    parser = FormulaParser("y ~ C(group) * x + np.log(z)")
    response, design, info = parser.eval(frame)
    assert info.column_names == [
        "Intercept",
        "C(group)[T.b]",
        "C(group)[T.c]",
        "x",
        "C(group)[T.b]:x",
        "C(group)[T.c]:x",
        "np.log(z)",
    ]
    rows = parser.row_positions
    retained = frame.iloc[rows]
    group_b = np.asarray(retained["group"] == "b", dtype=float)
    group_c = np.asarray(retained["group"] == "c", dtype=float)
    x = retained["x"].to_numpy()
    manual_X = np.column_stack(
        [group_b, group_c, x, group_b * x, group_c * x, np.log(retained["z"])]
    )
    np.testing.assert_array_equal(design[:, 0], np.ones(len(rows)))
    np.testing.assert_allclose(design[:, 1:], manual_X)
    aligned_weight = align_formula_sample_weight(
        weight, data_length=len(frame), retained_rows=rows, retained_length=len(rows)
    )
    controls = {
        "solver": "lbfgs",
        "C": 0.7,
        "fit_intercept": True,
        "device": "cpu",
        "max_iter": 500,
        "tol": 1e-9,
    }
    # The parser owns an explicit intercept column; remove it because this
    # estimator adds its unpenalized intercept itself.
    parsed = LogisticRegression(**controls).fit(
        design[:, 1:], response, sample_weight=aligned_weight
    )
    manual = LogisticRegression(**controls).fit(
        manual_X, y[rows], sample_weight=weight[rows]
    )
    # Different BLAS array layouts can take slightly different line searches;
    # compare converged numerical solutions rather than bitwise histories.
    assert parsed.converged_ and manual.converged_
    np.testing.assert_allclose(parsed._params, manual._params, rtol=2e-7, atol=2e-8)
    new_frame = retained.iloc[:12][["group", "z", "x", "y"]]
    transformed = parser.transform(new_frame)
    np.testing.assert_array_equal(transformed, design[:12])
    np.testing.assert_allclose(
        parsed.predict_proba(transformed[:, 1:]),
        manual.predict_proba(manual_X[:12]),
        rtol=2e-7,
        atol=2e-8,
    )


def _mixed_scale_problem(scale):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(500, 2))
    y = rng.binomial(1, expit(1.0 + base @ [0.5, 0.9])).astype(float)
    X = base.copy()
    X[:, 0] *= scale
    return X, y


@pytest.mark.parametrize("scale", [1000.0, 10000.0])
@pytest.mark.parametrize(
    "backend_name", ["numpy", "torch_cpu", "cupy_cuda", "torch_cuda"]
)
def test_lbfgs_mixed_scale_success_requires_gradient_stationarity(
    request, scale, backend_name
):
    X, y = _mixed_scale_problem(scale)
    reference = LogisticRegression(
        solver="irls",
        device="cpu",
        C=1.0,
        tol=1e-9,
        max_iter=500,
        compute_inference=False,
    ).fit(X, y)
    expected = reference.predict_proba(X)
    native = lambda a: a
    device = "cpu"
    if backend_name == "torch_cpu":
        _, backend = request.getfixturevalue("torch_cpu_logistic_backend")
        native, device = backend.asarray, "torch"
    elif backend_name == "cupy_cuda":
        request.getfixturevalue("cupy_available")
        import cupy as cp

        native, device = cp.asarray, "cuda"
    elif backend_name == "torch_cuda":
        request.getfixturevalue("torch_cuda_available")
        import torch

        native = lambda a: torch.as_tensor(a, dtype=torch.float64, device="cuda")
        device = "torch"
    model = LogisticRegression(
        solver="lbfgs",
        device=device,
        C=1.0,
        tol=1e-4,
        max_iter=500,
        compute_inference=False,
    ).fit(native(X), native(y))
    _, gradient, _ = _objective_functions(X, y, None, 1.0, True)
    assert model.converged_
    assert np.linalg.norm(gradient(_parameters(model))) / len(y) < model.tol
    np.testing.assert_allclose(
        _numpy(model.predict_proba(native(X))), expected, rtol=5e-4, atol=2e-4
    )


def test_shared_lbfgs_keeps_legacy_step_stop_unless_loss_requires_gradient():
    from statgpu.linear_model.wrappers._logistic_solver import _LogisticObjective
    from statgpu.solvers import lbfgs_solver

    X, y = _mixed_scale_problem(1000.0)
    design = np.column_stack([np.ones(len(y)), X])
    legacy = _LogisticObjective(None, 1.0, True, normalizer=len(y))
    legacy._require_gradient_convergence = False
    strict = _LogisticObjective(None, 1.0, True, normalizer=len(y))
    legacy_params, legacy_iterations = lbfgs_solver(
        legacy, None, design, y, tol=1e-4, max_iter=500
    )
    strict_params, strict_iterations = lbfgs_solver(
        strict, None, design, y, tol=1e-4, max_iter=500
    )
    assert legacy_iterations < strict_iterations
    assert (
        np.linalg.norm(legacy.fused_value_and_gradient(design, y, legacy_params)[1])
        > 1e-2
    )
    assert (
        np.linalg.norm(strict.fused_value_and_gradient(design, y, strict_params)[1])
        < 1e-4
    )


def test_lbfgs_early_nonstationary_return_is_not_converged(
    logistic_problem, monkeypatch
):
    from statgpu.linear_model.wrappers import _logistic_solver as adapter

    X, y, weight = logistic_problem
    monkeypatch.setattr(
        adapter,
        "lbfgs_solver",
        lambda loss, penalty, design, response, **kwargs: (
            np.zeros(design.shape[1]),
            1,
        ),
    )
    with pytest.warns(ConvergenceWarning, match="LBFGS"):
        model = LogisticRegression(
            solver="lbfgs", device="cpu", max_iter=100, compute_inference=False
        ).fit(X, y, sample_weight=weight)
    assert not model.converged_
