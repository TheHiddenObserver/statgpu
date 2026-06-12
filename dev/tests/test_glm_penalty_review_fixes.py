import warnings

import numpy as np
import pytest

from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
from statgpu.glm_core._gamma import GammaLoss
from statgpu.glm_core._logistic import LogisticLoss
from statgpu.glm_core._solver import (
    _fused_glm_value_and_gradient,
    _get_sqerr_proximal_cupy,
    admm_solver,
    fista_lla_path,
    fista_bb_solver,
    fista_solver,
    lbfgs_solver,
    newton_solver,
)
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.glm_core._tweedie import TweedieLoss
from statgpu.linear_model import (
    GeneralizedLinearModel,
    OrderedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
from statgpu.linear_model._penalized import (
    PenalizedGeneralizedLinearModel,
    _preferred_penalized_glm_solver,
    _resolve_loss_name,
)
from statgpu.penalties import (
    AdaptiveL1Penalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
    L2Penalty,
    MCPPenalty,
    SCADPenalty,
)


def _skip_if_cupy_cuda_unavailable():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            pytest.skip("CuPy CUDA unavailable")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA unavailable: {exc}")
    return cp


def test_negative_binomial_loss_alpha_matches_finite_difference_gradient():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(40, 4))
    y = rng.poisson(lam=2.0, size=40).astype(float)
    coef = rng.normal(scale=0.15, size=4)
    loss = NegativeBinomialLoss(alpha=0.35)

    analytical = loss.gradient(X, y, coef)
    numerical = np.empty_like(coef)
    eps = 1e-6
    for j in range(coef.size):
        step = np.zeros_like(coef)
        step[j] = eps
        numerical[j] = (
            loss.value(X, y, coef + step) - loss.value(X, y, coef - step)
        ) / (2.0 * eps)

    assert np.allclose(analytical, numerical, rtol=1e-4, atol=1e-5)


def test_fused_negative_binomial_uses_loss_alpha():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(32, 3))
    y = rng.poisson(lam=1.7, size=32).astype(float)
    coef = rng.normal(scale=0.2, size=3)
    loss = NegativeBinomialLoss(alpha=0.25)

    fused_value, fused_grad = _fused_glm_value_and_gradient(loss, X, y, coef)

    assert np.allclose(fused_value, loss.value(X, y, coef))
    assert np.allclose(fused_grad, loss.gradient(X, y, coef))


def test_logistic_loss_value_is_stable_for_large_linear_predictor():
    X = np.array([[1000.0], [-1000.0]])
    y = np.array([0.0, 1.0])
    coef = np.array([1.0])
    loss = LogisticLoss()

    assert np.isfinite(loss.value(X, y, coef))
    assert np.allclose(loss.value(X, y, coef), 1000.0)


def test_fused_tweedie_uses_loss_power():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(32, 3))
    y = np.exp(rng.normal(scale=0.3, size=32))
    coef = rng.normal(scale=0.2, size=3)
    loss = TweedieLoss(power=1.25)

    fused_value, fused_grad = _fused_glm_value_and_gradient(loss, X, y, coef)

    assert np.allclose(fused_value, loss.value(X, y, coef))
    assert np.allclose(fused_grad, loss.gradient(X, y, coef))


def test_penalized_glm_can_preserve_fold_cv_cache_across_repeated_fits():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(30, 5))
    y = X @ np.linspace(1.0, -0.5, 5) + rng.normal(scale=0.1, size=30)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        alpha=0.05,
        solver="fista",
        device="cpu",
        compute_inference=False,
        max_iter=3,
    )
    model._cv_cache = {"XtX": Xc.T @ Xc, "Xty": Xc.T @ yc}
    model._preserve_cv_cache = True

    model.fit(X, y)

    assert hasattr(model, "_cv_cache")


def test_penalized_glm_cv_uses_strict_logistic_solver_defaults():
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    assert PenalizedGLM_CV().random_state == 0
    assert PenalizedGLM_CV(loss="logistic", penalty="l2")._solver_for_cv() == "irls"
    assert PenalizedGLM_CV(loss="logistic", penalty="l1")._solver_for_cv() == "fista"
    assert (
        PenalizedGLM_CV(loss="logistic", penalty="elasticnet")._solver_for_cv()
        == "fista"
    )
    assert (
        PenalizedGLM_CV(loss="logistic", penalty="l2", solver="fista")._solver_for_cv()
        == "fista"
    )
    assert (
        PenalizedGLM_CV(loss="negative_binomial", penalty="l1")._solver_for_cv()
        == "fista_bb"
    )
    assert (
        PenalizedGLM_CV(loss="tweedie", penalty="l1")._solver_for_cv("cuda")
        == "fista"
    )


def test_private_auto_solver_policy_by_loss_penalty_backend():
    assert _preferred_penalized_glm_solver("squared_error", "l2") == "exact"
    assert _preferred_penalized_glm_solver("gamma", "l2") == "newton"
    assert _preferred_penalized_glm_solver("logistic", "l2") == "irls"
    assert (
        _preferred_penalized_glm_solver(
            "negative_binomial", "elasticnet", backend_name="torch", cv_mode=True
        )
        == "fista_bb"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson", "elasticnet", backend_name="cupy", cv_mode=True
        )
        == "fista_bb"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson", "l1", backend_name="cupy", cv_mode=True
        )
        == "fista_bb"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson",
            "l1",
            backend_name="cupy",
            cv_mode=True,
            problem_size=2_500_000,
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson", "l1", backend_name="numpy", cv_mode=True
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "negative_binomial",
            "l1",
            backend_name="cupy",
            cv_mode=True,
            problem_size=10_000,
        )
        == "fista_bb"
    )
    assert (
        _preferred_penalized_glm_solver(
            "gamma", "l1", backend_name="torch", cv_mode=True
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "gamma", "l1", backend_name="cupy", cv_mode=True
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson", "l2", backend_name="cupy", cv_mode=True
        )
        == "newton"
    )
    assert (
        _preferred_penalized_glm_solver(
            "poisson", "l2", backend_name="numpy", cv_mode=True
        )
        == "newton"
    )
    assert (
        _preferred_penalized_glm_solver(
            "gamma", "l2", backend_name="cupy", cv_mode=True
        )
        == "lbfgs"
    )
    assert (
        _preferred_penalized_glm_solver(
            "negative_binomial", "l2", backend_name="cupy", cv_mode=True
        )
        == "lbfgs"
    )
    assert (
        _preferred_penalized_glm_solver(
            "tweedie", "l2", backend_name="torch", cv_mode=True
        )
        == "newton"
    )
    assert (
        _preferred_penalized_glm_solver(
            "negative_binomial",
            "l1",
            backend_name="cupy",
            cv_mode=True,
            problem_size=400_000,
        )
        == "fista_bb"
    )
    assert (
        _preferred_penalized_glm_solver(
            "negative_binomial",
            "elasticnet",
            backend_name="torch",
            cv_mode=True,
            problem_size=400_000,
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "inverse_gaussian", "l1", backend_name="cupy", cv_mode=True
        )
        == "fista"
    )
    assert (
        _preferred_penalized_glm_solver(
            "tweedie", "elasticnet", backend_name="cupy", cv_mode=True
        )
        == "fista"
    )


def test_penalized_glm_cv_rejects_unknown_cv_strategy():
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    with pytest.raises(ValueError, match="cv_strategy"):
        PenalizedGLM_CV(cv_strategy="auto")


def test_penalized_glm_cv_strict_is_default_and_scores_all_alphas(monkeypatch):
    from statgpu.linear_model import _penalized_cv as cv_mod

    calls = []

    def fake_compute(self, X, y, alpha_grid, cv_device, folds, sample_weight=None,
                     max_iter=None, tol=None, strict=True):
        calls.append({
            "alpha_grid": np.asarray(alpha_grid, dtype=float).copy(),
            "max_iter": max_iter,
            "tol": tol,
            "strict": strict,
        })
        row = np.arange(len(alpha_grid), dtype=float)
        return np.tile(row, (self.cv, 1))

    class DummyEstimator:
        coef_ = np.zeros(2)
        intercept_ = 0.0

    monkeypatch.setattr(cv_mod.PenalizedGLM_CV, "_compute_cv_scores", fake_compute)
    monkeypatch.setattr(
        cv_mod.PenalizedGLM_CV,
        "_refit_best",
        lambda self, X, y, best_alpha, sample_weight=None: DummyEstimator(),
    )

    alpha_grid = np.array([0.3, 0.2, 0.1])
    model = cv_mod.PenalizedGLM_CV(
        alpha_grid=alpha_grid,
        cv=3,
        max_iter=123,
        tol=1e-7,
    )
    model.fit(np.zeros((6, 2)), np.zeros(6))

    assert model.cv_strategy_ == "strict"
    assert len(calls) == 1
    assert calls[0]["strict"] is True
    assert calls[0]["max_iter"] == 123
    assert calls[0]["tol"] == 1e-7
    assert np.array_equal(calls[0]["alpha_grid"], alpha_grid)
    assert model.alpha_ == alpha_grid[0]
    assert model.cv_results_["mean_score_stage1"] is None
    assert model.cv_results_["all_scores_stage1"] is None
    assert np.all(model.cv_results_["refined_mask"])


def test_penalized_glm_cv_strict_poisson_sparse_disables_cv_mode(monkeypatch):
    from statgpu.linear_model import _penalized_cv as cv_mod

    seen = []

    def fake_path(
        loss_name,
        X_train,
        y_train,
        alpha_sorted,
        penalty_name,
        l1_ratio,
        max_iter,
        tol,
        device,
        X_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        return_path=False,
        solver_name="fista",
        cv_mode=True,
    ):
        seen.append(bool(cv_mode))
        n_alphas = len(alpha_sorted)
        return {
            "scores": np.arange(n_alphas, dtype=np.float64),
            "coef": np.zeros((n_alphas, X_train.shape[1]), dtype=np.float64),
            "intercept": np.zeros(n_alphas, dtype=np.float64),
            "n_iter": np.ones(n_alphas, dtype=np.int64),
        }

    monkeypatch.setattr(cv_mod, "_glm_sparse_cv_path", fake_path)

    rng = np.random.default_rng(33)
    X = rng.normal(size=(12, 3))
    y = rng.poisson(lam=1.2, size=12).astype(float)
    folds = cv_mod.kfold_indices(X.shape[0], 2, 0)
    model = cv_mod.PenalizedGLM_CV(
        loss="poisson",
        penalty="elasticnet",
        alpha_grid=np.array([0.1, 0.02]),
        cv=2,
        device="cpu",
    )

    model._compute_cv_scores(
        X,
        y,
        np.array([0.1, 0.02]),
        "cpu",
        folds,
        strict=True,
    )
    model._compute_cv_scores(
        X,
        y,
        np.array([0.1, 0.02]),
        "cpu",
        folds,
        strict=False,
    )

    assert seen == [False, False, True, True]


def test_penalized_glm_cv_poisson_sparse_tie_break_prefers_larger_alpha():
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    alpha_grid = np.array([2.5e-4, 6.7e-5, 1.8e-5, 2.5e-8])
    scores = np.array([-2.22244263, -2.22245286, -2.22245248, -2.22245293])
    model = PenalizedGLM_CV(loss="poisson", penalty="l1")

    idx = model._best_index_from_scores(scores, alpha_grid, "fista")

    assert idx == 1


def test_penalized_glm_cv_two_stage_warns_and_acknowledge_suppresses():
    from statgpu.linear_model._penalized_cv import (
        ApproximateCVWarning,
        PenalizedGLM_CV,
    )

    rng = np.random.default_rng(31)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([0.5, -0.2, 0.1]) + rng.normal(scale=0.05, size=20)
    alpha_grid = np.array([0.2, 0.05, 0.01])

    with pytest.warns(ApproximateCVWarning):
        PenalizedGLM_CV(
            loss="squared_error",
            penalty="l2",
            alpha_grid=alpha_grid,
            cv=2,
            cv_strategy="two_stage",
        ).fit(X, y)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        PenalizedGLM_CV(
            loss="squared_error",
            penalty="l2",
            alpha_grid=alpha_grid,
            cv=2,
            cv_strategy="two_stage",
            acknowledge_approx=True,
        ).fit(X, y)

    assert not any(isinstance(item.message, ApproximateCVWarning) for item in seen)


def test_two_stage_candidate_mask_includes_top_neighbors_and_near_ties():
    from statgpu.linear_model._penalized_cv import _two_stage_candidate_mask

    scores = np.array([1.0, 0.5, 0.6, 0.502, 0.9])
    mask = _two_stage_candidate_mask(scores, refine_top_k=1)

    assert np.array_equal(mask, np.array([True, True, True, True, True]))


def test_penalized_glm_cv_two_stage_selects_from_refined_scores(monkeypatch):
    from statgpu.linear_model import _penalized_cv as cv_mod

    alpha_grid = np.array([1.0, 0.1, 0.01, 0.001])
    calls = []

    def fake_compute(self, X, y, grid, cv_device, folds, sample_weight=None,
                     max_iter=None, tol=None, strict=True):
        grid = np.asarray(grid, dtype=float)
        calls.append((strict, grid.copy(), max_iter, tol))
        if strict:
            assert np.array_equal(grid, alpha_grid)
            return np.tile(np.array([0.5, 0.6, 0.7, 0.8]), (self.cv, 1))
        return np.tile(np.array([0.2, 0.1, 0.3, 0.11]), (self.cv, 1))

    class DummyEstimator:
        coef_ = np.zeros(2)
        intercept_ = 0.0

    monkeypatch.setattr(cv_mod.PenalizedGLM_CV, "_compute_cv_scores", fake_compute)
    monkeypatch.setattr(
        cv_mod.PenalizedGLM_CV,
        "_refit_best",
        lambda self, X, y, best_alpha, sample_weight=None: DummyEstimator(),
    )

    model = cv_mod.PenalizedGLM_CV(
        alpha_grid=alpha_grid,
        cv=2,
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=200,
        tol=1e-5,
    )
    model.fit(np.zeros((8, 2)), np.zeros(8))

    assert model.alpha_ == 1.0
    assert np.array_equal(
        model.cv_results_["refined_mask"],
        np.array([True, True, True, True]),
    )
    assert calls[0][0] is False
    assert np.array_equal(calls[0][1], alpha_grid)
    assert calls[0][2] == 50
    assert calls[0][3] == 1e-4
    assert calls[1][0] is True
    assert calls[1][2] == 200
    assert calls[1][3] == 1e-5


def test_penalized_glm_cv_two_stage_refines_all_gaussian_nonconvex(monkeypatch):
    from statgpu.linear_model import _penalized_cv as cv_mod

    alpha_grid = np.array([1.0, 0.1, 0.01, 0.001])

    def fake_compute(self, X, y, grid, cv_device, folds, sample_weight=None,
                     max_iter=None, tol=None, strict=True):
        grid = np.asarray(grid, dtype=float)
        if strict:
            assert np.array_equal(grid, alpha_grid)
            return np.tile(np.array([0.4, 0.1, 0.2, 0.3]), (self.cv, 1))
        return np.tile(np.array([0.2, 0.1, 0.3, 0.11]), (self.cv, 1))

    class DummyEstimator:
        coef_ = np.zeros(2)
        intercept_ = 0.0

    monkeypatch.setattr(cv_mod.PenalizedGLM_CV, "_compute_cv_scores", fake_compute)
    monkeypatch.setattr(
        cv_mod.PenalizedGLM_CV,
        "_refit_best",
        lambda self, X, y, best_alpha, sample_weight=None: DummyEstimator(),
    )

    model = cv_mod.PenalizedGLM_CV(
        loss="squared_error",
        penalty="scad",
        alpha_grid=alpha_grid,
        cv=2,
        cv_strategy="two_stage",
        acknowledge_approx=True,
    )
    model.fit(np.zeros((8, 2)), np.zeros(8))

    assert np.all(model.cv_results_["refined_mask"])
    assert model.alpha_ == 0.1


def test_penalized_glm_cv_two_stage_explicit_gpu_device_is_not_overridden():
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV, _device_to_name

    model = PenalizedGLM_CV(device="cuda", cv_strategy="two_stage", acknowledge_approx=True)

    assert _device_to_name(model._effective_cv_device(np.zeros((20, 3)), "l2", 3)) == "cuda"


def test_cv_auto_routes_medium_tweedie_sparse_to_torch(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    monkeypatch.setattr(cv_mod, "_torch_cuda_available", lambda: True)
    cv = PenalizedGLM_CV(
        loss="tweedie",
        penalty="l1",
        device="auto",
        n_alphas=6,
        cv=3,
    )

    selected = cv._effective_cv_device(np.zeros((2000, 200)), "l1", 6)

    assert selected == "torch"


def test_cv_auto_sparse_glm_routes_by_benchmarked_break_even(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    monkeypatch.setattr(cv_mod, "_torch_cuda_available", lambda: True)

    logistic = PenalizedGLM_CV(loss="logistic", penalty="l1", device="auto", n_alphas=8, cv=3)
    assert logistic._effective_cv_device(np.zeros((2000, 100)), "l1", 8) == "cpu"
    assert logistic._effective_cv_device(np.zeros((2000, 500)), "l1", 8) == "torch"
    assert logistic._effective_cv_device(np.zeros((5000, 500)), "l1", 8) == "torch"

    poisson = PenalizedGLM_CV(loss="poisson", penalty="elasticnet", device="auto", n_alphas=8, cv=3)
    assert poisson._effective_cv_device(np.zeros((10000, 100)), "elasticnet", 8) == "cpu"
    assert poisson._effective_cv_device(np.zeros((2000, 500)), "elasticnet", 8) == "torch"

    gamma = PenalizedGLM_CV(loss="gamma", penalty="l1", device="auto", n_alphas=8, cv=3)
    assert gamma._effective_cv_device(np.zeros((2000, 500)), "l1", 8) == "cpu"
    assert gamma._effective_cv_device(np.zeros((5000, 500)), "l1", 8) == "torch"


def test_cv_auto_squared_error_sparse_uses_batched_torch_break_even(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    monkeypatch.setattr(cv_mod, "_torch_cuda_available", lambda: True)
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty="elasticnet",
        device="auto",
        n_alphas=8,
        cv=3,
    )

    assert cv._effective_cv_device(np.zeros((2000, 100)), "elasticnet", 8) == "cpu"
    assert cv._effective_cv_device(np.zeros((10000, 100)), "elasticnet", 8) == "cpu"
    assert cv._effective_cv_device(np.zeros((2000, 500)), "elasticnet", 8) == "torch"


def test_glm_sparse_final_refit_uses_strict_single_alpha_path(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    calls = []

    def fake_glm_path(loss_name, X, y, alpha_sorted, penalty_name, l1_ratio,
                      max_iter, tol, device, **kwargs):
        calls.append({
            "loss": loss_name,
            "alpha": np.asarray(alpha_sorted, dtype=float).copy(),
            "max_iter": max_iter,
            "tol": tol,
            "device": device,
            "return_path": kwargs.get("return_path"),
            "solver_name": kwargs.get("solver_name"),
            "cv_mode": kwargs.get("cv_mode"),
        })
        return {
            "coef": np.array([[0.1, -0.2]]),
            "intercept": np.array([0.3]),
            "n_iter": np.array([17]),
            "scores": None,
        }

    monkeypatch.setattr(cv_mod, "_glm_sparse_cv_path", fake_glm_path)
    model = PenalizedGLM_CV(
        loss="poisson",
        penalty="l1",
        solver="fista",
        device="cpu",
        max_iter=123,
        tol=1e-7,
    )._refit_best(np.zeros((5, 2)), np.ones(5), 0.04)

    assert len(calls) == 1
    call = calls[0]
    assert call["loss"] == "poisson"
    assert np.allclose(call["alpha"], [0.04])
    assert call["max_iter"] == 123
    assert call["tol"] == 1e-7
    assert cv_mod._device_to_name(call["device"]) == "cpu"
    assert call["return_path"] is True
    assert call["solver_name"] == "fista"
    assert call["cv_mode"] is False
    assert np.allclose(model.coef_, [0.1, -0.2])
    assert model.intercept_ == pytest.approx(0.3)
    assert model.n_iter_ == 17


def test_logistic_sparse_gpu_cv_caps_only_path_iterations_not_refit():
    from statgpu.linear_model._penalized_cv import _logistic_sparse_effective_max_iter

    assert _logistic_sparse_effective_max_iter(1000, "cuda", "l1") == 400
    assert _logistic_sparse_effective_max_iter(1000, "torch", "elasticnet") == 600
    assert _logistic_sparse_effective_max_iter(1000, "cpu", "l1") == 1000
    assert (
        _logistic_sparse_effective_max_iter(
            1000, "cuda", "l1", refit=True
        )
        == 1000
    )


def test_logistic_sparse_torch_cv_uses_fold_batched_path(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._cv_base import kfold_indices
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    calls = []

    def fake_fold_batched(
        X,
        y,
        folds,
        alpha_sorted,
        penalty_name,
        l1_ratio,
        max_iter,
        tol,
        loss_name,
        device_backend,
        sample_weight=None,
    ):
        calls.append(
            {
                "alpha_sorted": np.asarray(alpha_sorted, dtype=float).copy(),
                "penalty_name": penalty_name,
                "max_iter": max_iter,
                "tol": tol,
                "n_folds": len(folds),
                "loss_name": loss_name,
                "device_backend": device_backend,
            }
        )
        return {
            "scores": np.tile(np.array([0.4, 0.2]), (len(folds), 1)),
            "n_iter": np.ones((len(folds), 2), dtype=np.int64),
        }

    monkeypatch.setattr(cv_mod, "_glm_sparse_cv_folds", fake_fold_batched)

    rng = np.random.default_rng(36)
    X = rng.normal(size=(12, 3))
    y = (X[:, 0] > 0).astype(float)
    folds = kfold_indices(X.shape[0], 2, random_state=0)
    alpha_grid = np.array([0.01, 0.1])
    cv = PenalizedGLM_CV(
        loss="logistic",
        penalty="l1",
        alpha_grid=alpha_grid,
        cv=2,
        device="torch",
        max_iter=123,
        tol=1e-7,
    )

    scores = cv._compute_cv_scores(
        X,
        y,
        alpha_grid,
        "torch",
        folds,
        strict=False,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0]["alpha_sorted"], np.array([0.1, 0.01]))
    assert calls[0]["penalty_name"] == "l1"
    assert calls[0]["max_iter"] == 123
    assert calls[0]["tol"] == 1e-7
    assert calls[0]["n_folds"] == 2
    assert np.allclose(scores, np.tile(np.array([0.2, 0.4]), (2, 1)))


def test_logistic_sparse_cupy_cv_uses_fold_batched_path(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._cv_base import kfold_indices
    from statgpu.linear_model._penalized_cv import PenalizedGLM_CV

    calls = []

    def fake_fold_batched(
        X,
        y,
        folds,
        alpha_sorted,
        penalty_name,
        l1_ratio,
        max_iter,
        tol,
        loss_name,
        device_backend,
        sample_weight=None,
    ):
        calls.append(
            {
                "alpha_sorted": np.asarray(alpha_sorted, dtype=float).copy(),
                "penalty_name": penalty_name,
                "max_iter": max_iter,
                "tol": tol,
                "n_folds": len(folds),
                "loss_name": loss_name,
                "device_backend": device_backend,
            }
        )
        return {
            "scores": np.tile(np.array([0.5, 0.3]), (len(folds), 1)),
            "n_iter": np.ones((len(folds), 2), dtype=np.int64),
        }

    monkeypatch.setattr(cv_mod, "_glm_sparse_cv_folds", fake_fold_batched)

    rng = np.random.default_rng(39)
    X = rng.normal(size=(12, 3))
    y = (X[:, 0] > 0).astype(float)
    folds = kfold_indices(X.shape[0], 2, random_state=0)
    alpha_grid = np.array([0.01, 0.1])
    cv = PenalizedGLM_CV(
        loss="logistic",
        penalty="elasticnet",
        alpha_grid=alpha_grid,
        cv=2,
        device="cuda",
        max_iter=123,
        tol=1e-7,
    )

    scores = cv._compute_cv_scores(
        X,
        y,
        alpha_grid,
        "cuda",
        folds,
        strict=False,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0]["alpha_sorted"], np.array([0.1, 0.01]))
    assert calls[0]["penalty_name"] == "elasticnet"
    assert calls[0]["max_iter"] == 123
    assert calls[0]["tol"] == 1e-7
    assert calls[0]["n_folds"] == 2
    assert np.allclose(scores, np.tile(np.array([0.3, 0.5]), (2, 1)))


def test_logistic_sparse_fold_batched_torch_matches_per_fold_path():
    import platform

    if platform.system().lower() == "windows":
        pytest.skip("Torch fold-batched CV is validated on the remote Linux GPU gate")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device not available")

    from statgpu.linear_model._cv_base import kfold_indices
    from statgpu.linear_model._penalized_cv import (
        _glm_sparse_cv_folds,
        _logistic_sparse_cv_path,
    )

    rng = np.random.default_rng(37)
    X = rng.normal(scale=0.4, size=(45, 6))
    y = (X @ np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.2]) > 0).astype(float)
    folds = kfold_indices(X.shape[0], 3, random_state=0)
    alpha_sorted = np.array([0.1, 0.03, 0.01])

    expected = np.full((len(folds), alpha_sorted.size), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        path = _logistic_sparse_cv_path(
            X[train_idx],
            y[train_idx],
            alpha_sorted,
            "elasticnet",
            0.5,
            max_iter=80,
            tol=1e-4,
            device="torch",
            X_val=X[val_idx],
            y_val=y[val_idx],
            return_path=False,
        )
        expected[fold_idx, :] = path["scores"]

    got = _glm_sparse_cv_folds(
        X,
        y,
        folds,
        alpha_sorted,
        "elasticnet",
        0.5,
        max_iter=80,
        tol=1e-4,
        loss_name="logistic",
        device_backend="torch",
    )

    assert got is not None
    assert np.allclose(got["scores"], expected, rtol=1e-10, atol=1e-10)


def test_logistic_sparse_fold_batched_cupy_matches_per_fold_path():
    import platform

    if platform.system().lower() == "windows":
        pytest.skip("CuPy fold-batched CV is validated on the remote Linux GPU gate")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            pytest.skip("CuPy CUDA device not available")
    except Exception:
        pytest.skip("CuPy CUDA device not available")

    from statgpu.linear_model._cv_base import kfold_indices
    from statgpu.linear_model._penalized_cv import (
        _glm_sparse_cv_folds,
        _logistic_sparse_cv_path,
    )

    rng = np.random.default_rng(40)
    X = rng.normal(scale=0.4, size=(45, 6))
    y = (X @ np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.2]) > 0).astype(float)
    folds = kfold_indices(X.shape[0], 3, random_state=0)
    alpha_sorted = np.array([0.1, 0.03, 0.01])

    expected = np.full((len(folds), alpha_sorted.size), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        path = _logistic_sparse_cv_path(
            X[train_idx],
            y[train_idx],
            alpha_sorted,
            "l1",
            0.5,
            max_iter=80,
            tol=1e-4,
            device="cuda",
            X_val=X[val_idx],
            y_val=y[val_idx],
            return_path=False,
        )
        expected[fold_idx, :] = path["scores"]

    got = _glm_sparse_cv_folds(
        X,
        y,
        folds,
        alpha_sorted,
        "l1",
        0.5,
        max_iter=80,
        tol=1e-4,
        loss_name="logistic",
        device_backend="cupy",
    )

    assert got is not None
    assert np.allclose(got["scores"], expected, rtol=1e-10, atol=1e-10)


def test_squared_error_sparse_cv_path_scores_entire_alpha_grid():
    from statgpu.linear_model._penalized_cv import _squared_error_sparse_cv_path

    rng = np.random.default_rng(27)
    X = rng.normal(size=(36, 5))
    coef = np.array([1.0, -0.6, 0.0, 0.25, 0.0])
    y = X @ coef + 0.4 + rng.normal(scale=0.05, size=X.shape[0])
    alpha_grid = np.array([0.2, 0.05, 0.01])

    path = _squared_error_sparse_cv_path(
        X[:24],
        y[:24],
        alpha_grid,
        "elasticnet",
        0.5,
        max_iter=80,
        tol=1e-6,
        device="cpu",
        X_val=X[24:],
        y_val=y[24:],
    )

    assert path is not None
    assert path["coef"].shape == (alpha_grid.size, X.shape[1])
    assert path["intercept"].shape == (alpha_grid.size,)
    assert path["scores"].shape == (alpha_grid.size,)
    assert np.all(np.isfinite(path["scores"]))

    score_only = _squared_error_sparse_cv_path(
        X[:24],
        y[:24],
        alpha_grid,
        "elasticnet",
        0.5,
        max_iter=80,
        tol=1e-6,
        device="cpu",
        X_val=X[24:],
        y_val=y[24:],
        return_path=False,
    )
    assert "coef" not in score_only
    assert "intercept" not in score_only
    assert score_only["scores"].shape == (alpha_grid.size,)


def test_squared_error_sparse_gpu_score_path_batches_alpha_grid():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for torch")
    from statgpu.linear_model._penalized_cv import _squared_error_sparse_cv_path

    rng = np.random.default_rng(127)
    X = rng.normal(size=(72, 8))
    coef = np.linspace(0.8, -0.2, X.shape[1])
    y = X @ coef + 0.25 + rng.normal(scale=0.03, size=X.shape[0])
    alpha_grid = np.geomspace(0.2, 0.002, 5)

    sequential = _squared_error_sparse_cv_path(
        X[:48],
        y[:48],
        alpha_grid,
        "elasticnet",
        0.5,
        max_iter=300,
        tol=1e-7,
        device="torch",
        X_val=X[48:],
        y_val=y[48:],
        return_path=True,
    )
    batched = _squared_error_sparse_cv_path(
        X[:48],
        y[:48],
        alpha_grid,
        "elasticnet",
        0.5,
        max_iter=300,
        tol=1e-7,
        device="torch",
        X_val=X[48:],
        y_val=y[48:],
        return_path=False,
    )

    assert "coef" not in batched
    assert batched["scores"].shape == alpha_grid.shape
    assert np.allclose(batched["scores"], sequential["scores"], rtol=1e-4, atol=1e-5)


def test_negative_binomial_cupy_l2_iteration_cap_is_cv_only():
    from statgpu.linear_model._penalized_cv import _glm_cv_effective_max_iter

    assert _glm_cv_effective_max_iter(1000, "negative_binomial", "l2", "cuda") == 30
    assert (
        _glm_cv_effective_max_iter(
            1000, "negative_binomial", "l2", "cuda", refit=True
        )
        == 1000
    )
    assert _glm_cv_effective_max_iter(1000, "negative_binomial", "l2", "cpu") == 1000
    assert _glm_cv_effective_max_iter(1000, "negative_binomial", "l1", "cuda") == 1000


def test_tweedie_sparse_gpu_iteration_cap_is_cv_only():
    from statgpu.linear_model._penalized_cv import _glm_cv_effective_max_iter

    assert _glm_cv_effective_max_iter(1000, "tweedie", "l1", "cuda") == 200
    assert _glm_cv_effective_max_iter(1000, "tweedie", "elasticnet", "torch") == 200
    assert (
        _glm_cv_effective_max_iter(1000, "tweedie", "l1", "cuda", refit=True)
        == 1000
    )
    assert _glm_cv_effective_max_iter(1000, "gamma", "l1", "cuda") == 1000


def test_glm_sparse_cv_path_scores_entire_alpha_grid_on_cpu():
    from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path

    rng = np.random.default_rng(28)
    X = rng.normal(scale=0.3, size=(42, 4))
    beta = np.array([0.25, -0.15, 0.05, 0.0])
    y = rng.poisson(np.exp(np.clip(X @ beta, -2, 2))).astype(float)
    alpha_grid = np.array([0.05, 0.01, 0.002])

    path = _glm_sparse_cv_path(
        "poisson",
        X[:30],
        y[:30],
        alpha_grid,
        "elasticnet",
        0.5,
        max_iter=20,
        tol=1e-4,
        device="cpu",
        X_val=X[30:],
        y_val=y[30:],
        return_path=True,
    )

    assert path is not None
    assert path["scores"].shape == (alpha_grid.size,)
    assert path["coef"].shape == (alpha_grid.size, X.shape[1])
    assert path["intercept"].shape == (alpha_grid.size,)
    assert np.all(np.isfinite(path["scores"]))


def test_glm_sparse_cv_path_can_use_fista_bb_cv_mode(monkeypatch):
    import statgpu.glm_core._solver as solver_mod
    from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path

    calls = []

    def fake_fista_bb(loss, penalty, X_work, y_arr, **kwargs):
        init_coef = kwargs.get("init_coef")
        calls.append(
            {
                "cv_mode": kwargs.get("cv_mode"),
                "init": None if init_coef is None else np.asarray(init_coef, dtype=float).copy(),
            }
        )
        return np.zeros(X_work.shape[1]), 1

    monkeypatch.setattr(solver_mod, "fista_bb_solver", fake_fista_bb)

    rng = np.random.default_rng(30)
    X = rng.normal(scale=0.3, size=(36, 4))
    y = rng.negative_binomial(1.0, 0.5, size=36).astype(float)
    alpha_grid = np.array([0.1, 0.02])

    path = _glm_sparse_cv_path(
        "negative_binomial",
        X[:24],
        y[:24],
        alpha_grid,
        "l1",
        0.5,
        max_iter=20,
        tol=1e-4,
        device="cpu",
        X_val=X[24:],
        y_val=y[24:],
        return_path=True,
        solver_name="fista_bb",
    )

    assert path is not None
    assert [call["cv_mode"] for call in calls] == [True, True]
    assert calls[0]["init"] is not None
    assert np.isclose(calls[0]["init"][-1], np.log(max(np.mean(y[:24]), 1e-3)))
    assert np.allclose(calls[0]["init"][:-1], 0.0)
    assert np.allclose(calls[1]["init"], 0.0)


def test_glm_sparse_cv_path_can_disable_private_cv_mode(monkeypatch):
    import statgpu.glm_core._solver as solver_mod
    from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path

    seen = []

    def fake_fista(loss, penalty, X_work, y_arr, **kwargs):
        seen.append(kwargs.get("cv_mode"))
        return np.zeros(X_work.shape[1]), 1

    monkeypatch.setattr(solver_mod, "fista_solver", fake_fista)

    rng = np.random.default_rng(32)
    X = rng.normal(scale=0.3, size=(36, 4))
    y = rng.poisson(lam=1.3, size=36).astype(float)

    path = _glm_sparse_cv_path(
        "poisson",
        X[:24],
        y[:24],
        np.array([0.1, 0.02]),
        "elasticnet",
        0.5,
        max_iter=20,
        tol=1e-4,
        device="cpu",
        X_val=X[24:],
        y_val=y[24:],
        return_path=True,
        solver_name="fista",
        cv_mode=False,
    )

    assert path is not None
    assert seen == [False, False]


def test_glm_sparse_cv_path_reuses_fold_lipschitz(monkeypatch):
    import statgpu.glm_core._solver as solver_mod
    import statgpu.linear_model._penalized as penalized_mod
    from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path

    class FakePoissonLoss:
        name = "poisson"
        _lipschitz_at_init = False

        def __init__(self):
            self.lipschitz_calls = 0

        def lipschitz(self, X, coef, y=None):
            self.lipschitz_calls += 1
            return 7.0

        def value(self, X, y, coef):
            return float(np.mean(np.asarray(X) @ np.asarray(coef)))

    fake_loss = FakePoissonLoss()
    monkeypatch.setattr(
        penalized_mod,
        "_resolve_loss_name",
        lambda loss_name, loss_kwargs=None: fake_loss,
    )

    calls = {"fista": [], "fista_bb": []}

    def fake_fista(loss, penalty, X_work, y_arr, **kwargs):
        calls["fista"].append(kwargs.get("lipschitz_L"))
        return np.zeros(X_work.shape[1]), 1

    def fake_fista_bb(loss, penalty, X_work, y_arr, **kwargs):
        calls["fista_bb"].append(kwargs.get("lipschitz_L"))
        return np.zeros(X_work.shape[1]), 1

    monkeypatch.setattr(solver_mod, "fista_solver", fake_fista)
    monkeypatch.setattr(solver_mod, "fista_bb_solver", fake_fista_bb)

    rng = np.random.default_rng(34)
    X = rng.normal(scale=0.3, size=(32, 4))
    y = rng.poisson(lam=1.3, size=32).astype(float)
    alpha_grid = np.array([0.2, 0.05, 0.01])

    for solver_name in ("fista", "fista_bb"):
        fake_loss.lipschitz_calls = 0
        path = _glm_sparse_cv_path(
            "poisson",
            X[:24],
            y[:24],
            alpha_grid,
            "l1",
            0.5,
            max_iter=20,
            tol=1e-4,
            device="cpu",
            X_val=X[24:],
            y_val=y[24:],
            return_path=False,
            solver_name=solver_name,
        )

        assert path is not None
        assert fake_loss.lipschitz_calls == 1
        assert calls[solver_name] == [7.0, 7.0, 7.0]


def test_glm_sparse_cupy_score_batch_matches_returned_path():
    import platform

    if platform.system().lower() == "windows":
        pytest.skip("CUDA score batching is validated on the remote Linux GPU gate")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            pytest.skip("CUDA device not available")
    except Exception:
        pytest.skip("CUDA device not available")

    from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path

    rng = np.random.default_rng(35)
    X = rng.normal(scale=0.25, size=(48, 5))
    beta = np.array([0.2, -0.1, 0.05, 0.0, 0.03])
    y = rng.poisson(np.exp(np.clip(X @ beta, -2, 2))).astype(float)
    alpha_grid = np.array([0.05, 0.01, 0.002])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = _glm_sparse_cv_path(
            "poisson",
            X[:32],
            y[:32],
            alpha_grid,
            "elasticnet",
            0.5,
            max_iter=20,
            tol=1e-4,
            device="cuda",
            X_val=X[32:],
            y_val=y[32:],
            return_path=True,
            solver_name="fista",
        )

    assert path is not None
    eta = X[32:] @ path["coef"].T + path["intercept"].reshape(1, -1)
    mu = np.clip(np.exp(np.clip(eta, -30.0, 30.0)), 1e-10, 1e6)
    manual = np.mean(mu - y[32:].reshape(-1, 1) * np.log(mu), axis=0)
    assert np.allclose(path["scores"], manual, rtol=1e-10, atol=1e-10)


def test_glm_sparse_cv_uses_device_scores_without_returning_coef_path(monkeypatch):
    import statgpu.linear_model._penalized_cv as cv_mod
    from statgpu.linear_model._cv_base import kfold_indices

    calls = []

    def fake_path(*args, **kwargs):
        calls.append(kwargs)
        return {"scores": np.array([0.3, 0.1]), "n_iter": np.array([2, 2])}

    monkeypatch.setattr(cv_mod, "_glm_sparse_cv_path", fake_path)

    rng = np.random.default_rng(33)
    X = rng.normal(scale=0.2, size=(30, 4))
    y = rng.poisson(lam=1.2, size=30).astype(float)
    cv = cv_mod.PenalizedGLM_CV(
        loss="poisson",
        penalty="elasticnet",
        alpha_grid=np.array([0.1, 0.01]),
        cv=3,
        device="cpu",
        solver="fista",
    )

    scores = cv._compute_cv_scores(
        X,
        y,
        np.array([0.1, 0.01]),
        "cpu",
        kfold_indices(X.shape[0], 3, random_state=0),
        strict=True,
    )

    assert scores.shape == (3, 2)
    assert np.allclose(scores, np.array([[0.3, 0.1]] * 3))
    assert calls
    assert all(call["return_path"] is False for call in calls)
    assert all(call["cv_mode"] is False for call in calls)


def test_fista_solver_accepts_private_cv_mode_on_cpu():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(24, 3))
    y = rng.poisson(np.exp(np.clip(X @ np.array([0.1, -0.2, 0.05]), -2, 2)))

    coef, n_iter = fista_solver(
        _resolve_loss_name("poisson"),
        L2Penalty(alpha=0.01),
        X,
        y.astype(float),
        max_iter=3,
        tol=1e-4,
        cv_mode=True,
    )

    assert coef.shape == (X.shape[1],)
    assert n_iter > 0


def test_irls_backend_uses_cv_warm_start_for_logistic(monkeypatch):
    import statgpu.glm_core._irls as irls_mod

    captured = {}

    def fake_fit(self, X_work, y_arr, **kwargs):
        captured["init_coef"] = np.asarray(kwargs["init_coef"], dtype=float)
        return np.zeros(X_work.shape[1]), 1

    monkeypatch.setattr(irls_mod.IRLSSolver, "fit", fake_fit)

    rng = np.random.default_rng(23)
    X = rng.normal(size=(24, 3))
    y = rng.integers(0, 2, size=24).astype(float)
    model = PenalizedGeneralizedLinearModel(
        loss="logistic",
        penalty="l2",
        alpha=0.1,
        solver="irls",
        device="cpu",
        compute_inference=False,
    )
    model._init_coef = np.array([0.2, -0.1, 0.05])
    model._init_intercept = -0.3

    model.fit(X, y)

    assert np.allclose(captured["init_coef"], [-0.3, 0.2, -0.1, 0.05])


def test_explicit_fista_is_not_promoted_to_fista_bb_for_logistic_l2(monkeypatch):
    import statgpu.glm_core._solver as solver_mod

    calls = []

    def fake_fista(loss, penalty, X_work, y_arr, **kwargs):
        calls.append("fista")
        return np.zeros(X_work.shape[1]), 1

    def fake_fista_bb(loss, penalty, X_work, y_arr, **kwargs):
        calls.append("fista_bb")
        return np.zeros(X_work.shape[1]), 1

    monkeypatch.setattr(solver_mod, "fista_solver", fake_fista)
    monkeypatch.setattr(solver_mod, "fista_bb_solver", fake_fista_bb)

    rng = np.random.default_rng(24)
    X = rng.normal(size=(24, 3))
    y = rng.integers(0, 2, size=24).astype(float)
    model = PenalizedGeneralizedLinearModel(
        loss="logistic",
        penalty="l2",
        alpha=0.05,
        solver="fista",
        device="cpu",
        compute_inference=False,
    )

    model.fit(X, y)

    assert calls == ["fista"]


@pytest.mark.parametrize("penalty_cls", [SCADPenalty, MCPPenalty])
def test_fista_lla_warm_start_is_target_only_for_nb_continuation(penalty_cls):
    class RecordingPenalty(penalty_cls):
        def __init__(self):
            super().__init__(alpha=0.08)
            self.seen = []

        def lla_weights(self, coef):
            self.seen.append(np.asarray(coef, dtype=np.float64).copy())
            return super().lla_weights(coef)

    rng = np.random.default_rng(25)
    X = rng.normal(scale=0.2, size=(20, 3))
    y = rng.poisson(lam=1.6, size=20).astype(float)
    penalty = RecordingPenalty()

    fista_lla_path(
        NegativeBinomialLoss(alpha=0.4),
        penalty,
        X,
        y,
        alpha_path=np.array([0.2, 0.08]),
        max_lla_per_step=1,
        max_iter=[1, 1],
        tol=1e-4,
        fit_intercept=True,
        init_coef=np.full(X.shape[1], 20.0),
        init_intercept=5.0,
    )

    assert len(penalty.seen) >= 2
    assert np.allclose(penalty.seen[0], 0.0)
    assert np.linalg.norm(penalty.seen[1]) > 10.0


def test_glm_scad_mcp_cv_disables_cross_alpha_warm_start(monkeypatch):
    from statgpu.linear_model import _penalized_cv as cv_mod
    from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

    seen_init = []

    def fake_fit(self, X, y, sample_weight=None):
        seen_init.append(getattr(self, "_init_coef", None))
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(PenalizedGeneralizedLinearModel, "fit", fake_fit)

    rng = np.random.default_rng(26)
    X = rng.normal(size=(24, 4))
    y = rng.poisson(lam=1.4, size=24).astype(float)
    model = cv_mod.PenalizedGLM_CV(
        loss="negative_binomial",
        penalty="scad",
        alpha_grid=np.array([0.1, 0.01]),
        cv=2,
        device="cpu",
    )
    model.fit(X, y)

    assert seen_init
    assert all(init is None for init in seen_init)


@pytest.mark.parametrize(
    "factory",
    [
        lambda groups: GroupMCPPenalty(alpha=0.2, gamma=3.0, groups=groups),
        lambda groups: GroupSCADPenalty(alpha=0.2, a=3.7, groups=groups),
    ],
)
@pytest.mark.parametrize(
    "groups",
    [
        [[2, 0], [3, 1]],
        [[2, 0], [3], [1, 4]],
    ],
)
def test_group_nonconvex_vectorized_proximal_matches_loop_on_numpy(factory, groups):
    penalty = factory(groups)
    w = np.array([0.8, -0.4, 1.1, -0.2, 0.6], dtype=float)
    step = 0.15

    loop = penalty._proximal_loop(w, step, np)
    vectorized = penalty._proximal_vectorized(w, step, np)

    assert np.allclose(vectorized, loop, rtol=1e-12, atol=1e-12)


def test_cupy_sqerr_fused_proximal_uses_y_current_as_center():
    cp = _skip_if_cupy_cuda_unavailable()

    y_current = cp.asarray([0.45, -1.25, 0.04, 2.5, -0.9], dtype=cp.float64)
    coef_old = cp.asarray([0.2, -1.0, 0.1, 2.1, -0.4], dtype=cp.float64)
    grad = cp.asarray([0.5, -0.25, 1.0, -0.4, 0.3], dtype=cp.float64)
    thresh = cp.asarray([0.04, 0.12, 0.02, 0.3, 0.15], dtype=cp.float64)
    step = 0.2
    beta = 0.35

    fused = _get_sqerr_proximal_cupy()
    coef_new, y_next = fused(y_current, grad, step, thresh, coef_old, beta)

    w = y_current - step * grad
    expected_coef = cp.sign(w) * cp.maximum(cp.abs(w) - thresh, 0.0)
    expected_y = expected_coef + beta * (expected_coef - coef_old)

    assert cp.allclose(coef_new, expected_coef)
    assert cp.allclose(y_next, expected_y)


@pytest.mark.parametrize("penalty", ["l1", "scad", "mcp"])
def test_cupy_squared_error_penalties_match_cpu_after_fused_lla_fix(penalty):
    cp = _skip_if_cupy_cuda_unavailable()

    rng = np.random.RandomState(42)
    n, p = 500, 20
    X_np = rng.randn(n, p)
    y_np = X_np @ np.linspace(2.0, 0.5, p) + rng.randn(n) * 0.5
    X_cu = cp.asarray(X_np)
    y_cu = cp.asarray(y_np)

    common = dict(
        loss="squared_error",
        penalty=penalty,
        alpha=0.1,
        compute_inference=False,
        max_iter=200,
    )
    model_cpu = PenalizedGeneralizedLinearModel(device="cpu", **common)
    model_cu = PenalizedGeneralizedLinearModel(device="cuda", **common)

    model_cpu.fit(X_np, y_np)
    model_cu.fit(X_cu, y_cu)

    coef_cpu = np.asarray(model_cpu.coef_)
    coef_cu_raw = model_cu.coef_
    coef_cu = (
        cp.asnumpy(coef_cu_raw)
        if isinstance(coef_cu_raw, cp.ndarray)
        else np.asarray(coef_cu_raw)
    )
    corr = np.corrcoef(coef_cpu, coef_cu)[0, 1]

    assert np.all(np.isfinite(coef_cu))
    assert corr > 0.999
    # CuPy fused kernel may take more iterations due to different numerical
    # path, but the final result is correct (corr > 0.999).
    assert abs(model_cpu.n_iter_ - model_cu.n_iter_) < 300


def test_resolve_gamma_loss_forwards_link_kwargs():
    loss = _resolve_loss_name("gamma", {"link": "inverse_power"})

    assert isinstance(loss, GammaLoss)
    assert loss.link_name == "inverse_power"


def test_adaptive_l1_external_weights_respect_normalize_false():
    weights = np.array([1.0, 2.0, 5.0])
    penalty = AdaptiveL1Penalty(alpha=0.1, weights=weights, normalize=False)

    assert np.allclose(penalty.lla_weights(np.zeros_like(weights)), weights)


def test_general_glm_fit_invokes_cleanup_hook(monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        device="cpu",
        gpu_memory_cleanup=True,
    )
    calls = []
    monkeypatch.setattr(model, "_cleanup_backend_memory", calls.append)

    model.fit(X, y)

    assert calls == ["numpy"]


def test_ordered_glm_rejects_sample_weight():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(24, 3))
    y = rng.integers(0, 3, size=24)
    sample_weight = np.linspace(0.5, 1.5, X.shape[0])
    model = OrderedGeneralizedLinearModel(device="cpu", max_iter=5)

    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(X, y, sample_weight=sample_weight)


def test_ordered_glm_fit_invokes_cleanup_hook(monkeypatch):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, size=30)
    model = OrderedGeneralizedLinearModel(device="cpu", max_iter=5)
    calls = []

    def fake_cleanup(backend_name):
        calls.append(backend_name)

    monkeypatch.setattr(model, "_cleanup_backend_memory", fake_cleanup)
    model.fit(X, y)

    assert calls == ["numpy"]


def test_torch_irls_promotes_mixed_float_dtype_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    rng = np.random.default_rng(15)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    beta = np.array([0.1, -0.2, 0.05, 0.15])
    y = rng.poisson(np.exp(X.astype(np.float64) @ beta)).astype(np.float64)

    glm = GeneralizedLinearModel(
        family="poisson",
        solver="irls",
        device="torch",
        fit_intercept=False,
        max_iter=5,
    )
    glm.fit(X, y)
    assert np.all(np.isfinite(glm.coef_))

    pglm = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l2",
        alpha=0.01,
        solver="irls",
        device="torch",
        fit_intercept=True,
        max_iter=5,
    )
    pglm.fit(X, y)
    assert np.all(np.isfinite(pglm.coef_))


def test_torch_fista_lla_squared_error_promotes_mixed_float_dtype_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    rng = np.random.default_rng(16)
    X_np = rng.normal(size=(36, 4)).astype(np.float32)
    y_np = rng.normal(size=36).astype(np.float64)
    X = torch.as_tensor(X_np, device="cuda")
    y = torch.as_tensor(y_np, device="cuda")

    coef, intercept, n_iter = fista_lla_path(
        SquaredErrorLoss(),
        SCADPenalty(alpha=0.05),
        X,
        y,
        alpha_path=np.array([0.08, 0.05]),
        max_lla_per_step=1,
        max_iter=[3, 3],
        tol=1e-4,
        fit_intercept=True,
    )

    assert coef.shape == (X_np.shape[1],)
    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert n_iter > 0


@pytest.mark.parametrize(
    "loss, loss_kwargs",
    [
        ("poisson", {}),
        ("gamma", {}),
        ("inverse_gaussian", {}),
        ("negative_binomial", {"alpha": 0.4}),
        ("tweedie", {"power": 1.3}),
    ],
)
def test_penalized_glm_predict_uses_mean_scale_for_positive_families(loss, loss_kwargs):
    X = np.array([[0.2, -0.1], [1.0, 0.5], [-0.4, 0.7]])
    coef = np.array([0.8, -0.35])
    intercept = 0.15
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        penalty="l2",
        fit_intercept=True,
        device="cpu",
        loss_kwargs=loss_kwargs,
    )
    model.coef_ = coef
    model.intercept_ = intercept

    pred = model.predict(X)
    expected = np.exp(X @ coef + intercept)

    assert np.all(pred > 0.0)
    assert np.allclose(pred, expected)


def test_general_glm_irls_l2_scaling_matches_penalized_glm():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 5))
    beta = rng.normal(size=5)
    y = X @ beta + 0.3 + rng.normal(scale=0.2, size=80)
    C = 0.7
    alpha = 1.0 / (2.0 * C)

    glm = GeneralizedLinearModel(
        family="gaussian",
        fit_intercept=True,
        C=C,
        solver="irls",
        device="cpu",
        max_iter=100,
        tol=1e-10,
    ).fit(X, y)
    penalized = PenalizedLinearRegression(
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        solver="irls",
        device="cpu",
        max_iter=100,
        tol=1e-10,
    ).fit(X, y)

    assert np.allclose(glm.coef_, penalized.coef_, rtol=1e-6, atol=1e-6)
    assert np.allclose(glm.intercept_, penalized.intercept_, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "penalty, l1_ratio, alpha, seed",
    [
        ("l1", 1.0, 0.03, 140),
        ("elasticnet", 0.5, 0.02, 245),
    ],
)
def test_poisson_sparse_penalty_matches_statsmodels_equivalent_alpha(
    penalty, l1_ratio, alpha, seed
):
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(seed)
    n, p = 45, 3
    X = rng.normal(scale=0.3, size=(n, p))
    beta = np.array([0.30, -0.10, 0.15])
    y = rng.poisson(np.exp(np.clip(0.12 + X @ beta, -2.0, 2.0))).astype(float)

    X_sm = sm.add_constant(X, has_constant="add")
    # statsmodels penalizes every parameter unless alpha is a vector.  The
    # leading zero keeps the intercept unpenalized, matching statgpu.
    alpha_sm = np.concatenate([[0.0], np.full(p, alpha)])
    sm_res = sm.GLM(y, X_sm, family=sm.families.Poisson()).fit_regularized(
        alpha=alpha_sm,
        L1_wt=l1_ratio,
        maxiter=500,
        cnvrg_tol=1e-8,
        zero_tol=1e-10,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        solver="fista",
        device="cpu",
        max_iter=3000,
        tol=1e-8,
        compute_inference=False,
    ).fit(X, y)

    statgpu_params = np.concatenate([[model.intercept_], np.asarray(model.coef_)])

    assert np.allclose(statgpu_params, sm_res.params, rtol=2e-3, atol=5e-4)


def test_fista_uniform_sample_weight_is_noop():
    """Uniform sample_weight should produce same result as unweighted."""
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)

    coef_unweighted, _ = fista_solver(loss, penalty, X, y, max_iter=80, tol=1e-10)
    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10,
        sample_weight=np.full(X.shape[0], 3.0),
    )

    assert np.allclose(coef_uniform, coef_unweighted)


def test_fista_uniform_sample_weight_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for torch")
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)
    sample_weight = torch.full((X.shape[0],), 3.0, device="cuda")

    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10, sample_weight=sample_weight
    )

    assert coef_uniform.shape == (X.shape[1],)


def test_fista_uniform_sample_weight_accepts_cupy_array():
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy unavailable: {exc}")
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)
    sample_weight = cp.full(X.shape[0], 3.0)

    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10, sample_weight=sample_weight
    )

    assert coef_uniform.shape == (X.shape[1],)


@pytest.mark.parametrize("solver", [newton_solver, admm_solver, lbfgs_solver])
def test_non_irls_solvers_reject_nonuniform_sample_weight(solver):
    rng = np.random.default_rng(15)
    X = rng.normal(size=(24, 3))
    y = rng.normal(size=24)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.1)

    with pytest.raises(ValueError, match="non-uniform sample_weight"):
        solver(loss, penalty, X, y, sample_weight=np.linspace(1.0, 2.0, X.shape[0]))


def test_fista_accepts_nonuniform_sample_weight():
    """FISTA solver should now accept non-uniform sample_weight."""
    rng = np.random.default_rng(15)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * rng.normal(size=50)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.1)
    w = np.linspace(0.5, 2.0, 50)

    coef, n_iter = fista_solver(
        loss, penalty, X, y, max_iter=200, tol=1e-6, sample_weight=w
    )
    assert coef.shape == (X.shape[1],)
    assert all(np.isfinite(coef))
    # Weighted coef should differ from unweighted
    coef_unw, _ = fista_solver(
        loss, penalty, X, y, max_iter=200, tol=1e-6
    )
    assert not np.allclose(coef, coef_unw, atol=1e-3), "Weighted and unweighted coefs should differ"


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: SCADPenalty(**kw),
        lambda **kw: MCPPenalty(**kw),
        lambda **kw: GroupSCADPenalty(groups=[[0, 1]], **kw),
        lambda **kw: GroupMCPPenalty(groups=[[0, 1]], **kw),
    ],
)
def test_scad_mcp_penalties_validate_parameters(factory):
    with pytest.raises(ValueError):
        factory(alpha=0.0)
    if "SCAD" in factory().__class__.__name__:
        with pytest.raises(ValueError):
            factory(a=2.0)
    else:
        with pytest.raises(ValueError):
            factory(gamma=1.0)


def _configured_pglm(loss, penalty, device="auto", alpha=1.0):
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        solver="auto",
        device=device,
    )
    model._loss = model._resolve_loss()
    model._penalty = model._resolve_penalty()
    return model


def test_auto_backend_routing_prefers_torch_for_large_nb_l2(monkeypatch):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm("negative_binomial", "l2")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "torch"


def test_auto_backend_routing_uses_cpu_for_large_sparse_gaussian():
    model = _configured_pglm("squared_error", "l1")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_auto_backend_routing_uses_cpu_for_large_gaussian_exact(alpha):
    model = _configured_pglm("squared_error", "l2", alpha=alpha)
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


@pytest.mark.parametrize(
    "loss,penalty",
    [
        ("logistic", "l1"),
        ("logistic", "elasticnet"),
        ("gamma", "l2"),
        ("tweedie", "l1"),
        ("tweedie", "elasticnet"),
    ],
)
def test_auto_backend_routing_uses_cpu_for_large_guarded_slow_paths(monkeypatch, loss, penalty):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm(loss, penalty)
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


def test_auto_backend_routing_does_not_change_explicit_cuda(monkeypatch):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm("negative_binomial", "l2", device="cuda")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "cupy"


def test_predict_uses_selected_backend_after_auto_routing():
    model = _configured_pglm("poisson", "l2")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    model._selected_backend_name = "numpy"

    pred = model.predict(np.ones((3, 2)))

    assert isinstance(pred, np.ndarray)
    assert np.allclose(pred, np.exp(np.ones((3, 2)) @ model.coef_ + model.intercept_))


def test_predict_auto_falls_back_to_numpy_when_selected_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="auto")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    model._selected_backend_name = "cupy"
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_cupy_available",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: False),
    )

    pred = model.predict(np.ones((3, 2)))

    assert isinstance(pred, np.ndarray)
    assert np.allclose(pred, np.exp(np.ones((3, 2)) @ model.coef_ + model.intercept_))


def test_predict_explicit_cuda_raises_when_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="cuda")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_cupy_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="device='cuda'"):
        model.predict(np.ones((3, 2)))


def test_predict_explicit_torch_raises_when_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="torch")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="device='torch'"):
        model.predict(np.ones((3, 2)))
