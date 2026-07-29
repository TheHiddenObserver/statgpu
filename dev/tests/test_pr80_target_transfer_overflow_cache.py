"""Regression coverage for the final PR80 provenance/numerical boundaries."""

from __future__ import annotations

import numpy as np
import pytest

import statgpu
from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.survival import CoxFitNumericalError, CoxPH
from statgpu.survival import _cox_counting as cox_counting
from statgpu.survival import _cox_cv as cox_cv
from statgpu.survival import _numeric as survival_numeric
from statgpu.survival._cox_counting import (
    fit_counting_process_cox,
    prepare_right_censored_cox_fast_path,
)
from statgpu.survival._cox_cv import (
    _COXPH_CV_CACHE,
    _select_coxph_penalty_cv,
)


def _sample(seed=9081, n=42, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.35, -0.15, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=2.0, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[:6] = 1.0
    return X, stop, event


def test_public_numerical_error_has_one_consistent_export():
    assert statgpu.CoxFitNumericalError is CoxFitNumericalError
    assert "CoxFitNumericalError" in statgpu.__all__
    assert "CoxFitNumericalError" in statgpu.survival.__all__
    assert issubclass(CoxFitNumericalError, FloatingPointError)
    assert not hasattr(statgpu, "CoxCandidateNumericalError")


@pytest.mark.parametrize("value", [800.0, -800.0])
def test_safe_exp_rejects_unrepresentable_numpy_log_risk(value):
    with pytest.raises(FloatingPointError, match="finite positive float64"):
        survival_numeric._safe_exp_linear_predictor(
            np.array([value], dtype=np.float64)
        )


def test_safe_exp_accepts_representable_numpy_boundaries():
    values = np.array(
        [
            np.nextafter(
                survival_numeric._LOG_FLOAT64_MIN_POSITIVE, np.inf
            ),
            np.nextafter(survival_numeric._LOG_FLOAT64_MAX, -np.inf),
        ],
        dtype=np.float64,
    )
    result = survival_numeric._safe_exp_linear_predictor(values)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0.0)


def test_safe_exp_torch_branch_matches_numpy_contract():
    torch = pytest.importorskip("torch")
    finite = torch.tensor([0.0, 10.0], dtype=torch.float32)
    actual = survival_numeric._safe_exp_linear_predictor(finite)
    assert actual.dtype == torch.float64
    assert torch.allclose(actual, torch.exp(finite.to(dtype=torch.float64)))
    for value in (800.0, -800.0):
        with pytest.raises(FloatingPointError, match="finite positive float64"):
            survival_numeric._safe_exp_linear_predictor(
                torch.tensor([value], dtype=torch.float64)
            )


def test_torch_cpu_preparation_is_not_reported_as_device_to_host():
    torch = pytest.importorskip("torch")
    X, stop, event = _sample(n=18)
    prepared = prepare_right_censored_cox_fast_path(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(stop, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.float64),
        ties="breslow",
    )
    assert prepared.backend == "torch"
    assert prepared.full_target_host_transfer_performed is False


def test_group_encoding_can_skip_unused_label_materialization():
    codes, labels = CoxPH._encode_group_labels(
        np.array(["b", "a", "b"], dtype=object),
        3,
        "cluster",
        return_labels=False,
    )
    assert np.array_equal(codes, np.array([1, 0, 1], dtype=np.int64))
    assert labels is None


def test_safe_exp_cupy_branch_matches_numpy_contract():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA unavailable")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA unavailable: {exc}")
    finite = cupy.asarray([0.0, 10.0], dtype=cupy.float32)
    actual = survival_numeric._safe_exp_linear_predictor(finite)
    assert actual.dtype == cupy.float64
    assert bool(
        cupy.allclose(actual, cupy.exp(finite.astype(cupy.float64))).item()
    )
    for value in (800.0, -800.0):
        with pytest.raises(FloatingPointError, match="finite positive float64"):
            survival_numeric._safe_exp_linear_predictor(
                cupy.asarray([value], dtype=cupy.float64)
            )


def test_cox_public_hazard_ratio_raises_but_log_risk_remains_available():
    X, stop, event = _sample(p=1)
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, stop, event)
    model.coef_ = np.array([800.0])
    with pytest.raises(FloatingPointError, match="finite positive float64"):
        model.predict_hazard_ratio(np.ones((2, 1)))
    with pytest.raises(ValueError, match="real-valued"):
        model.predict_risk_score(
            np.ones((2, 1), dtype=np.complex128) + 1j
        )
    assert np.array_equal(
        model.predict_risk_score(np.ones((2, 1))), np.array([800.0, 800.0])
    )


def test_ordinary_survival_keeps_centered_log_baseline_for_extreme_risk():
    X, stop, event = _sample(p=1)
    model = CoxPH(
        device="cpu", compute_inference=True, compute_cindex=False
    ).fit(X, stop, event)
    assert model._baseline_by_stratum is None
    assert model._baseline_log_cumulative_hazard_centered is not None
    assert model._baseline_x_reference is not None
    model.coef_ = np.array([800.0])
    with np.errstate(over="raise"):
        survival, prediction_times = model.predict_survival(
            np.ones((2, 1))
        )
    assert prediction_times.size > 0
    assert np.all(np.isfinite(survival))
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    model._baseline_log_cumulative_hazard_centered = None
    model._baseline_x_reference = None
    with np.errstate(over="raise", divide="raise"):
        fallback_survival, _ = model.predict_survival(np.ones((2, 1)))
    assert np.all(np.isfinite(fallback_survival))
    assert np.all(
        (fallback_survival >= 0.0) & (fallback_survival <= 1.0)
    )


def test_penalized_cox_uses_same_strict_hazard_ratio_boundary():
    model = PenalizedCoxPHModel(device="cpu")
    model.coef_ = np.array([800.0])
    model._design_info = None
    model._selected_backend_name = "numpy"
    with pytest.raises(FloatingPointError, match="finite positive float64"):
        model.predict_hazard_ratio(np.ones((2, 1)))
    with pytest.raises(ValueError, match="real-valued"):
        model.predict_risk_score(
            np.ones((2, 1), dtype=np.complex128) + 1j
        )
    assert np.array_equal(
        model.predict_risk_score(np.ones((2, 1))), np.array([800.0, 800.0])
    )


def test_fit_hazard_ratio_overflow_uses_public_numerical_error(monkeypatch):
    def overflow_result(*args, **kwargs):
        return {"coef": np.array([800.0], dtype=np.float64)}

    monkeypatch.setattr(
        cox_counting, "fit_counting_process_cox", overflow_result
    )
    X, stop, event = _sample(p=1)
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    )
    with pytest.raises(CoxFitNumericalError, match="finite positive float64"):
        model.fit(X, stop, event)
    assert model.coef_ is None
    assert model.hazard_ratios_ is None
    assert model._fitted is False


def test_summary_rejects_unrepresentable_inverse_hazard_ratio():
    X, stop, event = _sample(p=1)
    model = CoxPH(
        device="cpu", compute_inference=True, compute_cindex=False
    ).fit(X, stop, event)
    model.coef_ = np.array([-720.0])
    model.hazard_ratios_ = survival_numeric._safe_exp_linear_predictor(
        model.coef_
    )
    with pytest.raises(FloatingPointError, match="finite positive float64"):
        model.summary()


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cv_reuses_one_right_censored_preparation_per_fold(monkeypatch, ties):
    from statgpu.losses import _cox_ph as cox_loss_module

    X, stop, event = _sample(n=36)
    calls = 0
    real_to_numpy = cox_loss_module._to_numpy

    def recording_to_numpy(value):
        nonlocal calls
        calls += 1
        return real_to_numpy(value)

    monkeypatch.setattr(cox_loss_module, "_to_numpy", recording_to_numpy)
    key = f"right-censored-fold-preparation-reuse-{ties}"
    _COXPH_CV_CACHE.pop(key, None)
    _, details = _select_coxph_penalty_cv(
        X,
        stop,
        event,
        penalties=np.array([0.2, 0.05, 0.01]),
        cv_folds=3,
        random_state=7,
        ties=ties,
        device="cpu",
        max_iter=80,
        tol=1e-7,
        return_details=True,
        cache_key=key,
    )

    # Each fold copies sorted time and event exactly once. A per-candidate loss
    # would make this 2 * folds * penalties instead of 2 * folds.
    assert calls == 6
    assert details["candidate_right_censored_preparation_count"] == 3
    assert details["candidate_target_host_transfer_count"] == 0
    assert details["candidate_target_host_vector_transfer_count"] == 0
    assert details["selection_cache_hit"] is False


@pytest.mark.parametrize(
    "cache_limit, cache_expected",
    [(str(1 << 30), True), ("0", False)],
)
def test_staged_cv_fold_state_cache_is_workspace_bounded(
    monkeypatch, cache_limit, cache_expected
):
    from statgpu.losses import _cox_ph as cox_loss_module

    class ConvergedCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            return self

    X, stop, event = _sample(n=42)
    loss_transfers = 0
    fold_preparations = 0
    real_to_numpy = cox_loss_module._to_numpy
    real_fold_prepare = cox_cv._prepare_cox_cv_fold_backend
    numpy_backend = cox_cv._cv_backend_for_device("cpu")

    def recording_to_numpy(value):
        nonlocal loss_transfers
        loss_transfers += 1
        return real_to_numpy(value)

    def recording_fold_prepare(*args, **kwargs):
        nonlocal fold_preparations
        fold_preparations += 1
        return real_fold_prepare(*args, **kwargs)

    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")
    monkeypatch.setenv(
        "STATGPU_COXPHCV_FOLD_CACHE_MAX_BYTES", cache_limit
    )
    monkeypatch.setattr(cox_cv, "CoxPH", ConvergedCoxPH)
    monkeypatch.setattr(
        cox_cv, "_cv_backend_for_device", lambda device: numpy_backend
    )
    monkeypatch.setattr(
        cox_cv, "_prepare_cox_cv_fold_backend", recording_fold_prepare
    )
    monkeypatch.setattr(cox_loss_module, "_to_numpy", recording_to_numpy)
    key = f"staged-fold-state-reuse-{cache_limit}"
    _COXPH_CV_CACHE.pop(key, None)

    _, details = _select_coxph_penalty_cv(
        X,
        stop,
        event,
        penalties=np.geomspace(1.0, 0.01, 8),
        cv_folds=3,
        random_state=3,
        device="cuda",
        return_details=True,
        cache_key=key,
    )

    assert details["fold_state_cache_enabled"] is cache_expected
    assert details["fold_state_cache_enabled_this_call"] is cache_expected
    assert details["fold_state_cache_limit_bytes"] == int(cache_limit)
    if cache_expected:
        assert fold_preparations == 3
        assert loss_transfers == 6
        assert details["fold_backend_preparation_count"] == 3
        assert details["candidate_right_censored_preparation_count"] == 3
    else:
        assert fold_preparations > 3
        assert loss_transfers > 6
        assert details["fold_backend_preparation_count"] > 3
        assert details["candidate_right_censored_preparation_count"] > 3


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_reused_right_censored_state_matches_fresh_solver(ties):
    X, stop, event = _sample(n=36)
    prepared = prepare_right_censored_cox_fast_path(
        X, stop, event, ties=ties
    )
    common = dict(
        ties=ties,
        penalty=0.05,
        max_iter=80,
        tol=1e-8,
        compute_baseline=False,
        compute_score_residuals=False,
        right_censored_fast_path=True,
    )
    fresh = fit_counting_process_cox(X, stop, event, **common)
    reused = fit_counting_process_cox(
        X,
        stop,
        event,
        right_censored_prepared=prepared,
        **common,
    )
    assert np.allclose(reused["coef"], fresh["coef"], rtol=1e-10, atol=1e-11)
    assert reused["log_likelihood"] == pytest.approx(
        fresh["log_likelihood"], rel=1e-11, abs=1e-11
    )


def test_public_fit_rejects_prepared_state_from_other_array_identity():
    X, stop, event = _sample(n=30)
    prepared = prepare_right_censored_cox_fast_path(
        X, stop, event, ties="breslow"
    )
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    )
    with pytest.raises(ValueError, match="does not match"):
        model.fit(
            X.copy(),
            stop,
            event,
            _right_censored_prepared=prepared,
        )


def test_public_dispatch_does_not_repeat_solver_input_normalization(monkeypatch):
    X, stop, event = _sample(n=30)

    def unexpected_solver_normalization(*args, **kwargs):
        raise AssertionError("solver repeated public input normalization")

    monkeypatch.setattr(
        cox_counting,
        "prepare_counting_process_inputs",
        unexpected_solver_normalization,
    )
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    ).fit(X, stop, event)
    assert np.all(np.isfinite(model.coef_))


def test_cache_hit_separates_origin_from_current_invocation(monkeypatch):
    class CountingCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            return self

    monkeypatch.setattr(cox_cv, "CoxPH", CountingCoxPH)
    X, stop, event = _sample(n=30)
    key = "cache-origin-versus-invocation"
    _COXPH_CV_CACHE.pop(key, None)
    common = dict(
        penalties=np.array([0.1]),
        cv_folds=3,
        random_state=2,
        return_details=True,
        cache_key=key,
    )
    _, first = _select_coxph_penalty_cv(
        X, stop, event, device="cpu", **common
    )
    _, second = _select_coxph_penalty_cv(
        X, stop, event, device="cuda", **common
    )

    assert first["selection_cache_hit"] is False
    assert first["selection_origin_device"] == "cpu"
    assert first["requested_fit_device"] == "cpu"
    assert first["candidate_preparation_origin_device"] == "cpu"
    assert first["fold_backend_preparation_count_this_call"] == 3
    assert second["selection_cache_hit"] is True
    assert second["selection_origin_device"] == "cpu"
    assert second["requested_fit_device"] == "cuda"
    assert second["candidate_preparation_origin_device"] == "cpu"
    assert second["effective_device"] == "cuda"
    assert second["scoring_device"] == "cpu"
    assert second["fold_backend_preparation_count"] == 3
    assert second["fold_backend_preparation_count_this_call"] == 0
    assert second["fold_state_cache_enabled_this_call"] is False
    assert second["candidate_right_censored_preparation_count_this_call"] == 0
    assert second["candidate_target_host_transfer_count_this_call"] == 0
    assert second["candidate_target_host_vector_transfer_count_this_call"] == 0


def test_successful_public_fit_resets_state_once(monkeypatch):
    X, stop, event = _sample()
    model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False
    )
    calls = 0
    real_reset = model._reset_fit_state

    def recording_reset():
        nonlocal calls
        calls += 1
        real_reset()

    monkeypatch.setattr(model, "_reset_fit_state", recording_reset)
    model.fit(X, stop, event)
    assert calls == 1
    assert isinstance(model._objective_history, np.ndarray)
    assert model._objective_history.size >= 1
