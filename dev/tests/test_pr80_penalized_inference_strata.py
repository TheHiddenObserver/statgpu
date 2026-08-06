"""Fixed-penalty Cox inference and shared public strata validation."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival._risk_sets import cox_counting_process_objective


def _sample(seed=12831, n=72, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.4, -0.2, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=2.0, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[: max(6, p + 2)] = 1.0
    return X, stop, event


def _backend_inputs(backend_name, *values):
    if backend_name == "numpy":
        return "cpu", tuple(np.asarray(value) for value in values)
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA device is unavailable: {exc}")
        return "cuda", tuple(cp.asarray(value) for value in values)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")
    converted = []
    for value in values:
        array = np.asarray(value)
        dtype = torch.float64 if array.dtype.kind == "f" else torch.int64
        converted.append(torch.as_tensor(array, dtype=dtype, device="cuda"))
    return "torch", tuple(converted)


def _to_numpy(backend_name, value):
    if backend_name == "numpy":
        return np.asarray(value)
    if backend_name == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return value.detach().cpu().numpy()


def _eventless_stratum_sample(seed=12840):
    rng = np.random.default_rng(seed)
    n_event_stratum = 48
    n_eventless_stratum = 16
    X = rng.normal(size=(n_event_stratum + n_eventless_stratum, 1))
    beta = np.array([0.35])
    failure = rng.exponential(
        scale=np.exp(-(X[:n_event_stratum] @ beta))
    ) + 0.05
    censor = rng.exponential(scale=2.0, size=n_event_stratum) + 0.05
    stop = np.empty(X.shape[0], dtype=np.float64)
    stop[:n_event_stratum] = np.minimum(failure, censor)
    stop[n_event_stratum:] = rng.uniform(
        0.1, 3.0, size=n_eventless_stratum
    )
    event = np.zeros(X.shape[0], dtype=np.float64)
    event[:n_event_stratum] = failure <= censor
    event[:16] = 1.0
    strata = np.zeros(X.shape[0], dtype=np.int64)
    strata[n_event_stratum:] = 1
    return X, stop, event, strata, n_event_stratum


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("ties", ["breslow", "efron", "exact"])
def test_penalized_nonrobust_uses_fixed_penalty_sandwich(backend_name, ties):
    seed = {
        "breslow": 12832,
        "efron": 12833,
        "exact": 12839,
    }[ties]
    X, stop, event = _sample(seed=seed)
    device, (Xb, stopb, eventb) = _backend_inputs(backend_name, X, stop, event)
    penalty = 0.4
    model = CoxPH(
        ties=ties,
        penalty=penalty,
        device=device,
        cov_type="nonrobust",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(Xb, stopb, eventb)

    objective = cox_counting_process_objective(model.coef_, X, stop, event, ties=ties)
    meat = np.asarray(objective["information"], dtype=np.float64)
    derivative = meat + 2.0 * penalty * np.eye(X.shape[1])
    bread = np.linalg.inv(derivative)
    expected = bread @ meat @ bread

    np.testing.assert_allclose(model._var_matrix, expected, rtol=2e-8, atol=2e-10)
    assert not np.allclose(model._var_matrix, bread, rtol=1e-7, atol=1e-10)
    assert model.inference_method_ == "m_estimation"
    assert model._inference_result.method == "m_estimation"
    assert model.inference_target_ == "penalized_estimating_equation"
    assert model.penalty_conditioning_ == "fixed_penalty"
    assert model.penalty_selection_adjusted_ is False
    metadata = model._inference_result.metadata
    assert metadata["bread_information"] == ("observed_information_plus_l2_curvature")
    assert metadata["meat_information"] == "unpenalized_observed_information"
    assert metadata["penalty_selection_adjusted"] is False
    assert metadata["meat_type"] == "nonrobust"
    assert metadata["covariance_convention"] == ("fixed_penalty_model_based_sandwich")

    assert metadata["score_test_contract"] == "suppressed_penalized_fit"
    assert metadata["likelihood_ratio_test_contract"] == "suppressed_penalized_fit"
    assert model.score_test_available_ is False
    assert model.score_test_failure_reason_ == (
        "classical score test is suppressed for penalized fit"
    )


def test_positive_l2_robust_uses_shared_m_estimation_result_name():
    X, stop, event = _sample(seed=12839, n=64, p=2)
    model = CoxPH(
        device="cpu",
        penalty=0.25,
        cov_type="hc0",
        compute_inference=True,
        compute_cindex=False,
    ).fit(X, stop, event)

    assert model.inference_method_ == "m_estimation"
    assert model._inference_result.method == "m_estimation"
    assert model._inference_result.metadata["meat_type"] == "hc0"
    assert model._inference_result.metadata["covariance_convention"] == (
        "fixed_penalty_robust_sandwich"
    )


def test_unpenalized_nonrobust_covariance_remains_inverse_information():
    X, stop, event = _sample(seed=12834)
    model = CoxPH(
        device="cpu",
        penalty=0.0,
        compute_inference=True,
        compute_cindex=False,
    ).fit(X, stop, event)
    objective = cox_counting_process_objective(
        model.coef_, X, stop, event, ties="breslow"
    )
    expected = np.linalg.inv(np.asarray(objective["information"]))
    np.testing.assert_allclose(model._var_matrix, expected, rtol=2e-10, atol=2e-12)
    assert model.inference_method_ == "observed_information"
    assert model.inference_target_ == "partial_likelihood_parameter"
    assert model.penalty_selection_adjusted_ is None


def test_coxphcv_copies_fixed_penalty_inference_provenance():
    X, stop, event = _sample(seed=12835, n=60, p=2)
    model = CoxPHCV(
        penalties=[0.25],
        cv=2,
        random_state=17,
        device="cpu",
        compute_inference=True,
        max_iter=80,
    ).fit(X, stop, event)
    assert model.penalty_ == pytest.approx(0.25)
    assert model.inference_method_ == "m_estimation"
    assert model.inference_target_ == "penalized_estimating_equation"
    assert model.penalty_conditioning_ == "fixed_penalty"
    assert model.penalty_selection_adjusted_ is False
    assert model._inference_result.metadata["penalty_selection_adjusted"] is False


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_score_and_survival_share_strata_shape_and_label_contract(backend_name):
    X, stop, event = _sample(seed=12836, n=48, p=2)
    strata = np.arange(X.shape[0], dtype=np.int64) % 2
    device, (Xb, stopb, eventb, stratab) = _backend_inputs(
        backend_name, X, stop, event, strata
    )
    model = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(Xb, stopb, eventb, strata=stratab)

    score = model.score(Xb, stopb, eventb, strata=stratab)
    assert np.isfinite(score)
    survival, returned_times = model.predict_survival(Xb[:4], strata=stratab[:4])
    assert tuple(survival.shape) == (4, int(returned_times.shape[0]))

    malformed = (
        stratab[0],
        stratab.reshape(-1, 1),
        stratab[:-1],
    )
    for value in malformed:
        with pytest.raises(ValueError, match=r"strata must have shape \(n_samples,\)"):
            model.score(Xb, stopb, eventb, strata=value)

    with pytest.raises(ValueError, match=r"strata must have shape \(n_samples,\)"):
        model.predict_survival(Xb[:4], strata=stratab[:4].reshape(-1, 1))

    unknown = stratab + 10
    with pytest.raises(ValueError, match="unknown scoring stratum"):
        model.score(Xb, stopb, eventb, strata=unknown)
    with pytest.raises(ValueError, match="unknown prediction stratum"):
        model.predict_survival(Xb[:4], strata=unknown[:4])


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_eventless_stratum_survival_is_one_for_all_time_modes(backend_name):
    X, stop, event, strata, split = _eventless_stratum_sample()
    device, (Xb, stopb, eventb, stratab) = _backend_inputs(
        backend_name, X, stop, event, strata
    )
    model = CoxPH(
        ties="efron",
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(Xb, stopb, eventb, strata=stratab)

    empty_baseline = model._baseline_by_stratum[1]
    assert empty_baseline["time"].shape == (0,)
    assert empty_baseline["cumulative_hazard"].shape == (0,)

    explicit_times = np.array(
        [0.0, np.median(stop[event == 1.0]), np.max(stop[event == 1.0]) + 1.0]
    )
    eventless_survival, returned_times = model.predict_survival(
        Xb[split : split + 3],
        times=explicit_times,
        strata=stratab[split : split + 3],
    )
    eventless_np = _to_numpy(backend_name, eventless_survival)
    assert eventless_np.shape == (3, explicit_times.size)
    assert np.all(np.isfinite(eventless_np))
    np.testing.assert_allclose(eventless_np, 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        _to_numpy(backend_name, returned_times), explicit_times
    )

    automatic_survival, automatic_times = model.predict_survival(
        Xb[split : split + 3],
        strata=stratab[split : split + 3],
    )
    automatic_np = _to_numpy(backend_name, automatic_survival)
    assert automatic_np.shape == (
        3,
        int(automatic_times.shape[0]),
    )
    assert automatic_np.shape[1] > 0
    assert np.all(np.isfinite(automatic_np))
    np.testing.assert_allclose(automatic_np, 1.0, rtol=0.0, atol=0.0)

    mixed_indices = np.array([0, split])
    _, (mixed_X, mixed_strata) = _backend_inputs(
        backend_name, X[mixed_indices], strata[mixed_indices]
    )
    mixed_survival, mixed_times = model.predict_survival(
        mixed_X, times=explicit_times, strata=mixed_strata
    )
    mixed_np = _to_numpy(backend_name, mixed_survival)
    assert mixed_np.shape == (2, int(mixed_times.shape[0]))
    assert np.all(np.isfinite(mixed_np))
    assert np.any(mixed_np[0] < 1.0)
    np.testing.assert_allclose(mixed_np[1], 1.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_coxphcv_delegates_eventless_stratum_survival(backend_name):
    X, stop, event, strata, split = _eventless_stratum_sample(seed=12841)
    device, (Xb, stopb, eventb, stratab) = _backend_inputs(
        backend_name, X, stop, event, strata
    )
    model = CoxPHCV(
        penalties=[0.2],
        cv=2,
        random_state=23,
        ties="efron",
        device=device,
        compute_inference=True,
        max_iter=100,
        tol=1e-8,
    ).fit(Xb, stopb, eventb, strata=stratab)

    assert model.estimator_._baseline_by_stratum[1]["time"].shape == (0,)
    times = np.array([0.0, 0.5, np.max(stop[event == 1.0]) + 1.0])
    survival, returned_times = model.predict_survival(
        Xb[split : split + 2],
        times=times,
        strata=stratab[split : split + 2],
    )
    survival_np = _to_numpy(backend_name, survival)
    assert survival_np.shape == (2, times.size)
    assert np.all(np.isfinite(survival_np))
    np.testing.assert_allclose(survival_np, 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(_to_numpy(backend_name, returned_times), times)


def test_eventless_stratum_still_rejects_mismatched_baseline_shapes():
    X, stop, event, strata, split = _eventless_stratum_sample(seed=12842)
    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(X, stop, event, strata=strata)
    model._baseline_by_stratum[1]["cumulative_hazard"] = np.array([0.0])

    with pytest.raises(
        RuntimeError, match="Stored baseline hazard state is inconsistent"
    ):
        model.predict_survival(
            X[split : split + 1],
            times=[0.0, 1.0],
            strata=strata[split : split + 1],
        )


def test_scalar_string_scoring_strata_has_public_shape_error():
    X, stop, event = _sample(seed=12837, n=36, p=2)
    labels = np.where(np.arange(X.shape[0]) % 2, "south", "north")
    model = CoxPH(device="cpu", compute_inference=False, compute_cindex=False).fit(
        X, stop, event, strata=labels
    )
    with pytest.raises(ValueError, match=r"strata must have shape \(n_samples,\)"):
        model.score(X, stop, event, strata="north")


def test_penalized_summary_states_fixed_penalty_limitations(capsys):
    X, stop, event = _sample(seed=12838, n=54, p=2)
    model = CoxPH(
        device="cpu",
        penalty=0.3,
        compute_inference=True,
        compute_cindex=False,
    ).fit(X, stop, event)
    model.summary()
    output = capsys.readouterr().out
    assert "fixed-penalty frequentist estimating-equation sandwich" in output
    assert "CV selection and shrinkage bias are not included" in output
    assert "Penalized estimating-equation Wald test" in output
    assert "Classical LR/Score/AIC/BIC diagnostics suppressed" in output
