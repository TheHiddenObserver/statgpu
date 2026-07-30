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
