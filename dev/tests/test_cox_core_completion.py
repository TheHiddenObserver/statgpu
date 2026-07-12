"""Focused regression tests for the completed CoxPH core contracts."""

import numpy as np
import pytest

from statgpu.survival import CoxPH


def _survival_data(n=260, p=4, seed=2701, tied=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.45, -0.25, p)
    event_time = rng.exponential(scale=np.exp(-(X @ beta)))
    censor_time = rng.exponential(scale=1.8, size=n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int32)
    if tied:
        # Deliberately create dense tied failure groups while retaining the
        # covariate-dependent event-time ordering.
        time = np.maximum(1.0, np.ceil(5.0 * time)).astype(np.float64)
    return X.astype(np.float64), time.astype(np.float64), event


def _require_backend(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:  # pragma: no cover - host-specific driver error
            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")
        return

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")


def test_refit_resets_convergence_and_inference_state():
    X, time, event = _survival_data(seed=2702)
    model = CoxPH(
        device="cpu", compute_inference=True, compute_cindex=False, max_iter=60
    ).fit(X, time, event)
    assert model._baseline_cumulative_hazard is not None

    # Make stale state unmistakable, then perform a zero-iteration,
    # estimation-only refit on the same object.
    model._converged = True
    model.max_iter = 0
    model.compute_inference = False
    model.fit(X[:120], time[:120], event[:120])

    assert model._fitted
    assert model._iterations == 0
    assert model._converged is False
    assert model._bse is None
    assert model._var_matrix is None
    assert model._baseline_cumulative_hazard is None
    assert model._nobs == 120


def test_failed_refit_clears_partially_computed_cox_state():
    X, time, event = _survival_data(n=90, p=2, seed=2711)
    model = CoxPH(device="cpu", compute_inference=True, compute_cindex=False).fit(
        X, time, event
    )
    assert model.coef_ is not None

    singular_X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    singular_time = np.ones(4)
    singular_event = np.ones(4, dtype=np.int64)
    model.set_params(ties="exact")
    with pytest.raises(RuntimeError, match="information is singular"):
        model.fit(singular_X, singular_time, singular_event)
    assert model._fitted is False
    assert model.coef_ is None
    assert model.hazard_ratios_ is None
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(singular_X)


def test_efron_heavy_ties_bse_matches_statsmodels():
    smd = pytest.importorskip("statsmodels.duration.api")
    X, time, event = _survival_data(n=520, p=4, seed=2703, tied=True)

    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(X, time, event)
    reference = smd.PHReg(time, X, status=event, ties="efron").fit(disp=0)

    assert np.all(np.isfinite(model._bse))
    assert np.all(model._bse > 0)
    np.testing.assert_allclose(model.coef_, reference.params, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(model._bse, reference.bse, rtol=5e-2, atol=3e-3)


def test_torch_efron_private_path_is_exact_and_native(monkeypatch):
    torch = pytest.importorskip("torch")
    X, time, event = _survival_data(n=180, p=4, seed=2704, tied=True)
    order = np.argsort(time, kind="stable")
    X, time, event = X[order], time[order], event[order]

    model = CoxPH(ties="efron", device="cpu", compute_inference=False)
    efron_pre = model._efron_unique_failure_indices(time, event)
    model._efron_pre = efron_pre
    model._efron_all_singletons = False
    monkeypatch.delenv("STATGPU_EFRON_TRITON", raising=False)

    beta = np.linspace(-0.12, 0.15, X.shape[1])
    grad_np, hess_np = model._compute_gradient_hessian(beta, X, time, event, efron_pre)
    grad_t, hess_t = model._compute_gradient_hessian_torch(
        torch.as_tensor(beta, dtype=torch.float64),
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(time, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.int32),
        efron_pre,
    )

    assert grad_t.device.type == "cpu"
    np.testing.assert_allclose(grad_t.numpy(), grad_np, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        model._observed_information_torch(hess_t).numpy(),
        model._observed_information(hess_np),
        rtol=2e-11,
        atol=2e-11,
    )


def test_torch_breslow_hessian_uses_sample_dimension():
    """Guard the Torch outer-product reshape against the former undefined ``n``."""
    torch = pytest.importorskip("torch")
    X, time, event = _survival_data(n=90, p=3, seed=2710)
    order = np.argsort(time, kind="stable")
    X, time, event = X[order], time[order], event[order]
    beta = np.array([0.08, -0.04, 0.11])

    model = CoxPH(ties="breslow", device="cpu", compute_inference=False)
    grad_np, hess_np = model._compute_gradient_hessian(beta, X, time, event)
    grad_t, hess_t = model._compute_gradient_hessian_torch(
        torch.as_tensor(beta, dtype=torch.float64),
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(time, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.int32),
    )

    assert hess_t.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(grad_t.numpy(), grad_np, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(hess_t.numpy(), hess_np, rtol=2e-11, atol=2e-11)


def test_torch_fit_core_matches_cpu_and_keeps_full_covariance():
    """Exercise the complete Torch implementation locally on Torch CPU.

    Public ``device='torch'`` correctly requires CUDA, but the private core is
    device-generic and can therefore provide non-skipped local regression
    coverage for exact Efron optimization and inference.
    """
    torch = pytest.importorskip("torch")
    X, time, event = _survival_data(n=220, p=4, seed=2709, tied=True)
    reference = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(X, time, event)

    model = CoxPH(
        ties="efron",
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    )
    model._reset_fit_state()
    model._nobs = X.shape[0]
    model._nevents = int(event.sum())
    model._feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
    model._X = X.copy()
    model._time = time.copy()
    model._event = event.copy()
    model._fit_torch(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(time, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.int32),
        torch_device="cpu",
    )
    model._fitted = True

    np.testing.assert_allclose(model.coef_, reference.coef_, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(model._bse, reference._bse, rtol=2e-8, atol=2e-9)
    assert model._var_matrix.shape == (X.shape[1], X.shape[1])
    assert (
        np.linalg.norm(model._var_matrix - np.diag(np.diag(model._var_matrix))) > 1e-10
    )
    assert model._baseline_cumulative_hazard is not None


def test_predict_survival_custom_times_use_step_lookup():
    X, time, event = _survival_data(seed=2705)
    model = CoxPH(
        device="cpu", compute_inference=True, compute_cindex=False, max_iter=80
    ).fit(X, time, event)
    base_times = np.asarray(model._unique_times)
    assert base_times.size >= 2

    custom_times = np.array(
        [
            base_times[0] - 1.0,
            0.5 * (base_times[0] + base_times[1]),
            base_times[1],
            base_times[-1] + 1.0,
        ]
    )
    survival, returned_times = model.predict_survival(X[:3], times=custom_times)

    indices = np.searchsorted(base_times, custom_times, side="right") - 1
    expected_h0 = np.zeros(custom_times.size)
    valid = indices >= 0
    expected_h0[valid] = model._baseline_cumulative_hazard[indices[valid]]
    expected = np.exp(-np.exp(X[:3] @ model.coef_)[:, None] * expected_h0[None, :])

    assert survival.shape == (3, custom_times.size)
    np.testing.assert_array_equal(returned_times, custom_times)
    np.testing.assert_allclose(survival, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(survival[:, 0], 1.0, rtol=0, atol=0)

    scalar_survival, scalar_time = model.predict_survival(X[0], times=custom_times[2])
    assert scalar_survival.shape == (1, 1)
    assert scalar_time.shape == (1,)


def test_predict_survival_requires_fitted_baseline():
    X, time, event = _survival_data(seed=2706)
    model = CoxPH(device="cpu", compute_inference=False, compute_cindex=False).fit(
        X, time, event
    )
    with pytest.raises(RuntimeError, match="compute_inference=True"):
        model.predict_survival(X[:2], times=[0.1, 0.5])


@pytest.mark.parametrize("device,ties", [("cuda", "breslow"), ("torch", "efron")])
def test_gpu_nonrobust_inference_keeps_full_covariance_and_baseline(device, ties):
    _require_backend(device)
    X, time, event = _survival_data(
        n=220, p=4, seed=2707 if device == "cuda" else 2708, tied=ties == "efron"
    )
    model = CoxPH(
        ties=ties,
        device=device,
        cov_type="nonrobust",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-8,
    ).fit(X, time, event)

    assert model._var_matrix.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(model._var_matrix, model._var_matrix.T, atol=1e-10)
    np.testing.assert_allclose(
        np.diag(model._var_matrix), np.square(model._bse), rtol=1e-10, atol=1e-12
    )
    assert (
        np.linalg.norm(model._var_matrix - np.diag(np.diag(model._var_matrix))) > 1e-10
    )
    assert model._unique_times is not None
    assert model._baseline_cumulative_hazard is not None

    custom_times = np.array(
        [model._unique_times[0] - 1.0, model._unique_times[0], model._unique_times[-1]]
    )
    survival, returned_times = model.predict_survival(X[:5], custom_times)
    assert survival.shape == (5, 3)
    np.testing.assert_array_equal(returned_times, custom_times)
