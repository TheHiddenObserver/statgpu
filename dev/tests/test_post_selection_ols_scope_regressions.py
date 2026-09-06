import warnings

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import Lasso, PenalizedGeneralizedLinearModel, Ridge
import statgpu.linear_model._penalized_inference_api_contract as inference_contract


def _gaussian_data(seed=137, n=80, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 0.3 + X @ np.array([1.0, -0.6, 0.0, 0.25]) + rng.normal(
        scale=0.25, size=n
    )
    return X, y


def test_sparse_gaussian_inherited_fit_actually_installs_auto_native_device_wrapper(
    monkeypatch,
):
    X, y = _gaussian_data()
    model = Lasso(device="auto", compute_inference=False)
    observed = {}

    monkeypatch.setattr(
        inference_contract,
        "_input_native_device",
        lambda owner, value: Device.CUDA if owner is model else None,
    )

    def stop_at_backend_resolution(backend="auto"):
        observed["device"] = model._device
        raise RuntimeError("synthetic backend-resolution sentinel")

    monkeypatch.setattr(model, "_get_backend", stop_at_backend_resolution)

    with pytest.raises(RuntimeError, match="synthetic backend-resolution sentinel"):
        model.fit(X, y)

    assert observed["device"] == Device.CUDA
    # The temporary input-native pin is transactional and never mutates the
    # estimator's public device request after success or failure.
    assert model._device == Device.AUTO
    assert model.device == "auto"


def test_ridge_is_outside_sparse_gaussian_auto_native_pin_scope(monkeypatch):
    ridge = Ridge(device="auto", compute_inference=False)
    fake_cupy = object()
    monkeypatch.setattr(inference_contract, "_is_cupy_array", lambda x: x is fake_cupy)
    monkeypatch.setattr(inference_contract, "_get_configured_device", lambda: Device.AUTO)

    assert inference_contract._input_native_device(ridge, fake_cupy) is None


def test_non_gaussian_legacy_ols_alias_keeps_current_master_rejection_contract():
    rng = np.random.default_rng(138)
    X = rng.normal(size=(120, 3))
    eta = 0.2 + 0.25 * X[:, 0] - 0.15 * X[:, 1]
    y = rng.poisson(np.exp(eta)).astype(float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PenalizedGeneralizedLinearModel(
            loss="poisson",
            penalty="l1",
            alpha=0.01,
            inference_method="cpu_ols",
            compute_inference=True,
            device="cpu",
            solver="fista",
            max_iter=1000,
            tol=1e-7,
        )
        with pytest.raises(
            NotImplementedError,
            match="does not support inference_method='cpu_ols'",
        ):
            model.fit(X, y)

    # #138 must not reinterpret this unrelated non-Gaussian spelling as the new
    # Gaussian-only post_selection_ols migration or emit its deprecation warning.
    assert not any(
        issubclass(item.category, FutureWarning)
        and "post_selection_ols" in str(item.message)
        for item in caught
    )
    assert model.inference_method == "cpu_ols"
    assert model._inference_method == "cpu_ols"


def test_explicit_canonical_method_remains_gaussian_sparse_only():
    rng = np.random.default_rng(139)
    X = rng.normal(size=(40, 2))
    y = rng.poisson(np.exp(0.1 + 0.1 * X[:, 0])).astype(float)
    model = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l1",
        alpha=0.02,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
    )
    with pytest.raises(NotImplementedError, match="squared_error"):
        model.fit(X, y)
