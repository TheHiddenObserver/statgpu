"""Behavioral characterization for definition-owned linear fit transactions."""

import importlib
import inspect
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.linear_model import LinearRegression, Ridge


def _data():
    rng = np.random.default_rng(261)
    X = rng.normal(size=(24, 3))
    return X, 0.8 + X @ np.array([1.2, -0.4, 0.7]) + rng.normal(size=24) * 0.1


def _chain(function):
    result = []
    while function is not None:
        result.append(function)
        function = getattr(function, "__wrapped__", None)
    return result


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
@pytest.mark.parametrize("inference", [False, True])
def test_fit_refit_failure_recovery_and_pickle(cls, inference):
    X, y = _data()
    model = cls(device="cpu", compute_inference=inference).fit(X, y)
    prediction = model.predict(X).copy()
    assert model.fit(X, y) is model
    assert not {"_get_backend", "_to_array", "_cleanup_cuda_memory", "_cleanup_torch_memory"} & model.__dict__.keys()
    assert_allclose(model.predict(X), prediction)
    restored = pickle.loads(pickle.dumps(model))
    assert_allclose(restored.predict(X), prediction)
    assert pickle.loads(pickle.dumps(cls)) is cls
    assert pickle.loads(pickle.dumps(cls.fit)) is cls.fit
    assert pickle.loads(pickle.dumps(model.fit)).__func__ is cls.fit
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, np.full_like(y, np.nan))
    assert model._fitted is False
    assert model.coef_ is None
    with pytest.raises(Exception, match="fitted|fit"):
        model.predict(X)
    model.fit(X, y)
    assert_allclose(model.predict(X), prediction)


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_formula_then_direct_fit_resets_intercept_state(cls):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, y = _data()
    data = pd.DataFrame({"y": y, "x": X[:, 0]})
    model = cls(device="cpu").fit(formula="y ~ x - 1", data=data)
    assert model.fit(X, y) is model
    fresh = cls(device="cpu").fit(X, y)
    assert_allclose(model.predict(X), fresh.predict(X))
    assert model._design_info is None
    assert model._feature_names is None


@pytest.mark.parametrize("failure_phase", ["finite", "conversion", "backend"])
def test_linear_preserves_dynamic_delegates_cleanup_and_error_order(monkeypatch, failure_phase):
    from statgpu.linear_model.penalized import _latest_review_fix_contract as attempt
    from statgpu.linear_model.wrappers import _linear_fit_transaction_contract as transaction

    X, y = _data()
    model = LinearRegression(device="cpu", compute_inference=False).fit(X, y)
    events = []
    error = RuntimeError(failure_phase)
    error._statgpu_finite_backend = "cupy"
    error._statgpu_finite_device = "cuda:4"

    def get_backend(*args, **kwargs):
        events.append("backend")
        if failure_phase == "backend":
            raise error
        return SimpleNamespace(name="cupy")

    def to_array(*args, **kwargs):
        events.append("convert")
        raise error

    def finite(value, **kwargs):
        events.append("finite:" + kwargs["name"])
        if failure_phase == "finite":
            raise error

    def cleanup():
        events.append("cleanup")

    def device_label(*args, **kwargs):
        events.append("device")
        return "cuda:7"

    def on_device(callback, label):
        events.append("cleanup_device:" + label)
        return callback()

    monkeypatch.setattr(model, "_get_backend", get_backend)
    monkeypatch.setattr(model, "_to_array", to_array)
    monkeypatch.setattr(model, "_check_public_input_finite", finite)
    monkeypatch.setattr(model, "_cleanup_cuda_memory", cleanup)
    monkeypatch.setattr(attempt, "_device_label", device_label)
    monkeypatch.setattr(transaction, "_run_cupy_cleanup_on_device", on_device)
    with pytest.raises(RuntimeError) as caught:
        model.fit(X, y)
    assert caught.value is error
    assert model._get_backend is get_backend
    assert model._to_array is to_array
    assert model._cleanup_cuda_memory is cleanup
    assert model._fitted is False and model.coef_ is None
    if failure_phase == "finite":
        assert events == ["finite:X", "cleanup_device:cuda:4", "cleanup"]
    elif failure_phase == "conversion":
        assert events == ["finite:X", "finite:y", "backend", "device", "convert", "cleanup_device:cuda:7", "cleanup"]
    else:
        assert events == ["finite:X", "finite:y", "backend", "cleanup_device:cuda:4", "cleanup"]


@pytest.mark.parametrize("cls", [LinearRegression, Ridge])
def test_subclass_override_and_inherited_fit_keep_public_contract(cls):
    class Child(cls):
        def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
            self.attempts = getattr(self, "attempts", 0) + 1
            return super().fit(X, y, sample_weight, formula, data)

    X, y = _data()
    model = Child(device="cpu", compute_inference=False)
    assert model.fit(X, y) is model
    assert model.attempts == 1
    assert_allclose(model.predict(X), cls(device="cpu", compute_inference=False).fit(X, y).predict(X))
    assert str(inspect.signature(cls.fit)) == "(self, X=None, y=None, sample_weight=None, formula=None, data=None)"
    source = inspect.getsource(inspect.unwrap(cls.fit))
    assert "def fit(" in source
    if cls is LinearRegression:
        assert "X_arr = X" in source and "y_arr = y" in source
    else:
        assert "XtX = X_np.T @ X_np" in source


def test_compatibility_installers_are_idempotent_and_preserve_wrapped_chain():
    from statgpu.linear_model.wrappers import _linear_fit_transaction_contract as linear
    from statgpu.linear_model.wrappers import _ridge as ridge
    from statgpu.linear_model.penalized import _latest_review_fix_contract as attempt

    before = LinearRegression.fit, Ridge.fit
    for _ in range(2):
        linear._install_linear_fit_transaction_contract()
        ridge._install_ridge_failed_refit_contract()
        attempt._install_linear_attempt_device_contract()
    importlib.reload(linear)
    assert (LinearRegression.fit, Ridge.fit) == before
    assert len(_chain(LinearRegression.fit)) == 4
    assert len(_chain(Ridge.fit)) == 5
    for cls, expected in ((LinearRegression, [1, 2, None, None]), (Ridge, [3, 3, 3, None, None])):
        chain = _chain(cls.fit)
        assert [chain.index(function._statgpu_original) if hasattr(function, "_statgpu_original") else None for function in chain] == expected


@pytest.mark.parametrize("first", [
    "statgpu.linear_model.wrappers._linear_fit_transaction_contract",
    "statgpu.linear_model.penalized._latest_review_fix_contract",
    "statgpu.linear_model.wrappers._ridge",
])
def test_fresh_import_orders(first):
    code = f"""
import importlib, pickle
importlib.import_module({first!r})
from statgpu.linear_model import LinearRegression, Ridge
for cls in (LinearRegression, Ridge):
    model = cls(device='cpu', compute_inference=False).fit([[0.], [1.], [2.]], [1., 2., 3.])
    assert model._fitted
    assert pickle.loads(pickle.dumps(cls.fit)) is cls.fit
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
