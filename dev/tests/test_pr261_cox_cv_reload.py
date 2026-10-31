"""Reload the split boundary without recapturing its installed wrappers."""

from functools import wraps
import importlib
import inspect
import pickle
import subprocess
import sys

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv_split_lifecycle_contract as lifecycle


def _folds():
    return [(np.array([2, 3]), np.array([0, 1])),
            (np.array([0, 1]), np.array([2, 3]))]


def _model(splits=None):
    return CoxPHCV(penalties=[0.1], cv=2, cv_splits=splits,
                   device="cpu", compute_inference=False)


@pytest.mark.parametrize("reloads", [1, 2, 5])
def test_reload_preserves_installed_methods_and_existing_instances(reloads):
    existing = _model()
    methods = {name: getattr(CoxPHCV, name) for name in
               ("__init__", "_fit_cv", "get_params", "set_params",
                "__sklearn_clone__", "__getstate__")}
    signature = inspect.signature(CoxPHCV)
    documentation = CoxPHCV.__doc__
    original_init = lifecycle._ORIGINAL_COXPHCV_INIT
    original_fit = lifecycle._ORIGINAL_COXPHCV_FIT_CV
    for _ in range(reloads):
        importlib.reload(lifecycle)
        assert _model().get_params() == existing.get_params()
    assert lifecycle._ORIGINAL_COXPHCV_INIT is original_init
    assert lifecycle._ORIGINAL_COXPHCV_FIT_CV is original_fit
    assert all(getattr(CoxPHCV, name) is method
               for name, method in methods.items())
    assert inspect.signature(CoxPHCV) == signature
    assert CoxPHCV.__doc__ == documentation
    assert existing.__sklearn_clone__().get_params() == existing.get_params()


@pytest.mark.parametrize("fail", [False, True])
def test_reload_preserves_dynamic_fit_hook_and_split_restoration(monkeypatch, fail):
    source = iter(_folds())
    model = _model(source)
    observed = []

    def fit_hook(self, *args, **kwargs):
        observed.append(self.cv_splits)
        if fail:
            raise RuntimeError("fit hook failed")
        return self

    monkeypatch.setattr(lifecycle, "_ORIGINAL_COXPHCV_FIT_CV", fit_hook)
    old_bound_fit = model._fit_cv
    for _ in range(3):
        importlib.reload(lifecycle)
        if fail:
            with pytest.raises(RuntimeError, match="fit hook failed"):
                old_bound_fit(None, None, None)
        else:
            assert old_bound_fit(None, None, None) is model
        assert model.cv_splits is source
    assert len(observed) == 3
    assert all(snapshot is observed[0] for snapshot in observed)
    assert len(observed[0]) == 2


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_existing_pickle_and_callable_paths_survive_reload(protocol):
    source = iter(_folds())
    model = _model(source)
    model.coef_ = np.array([0.25, -0.1], dtype=np.float32)
    model._fitted = True
    serialized = pickle.dumps(model, protocol=protocol)
    methods = [CoxPHCV.__sklearn_clone__, CoxPHCV.__getstate__]
    callables = [pickle.dumps(method, protocol=protocol) for method in methods]
    importlib.reload(lifecycle)
    restored = pickle.loads(serialized)
    np.testing.assert_array_equal(restored.coef_, model.coef_)
    assert restored.coef_.dtype == np.float32
    assert restored._fitted is True
    assert len(restored.cv_splits) == 2
    assert model.cv_splits is source
    for payload, method in zip(callables, methods):
        assert pickle.loads(payload) is method
    from sklearn.base import clone
    cloned = clone(model)
    assert cloned._fitted is False
    assert len(cloned.cv_splits) == 2


def test_reload_preserves_subclass_and_later_constructor_wrapper(monkeypatch):
    events = []
    original_init = CoxPHCV.__init__

    @wraps(original_init)
    def custom_init(self, *args, **kwargs):
        events.append("dynamic")
        original_init(self, *args, **kwargs)

    class Child(CoxPHCV):
        @wraps(original_init)
        def __init__(self, *args, **kwargs):
            events.append("child")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(CoxPHCV, "__init__", custom_init)
    importlib.reload(lifecycle)
    model = Child(device="cpu")
    assert events == ["child", "dynamic"]
    assert model._cox_cv_split_snapshot is None
    assert CoxPHCV.__init__ is custom_init
    assert model.__sklearn_clone__().get_params() == model.get_params()
    assert events == ["child", "dynamic", "child", "dynamic"]


@pytest.mark.parametrize("entrypoint", [
    "statgpu", "statgpu.survival", "statgpu.survival._cox_cv_split_lifecycle_contract",
])
def test_import_entrypoint_and_repeated_package_reload(entrypoint):
    code = f"""
import importlib
importlib.import_module({entrypoint!r})
import statgpu.survival as survival
from statgpu.survival import _cox_cv_split_lifecycle_contract as lifecycle
init = survival.CoxPHCV.__init__
fit = survival.CoxPHCV._fit_cv
for _ in range(3):
    importlib.reload(lifecycle)
    importlib.reload(survival)
    assert survival.CoxPHCV(device='cpu').cv == 5
assert survival.CoxPHCV.__init__ is init
assert survival.CoxPHCV._fit_cv is fit
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("one_shot", [False, True])
def test_fit_before_and_after_reload_keeps_same_split_snapshot(one_shot):
    rng = np.random.default_rng(1261)
    X = rng.normal(size=(24, 2))
    time = np.arange(1.0, 25.0)
    event = (np.arange(24) % 3 != 0).astype(float)
    folds = [(np.arange(12, 24), np.arange(12)),
             (np.arange(12), np.arange(12, 24))]
    splits = iter(folds) if one_shot else folds
    model = _model(splits)
    model.fit(X, time, event)
    expected = model.coef_.copy()
    penalty = model.penalty_
    snapshot = model._cox_cv_split_snapshot
    importlib.reload(lifecycle)
    model.fit(X, time, event)
    np.testing.assert_array_equal(model.coef_, expected)
    assert model.penalty_ == penalty
    assert model.cv_splits is splits
    assert model._cox_cv_split_snapshot is snapshot
    reconstructed = model.__sklearn_clone__().fit(X, time, event)
    np.testing.assert_array_equal(reconstructed.coef_, expected)
