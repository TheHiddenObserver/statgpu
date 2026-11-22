"""Preserve CoxPHCV split lifecycle while defining parameter methods once."""

import inspect
import pickle

import numpy as np
import pytest

from statgpu._base import BaseEstimator
from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv_split_lifecycle_contract as lifecycle


def _folds():
    return [(np.array([2, 3]), np.array([0, 1])),
            (np.array([0, 1]), np.array([2, 3]))]


def _model(splits=None):
    return CoxPHCV(penalties=[0.1], cv=2, cv_splits=splits,
                   device="cpu", compute_inference=False)


@pytest.mark.parametrize("name", ["get_params", "set_params"])
def test_parameter_metadata_and_captured_base_call_survive(name, monkeypatch):
    method = getattr(CoxPHCV, name)
    original = getattr(BaseEstimator, name)
    assert method.__wrapped__ is original
    assert method.__module__ == original.__module__
    assert method.__qualname__ == original.__qualname__
    assert method.__doc__ == original.__doc__
    assert inspect.signature(method) == inspect.signature(original)

    def unexpected(*args, **kwargs):
        raise AssertionError("the captured base implementation must remain stable")

    monkeypatch.setattr(BaseEstimator, name, unexpected)
    model = _model()
    if name == "get_params":
        assert model.get_params(deep=False)["cv"] == 2
    else:
        assert model.set_params() is model


@pytest.mark.parametrize("method_name, legacy_name", [
    ("__sklearn_clone__", "_clone_with_reusable_splits"),
    ("__getstate__", "_getstate_with_reusable_splits"),
])
def test_lifecycle_callable_pickle_identity(method_name, legacy_name):
    method = getattr(CoxPHCV, method_name)
    assert getattr(lifecycle, legacy_name) is method
    assert method.__module__ == lifecycle.__name__
    assert method.__name__ == method.__qualname__ == legacy_name
    assert pickle.loads(pickle.dumps(method)) is method


def test_failed_split_materialization_does_not_commit_partial_snapshot():
    class SplitFailure(Exception):
        pass

    def broken():
        yield _folds()[0]
        raise SplitFailure("split enumeration failed")

    source = broken()
    model = _model(source)
    with pytest.raises(SplitFailure, match="split enumeration failed"):
        model.get_params(deep=False)
    assert model.cv_splits is source
    assert model._cox_cv_split_source is None
    assert model._cox_cv_split_snapshot is None


def test_clone_dispatches_subclass_get_params_and_keeps_source_unfitted():
    events = []

    class Child(CoxPHCV):
        def get_params(self, deep=True):
            events.append(deep)
            return super().get_params(deep=deep)

    source = (fold for fold in _folds())
    model = Child(penalties=[0.1], cv=2, cv_splits=source,
                  device="cpu", compute_inference=False)
    cloned = model.__sklearn_clone__()
    assert type(cloned) is Child
    assert events == [False]
    assert model.cv_splits is source
    assert isinstance(cloned.cv_splits, list)
    assert cloned.cv_splits is not model._cox_cv_split_snapshot
    assert cloned._fitted is False
    assert model._fitted is False


def test_serialization_detaches_split_snapshot_and_preserves_other_state():
    source = (fold for fold in _folds())
    model = _model(source)
    model.coef_ = np.array([0.3, -0.1], dtype=np.float32)
    model._fitted = True
    state = model.__getstate__()
    assert state["coef_"] is model.coef_
    assert state["_fitted"] is True
    assert model.cv_splits is source
    assert state["cv_splits"] is not model._cox_cv_split_snapshot
    assert state["_constructor_params_raw"] is not model._constructor_params_raw
    assert state["_constructor_params_raw"]["cv_splits"] is not state["cv_splits"]
    assert state["_cox_cv_split_source"] is None
    assert state["_cox_cv_split_snapshot"] is None
    restored = pickle.loads(pickle.dumps(model))
    assert restored.coef_.dtype == np.float32
    np.testing.assert_array_equal(restored.coef_, model.coef_)
    assert restored._fitted is True


def test_failed_set_params_preserves_existing_split_snapshot():
    source = (fold for fold in _folds())
    model = _model(source)
    snapshot = model.get_params(deep=False)["cv_splits"]
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_parameter=True)
    assert model.cv_splits is source
    assert model._cox_cv_split_source is source
    assert model._cox_cv_split_snapshot is snapshot
    assert model.set_params() is model
    assert model._cox_cv_split_snapshot is snapshot


@pytest.mark.parametrize("name", [
    "get_params", "set_params", "__sklearn_clone__", "__getstate__",
])
def test_parameter_lifecycle_has_one_canonical_definition(name):
    from pathlib import Path
    from statgpu.survival import _cox_cv

    method = getattr(CoxPHCV, name)
    assert Path(method.__code__.co_filename).resolve() == Path(_cox_cv.__file__).resolve()
    for helper in ("_is_one_shot_iterator", "_materialize_cv_splits",
                   "_clear_materialized_split_state"):
        assert getattr(lifecycle, helper) is getattr(_cox_cv, helper)


@pytest.mark.parametrize("name, expected", [
    ("_is_one_shot_iterator", "(value) -> 'bool'"),
    ("_clear_materialized_split_state", "(estimator) -> 'None'"),
])
def test_legacy_helper_annotations_retain_postponed_form(name, expected):
    assert str(inspect.signature(getattr(lifecycle, name))) == expected
