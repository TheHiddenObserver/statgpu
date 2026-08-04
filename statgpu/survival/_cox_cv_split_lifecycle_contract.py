"""Reusable custom-fold lifecycle boundary for :class:`CoxPHCV`."""

from __future__ import annotations

import copy
from functools import wraps

from . import _cox_cv as _module


_ORIGINAL_COXPHCV_INIT = _module.CoxPHCV.__init__
_ORIGINAL_COXPHCV_FIT_CV = _module.CoxPHCV._fit_cv
_ORIGINAL_COXPHCV_GET_PARAMS = _module.CoxPHCV.get_params
_ORIGINAL_COXPHCV_SET_PARAMS = _module.CoxPHCV.set_params
_ORIGINAL_COXPHCV_GETSTATE = _module.CoxPHCV.__dict__.get("__getstate__")


def _is_one_shot_iterator(value) -> bool:
    """Return whether ``value`` is consumed by iterating it once."""
    if value is None:
        return False
    try:
        return iter(value) is value
    except TypeError:
        return False


def _clear_materialized_split_state(estimator) -> None:
    estimator._cox_cv_split_source = None
    estimator._cox_cv_split_snapshot = None


def _materialize_cv_splits(estimator):
    """Materialize a one-shot splitter once and reuse it across estimator use."""
    splits = estimator.cv_splits
    if splits is None or not _is_one_shot_iterator(splits):
        return splits

    source = getattr(estimator, "_cox_cv_split_source", None)
    snapshot = getattr(estimator, "_cox_cv_split_snapshot", None)
    if source is splits and snapshot is not None:
        return snapshot

    snapshot = list(splits)
    estimator._cox_cv_split_source = splits
    estimator._cox_cv_split_snapshot = snapshot
    return snapshot


@wraps(_ORIGINAL_COXPHCV_INIT)
def _init_with_split_lifecycle(self, *args, **kwargs):
    _ORIGINAL_COXPHCV_INIT(self, *args, **kwargs)
    _clear_materialized_split_state(self)


@wraps(_ORIGINAL_COXPHCV_FIT_CV)
def _fit_cv_with_reusable_splits(self, *args, **kwargs):
    """Use one private reusable snapshot without rewriting constructor state."""
    public_splits = self.cv_splits
    effective_splits = _materialize_cv_splits(self)
    if effective_splits is public_splits:
        return _ORIGINAL_COXPHCV_FIT_CV(self, *args, **kwargs)

    self.cv_splits = effective_splits
    try:
        return _ORIGINAL_COXPHCV_FIT_CV(self, *args, **kwargs)
    finally:
        self.cv_splits = public_splits


@wraps(_ORIGINAL_COXPHCV_GET_PARAMS)
def _get_params_with_reusable_splits(self, deep=True):
    """Expose a reusable equivalent of a one-shot constructor iterator."""
    params = _ORIGINAL_COXPHCV_GET_PARAMS(self, deep=deep)
    if _is_one_shot_iterator(params.get("cv_splits")):
        params["cv_splits"] = _materialize_cv_splits(self)
    return params


@wraps(_ORIGINAL_COXPHCV_SET_PARAMS)
def _set_params_with_split_invalidation(self, **params):
    result = _ORIGINAL_COXPHCV_SET_PARAMS(self, **params)
    if "cv_splits" in params:
        _clear_materialized_split_state(self)
    return result


def _clone_with_reusable_splits(self):
    """Return an unfitted clone even when ``cv_splits`` is a generator."""
    params = self.get_params(deep=False).copy()
    if _is_one_shot_iterator(params.get("cv_splits")):
        params["cv_splits"] = _materialize_cv_splits(self)
    return type(self)(**copy.deepcopy(params))


def _getstate_with_reusable_splits(self):
    """Serialize one-shot custom folds as a reusable constructor sequence."""
    if _ORIGINAL_COXPHCV_GETSTATE is None:
        state = self.__dict__.copy()
    else:
        state = dict(_ORIGINAL_COXPHCV_GETSTATE(self))

    if _is_one_shot_iterator(state.get("cv_splits")):
        state["cv_splits"] = copy.deepcopy(_materialize_cv_splits(self))
    state["_cox_cv_split_source"] = None
    state["_cox_cv_split_snapshot"] = None
    return state


_module.CoxPHCV.__init__ = _init_with_split_lifecycle
_module.CoxPHCV._fit_cv = _fit_cv_with_reusable_splits
_module.CoxPHCV.get_params = _get_params_with_reusable_splits
_module.CoxPHCV.set_params = _set_params_with_split_invalidation
_module.CoxPHCV.__sklearn_clone__ = _clone_with_reusable_splits
_module.CoxPHCV.__getstate__ = _getstate_with_reusable_splits

_SPLIT_DOC = """

    Custom split lifecycle
    ----------------------
    ``cv_splits`` may be a reusable sequence or a one-shot iterator. A one-shot
    iterator is materialized privately on first fit, parameter inspection,
    clone, or serialization and then reused for repeated fits. The public
    ``cv_splits`` attribute is not rewritten during fit; ``get_params()``
    exports the reusable equivalent required by legacy sklearn cloning.
"""
if _SPLIT_DOC.strip() not in (_module.CoxPHCV.__doc__ or ""):
    _module.CoxPHCV.__doc__ = (_module.CoxPHCV.__doc__ or "") + _SPLIT_DOC


__all__ = [
    "_is_one_shot_iterator",
    "_materialize_cv_splits",
]
