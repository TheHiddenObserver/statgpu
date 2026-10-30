"""Reusable custom-fold lifecycle boundary for :class:`CoxPHCV`."""

from __future__ import annotations

from functools import wraps

from . import _cox_cv as _module
from ._cox_cv import (
    _clear_materialized_split_state,
    _is_one_shot_iterator,
    _materialize_cv_splits,
)


_ORIGINAL_COXPHCV_INIT = _module.CoxPHCV.__init__
_ORIGINAL_COXPHCV_FIT_CV = _module.CoxPHCV._fit_cv
_ORIGINAL_COXPHCV_GET_PARAMS = _module._COXPHCV_BASE_GET_PARAMS
_ORIGINAL_COXPHCV_SET_PARAMS = _module._COXPHCV_BASE_SET_PARAMS
_ORIGINAL_COXPHCV_GETSTATE = None


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


_module.CoxPHCV.__init__ = _init_with_split_lifecycle
_module.CoxPHCV._fit_cv = _fit_cv_with_reusable_splits
# Historical callable paths and original-function snapshots remain importable.
# Parameter inspection, reconstruction, clone and serialization are now defined
# by CoxPHCV itself, rather than installed as mutable class replacements.
_get_params_with_reusable_splits = _module.CoxPHCV.get_params
_set_params_with_split_invalidation = _module.CoxPHCV.set_params
_clone_with_reusable_splits = _module.CoxPHCV.__sklearn_clone__
_getstate_with_reusable_splits = _module.CoxPHCV.__getstate__

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
