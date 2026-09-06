"""Compatibility and dispatch contract for penalized post-selection OLS."""

from __future__ import annotations

import functools
import inspect
import warnings

from statgpu._config import Device, _get_configured_device
from statgpu.backends import _is_cupy_array, _is_torch_array
from statgpu.linear_model._penalized_inference_api import (
    POST_SELECTION_OLS,
    deprecated_post_selection_alias,
    normalize_penalized_inference_method,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._post_selection_ols import (
    compute_post_selection_ols_inference,
)

_CONSTRUCTOR_DEPTH_ATTR = "_statgpu_penalized_inference_deprecation_depth"
_FIT_DEPTH_ATTR = "_statgpu_post_selection_fit_device_depth"
_CONSTRUCTOR_MARKER = "__statgpu_penalized_inference_deprecation__"
_FIT_MARKER = "__statgpu_post_selection_fit_device__"
_ROUTER_MARKER = "__statgpu_post_selection_router__"
_VALIDATOR_MARKER = "__statgpu_post_selection_validator__"


def _iter_subclasses(cls):
    seen = set()
    stack = [cls]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        stack.extend(current.__subclasses__())


def _explicit_argument(signature, self, args, kwargs, name):
    try:
        bound = signature.bind_partial(self, *args, **kwargs)
    except TypeError:
        return False, None
    if name in bound.arguments:
        return True, bound.arguments[name]
    for param_name, parameter in signature.parameters.items():
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            continue
        extra = bound.arguments.get(param_name, {})
        if isinstance(extra, dict) and name in extra:
            return True, extra[name]
    return False, None


def _constructor_warning_policy():
    """Return ``(suppress_warning, stacklevel)`` for constructor replay."""
    frame = inspect.currentframe()
    try:
        for _ in range(20):
            if frame is None:
                return False, 2
            frame = frame.f_back
            if frame is None:
                return False, 2
            module_name = str(frame.f_globals.get("__name__", ""))
            function_name = frame.f_code.co_name
            if module_name == "statgpu._base" and function_name == "__sklearn_clone__":
                return True, 2
            if module_name == "statgpu._base" and function_name == "set_params":
                updates = frame.f_locals.get("direct_updates", {})
                if isinstance(updates, dict) and "inference_method" in updates:
                    return False, 3
                return True, 2
            if module_name == "sklearn.base" and function_name in {
                "clone",
                "_clone_parametrized",
            }:
                return True, 2
            if module_name.startswith("statgpu.") and module_name != __name__:
                return True, 2
        return False, 2
    finally:
        del frame


def _install_constructor_contract(cls, *, allow_lassocv_legacy=False):
    original = cls.__dict__.get("__init__")
    if original is None or getattr(original, _CONSTRUCTOR_MARKER, False):
        return
    if cls.__module__.startswith("statgpu.linear_model.legacy"):
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        depth = int(getattr(self, _CONSTRUCTOR_DEPTH_ATTR, 0))
        explicit, value = _explicit_argument(
            signature, self, args, kwargs, "inference_method"
        )
        alias = (
            deprecated_post_selection_alias(
                value, allow_lassocv_legacy=allow_lassocv_legacy
            )
            if explicit
            else None
        )
        suppress, stacklevel = _constructor_warning_policy()
        if depth == 0 and alias is not None and not suppress:
            warnings.warn(
                f"{cls.__name__}(inference_method={alias!r}) is deprecated. "
                f"Use inference_method={POST_SELECTION_OLS!r}. Statistical "
                "method selection no longer encodes CPU/GPU execution; use "
                "device=... for backend selection. The deprecated spelling "
                "will be removed in a future breaking release.",
                FutureWarning,
                stacklevel=stacklevel,
            )

        setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth + 1)
        try:
            result = original(self, *args, **kwargs)
        finally:
            if depth == 0:
                self.__dict__.pop(_CONSTRUCTOR_DEPTH_ATTR, None)
            else:
                setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth)

        runtime_value = getattr(self, "inference_method", value)
        self._inference_method = normalize_penalized_inference_method(
            runtime_value,
            allow_lassocv_legacy=allow_lassocv_legacy,
        )
        return result

    setattr(wrapped, _CONSTRUCTOR_MARKER, True)
    cls.__init__ = wrapped


def _install_inference_validator():
    original = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(original, _VALIDATOR_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(self):
        if not self._compute_inference_enabled:
            return original(self)
        method = normalize_penalized_inference_method(
            getattr(self, "_inference_method", getattr(self, "inference_method", ""))
        )
        self._inference_method = method
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        if method == POST_SELECTION_OLS:
            if (
                self.loss == "squared_error"
                and penalty_name in {"l1", "elasticnet", "en"}
            ):
                return None
            raise NotImplementedError(
                "inference_method='post_selection_ols' is supported only for "
                "squared_error with L1 or ElasticNet penalties. "
                "Choose a supported inference method for this loss/penalty "
                "combination or set compute_inference=False."
            )
        return original(self)

    setattr(wrapped, _VALIDATOR_MARKER, True)
    PenalizedGeneralizedLinearModel._validate_inference_request = wrapped


def _install_post_fit_router():
    original = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    if getattr(original, _ROUTER_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(self, X, y, sample_weight=None):
        method = normalize_penalized_inference_method(
            getattr(self, "_inference_method", getattr(self, "inference_method", ""))
        )
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        if (
            self._compute_inference_enabled
            and self.loss == "squared_error"
            and penalty_name in {"l1", "elasticnet", "en"}
            and method == POST_SELECTION_OLS
        ):
            return compute_post_selection_ols_inference(
                self, X, y, sample_weight=sample_weight
            )
        return original(self, X, y, sample_weight=sample_weight)

    setattr(wrapped, _ROUTER_MARKER, True)
    PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = wrapped


def _input_native_device(self, X):
    """Return an explicit temporary device only for genuine AUTO/AUTO input routing."""
    if getattr(self, "_device", None) != Device.AUTO:
        return None
    if _get_configured_device() != Device.AUTO or X is None:
        return None
    if _is_cupy_array(X):
        return Device.CUDA
    if _is_torch_array(X):
        try:
            return Device.TORCH if bool(X.is_cuda) else None
        except Exception:
            return None
    return None


def _install_fit_device_contract(cls):
    original = cls.__dict__.get("fit")
    if original is None or getattr(original, _FIT_MARKER, False):
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        depth = int(getattr(self, _FIT_DEPTH_ATTR, 0))
        explicit_x, X = _explicit_argument(signature, self, args, kwargs, "X")
        target = _input_native_device(self, X if explicit_x else None) if depth == 0 else None
        prior_device = getattr(self, "_device", None)
        setattr(self, _FIT_DEPTH_ATTR, depth + 1)
        if target is not None:
            self._device = target
        try:
            return original(self, *args, **kwargs)
        finally:
            if target is not None:
                self._device = prior_device
            if depth == 0:
                self.__dict__.pop(_FIT_DEPTH_ATTR, None)
            else:
                setattr(self, _FIT_DEPTH_ATTR, depth)

    setattr(wrapped, _FIT_MARKER, True)
    cls.fit = wrapped


def install_penalized_inference_api_contract():
    """Install migration, device-routing, validation, and inference dispatch."""
    _install_inference_validator()
    _install_post_fit_router()

    for cls in _iter_subclasses(PenalizedGeneralizedLinearModel):
        if not cls.__module__.startswith("statgpu.linear_model"):
            continue
        _install_constructor_contract(cls)
        _install_fit_device_contract(cls)

    from statgpu.linear_model.cv._lasso_cv import LassoCV

    _install_constructor_contract(LassoCV, allow_lassocv_legacy=True)
    _install_fit_device_contract(LassoCV)


__all__ = ["install_penalized_inference_api_contract"]
