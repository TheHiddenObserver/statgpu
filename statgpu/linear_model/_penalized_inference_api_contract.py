"""Compatibility and dispatch contract for penalized post-selection OLS."""

from __future__ import annotations

import functools
import inspect
import math
import operator
import warnings

from statgpu._config import Device, _get_configured_device
from statgpu.backends import _is_cupy_array, _is_torch_array
from statgpu.linear_model import _penalized_solver_api_contract as _solver_api_contract
from statgpu.linear_model._penalized_inference_api import (
    POST_SELECTION_OLS,
    deprecated_post_selection_alias,
    normalize_penalized_inference_method,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.linear_model.penalized._post_selection_ols import (
    compute_post_selection_ols_inference,
)

_CONSTRUCTOR_DEPTH_ATTR = "_statgpu_penalized_inference_deprecation_depth"
_FIT_DEPTH_ATTR = "_statgpu_post_selection_fit_device_depth"
_CONSTRUCTOR_MARKER = "__statgpu_penalized_inference_deprecation__"
_FIT_MARKER = "__statgpu_post_selection_fit_device__"
_ROUTER_MARKER = "__statgpu_post_selection_router__"
_VALIDATOR_MARKER = "__statgpu_post_selection_validator__"
_STATE_CLEANUP_MARKER = "__statgpu_post_selection_state_cleanup__"
_SOLVER_WARNING_MARKER = "__statgpu_post_selection_solver_warning_bridge__"
_SPARSE_GAUSSIAN_PENALTIES = frozenset({"l1", "elasticnet", "en"})
_POST_SELECTION_STATE_FIELDS = (
    "_post_selection_X_design",
    "_post_selection_y",
    "_post_selection_resid",
    "_post_selection_scale",
    "_post_selection_df_resid",
    "_post_selection_nobs",
)


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


def _supports_sparse_gaussian_migration(self) -> bool:
    """Whether #137's hardware-neutral migration applies to this estimator."""
    loss_name = str(getattr(self, "loss", "squared_error")).strip().lower()
    penalty_obj = getattr(self, "_penalty", None)
    penalty_name = str(
        getattr(penalty_obj, "name", getattr(self, "penalty", ""))
    ).strip().lower()
    return loss_name == "squared_error" and penalty_name in _SPARSE_GAUSSIAN_PENALTIES


def _is_lassocv_instance(self) -> bool:
    cls = type(self)
    return cls.__name__ == "LassoCV" and cls.__module__.startswith(
        "statgpu.linear_model.cv"
    )


def _validate_lasso_simultaneous_controls(cls, self) -> None:
    """Validate modern Lasso joint-inference controls in its existing wrapper."""
    if cls.__name__ != "Lasso" or cls.__module__ != "statgpu.linear_model.wrappers._lasso":
        return
    if not bool(getattr(self, "enable_simultaneous_inference", False)):
        return

    alpha = float(getattr(self, "simultaneous_alpha", 0.05))
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("simultaneous_alpha must be in (0, 1).")

    raw_n_bootstrap = getattr(self, "simultaneous_n_bootstrap", 1000)
    if isinstance(raw_n_bootstrap, bool):
        raise ValueError("simultaneous_n_bootstrap must be a positive integer.")
    try:
        n_bootstrap = operator.index(raw_n_bootstrap)
    except TypeError as exc:
        raise ValueError(
            "simultaneous_n_bootstrap must be a positive integer."
        ) from exc
    if n_bootstrap <= 0:
        raise ValueError("simultaneous_n_bootstrap must be a positive integer.")


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


def _solver_constructor_warning_policy():
    """Preserve #135 cpu_solver warning semantics through this wrapper layer."""
    frame = inspect.currentframe()
    transparent_frames = 0
    try:
        for _ in range(24):
            if frame is None:
                return False, 2 + transparent_frames
            frame = frame.f_back
            if frame is None:
                return False, 2 + transparent_frames
            module_name = str(frame.f_globals.get("__name__", ""))
            function_name = frame.f_code.co_name

            if module_name == _solver_api_contract.__name__:
                continue
            if module_name == __name__:
                transparent_frames += 1
                continue

            if module_name == "statgpu._base" and function_name == "__sklearn_clone__":
                return True, 2
            if module_name == "statgpu._base" and function_name == "set_params":
                direct_updates = frame.f_locals.get("direct_updates", {})
                if isinstance(direct_updates, dict) and "cpu_solver" in direct_updates:
                    return False, 3 + transparent_frames
                return True, 2
            if module_name == "sklearn.base" and function_name in {
                "clone",
                "_clone_parametrized",
            }:
                return True, 2
            if module_name.startswith("statgpu."):
                return True, 2
            return False, 2 + transparent_frames
        return False, 2 + transparent_frames
    finally:
        del frame


def _install_solver_warning_bridge():
    current = _solver_api_contract._constructor_warning_policy
    if getattr(current, _SOLVER_WARNING_MARKER, False):
        return
    setattr(_solver_constructor_warning_policy, _SOLVER_WARNING_MARKER, True)
    _solver_api_contract._constructor_warning_policy = _solver_constructor_warning_policy


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

        setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth + 1)
        try:
            result = original(self, *args, **kwargs)
        finally:
            if depth == 0:
                self.__dict__.pop(_CONSTRUCTOR_DEPTH_ATTR, None)
            else:
                setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth)

        if depth != 0:
            return result

        _validate_lasso_simultaneous_controls(cls, self)

        runtime_value = getattr(self, "inference_method", value)
        migration_applies = allow_lassocv_legacy or _supports_sparse_gaussian_migration(self)
        if not migration_applies:
            self._inference_method = str(runtime_value).strip().lower()
            return result

        alias = (
            deprecated_post_selection_alias(
                runtime_value, allow_lassocv_legacy=allow_lassocv_legacy
            )
            if explicit
            else None
        )
        suppress, stacklevel = _constructor_warning_policy()
        if alias is not None and not suppress:
            warnings.warn(
                f"{cls.__name__}(inference_method={alias!r}) is deprecated. "
                f"Use inference_method={POST_SELECTION_OLS!r}. Statistical "
                "method selection no longer encodes CPU/GPU execution; use "
                "device=... for backend selection. The deprecated spelling "
                "will be removed in a future breaking release.",
                FutureWarning,
                stacklevel=stacklevel,
            )

        self._inference_method = normalize_penalized_inference_method(
            runtime_value,
            allow_lassocv_legacy=allow_lassocv_legacy,
        )
        return result

    setattr(wrapped, _CONSTRUCTOR_MARKER, True)
    cls.__init__ = wrapped


def _install_state_cleanup():
    original = PenalizedGeneralizedLinearModel._clear_inference_state
    if getattr(original, _STATE_CLEANUP_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(self):
        result = original(self)
        for name in _POST_SELECTION_STATE_FIELDS:
            self.__dict__.pop(name, None)
        return result

    setattr(wrapped, _STATE_CLEANUP_MARKER, True)
    PenalizedGeneralizedLinearModel._clear_inference_state = wrapped


def _install_inference_validator():
    original = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(original, _VALIDATOR_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(self):
        if not self._compute_inference_enabled:
            return original(self)
        method = str(
            getattr(self, "_inference_method", getattr(self, "inference_method", ""))
        ).strip().lower()
        if method == POST_SELECTION_OLS:
            if _supports_sparse_gaussian_migration(self):
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
        method = str(
            getattr(self, "_inference_method", getattr(self, "inference_method", ""))
        ).strip().lower()
        if (
            self._compute_inference_enabled
            and method == POST_SELECTION_OLS
            and _supports_sparse_gaussian_migration(self)
        ):
            return compute_post_selection_ols_inference(
                self, X, y, sample_weight=sample_weight
            )
        return original(self, X, y, sample_weight=sample_weight)

    setattr(wrapped, _ROUTER_MARKER, True)
    PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = wrapped


def _input_native_device(self, X):
    """Return temporary native device only for genuine AUTO/AUTO target scope."""
    if not (_supports_sparse_gaussian_migration(self) or _is_lassocv_instance(self)):
        return None
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


def _fit_input_from_call(args, kwargs):
    if "X" in kwargs:
        return kwargs["X"]
    if args:
        return args[0]
    return None


def _install_fit_device_contract(cls):
    original = getattr(cls, "fit", None)
    if original is None:
        return
    local_fit = cls.__dict__.get("fit")
    if local_fit is not None and getattr(local_fit, _FIT_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        depth = int(getattr(self, _FIT_DEPTH_ATTR, 0))
        X = _fit_input_from_call(args, kwargs)
        target = _input_native_device(self, X) if depth == 0 else None
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
    """Install the #137 Gaussian-sparse migration without widening old GLM scope."""
    _install_solver_warning_bridge()
    _install_state_cleanup()
    _install_inference_validator()
    _install_post_fit_router()

    _install_fit_device_contract(PenalizedLinearRegression)
    for cls in _iter_subclasses(PenalizedLinearRegression):
        if not cls.__module__.startswith("statgpu.linear_model"):
            continue
        _install_constructor_contract(cls)
        if cls is not PenalizedLinearRegression:
            _install_fit_device_contract(cls)

    from statgpu.linear_model.cv._lasso_cv import LassoCV

    _install_constructor_contract(LassoCV, allow_lassocv_legacy=True)
    _install_fit_device_contract(LassoCV)


__all__ = ["install_penalized_inference_api_contract"]
