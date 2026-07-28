"""Public CoxPH adapters for backend-native survival and prediction inputs."""

from __future__ import annotations

from functools import wraps
import inspect

import numpy as np

from statgpu._config import Device
from statgpu.backends._utils import _require_real_array


_NATIVE_ARRAY_MODULES = ("cupy", "torch")
_TIE_METHODS = ("breslow", "efron", "exact")
_COVARIANCE_TYPES = ("nonrobust", "hc0", "hc1", "cluster")
_INFERENCE_MODES = ("strict", "approx")


def _is_native_backend_array(value) -> bool:
    """Return whether slicing ``value`` preserves a CuPy/Torch backend."""
    return type(value).__module__.startswith(_NATIVE_ARRAY_MODULES)


def _normalize_boolean_control(value, name: str) -> bool:
    """Normalize an actual boolean or integer 0/1 without truthy strings."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or integer 0/1")


def _validate_constructor_boolean_controls(
    original_init, args, kwargs, names
) -> None:
    """Reject truthy strings before an estimator constructor can coerce them.

    The validation is intentionally non-mutating.  Integer ``0``/``1`` values
    remain the exact constructor objects supplied by the caller, preserving the
    legacy scikit-learn clone identity contract for estimators that store their
    constructor parameters verbatim.
    """
    bound = inspect.signature(original_init).bind(*args, **kwargs)
    for name in names:
        if name in bound.arguments:
            _normalize_boolean_control(bound.arguments[name], name)


def _normalize_device_control(value) -> Device:
    """Normalize a public device value without silently selecting CPU."""
    try:
        return value if isinstance(value, Device) else Device(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "device must be one of: 'auto', 'cpu', 'cuda', or 'torch'"
        ) from exc


def _normalize_choice_control(value, choices, name: str) -> str:
    """Lowercase and validate a finite string-like choice control."""
    normalized = str(value).lower()
    if normalized not in choices:
        if name == "ties":
            raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
        if name == "cov_type":
            raise ValueError(
                "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'"
            )
        raise ValueError("inference_mode must be strict or approx")
    return normalized


def _normalize_mutable_fit_controls(estimator) -> None:
    """Revalidate CoxPH controls that may have changed through ``set_params``."""
    estimator._validate_optimization_controls()
    estimator.tol = float(estimator.tol)
    estimator.penalty = float(estimator.penalty)
    estimator.ties = _normalize_choice_control(
        estimator.ties, _TIE_METHODS, "ties"
    )
    estimator.cov_type = _normalize_choice_control(
        estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
    )
    estimator.inference_mode = _normalize_choice_control(
        estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
    )
    estimator.device = _normalize_device_control(estimator.device)
    estimator.compute_inference = _normalize_boolean_control(
        estimator.compute_inference, "compute_inference"
    )
    estimator.compute_cindex = _normalize_boolean_control(
        estimator.compute_cindex, "compute_cindex"
    )
    estimator.gpu_memory_cleanup = _normalize_boolean_control(
        estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
    )


def _normalize_mutable_cv_controls(estimator) -> None:
    """Validate CoxPHCV controls before any fold fitting is attempted."""
    estimator.ties = _normalize_choice_control(
        estimator.ties, _TIE_METHODS, "ties"
    )
    estimator.cov_type = _normalize_choice_control(
        estimator.cov_type, _COVARIANCE_TYPES, "cov_type"
    )
    estimator.inference_mode = _normalize_choice_control(
        estimator.inference_mode, _INFERENCE_MODES, "inference_mode"
    )
    estimator.device = _normalize_device_control(estimator.device)
    estimator.compute_inference = _normalize_boolean_control(
        estimator.compute_inference, "compute_inference"
    )
    estimator.gpu_memory_cleanup = _normalize_boolean_control(
        estimator.gpu_memory_cleanup, "gpu_memory_cleanup"
    )
    if (
        estimator.ties == "exact"
        and estimator.compute_inference
        and estimator.cov_type != "nonrobust"
    ):
        raise NotImplementedError(
            "robust covariance is not yet defined for ties='exact'; "
            "use cov_type='nonrobust' or compute_inference=False"
        )


def install_coxph_fit_adapter(coxph_class) -> None:
    """Install public CoxPH boundary adapters exactly once.

    Packed CuPy/Torch survival targets are unpacked by backend-native slicing,
    while ordinary array-likes (including pandas DataFrames) retain the historical
    NumPy normalization contract. Adapter-level validation is transactional: a
    failed refit clears any previously fitted state just like ``CoxPH.fit``.
    Constructor and mutable sklearn-style boolean parameters reject truthy strings;
    mutable controls are normalized and revalidated before every fit. Prediction
    adapters reject complex arrays before a real-dtype cast can discard their
    imaginary components.
    """
    original_init = coxph_class.__init__
    if not getattr(original_init, "_statgpu_validated_boolean_constructor", False):

        @wraps(original_init)
        def init(*args, **kwargs):
            _validate_constructor_boolean_controls(
                original_init,
                args,
                kwargs,
                ("compute_inference", "compute_cindex", "gpu_memory_cleanup"),
            )
            original_init(*args, **kwargs)

        init._statgpu_validated_boolean_constructor = True
        coxph_class.__init__ = init

    original_fit = coxph_class.fit
    if not getattr(original_fit, "_statgpu_backend_native_packed_target", False):

        @wraps(original_fit)
        def fit(
            self,
            X=None,
            time=None,
            event=None,
            entry=None,
            cluster=None,
            init_coef=None,
            formula=None,
            data=None,
            *,
            start=None,
            strata=None,
            subject_id=None,
        ):
            self._reset_fit_state()
            try:
                _normalize_mutable_fit_controls(self)

                if formula is None and X is not None:
                    x_shape = getattr(X, "shape", None)
                    if x_shape is None:
                        x_shape = np.asarray(X).shape
                    if len(x_shape) not in (1, 2):
                        raise ValueError("X must be a one- or two-dimensional array")
                    if len(x_shape) == 2 and int(x_shape[1]) < 1:
                        raise ValueError("X must contain at least one feature")

                if formula is None and event is None and time is not None:
                    _require_real_array(time, "packed survival target")
                    target = time
                    if not _is_native_backend_array(target):
                        target = np.asarray(target)
                    target_shape = getattr(target, "shape", None)
                    if (
                        target_shape is None
                        or len(target_shape) != 2
                        or int(target_shape[1]) not in (2, 3)
                    ):
                        raise ValueError(
                            "When event is omitted, time must be a survival target "
                            "with columns [time, event] or [start, stop, event]"
                        )
                    if int(target_shape[1]) == 2:
                        time, event = target[:, 0], target[:, 1]
                    else:
                        if entry is not None or start is not None:
                            raise ValueError(
                                "Do not pass entry/start separately when the target "
                                "already has [start, stop, event] columns"
                            )
                        start, time, event = target[:, 0], target[:, 1], target[:, 2]

                result = original_fit(
                    self,
                    X=X,
                    time=time,
                    event=event,
                    entry=entry,
                    cluster=cluster,
                    init_coef=init_coef,
                    formula=formula,
                    data=data,
                    start=start,
                    strata=strata,
                    subject_id=subject_id,
                )
                if not getattr(self, "_is_counting_process", False):
                    self._entry = None
                return result
            except Exception:
                self._reset_fit_state()
                raise

        fit._statgpu_backend_native_packed_target = True
        coxph_class.fit = fit

    original_prepare_prediction = coxph_class._prepare_prediction_X
    if not getattr(
        original_prepare_prediction, "_statgpu_real_prediction_guard", False
    ):

        @wraps(original_prepare_prediction)
        def prepare_prediction_X(self, X):
            _require_real_array(X, "X")
            return original_prepare_prediction(self, X)

        prepare_prediction_X._statgpu_real_prediction_guard = True
        coxph_class._prepare_prediction_X = prepare_prediction_X

    original_predict_survival = coxph_class.predict_survival
    if not getattr(original_predict_survival, "_statgpu_real_times_guard", False):

        @wraps(original_predict_survival)
        def predict_survival(self, X, times=None, strata=None):
            _require_real_array(times, "times")
            return original_predict_survival(
                self, X, times=times, strata=strata
            )

        predict_survival._statgpu_real_times_guard = True
        coxph_class.predict_survival = predict_survival


def install_coxphcv_fit_adapter(coxphcv_class) -> None:
    """Install constructor and transactional fit validation on ``CoxPHCV``."""
    original_init = coxphcv_class.__init__
    if not getattr(original_init, "_statgpu_validated_boolean_constructor", False):

        @wraps(original_init)
        def init(*args, **kwargs):
            _validate_constructor_boolean_controls(
                original_init,
                args,
                kwargs,
                ("compute_inference", "gpu_memory_cleanup"),
            )
            original_init(*args, **kwargs)

        init._statgpu_validated_boolean_constructor = True
        coxphcv_class.__init__ = init

    original_fit = coxphcv_class.fit
    if not getattr(original_fit, "_statgpu_validated_cv_controls", False):

        @wraps(original_fit)
        def fit(
            self,
            X,
            time,
            event=None,
            entry=None,
            cluster=None,
            *,
            start=None,
            strata=None,
            subject_id=None,
        ):
            self._reset_fit_state()
            try:
                _normalize_mutable_cv_controls(self)
                return original_fit(
                    self,
                    X,
                    time,
                    event=event,
                    entry=entry,
                    cluster=cluster,
                    start=start,
                    strata=strata,
                    subject_id=subject_id,
                )
            except Exception:
                self._reset_fit_state()
                raise

        fit._statgpu_validated_cv_controls = True
        coxphcv_class.fit = fit


__all__ = ["install_coxph_fit_adapter", "install_coxphcv_fit_adapter"]
