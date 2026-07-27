"""Public CoxPH adapters for backend-native survival and prediction inputs."""

from __future__ import annotations

from functools import wraps

import numpy as np

from statgpu.backends._utils import _require_real_array


_NATIVE_ARRAY_MODULES = ("cupy", "torch")


def _is_native_backend_array(value) -> bool:
    """Return whether slicing ``value`` preserves a CuPy/Torch backend."""
    return type(value).__module__.startswith(_NATIVE_ARRAY_MODULES)


def install_coxph_fit_adapter(coxph_class) -> None:
    """Install public CoxPH boundary adapters exactly once.

    Packed CuPy/Torch survival targets are unpacked by backend-native slicing,
    while ordinary array-likes (including pandas DataFrames) retain the historical
    NumPy normalization contract. Adapter-level validation is transactional: a
    failed refit clears any previously fitted state just like ``CoxPH.fit``.
    Prediction adapters reject complex arrays before a real-dtype cast can discard
    their imaginary components.
    """
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


__all__ = ["install_coxph_fit_adapter"]
