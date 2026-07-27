"""Public CoxPH fit boundary for backend-native packed survival targets."""

from __future__ import annotations

from functools import wraps

import numpy as np

from statgpu.backends._utils import _require_real_array


def install_coxph_fit_adapter(coxph_class) -> None:
    """Install the packed-target adapter exactly once on ``CoxPH``.

    The historical implementation materializes a packed CuPy/Torch target on
    NumPy before dispatch.  This narrow adapter unpacks two- or three-column
    targets by backend-native slicing, then calls the existing validated fit
    implementation with separate arrays.  It also restores ``_entry is None``
    for an ordinary right-censored fit instead of caching a transferred all-zero
    start vector.
    """
    original_fit = coxph_class.fit
    if getattr(original_fit, "_statgpu_backend_native_packed_target", False):
        return

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
        if formula is None and X is not None:
            x_shape = getattr(X, "shape", None)
            if x_shape is None:
                x_shape = np.asarray(X).shape
            if len(x_shape) == 0:
                raise ValueError("X must be a one- or two-dimensional array")

        if formula is None and event is None and time is not None:
            _require_real_array(time, "packed survival target")
            target = time
            target_shape = getattr(target, "shape", None)
            if target_shape is None:
                target = np.asarray(target)
                target_shape = target.shape
            if len(target_shape) != 2 or int(target_shape[1]) not in (2, 3):
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

    fit._statgpu_backend_native_packed_target = True
    coxph_class.fit = fit


__all__ = ["install_coxph_fit_adapter"]
