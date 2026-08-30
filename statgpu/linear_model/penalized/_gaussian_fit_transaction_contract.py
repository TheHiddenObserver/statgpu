"""Transactional conversion contract for ordinary Gaussian L2 inference fits.

The public penalized estimator currently starts its inner fit/inference failure
transaction after backend conversion and cross-device alignment.  For
squared-error L2 fits with inference enabled, this module installs a narrow
wrapper that closes that gap without changing unrelated estimators.

The wrapper records the *final arrays handed to the backend dispatcher* rather
than the earlier ``_to_array`` results.  Post-fit Gaussian inference therefore
reuses the solver-facing, post-alignment arrays.  For GPU fits those arrays are
normalized to the dtype/device of the retained native fitted coefficients before
inference, preserving the float64 numerical contract used by ordinary GPU L2
solvers even when the public inputs were float32.

Failures before the inner transaction still fail closed.  Cleanup methods are
tracked while the wrapped fit executes, so the outer guard only supplies cleanup
when the inner transaction has not already done so; this also covers CuPy device
context-entry failures without double cleanup.
"""

from __future__ import annotations

from statgpu.backends._utils import _cupy_asarray_on_device

from ._base import PenalizedGeneralizedLinearModel


_MISSING = object()


def _penalty_name(estimator) -> str:
    return str(getattr(estimator.penalty, "name", estimator.penalty)).lower().strip()


def _needs_gaussian_conversion_contract(estimator) -> bool:
    return (
        bool(getattr(estimator, "compute_inference", False))
        and str(getattr(estimator, "loss", "")).lower().strip() == "squared_error"
        and _penalty_name(estimator) == "l2"
    )


def _cleanup_selected_backend(estimator) -> None:
    backend_name = str(getattr(estimator, "_selected_backend_name", "") or "").lower()
    if backend_name == "cupy":
        estimator._cleanup_cuda_memory()
    elif backend_name == "torch":
        estimator._cleanup_torch_memory()


def _normalize_inference_arrays(estimator, X_value, y_value):
    """Match reused GPU fit arrays to retained native parameter precision/device."""
    coef = getattr(estimator, "_native_fit_coef", None)
    backend_name = str(getattr(estimator, "_selected_backend_name", "") or "").lower()
    if coef is None or backend_name not in ("cupy", "torch"):
        return X_value, y_value

    if backend_name == "torch":
        target_device = coef.device
        target_dtype = coef.dtype
        return (
            X_value.to(device=target_device, dtype=target_dtype),
            y_value.to(device=target_device, dtype=target_dtype),
        )

    target_device = int(coef.device.id)
    target_dtype = coef.dtype
    return (
        _cupy_asarray_on_device(X_value, target_device, dtype=target_dtype),
        _cupy_asarray_on_device(y_value, target_device, dtype=target_dtype),
    )


def _install_gaussian_fit_transaction_contract() -> None:
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_gaussian_fit_transaction", False):
        return

    def _fit_with_gaussian_conversion_transaction(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        if not _needs_gaussian_conversion_contract(self):
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )

        dispatch_arrays = {"X": None, "y": None}
        cleanup_calls = {"cupy": 0, "torch": 0}

        original_post_fit_inference = self._compute_post_fit_gaussian_inference
        original_fit_cpu = self._fit_cpu
        original_fit_gpu = self._fit_gpu
        original_fit_torch = self._fit_torch
        original_cleanup_cuda = self._cleanup_cuda_memory
        original_cleanup_torch = self._cleanup_torch_memory

        saved_instance_attrs = {
            name: self.__dict__.get(name, _MISSING)
            for name in (
                "_compute_post_fit_gaussian_inference",
                "_fit_cpu",
                "_fit_gpu",
                "_fit_torch",
                "_cleanup_cuda_memory",
                "_cleanup_torch_memory",
            )
        }

        def _remember_dispatch_arrays(X_value, y_value):
            dispatch_arrays["X"] = X_value
            dispatch_arrays["y"] = y_value

        def _post_fit_with_dispatch_arrays(X_value, y_value, sample_weight=None):
            X_dispatch = dispatch_arrays["X"]
            y_dispatch = dispatch_arrays["y"]
            if X_dispatch is None or y_dispatch is None:
                # Defensive fallback for a future path that invokes post-fit
                # inference without passing through one of the backend dispatchers.
                X_dispatch, y_dispatch = X_value, y_value
            X_inference, y_inference = _normalize_inference_arrays(
                self, X_dispatch, y_dispatch
            )
            return original_post_fit_inference(
                X_inference,
                y_inference,
                sample_weight=sample_weight,
            )

        def _fit_cpu_started(X_value, y_value, sample_weight=None):
            _remember_dispatch_arrays(X_value, y_value)
            return original_fit_cpu(X_value, y_value, sample_weight)

        def _fit_gpu_started(X_value, y_value, sample_weight=None):
            _remember_dispatch_arrays(X_value, y_value)
            return original_fit_gpu(X_value, y_value, sample_weight)

        def _fit_torch_started(X_value, y_value, sample_weight=None):
            _remember_dispatch_arrays(X_value, y_value)
            return original_fit_torch(X_value, y_value, sample_weight)

        def _cleanup_cuda_tracked():
            cleanup_calls["cupy"] += 1
            return original_cleanup_cuda()

        def _cleanup_torch_tracked():
            cleanup_calls["torch"] += 1
            return original_cleanup_torch()

        self._compute_post_fit_gaussian_inference = _post_fit_with_dispatch_arrays
        self._fit_cpu = _fit_cpu_started
        self._fit_gpu = _fit_gpu_started
        self._fit_torch = _fit_torch_started
        self._cleanup_cuda_memory = _cleanup_cuda_tracked
        self._cleanup_torch_memory = _cleanup_torch_tracked

        try:
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        except Exception:
            # The fit clears/replaces result-bearing state before conversion.
            # Never leave a failed refit advertising an older successful fit.
            self._native_fit_coef = None
            self._native_fit_intercept = None
            self._fitted = False

            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            if backend_name in cleanup_calls and cleanup_calls[backend_name] == 0:
                _cleanup_selected_backend(self)
            raise
        finally:
            for name, previous in saved_instance_attrs.items():
                if previous is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = previous

    _fit_with_gaussian_conversion_transaction._statgpu_gaussian_fit_transaction = True
    _fit_with_gaussian_conversion_transaction._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_gaussian_conversion_transaction


_install_gaussian_fit_transaction_contract()
