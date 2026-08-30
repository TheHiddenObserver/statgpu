"""Transactional conversion contract for ordinary Gaussian L2 inference fits.

The public penalized estimator converts input arrays before its existing backend
fit/inference transaction.  For squared-error L2 fits with inference enabled,
that ordering has two important consequences:

* a conversion or cross-device alignment failure can occur after fit state was
  cleared but before the existing rollback/cleanup guard starts; and
* post-fit Gaussian inference can receive the original inputs and repeat a
  full host/device conversion that the fit already performed.

This module installs a narrow wrapper around the existing public fit method.  It
uses the repository's established contract-hook pattern: conversions performed
by the fit are recorded by input identity, post-fit inference reuses those
arrays, and failures that occur before backend dispatch fail closed and execute
the configured accelerator cleanup.  Once backend dispatch has started, the
existing fit mixin owns rollback and cleanup, so cleanup is not duplicated.
"""

from __future__ import annotations

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

        converted_by_identity = {}
        dispatch_started = False

        original_to_array = self._to_array
        original_post_fit_inference = self._compute_post_fit_gaussian_inference
        original_fit_cpu = self._fit_cpu
        original_fit_gpu = self._fit_gpu
        original_fit_torch = self._fit_torch

        saved_instance_attrs = {
            name: self.__dict__.get(name, _MISSING)
            for name in (
                "_to_array",
                "_compute_post_fit_gaussian_inference",
                "_fit_cpu",
                "_fit_gpu",
                "_fit_torch",
            )
        }

        def _capture_to_array(value, device=None, backend=None):
            converted = original_to_array(value, device=device, backend=backend)
            converted_by_identity[id(value)] = converted
            return converted

        def _post_fit_with_converted_arrays(X_value, y_value, sample_weight=None):
            return original_post_fit_inference(
                converted_by_identity.get(id(X_value), X_value),
                converted_by_identity.get(id(y_value), y_value),
                sample_weight=sample_weight,
            )

        def _fit_cpu_started(X_value, y_value, sample_weight=None):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_cpu(X_value, y_value, sample_weight)

        def _fit_gpu_started(X_value, y_value, sample_weight=None):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_gpu(X_value, y_value, sample_weight)

        def _fit_torch_started(X_value, y_value, sample_weight=None):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_torch(X_value, y_value, sample_weight)

        self._to_array = _capture_to_array
        self._compute_post_fit_gaussian_inference = _post_fit_with_converted_arrays
        self._fit_cpu = _fit_cpu_started
        self._fit_gpu = _fit_gpu_started
        self._fit_torch = _fit_torch_started

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
            if not dispatch_started:
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
