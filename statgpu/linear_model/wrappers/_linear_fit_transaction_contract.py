"""Fail-closed refit transaction for public LinearRegression."""

from __future__ import annotations

import functools

from ._linear import LinearRegression


_MISSING = object()


def _invalidate_failed_linear_fit(estimator) -> None:
    """Remove prediction/inference state after a failed public refit."""
    estimator.coef_ = None
    estimator.intercept_ = None
    estimator.rank_ = None
    estimator._effective_rank = None
    estimator._df_model = None
    estimator._df_resid = None
    estimator._X_design = None
    estimator._y = None
    estimator._resid = None
    estimator._scale = None
    estimator._nobs = None
    estimator._params = None
    estimator._sample_weight_fit = None
    estimator._raw_resid = None
    estimator._feature_names = None
    estimator._design_info = None
    estimator._formula_has_intercept = None
    estimator._selected_backend_name = None
    estimator._selected_backend_device = None
    estimator._is_multi_output = False
    estimator._effective_fit_intercept = bool(estimator._fit_intercept)
    estimator._clear_inference_result()
    estimator.__dict__.pop("n_features_in_", None)
    estimator._fitted = False


def _install_linear_fit_transaction_contract() -> None:
    """Fail closed if conversion/alignment/dispatch aborts a refit."""
    current = LinearRegression.fit
    if getattr(current, "_statgpu_linear_fit_transaction", False):
        return

    @functools.wraps(current)
    def _fit_with_fail_closed_transaction(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        # Preserve the PR79 backend-boundary source contract: the wrapped
        # implementation still owns the literal `X_arr = X` and `y_arr = y`
        # staging until backend resolution. This wrapper does not convert either
        # public input before delegating; it only supplies failure transactionality.
        dispatch_started = False
        original_fit_cpu = self._fit_cpu
        original_fit_gpu = self._fit_gpu
        original_fit_torch = self._fit_torch
        saved_instance_attrs = {
            name: self.__dict__.get(name, _MISSING)
            for name in ("_fit_cpu", "_fit_gpu", "_fit_torch")
        }

        def _fit_cpu_started(*args, **kwargs):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_cpu(*args, **kwargs)

        def _fit_gpu_started(*args, **kwargs):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_gpu(*args, **kwargs)

        def _fit_torch_started(*args, **kwargs):
            nonlocal dispatch_started
            dispatch_started = True
            return original_fit_torch(*args, **kwargs)

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
            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            # A CuPy conversion/alignment failure can happen before _fit_gpu
            # takes ownership. Honor gpu_memory_cleanup for those partial
            # allocations; once dispatch starts, the backend remains owner.
            if backend_name == "cupy" and not dispatch_started:
                self._cleanup_cuda_memory()
            elif (
                backend_name == "torch"
                and not dispatch_started
                and bool(getattr(self, "gpu_memory_cleanup", False))
            ):
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            _invalidate_failed_linear_fit(self)
            raise
        finally:
            for name, previous in saved_instance_attrs.items():
                if previous is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = previous

    _fit_with_fail_closed_transaction._statgpu_linear_fit_transaction = True
    _fit_with_fail_closed_transaction._statgpu_original = current
    LinearRegression.fit = _fit_with_fail_closed_transaction


_install_linear_fit_transaction_contract()
