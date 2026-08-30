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
        backend_resolved = False
        cleanup_calls = {"cupy": 0, "torch": 0}
        original_get_backend = self._get_backend
        original_cleanup_cuda = self._cleanup_cuda_memory
        original_cleanup_torch = self._cleanup_torch_memory
        saved_instance_attrs = {
            name: self.__dict__.get(name, _MISSING)
            for name in (
                "_get_backend",
                "_cleanup_cuda_memory",
                "_cleanup_torch_memory",
            )
        }

        def _get_backend_tracked(*args, **kwargs):
            nonlocal backend_resolved
            result = original_get_backend(*args, **kwargs)
            backend_resolved = True
            return result

        def _tracked_cleanup(backend_name, cleanup):
            cleanup_calls[backend_name] += 1
            return cleanup()

        def _cleanup_cuda_tracked():
            return _tracked_cleanup("cupy", original_cleanup_cuda)

        def _cleanup_torch_tracked():
            return _tracked_cleanup("torch", original_cleanup_torch)

        self._get_backend = _get_backend_tracked
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
            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            # Conversion/alignment and backend solve/inference can both fail
            # before the backend reaches its success-only cleanup. Once this
            # fit has actually resolved a GPU backend, supply cleanup iff no
            # cleanup has executed yet in this transaction.
            if (
                backend_resolved
                and backend_name == "cupy"
                and cleanup_calls["cupy"] == 0
            ):
                self._cleanup_cuda_memory()
            elif (
                backend_resolved
                and backend_name == "torch"
                and cleanup_calls["torch"] == 0
            ):
                self._cleanup_torch_memory()
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
