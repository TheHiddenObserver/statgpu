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
context-entry failures without double cleanup. Successful exact/precomputed GPU
fits are likewise deduplicated: the backend-owned cleanup after precomputed
inference is retained, while the later outer ``finally`` cleanup becomes a no-op
even after the post-fit reporting handler consumes the precomputed-state flag.
Ordinary deferred-inference fits still keep their intentional solver cleanup and
post-inference cleanup as two separate phases.

After numerical inference has completed, the wrapper also retains the raw
response, raw residual, and validated analytic weights used by public weighted
fit diagnostics.  The numerical ``_resid`` state remains sqrt(weight)-scaled for
covariance, scale, and likelihood calculations.

BaseEstimator's public finite-input guard can sit outside the fit transaction on
typed subclasses.  A narrow ``_reset_fit_state`` hook therefore invalidates the
same Gaussian-L2 state if finite validation rejects a refit before this wrapper
is entered; unrelated PGLM configurations retain their existing reset behavior.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _cupy_asarray_on_device

from ._base import PenalizedGeneralizedLinearModel


_MISSING = object()
# These public spellings all resolve to the same ordinary L2 Gaussian path.
# ``None`` stringifies to ``"none"`` in _penalty_name and is covered too.
_GAUSSIAN_L2_PUBLIC_PENALTY_NAMES = frozenset(
    {"l2", "l2_squared", "ridge", "none", "null", ""}
)


def _penalty_name(estimator) -> str:
    return str(getattr(estimator.penalty, "name", estimator.penalty)).lower().strip()


def _needs_gaussian_conversion_contract(estimator) -> bool:
    return (
        bool(getattr(estimator, "compute_inference", False))
        and str(getattr(estimator, "loss", "")).lower().strip() == "squared_error"
        and _penalty_name(estimator) in _GAUSSIAN_L2_PUBLIC_PENALTY_NAMES
    )


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


def _store_weighted_diagnostic_state(estimator, dispatch_arrays) -> None:
    """Retain raw outcome/residual state after backend-native inference finishes."""
    y_value = dispatch_arrays["y"]
    if y_value is None or getattr(estimator, "_resid", None) is None:
        return

    y_raw = np.asarray(_to_numpy(y_value), dtype=float)
    if y_raw.ndim == 2 and int(y_raw.shape[1]) == 1:
        y_raw = y_raw.reshape(-1)
    estimator._y = y_raw

    weighted_resid = np.asarray(estimator._resid, dtype=float)
    sw_value = dispatch_arrays["sample_weight"]
    if sw_value is None:
        estimator._sample_weight_fit = None
        estimator._raw_resid = weighted_resid.copy()
        return

    weights = np.asarray(_to_numpy(sw_value), dtype=float).reshape(-1)
    if int(weights.shape[0]) != int(weighted_resid.shape[0]):
        raise RuntimeError(
            "Gaussian diagnostic weights no longer match the fitted residual state."
        )
    estimator._sample_weight_fit = weights

    sqrt_w = np.sqrt(weights)
    raw_resid = np.zeros_like(weighted_resid, dtype=float)
    if weighted_resid.ndim == 1:
        np.divide(
            weighted_resid,
            sqrt_w,
            out=raw_resid,
            where=sqrt_w > 0,
        )
    else:
        denom = sqrt_w.reshape((sqrt_w.shape[0],) + (1,) * (weighted_resid.ndim - 1))
        np.divide(
            weighted_resid,
            denom,
            out=raw_resid,
            where=denom > 0,
        )
    estimator._raw_resid = raw_resid


def _invalidate_failed_fit_state(estimator) -> None:
    """Fail closed after any Gaussian-L2 refit failure, including early validation."""
    estimator._native_fit_coef = None
    estimator._native_fit_intercept = None
    estimator.coef_ = None
    estimator.intercept_ = None
    estimator.n_iter_ = 0

    clear_inference = getattr(estimator, "_clear_inference_state", None)
    if callable(clear_inference):
        clear_inference()
    else:
        for name in (
            "_X_design",
            "_y",
            "_resid",
            "_scale",
            "_nobs",
            "_df_resid",
            "_params",
            "_bse",
            "_tvalues",
            "_zvalues",
            "_pvalues",
            "_conf_int",
            "_inference_result",
        ):
            setattr(estimator, name, None)

    estimator._raw_resid = None
    estimator._sample_weight_fit = None
    estimator._inference_precomputed = False
    estimator._precomputed_gaussian_state = None
    estimator._feature_names = None
    estimator._design_info = None
    estimator._formula_has_intercept = None
    estimator._use_intercept = None
    estimator._selected_solver = None
    estimator._selected_backend_name = None
    estimator._selected_backend_device = None
    estimator._fitted = False


def _install_gaussian_public_validation_reset_contract() -> None:
    """Fail closed when an outer public finite guard rejects an L2 refit."""
    current_reset = getattr(PenalizedGeneralizedLinearModel, "_reset_fit_state", None)
    if getattr(current_reset, "_statgpu_gaussian_fit_reset", False):
        return

    def _reset_fit_state(self):
        if _needs_gaussian_conversion_contract(self):
            _invalidate_failed_fit_state(self)
            return None
        if callable(current_reset):
            return current_reset(self)
        return None

    _reset_fit_state._statgpu_gaussian_fit_reset = True
    _reset_fit_state._statgpu_original = current_reset
    PenalizedGeneralizedLinearModel._reset_fit_state = _reset_fit_state


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

        dispatch_arrays = {"X": None, "y": None, "sample_weight": None}
        cleanup_calls = {"cupy": 0, "torch": 0}
        precomputed_cleanup_owned = {"cupy": False, "torch": False}

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

        def _remember_dispatch_arrays(X_value, y_value, sample_weight_value):
            dispatch_arrays["X"] = X_value
            dispatch_arrays["y"] = y_value
            dispatch_arrays["sample_weight"] = sample_weight_value

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
            _remember_dispatch_arrays(X_value, y_value, sample_weight)
            return original_fit_cpu(X_value, y_value, sample_weight)

        def _fit_gpu_started(X_value, y_value, sample_weight=None):
            _remember_dispatch_arrays(X_value, y_value, sample_weight)
            return original_fit_gpu(X_value, y_value, sample_weight)

        def _fit_torch_started(X_value, y_value, sample_weight=None):
            _remember_dispatch_arrays(X_value, y_value, sample_weight)
            return original_fit_torch(X_value, y_value, sample_weight)

        def _cleanup_backend_tracked(backend_name, cleanup):
            cleanup_calls[backend_name] += 1
            # If the backend already released exact/precomputed inference
            # temporaries, later reporting may consume/reset
            # _inference_precomputed before the generic outer finally runs.
            # Preserve ownership in this transaction-local flag rather than
            # re-reading the mutable estimator flag on the second cleanup call.
            if precomputed_cleanup_owned[backend_name]:
                return None

            owns_precomputed_cleanup = bool(
                getattr(self, "_inference_precomputed", False)
            )
            result = cleanup()
            if owns_precomputed_cleanup:
                precomputed_cleanup_owned[backend_name] = True
            return result

        def _cleanup_cuda_tracked():
            return _cleanup_backend_tracked("cupy", original_cleanup_cuda)

        def _cleanup_torch_tracked():
            return _cleanup_backend_tracked("torch", original_cleanup_torch)

        self._compute_post_fit_gaussian_inference = _post_fit_with_dispatch_arrays
        self._fit_cpu = _fit_cpu_started
        self._fit_gpu = _fit_gpu_started
        self._fit_torch = _fit_torch_started
        self._cleanup_cuda_memory = _cleanup_cuda_tracked
        self._cleanup_torch_memory = _cleanup_torch_tracked

        try:
            result = current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
            _store_weighted_diagnostic_state(self, dispatch_arrays)
            return result
        except Exception:
            # Capture the executed backend before invalidation clears provenance.
            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            # Never leave a failed refit advertising or using an older successful
            # fit. This applies even when validation failed before inner fit state
            # was mutated.
            _invalidate_failed_fit_state(self)

            if backend_name in cleanup_calls and cleanup_calls[backend_name] == 0:
                if backend_name == "cupy":
                    self._cleanup_cuda_memory()
                elif backend_name == "torch":
                    self._cleanup_torch_memory()
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


_install_gaussian_public_validation_reset_contract()
_install_gaussian_fit_transaction_contract()
