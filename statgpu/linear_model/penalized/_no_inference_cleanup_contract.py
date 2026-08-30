"""Fail-closed transaction and cleanup ownership for estimation-only PGLM fits."""

from __future__ import annotations

import functools

from ._base import PenalizedGeneralizedLinearModel


_MISSING = object()


def _cupy_device_id(device_label: str) -> int | None:
    text = str(device_label or "").lower()
    if not text.startswith("cuda:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def _run_cupy_cleanup_on_device(cleanup, device_label: str):
    """Run estimation-only cleanup on the executed CuPy device when available."""
    device_id = _cupy_device_id(device_label)
    if device_id is None:
        return cleanup()
    try:
        import cupy as cp
    except ImportError:
        # Hosted synthetic-CuPy regressions intentionally run without CuPy.
        return cleanup()

    context = cp.cuda.Device(device_id)
    try:
        context.__enter__()
    except Exception:
        # Preserve the established best-effort behavior if the device context
        # itself cannot be entered; never mask the original fit failure.
        return cleanup()
    try:
        return cleanup()
    finally:
        context.__exit__(None, None, None)


def _invalidate_failed_no_inference_fit(estimator) -> None:
    """Fail closed so a rejected refit cannot expose stale predictions."""
    estimator._native_fit_coef = None
    estimator._native_fit_intercept = None
    estimator.coef_ = None
    estimator.intercept_ = None
    estimator.n_iter_ = 0
    estimator._params = None
    estimator._inference_precomputed = False
    estimator._precomputed_gaussian_state = None
    estimator._feature_names = None
    estimator._design_info = None
    estimator._formula_has_intercept = None
    estimator._use_intercept = None
    estimator._selected_solver = None
    estimator._selected_backend_name = None
    estimator._selected_backend_device = None
    clear_inference = getattr(estimator, "_clear_inference_state", None)
    if callable(clear_inference):
        clear_inference()
    estimator.__dict__.pop("n_features_in_", None)
    estimator._fitted = False


def _public_fit_introspection_source(function):
    """Return the original public fit implementation beneath contract wrappers."""
    current = function
    seen = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "_statgpu_original", None)
        if not callable(wrapped):
            wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped):
            break
        current = wrapped
    return current


def _install_no_inference_cleanup_contract() -> None:
    """Fail close every estimation-only refit and deduplicate GPU cleanup."""
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_no_inference_cleanup_contract", False):
        return

    metadata_source = _public_fit_introspection_source(current)

    @functools.wraps(metadata_source)
    def _fit_with_no_inference_cleanup_contract(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        if bool(getattr(self, "compute_inference", False)):
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )

        cleanup_calls = {"cupy": 0, "torch": 0}
        original_cleanup_cuda = self._cleanup_cuda_memory
        original_cleanup_torch = self._cleanup_torch_memory
        saved_instance_attrs = {
            "_cleanup_cuda_memory": self.__dict__.get(
                "_cleanup_cuda_memory", _MISSING
            ),
            "_cleanup_torch_memory": self.__dict__.get(
                "_cleanup_torch_memory", _MISSING
            ),
        }

        def _tracked_cleanup(backend_name, cleanup):
            cleanup_calls[backend_name] += 1
            if cleanup_calls[backend_name] > 1:
                return None
            return cleanup()

        def _cleanup_cuda_tracked():
            device_label = str(
                getattr(self, "_selected_backend_device", "") or ""
            ).lower()
            return _tracked_cleanup(
                "cupy",
                lambda: _run_cupy_cleanup_on_device(
                    original_cleanup_cuda, device_label
                ),
            )

        def _cleanup_torch_tracked():
            return _tracked_cleanup("torch", original_cleanup_torch)

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
            # Conversion/device-alignment can fail before the inner fit
            # transaction reaches its own backend cleanup. If the executed
            # backend is already known and no cleanup has run yet, invoke the
            # configured cleanup method once on that backend's concrete device.
            # With gpu_memory_cleanup=False the method remains a no-op, while
            # state invalidation still applies.
            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            if backend_name == "cupy" and cleanup_calls["cupy"] == 0:
                self._cleanup_cuda_memory()
            elif backend_name == "torch" and cleanup_calls["torch"] == 0:
                self._cleanup_torch_memory()
            _invalidate_failed_no_inference_fit(self)
            raise
        finally:
            for name, previous in saved_instance_attrs.items():
                if previous is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = previous

    _fit_with_no_inference_cleanup_contract._statgpu_no_inference_cleanup_contract = True
    _fit_with_no_inference_cleanup_contract._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_no_inference_cleanup_contract


_install_no_inference_cleanup_contract()
