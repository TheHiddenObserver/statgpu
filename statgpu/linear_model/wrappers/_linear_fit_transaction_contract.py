"""Fail-closed refit transaction for public LinearRegression."""

from __future__ import annotations

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


def _cupy_device_id(device_label: str) -> int | None:
    text = str(device_label or "").lower()
    if not text.startswith("cuda:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def _run_cupy_cleanup_on_device(cleanup, device_label: str):
    """Run best-effort cleanup on a concrete CuPy device without masking failures."""
    device_id = _cupy_device_id(device_label)
    if device_id is None:
        return cleanup()
    try:
        import cupy as cp
    except ImportError:
        # Lightweight hosted regressions can exercise a synthetic CuPy backend
        # without installing CuPy. Preserve the established best-effort cleanup
        # behavior in that test-only/no-runtime-backend situation.
        return cleanup()

    context = cp.cuda.Device(device_id)
    try:
        context.__enter__()
    except Exception:
        # A fit can fail because the concrete device is no longer enterable
        # (for example after a reset). Cleanup must remain best effort: preserve
        # the original fit exception and let the outer transaction invalidate
        # stale fitted state instead of replacing it with this context error.
        return cleanup()
    try:
        return cleanup()
    finally:
        context.__exit__(None, None, None)


def _install_linear_fit_transaction_contract() -> None:
    """Compatibility no-op; LinearRegression owns its transaction at definition."""
