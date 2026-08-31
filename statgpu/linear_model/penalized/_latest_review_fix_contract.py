"""Final review-fix contracts for Gaussian inference device provenance and exact reporting."""

from __future__ import annotations

import functools
import sys

import numpy as np

from statgpu.backends._array_ops import _linalg_exception_is_rank_failure

from ..wrappers._linear import LinearRegression
from ._base import PenalizedGeneralizedLinearModel
from ._inference_mixin import _PenalizedInferenceMixin
from . import _no_inference_public_validation_reset_contract as _public_reset_contract


_MISSING = object()


def _backend_name(value) -> str:
    return str(value or "").lower().strip()


def _device_label(value, backend_name: str, *, allow_runtime_fallback: bool = False):
    backend_name = _backend_name(backend_name)
    if backend_name == "cupy":
        device_id = getattr(getattr(value, "device", None), "id", None)
        if device_id is not None:
            try:
                return f"cuda:{int(device_id)}"
            except (TypeError, ValueError):
                pass
        if allow_runtime_fallback:
            try:
                import cupy as cp

                return f"cuda:{int(cp.cuda.runtime.getDevice())}"
            except Exception:
                return None
        return None

    if backend_name == "torch":
        device = getattr(value, "device", None)
        if device is not None:
            return str(device)
        if allow_runtime_fallback:
            try:
                import torch

                if torch.cuda.is_available():
                    return f"cuda:{int(torch.cuda.current_device())}"
                return "cpu"
            except Exception:
                return None
        return None

    if backend_name in ("numpy", "cpu"):
        return "cpu"
    return None


def _restore_instance_attr(instance, name, previous):
    if previous is _MISSING:
        instance.__dict__.pop(name, None)
    else:
        instance.__dict__[name] = previous


def _install_linear_attempt_device_contract() -> None:
    """Publish the current LinearRegression attempt device before y conversion."""
    current = LinearRegression.fit
    if getattr(current, "_statgpu_attempt_device_contract", False):
        return

    @functools.wraps(current)
    def _fit_with_attempt_device(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        original_get_backend = self._get_backend
        original_to_array = self._to_array
        saved_get_backend = self.__dict__.get("_get_backend", _MISSING)
        saved_to_array = self.__dict__.get("_to_array", _MISSING)
        direct_X = X
        selected_backend = {"name": None}

        def _get_backend_tracked(*args, **kwargs):
            backend = original_get_backend(*args, **kwargs)
            name = _backend_name(getattr(backend, "name", None))
            selected_backend["name"] = name
            self._selected_backend_name = name
            # Clear stale provenance from a previous fit before any conversion.
            self._selected_backend_device = None
            label = _device_label(
                direct_X,
                name,
                allow_runtime_fallback=True,
            )
            if label is not None:
                self._selected_backend_device = label
            return backend

        def _to_array_tracked(value, *args, **kwargs):
            arr = original_to_array(value, *args, **kwargs)
            # In the direct X/y path, X is the first converted public operand.
            # Publish its actual device before y conversion/alignment can fail.
            if direct_X is not None and value is direct_X:
                name = _backend_name(
                    kwargs.get("backend") or selected_backend["name"]
                )
                label = _device_label(arr, name)
                if label is not None:
                    self._selected_backend_device = label
            return arr

        self._get_backend = _get_backend_tracked
        self._to_array = _to_array_tracked
        try:
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        finally:
            _restore_instance_attr(self, "_get_backend", saved_get_backend)
            _restore_instance_attr(self, "_to_array", saved_to_array)

    _fit_with_attempt_device._statgpu_attempt_device_contract = True
    _fit_with_attempt_device._statgpu_original = current
    LinearRegression.fit = _fit_with_attempt_device


def _install_pglm_attempt_device_contract() -> None:
    """Publish final PGLM backend/device provenance before remaining conversions."""
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_attempt_device_contract", False):
        return

    @functools.wraps(current)
    def _fit_with_attempt_device(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        original_select_solver = self._select_solver
        original_to_array = self._to_array
        saved_select_solver = self.__dict__.get("_select_solver", _MISSING)
        saved_to_array = self.__dict__.get("_to_array", _MISSING)
        attempt_X = {"value": None}
        selected_backend = {"name": None}

        def _select_solver_tracked(*args, **kwargs):
            # _fit_mixin calls _select_solver only after auto backend overrides
            # have produced the final backend, and before sample_weight conversion.
            name = _backend_name(kwargs.get("backend_name"))
            X_value = kwargs.get("X")
            attempt_X["value"] = X_value
            selected_backend["name"] = name
            self._selected_backend_name = name
            self._selected_backend_device = None
            label = _device_label(
                X_value,
                name,
                allow_runtime_fallback=True,
            )
            if label is not None:
                self._selected_backend_device = label
            return original_select_solver(*args, **kwargs)

        def _to_array_tracked(value, *args, **kwargs):
            arr = original_to_array(value, *args, **kwargs)
            # sample_weight may be converted before X. Do not mistake it for
            # design provenance; overwrite tentative provenance only when the
            # final solver-facing design itself has converted successfully.
            if attempt_X["value"] is not None and value is attempt_X["value"]:
                name = _backend_name(
                    kwargs.get("backend") or selected_backend["name"]
                )
                label = _device_label(arr, name)
                if label is not None:
                    self._selected_backend_device = label
            return arr

        self._select_solver = _select_solver_tracked
        self._to_array = _to_array_tracked
        try:
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        finally:
            _restore_instance_attr(self, "_select_solver", saved_select_solver)
            _restore_instance_attr(self, "_to_array", saved_to_array)

    _fit_with_attempt_device._statgpu_attempt_device_contract = True
    _fit_with_attempt_device._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_attempt_device


def _best_effort_cleanup(cleanup) -> None:
    try:
        cleanup()
    except Exception:
        # Cleanup is advisory; never replace the public validation exception.
        return None
    return None


def _cleanup_failed_public_finite_validation_safe(estimator) -> None:
    """Preserve the validation exception if CuPy cleanup context entry fails."""
    exc = sys.exc_info()[1]
    backend = _backend_name(getattr(exc, "_statgpu_finite_backend", None))
    device = _backend_name(getattr(exc, "_statgpu_finite_device", None))

    if backend == "torch":
        _best_effort_cleanup(estimator._cleanup_torch_memory)
        return

    if backend != "cupy":
        return

    cleanup = estimator._cleanup_cuda_memory
    if not device.startswith("cuda:"):
        _best_effort_cleanup(cleanup)
        return

    try:
        device_id = int(device.split(":", 1)[1])
    except (TypeError, ValueError):
        _best_effort_cleanup(cleanup)
        return

    try:
        import cupy as cp
    except ImportError:
        _best_effort_cleanup(cleanup)
        return

    try:
        context = cp.cuda.Device(device_id)
        context.__enter__()
    except Exception:
        # The concrete device can itself be unavailable during error recovery.
        # Fall back to the captured cleanup semantics once and preserve the
        # original finite-validation exception.
        _best_effort_cleanup(cleanup)
        return

    try:
        _best_effort_cleanup(cleanup)
    finally:
        try:
            context.__exit__(None, None, None)
        except Exception:
            pass


def _install_public_validation_cleanup_contract() -> None:
    _public_reset_contract._cleanup_failed_public_finite_validation = (
        _cleanup_failed_public_finite_validation_safe
    )


def _precompute_exact_l2_inference_cupy_raw_y(
    self,
    X,
    y,
    XtX_centered,
    X_mean,
    coef_full,
    n_samples,
    sample_weight=None,
    normalization=None,
):
    """Exact CuPy L2 inference with one raw-response reporting snapshot."""
    import cupy as cp
    from statgpu.inference._distributions_backend import t

    p = XtX_centered.shape[0]
    normalization = float(n_samples if normalization is None else normalization)
    ridge_alpha = normalization * self._ridge_alpha_for_exact()
    device_id = int(X.device.id)
    with cp.cuda.Device(device_id):
        sw = (
            None
            if sample_weight is None
            else cp.asarray(sample_weight, dtype=X.dtype).reshape(-1)
        )
        eye_p = cp.eye(p, dtype=XtX_centered.dtype)
    sqrt_sw = None if sw is None else cp.sqrt(sw)

    if X_mean is None:
        xtx_full = XtX_centered
        bread = xtx_full + ridge_alpha * eye_p
    else:
        sum_x = normalization * X_mean
        xtx_orig = XtX_centered + normalization * cp.outer(X_mean, X_mean)
        with cp.cuda.Device(device_id):
            xtx_full = cp.empty((p + 1, p + 1), dtype=XtX_centered.dtype)
        xtx_full[0, 0] = normalization
        xtx_full[0, 1:] = sum_x
        xtx_full[1:, 0] = sum_x
        xtx_full[1:, 1:] = xtx_orig
        bread = xtx_full.copy()
        bread[1:, 1:] = xtx_orig + ridge_alpha * eye_p
    try:
        chol = cp.linalg.cholesky(bread)
        with cp.cuda.Device(device_id):
            identity = cp.eye(bread.shape[0], dtype=bread.dtype)
        bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, identity))
    except Exception as exc:
        if not _linalg_exception_is_rank_failure(exc):
            raise
        bread_inv = cp.linalg.pinv(bread)

    y_pred = X @ coef_full if X_mean is None else coef_full[0] + X @ coef_full[1:]
    resid_raw = y - y_pred
    resid = resid_raw if sqrt_sw is None else resid_raw * sqrt_sw
    df_resid = int(n_samples - coef_full.shape[0])
    if df_resid > 0:
        scale = cp.sum(resid ** 2) / df_resid
    else:
        with cp.cuda.Device(device_id):
            scale = cp.asarray(cp.nan, dtype=X.dtype)

    if X_mean is None:
        X_design_gpu = X if sqrt_sw is None else X * sqrt_sw[:, None]
    else:
        if sqrt_sw is None:
            with cp.cuda.Device(device_id):
                intercept_col = cp.ones(int(n_samples), dtype=X.dtype)
        else:
            intercept_col = sqrt_sw
        feature_block = X if sqrt_sw is None else X * sqrt_sw[:, None]
        X_design_gpu = cp.column_stack([intercept_col, feature_block])

    if df_resid <= 0:
        self._inference_precomputed = True
        self._precomputed_gaussian_state = {
            "params": coef_full.get(),
            "X_design": X_design_gpu.get(),
            "y": y.get(),
            "resid": resid.get(),
            "scale": np.nan,
            "nobs": int(n_samples),
            "df_resid": int(df_resid),
        }
        return

    if self._cov_type == "nonrobust":
        cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
        distribution, method = "t", "classical"
    else:
        from statgpu.linear_model._gaussian_inference import robust_covariance_gpu

        cov_params = robust_covariance_gpu(
            X_design_gpu,
            resid,
            bread_inv,
            self._cov_type,
            cp,
            hac_maxlags=self._hac_maxlags,
        )
        distribution, method = "normal", "sandwich"

    bse = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
    tvalues = coef_full / (bse + 1e-30)
    with cp.cuda.Device(device_id):
        if distribution == "t":
            pvalues = t.two_sided_pvalue(
                tvalues,
                df=df_resid,
                backend="cupy",
            )
            critical = cp.asarray(
                t.two_sided_critical_value(
                    0.05,
                    df=df_resid,
                    backend="cupy",
                ),
                dtype=bse.dtype,
            )
        else:
            from statgpu.inference._distributions_backend import norm

            pvalues = 2.0 * norm.sf(cp.abs(tvalues), backend="cupy")
            critical = cp.asarray(
                norm.ppf(0.975, backend="cupy"),
                dtype=bse.dtype,
            )
    conf_int = cp.stack(
        [coef_full - critical * bse, coef_full + critical * bse],
        axis=1,
    )

    from statgpu.inference._results import GaussianInferenceResult

    result = GaussianInferenceResult(
        params=coef_full.get(),
        bse=bse.get(),
        statistic=tvalues.get(),
        pvalues=pvalues.get(),
        conf_int=conf_int.get(),
        cov_type=self._cov_type,
        distribution=distribution,
        df=df_resid,
        method=method,
        metadata={
            "ridge_alpha": ridge_alpha,
            "alpha": 0.05,
            "numerical_backend": "cupy",
            "numerical_device": f"cuda:{device_id}",
            "reporting_backend": "numpy",
            "reporting_boundary": "post_numerical_inference",
        },
    )
    result.apply_to(self)
    self._inference_precomputed = True
    self._precomputed_gaussian_state = {
        "params": coef_full.get(),
        "X_design": X_design_gpu.get(),
        "y": y.get(),
        "resid": resid.get(),
        "scale": float(scale.get()),
        "nobs": int(n_samples),
        "df_resid": int(df_resid),
    }


def _precompute_exact_l2_inference_torch_raw_y(
    self,
    X,
    y,
    XtX_centered,
    X_mean,
    coef_full,
    n_samples,
    sample_weight=None,
    normalization=None,
):
    """Exact Torch L2 inference with one raw-response reporting snapshot."""
    import torch
    from statgpu.inference._distributions_backend import get_distribution

    p = XtX_centered.shape[0]
    normalization = float(n_samples if normalization is None else normalization)
    ridge_alpha = normalization * self._ridge_alpha_for_exact()
    eye_p = torch.eye(
        p,
        dtype=XtX_centered.dtype,
        device=XtX_centered.device,
    )
    sw = (
        None
        if sample_weight is None
        else torch.as_tensor(
            sample_weight,
            dtype=X.dtype,
            device=X.device,
        ).reshape(-1)
    )
    sqrt_sw = None if sw is None else torch.sqrt(sw)

    if X_mean is None:
        xtx_full = XtX_centered
        bread = xtx_full + ridge_alpha * eye_p
    else:
        sum_x = normalization * X_mean
        xtx_orig = XtX_centered + normalization * torch.outer(X_mean, X_mean)
        xtx_full = torch.empty(
            (p + 1, p + 1),
            dtype=XtX_centered.dtype,
            device=XtX_centered.device,
        )
        xtx_full[0, 0] = normalization
        xtx_full[0, 1:] = sum_x
        xtx_full[1:, 0] = sum_x
        xtx_full[1:, 1:] = xtx_orig
        bread = xtx_full.clone()
        bread[1:, 1:] = xtx_orig + ridge_alpha * eye_p
    try:
        chol = torch.linalg.cholesky(bread)
        bread_inv = torch.cholesky_inverse(chol)
    except RuntimeError as exc:
        if not _linalg_exception_is_rank_failure(exc):
            raise
        bread_inv = torch.linalg.pinv(bread)

    y_pred = X @ coef_full if X_mean is None else coef_full[0] + X @ coef_full[1:]
    resid_raw = y - y_pred
    resid = resid_raw if sqrt_sw is None else resid_raw * sqrt_sw
    df_resid = int(n_samples - coef_full.shape[0])
    scale = (
        torch.sum(resid ** 2) / df_resid
        if df_resid > 0
        else torch.tensor(
            float("nan"),
            dtype=X.dtype,
            device=X.device,
        )
    )

    if X_mean is None:
        X_design_gpu = X if sqrt_sw is None else X * sqrt_sw[:, None]
    else:
        intercept_col = (
            torch.ones(
                int(n_samples),
                dtype=X.dtype,
                device=X.device,
            )
            if sqrt_sw is None
            else sqrt_sw
        )
        feature_block = X if sqrt_sw is None else X * sqrt_sw[:, None]
        X_design_gpu = torch.cat(
            [intercept_col.reshape(-1, 1), feature_block],
            dim=1,
        )

    if df_resid <= 0:
        self._inference_precomputed = True
        self._precomputed_gaussian_state = {
            "params": coef_full.detach().cpu().numpy(),
            "X_design": X_design_gpu.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
            "resid": resid.detach().cpu().numpy(),
            "scale": np.nan,
            "nobs": int(n_samples),
            "df_resid": int(df_resid),
        }
        return

    if self._cov_type == "nonrobust":
        cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
        distribution, method = "t", "classical"
    else:
        from statgpu.linear_model._gaussian_inference import robust_covariance_gpu

        cov_params = robust_covariance_gpu(
            X_design_gpu,
            resid,
            bread_inv,
            self._cov_type,
            torch,
            hac_maxlags=self._hac_maxlags,
        )
        distribution, method = "normal", "sandwich"

    bse = torch.sqrt(torch.clamp(torch.diag(cov_params), min=0.0))
    tvalues = coef_full / (bse + 1e-30)
    if distribution == "t":
        dist = get_distribution("t", backend="torch", device=X.device)
        pvalues = dist.two_sided_pvalue(tvalues, df=df_resid)
        critical = dist.two_sided_critical_value(0.05, df=df_resid)
    else:
        dist = get_distribution("norm", backend="torch", device=X.device)
        pvalues = 2.0 * dist.sf(torch.abs(tvalues))
        critical = dist.ppf(0.975)
    conf_int = torch.stack(
        [coef_full - critical * bse, coef_full + critical * bse],
        dim=1,
    )

    from statgpu.inference._results import GaussianInferenceResult

    result = GaussianInferenceResult(
        params=coef_full.detach().cpu().numpy(),
        bse=bse.detach().cpu().numpy(),
        statistic=tvalues.detach().cpu().numpy(),
        pvalues=pvalues.detach().cpu().numpy(),
        conf_int=conf_int.detach().cpu().numpy(),
        cov_type=self._cov_type,
        distribution=distribution,
        df=df_resid,
        method=method,
        metadata={
            "ridge_alpha": ridge_alpha,
            "alpha": 0.05,
            "numerical_backend": "torch",
            "numerical_device": str(X.device),
            "reporting_backend": "numpy",
            "reporting_boundary": "post_numerical_inference",
        },
    )
    result.apply_to(self)
    self._inference_precomputed = True
    self._precomputed_gaussian_state = {
        "params": coef_full.detach().cpu().numpy(),
        "X_design": X_design_gpu.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "resid": resid.detach().cpu().numpy(),
        "scale": float(scale.detach().cpu().numpy()),
        "nobs": int(n_samples),
        "df_resid": int(df_resid),
    }


def _install_exact_raw_response_contract() -> None:
    _PenalizedInferenceMixin._precompute_exact_l2_inference_cupy = (
        _precompute_exact_l2_inference_cupy_raw_y
    )
    _PenalizedInferenceMixin._precompute_exact_l2_inference_torch = (
        _precompute_exact_l2_inference_torch_raw_y
    )


_install_public_validation_cleanup_contract()
_install_exact_raw_response_contract()
_install_linear_attempt_device_contract()
_install_pglm_attempt_device_contract()
