"""Public node-wise-alpha contract for sparse Gaussian debiased inference.

Installed after the existing #138 sparse-inference contracts.  Those wrappers
continue to own centering, analytic-weight transformation, intercept recovery,
and backend-native simultaneous inference; this module replaces only the
node-wise precision/tuning contract they call internally and exposes its public
configuration.
"""

from __future__ import annotations

from contextvars import ContextVar
import functools
import inspect
import math
from typing import Optional

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.inference._results import DebiasedInferenceResult
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.linear_model.penalized._nodewise_precision import (
    build_nodewise_precision_cupy,
    build_nodewise_precision_numpy,
    build_nodewise_precision_torch,
    resolve_effective_n,
    validate_nodewise_alpha,
)
from statgpu.linear_model.wrappers._lasso import Lasso
from statgpu.linear_model.wrappers._elasticnet import ElasticNet
from statgpu.linear_model.cv._lasso_cv import LassoCV
from statgpu.linear_model.cv._elasticnet_cv import ElasticNetCV
from statgpu.linear_model import _post_selection_ols_fifth_review_contract as _fifth

_INSTALL_MARKER = "__statgpu_nodewise_alpha_contract__"
_NODEWISE_CONTEXT = ContextVar("statgpu_nodewise_precision_context", default=None)
_CV_CONTEXT = ContextVar("statgpu_nodewise_cv_final_refit_context", default=None)


def _context_for(model, n_rows: int):
    ctx = _NODEWISE_CONTEXT.get()
    if isinstance(ctx, tuple) and len(ctx) == 3 and ctx[0] == id(model):
        return float(ctx[1]), bool(ctx[2])
    return float(n_rows), False


def _gpu_effective_n(n_rows: int, sample_weight, backend_name: str, ref_arr) -> float:
    """Compute Kish effective n with at most scalar device-to-host transfers."""
    n = int(n_rows)
    if sample_weight is None:
        return float(n)
    name = str(backend_name).lower()
    if name == "cupy":
        import cupy as cp

        device_id = int(ref_arr.device.id)
        with cp.cuda.Device(device_id):
            w = cp.asarray(sample_weight, dtype=cp.float64).reshape(-1)
            if int(w.shape[0]) != n:
                raise ValueError("sample_weight must match n_samples")
            if not bool(cp.all(cp.isfinite(w)).item()) or bool(cp.any(w < 0.0).item()):
                raise ValueError("sample_weight must be finite and non-negative")
            total = float(cp.sum(w).item())
            sumsq = float(cp.sum(w * w).item())
    elif name == "torch":
        import torch

        device = ref_arr.device
        w = (
            sample_weight.to(device=device, dtype=torch.float64)
            if isinstance(sample_weight, torch.Tensor)
            else torch.as_tensor(sample_weight, dtype=torch.float64, device=device)
        ).reshape(-1)
        if int(w.shape[0]) != n:
            raise ValueError("sample_weight must match n_samples")
        if not bool(torch.all(torch.isfinite(w)).item()) or bool(torch.any(w < 0.0).item()):
            raise ValueError("sample_weight must be finite and non-negative")
        total = float(torch.sum(w).item())
        sumsq = float(torch.sum(w * w).item())
    else:
        return resolve_effective_n(n, sample_weight)
    if not math.isfinite(total) or not math.isfinite(sumsq) or total <= 0.0 or sumsq <= 0.0:
        raise ValueError("sample_weight must have positive finite total weight")
    value = total * total / sumsq
    tol = 64.0 * np.finfo(np.float64).eps * max(1.0, float(n))
    if not math.isfinite(value) or value <= 0.0 or value > float(n) + tol:
        raise FloatingPointError("invalid node-wise effective sample size")
    return float(min(value, float(n)))


def _set_candidate(model, resolved, metadata):
    model.__dict__["_nodewise_alpha_candidate"] = resolved
    model.__dict__["_nodewise_metadata_candidate"] = dict(metadata)


def _promote_candidate(model):
    result = getattr(model, "_inference_result", None)
    if result is None or str(getattr(result, "method", "")).lower() != "debiased":
        model.nodewise_alpha_ = None
        model.__dict__.pop("_nodewise_alpha_candidate", None)
        model.__dict__.pop("_nodewise_metadata_candidate", None)
        return
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(dict(model.__dict__.get("_nodewise_metadata_candidate", {}) or {}))
    result.metadata = metadata
    resolved = model.__dict__.get("_nodewise_alpha_candidate", None)
    model.nodewise_alpha_ = None if resolved is None else float(resolved)
    model.__dict__.pop("_nodewise_alpha_candidate", None)
    model.__dict__.pop("_nodewise_metadata_candidate", None)


def _validate_feature_report(params, bse, pvalues, conf_int):
    params_np = np.asarray(params, dtype=np.float64).reshape(-1)
    bse_np = np.asarray(bse, dtype=np.float64).reshape(-1)
    p_np = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    ci_np = np.asarray(conf_int, dtype=np.float64)
    if bse_np.shape != params_np.shape or p_np.shape != params_np.shape or ci_np.shape != (params_np.size, 2):
        raise RuntimeError("debiased node-wise reporting arrays have inconsistent shapes")
    if not (
        np.all(np.isfinite(params_np))
        and np.all(np.isfinite(bse_np))
        and np.all(np.isfinite(p_np))
        and np.all(np.isfinite(ci_np))
    ):
        raise FloatingPointError("debiased node-wise inference produced non-finite reporting values")
    if np.any(bse_np < 0.0) or np.any((p_np < 0.0) | (p_np > 1.0)) or np.any(ci_np[:, 0] > ci_np[:, 1]):
        raise FloatingPointError("debiased node-wise inference produced invalid reporting values")


def _base_result(model, *, params, bse, statistic, pvalues, conf_int, precision_meta, backend_path):
    metadata = {
        "backend_path": backend_path,
        "numerical_backend": "numpy" if backend_path.startswith("cpu") else backend_path.split("_", 1)[0],
        "numerical_device": getattr(model, "_selected_backend_device", "cpu"),
        "reporting_backend": "numpy",
        "reporting_boundary": "post_numerical_inference",
        "precision_cache_hit": False,
        **precision_meta,
    }
    result = DebiasedInferenceResult(
        method="debiased",
        feature_names=model._inference_feature_names(),
        params=params,
        bse=bse,
        statistic=statistic,
        statistic_name="z",
        pvalues=pvalues,
        conf_int=conf_int,
        distribution="normal",
        precision_method=precision_meta["precision_method"],
        metadata=metadata,
        simultaneous_conf_int=getattr(model, "_conf_int_simultaneous", None),
        simultaneous_method=getattr(model, "simultaneous_method", None),
        simultaneous_alpha=getattr(model, "simultaneous_alpha", None),
        simultaneous_n_bootstrap=getattr(model, "simultaneous_n_bootstrap", None),
        simultaneous_critical_value=getattr(model, "_simultaneous_critical_value", None),
        simultaneous_target_mask=getattr(model, "_simultaneous_target_mask", None),
    )
    result.apply_to(model)
    return result


def _nodewise_cpu_debiased(self, X, y, sample_weight=None):
    if sample_weight is not None:
        raise RuntimeError("node-wise CPU core expects the canonical unweighted working design")
    from statgpu.inference._distributions_backend import get_distribution

    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    y_np = np.asarray(_to_numpy(y), dtype=np.float64).reshape(-1)
    n, p = X_np.shape
    effective_n, weighted = _context_for(self, n)
    requested = getattr(self, "nodewise_alpha", None)
    M, resolved, precision_meta = build_nodewise_precision_numpy(
        X_np, requested_alpha=requested, effective_n=effective_n, weighted=weighted
    )
    _set_candidate(self, resolved, precision_meta)

    coef = np.asarray(self.coef_, dtype=np.float64).reshape(-1)
    sigma_hat = X_np.T @ X_np / float(n)
    resid = y_np - X_np @ coef
    s_hat = int(np.sum(np.abs(coef) > 0.0))
    sigma2 = float(np.sum(resid * resid) / float(max(n - s_hat, 1)))
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise FloatingPointError("debiased inference produced an invalid residual variance")
    theta, se, z_stats, _, _, _ = self._debiased_stats_from_M(
        M, sigma_hat, sigma2, coef, X_np, y_np, 0.0, False, n, np, np.linalg.norm
    )
    dist = get_distribution("norm", backend="numpy")
    pvalues = np.minimum(1.0, 2.0 * dist.sf(np.abs(z_stats)))
    critical = float(dist.ppf(0.975))
    ci = np.column_stack([theta - critical * se, theta + critical * se])
    _validate_feature_report(theta, se, pvalues, ci)

    self._params = np.asarray(theta, dtype=np.float64)
    self._bse = np.asarray(se, dtype=np.float64)
    self._tvalues = np.asarray(z_stats, dtype=np.float64)
    self._zvalues = self._tvalues.copy()
    self._pvalues = np.asarray(pvalues, dtype=np.float64)
    self._conf_int = np.asarray(ci, dtype=np.float64)
    self._debiased_M_cpu = np.asarray(M, dtype=np.float64)
    self._X_design = X_np.copy()
    self._y = y_np.copy()
    self._resid = resid.copy()
    self._scale = sigma2
    self._nobs = int(n)
    self._df_resid = int(n - p)

    if bool(getattr(self, "enable_simultaneous_inference", False)):
        self._compute_simultaneous_ci_maxz_bootstrap()

    _base_result(
        self,
        params=self._params.copy(),
        bse=self._bse.copy(),
        statistic=self._tvalues.copy(),
        pvalues=self._pvalues.copy(),
        conf_int=self._conf_int.copy(),
        precision_meta=precision_meta,
        backend_path="cpu_debiased",
    )
    return None


def _nodewise_cupy_debiased(self, X_gpu, y_gpu, coef_gpu):
    import cupy as cp
    from statgpu.inference._distributions_backend import norm as dist

    X = cp.asarray(X_gpu, dtype=cp.float64)
    y = cp.asarray(y_gpu, dtype=cp.float64).reshape(-1)
    coef = cp.asarray(coef_gpu, dtype=cp.float64).reshape(-1)
    n, p = map(int, X.shape)
    effective_n, weighted = _context_for(self, n)
    requested = getattr(self, "nodewise_alpha", None)
    M, resolved, precision_meta = build_nodewise_precision_cupy(
        X, requested_alpha=requested, effective_n=effective_n, weighted=weighted
    )
    _set_candidate(self, resolved, precision_meta)

    sigma_hat = X.T @ X / float(n)
    resid = y - X @ coef
    s_hat = int(cp.sum(cp.abs(coef) > 0.0).item())
    sigma2 = float(cp.sum(resid * resid).item()) / float(max(n - s_hat, 1))
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise FloatingPointError("debiased inference produced an invalid residual variance")
    theta, se, z_stats, _, _, _ = self._debiased_stats_from_M(
        M, sigma_hat, sigma2, coef, X, y, 0.0, False, n, cp, cp.linalg.norm
    )
    pvalues = cp.minimum(1.0, 2.0 * dist.sf(cp.abs(z_stats)))
    critical = dist.ppf(0.975)
    ci = cp.stack([theta - critical * se, theta + critical * se], axis=1)
    params_np, bse_np, z_np, p_np, ci_np = (
        cp.asnumpy(theta), cp.asnumpy(se), cp.asnumpy(z_stats), cp.asnumpy(pvalues), cp.asnumpy(ci)
    )
    _validate_feature_report(params_np, bse_np, p_np, ci_np)

    self._params, self._bse, self._tvalues, self._pvalues, self._conf_int = (
        params_np, bse_np, z_np, p_np, ci_np
    )
    self._zvalues = z_np.copy()
    self._debiased_M_cpu = cp.asnumpy(M)
    self._X_design = cp.asnumpy(X)
    self._y = cp.asnumpy(y)
    self._resid = cp.asnumpy(resid)
    self._scale = sigma2
    self._nobs = int(n)
    self._df_resid = int(n - p)

    if bool(getattr(self, "enable_simultaneous_inference", False)):
        self._compute_simultaneous_ci_maxz_bootstrap()

    _base_result(
        self,
        params=params_np,
        bse=bse_np,
        statistic=z_np,
        pvalues=p_np,
        conf_int=ci_np,
        precision_meta=precision_meta,
        backend_path="cupy_debiased",
    )
    return None


def _nodewise_torch_debiased(self, X_torch, y_torch, coef_torch):
    import torch
    from statgpu.inference._distributions_backend import norm as dist

    if not isinstance(X_torch, torch.Tensor):
        raise TypeError("Torch debiased inference requires torch.Tensor design")
    X = X_torch.to(dtype=torch.float64)
    y = y_torch.to(dtype=torch.float64, device=X.device).reshape(-1)
    coef = coef_torch.to(dtype=torch.float64, device=X.device).reshape(-1)
    n, p = map(int, X.shape)
    effective_n, weighted = _context_for(self, n)
    requested = getattr(self, "nodewise_alpha", None)
    M, resolved, precision_meta = build_nodewise_precision_torch(
        X, requested_alpha=requested, effective_n=effective_n, weighted=weighted
    )
    _set_candidate(self, resolved, precision_meta)

    sigma_hat = X.T @ X / float(n)
    resid = y - X @ coef
    s_hat = int(torch.sum(torch.abs(coef) > 0.0).item())
    sigma2 = float(torch.sum(resid * resid).item()) / float(max(n - s_hat, 1))
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise FloatingPointError("debiased inference produced an invalid residual variance")
    theta, se, z_stats, _, _, _ = self._debiased_stats_from_M(
        M, sigma_hat, sigma2, coef, X, y, 0.0, False, n, torch, torch.linalg.norm
    )
    pvalues = torch.minimum(torch.ones_like(z_stats), 2.0 * dist.sf(torch.abs(z_stats)))
    critical = dist.ppf(0.975)
    if not isinstance(critical, torch.Tensor):
        critical = torch.as_tensor(critical, dtype=X.dtype, device=X.device)
    ci = torch.stack([theta - critical * se, theta + critical * se], dim=1)
    params_np = theta.detach().cpu().numpy()
    bse_np = se.detach().cpu().numpy()
    z_np = z_stats.detach().cpu().numpy()
    p_np = pvalues.detach().cpu().numpy()
    ci_np = ci.detach().cpu().numpy()
    _validate_feature_report(params_np, bse_np, p_np, ci_np)

    self._params, self._bse, self._tvalues, self._pvalues, self._conf_int = (
        params_np, bse_np, z_np, p_np, ci_np
    )
    self._zvalues = z_np.copy()
    self._debiased_M_cpu = M.detach().cpu().numpy()
    self._X_design = X.detach().cpu().numpy()
    self._y = y.detach().cpu().numpy()
    self._resid = resid.detach().cpu().numpy()
    self._scale = sigma2
    self._nobs = int(n)
    self._df_resid = int(n - p)

    if bool(getattr(self, "enable_simultaneous_inference", False)):
        self._compute_simultaneous_ci_maxz_bootstrap()

    _base_result(
        self,
        params=params_np,
        bse=bse_np,
        statistic=z_np,
        pvalues=p_np,
        conf_int=ci_np,
        precision_meta=precision_meta,
        backend_path="torch_debiased",
    )
    return None


def _add_nodewise_parameter(cls):
    current = cls.__init__
    if getattr(current, _INSTALL_MARKER, False):
        return
    signature = inspect.signature(current)
    if "nodewise_alpha" in signature.parameters:
        return
    params = list(signature.parameters.values())
    new_param = inspect.Parameter(
        "nodewise_alpha",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=Optional[float],
    )
    insert_at = next(
        (i for i, parameter in enumerate(params) if parameter.kind is inspect.Parameter.VAR_KEYWORD),
        len(params),
    )
    params.insert(insert_at, new_param)
    new_signature = signature.replace(parameters=params)

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        explicit = "nodewise_alpha" in kwargs
        requested = kwargs.pop("nodewise_alpha", None)
        if not explicit:
            cv_ctx = _CV_CONTEXT.get()
            if isinstance(cv_ctx, tuple) and len(cv_ctx) == 2:
                target_name, inherited_value = cv_ctx
                if type(self).__name__ == target_name:
                    requested = inherited_value
        validate_nodewise_alpha(requested)
        result = current(self, *args, **kwargs)
        self.nodewise_alpha = requested
        self.nodewise_alpha_ = None
        raw = getattr(self, "_constructor_params_raw", None)
        if raw is None:
            raw = {}
            self._constructor_params_raw = raw
        raw["nodewise_alpha"] = requested
        return result

    wrapped.__signature__ = new_signature
    setattr(wrapped, _INSTALL_MARKER, True)
    cls.__init__ = wrapped


def _install_clear_state():
    current = PenalizedGeneralizedLinearModel._clear_inference_state
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self):
        result = current(self)
        self.nodewise_alpha_ = None
        self.__dict__.pop("_nodewise_alpha_candidate", None)
        self.__dict__.pop("_nodewise_metadata_candidate", None)
        return result

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGeneralizedLinearModel._clear_inference_state = wrapped


def _install_cpu_context():
    current = PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None):
        n = int(X.shape[0])
        effective_n = resolve_effective_n(
            n, None if sample_weight is None else _to_numpy(sample_weight)
        )
        token = _NODEWISE_CONTEXT.set((id(self), effective_n, sample_weight is not None))
        try:
            result = current(self, X, y, sample_weight=sample_weight)
            _promote_candidate(self)
            return result
        except Exception:
            self.nodewise_alpha_ = None
            self.__dict__.pop("_nodewise_alpha_candidate", None)
            self.__dict__.pop("_nodewise_metadata_candidate", None)
            raise
        finally:
            _NODEWISE_CONTEXT.reset(token)

    setattr(wrapped, _INSTALL_MARKER, True)
    PenalizedGeneralizedLinearModel._compute_post_fit_debiased_inference = wrapped


def _wrap_gpu_fit_for_context(cls):
    current = cls._fit_gpu_backend
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None, backend_name="cupy"):
        n = int(X.shape[0])
        effective_n = _gpu_effective_n(n, sample_weight, backend_name, X)
        token = _NODEWISE_CONTEXT.set((id(self), effective_n, sample_weight is not None))
        try:
            result = current(self, X, y, sample_weight, backend_name=backend_name)
            if bool(getattr(self, "_compute_inference_enabled", False)):
                _promote_candidate(self)
            return result
        except Exception:
            self.nodewise_alpha_ = None
            self.__dict__.pop("_nodewise_alpha_candidate", None)
            self.__dict__.pop("_nodewise_metadata_candidate", None)
            raise
        finally:
            _NODEWISE_CONTEXT.reset(token)

    setattr(wrapped, _INSTALL_MARKER, True)
    cls._fit_gpu_backend = wrapped


def _wrap_cv_reset(cls):
    current = getattr(cls, "_reset_cv_fit_state", None)
    if current is None or getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        result = current(self, *args, **kwargs)
        self.nodewise_alpha_ = None
        return result

    setattr(wrapped, _INSTALL_MARKER, True)
    cls._reset_cv_fit_state = wrapped


def _wrap_cv_fit(cls, target_name: str):
    current = cls.fit
    if getattr(current, _INSTALL_MARKER, False):
        return

    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        self.nodewise_alpha_ = None
        token = _CV_CONTEXT.set((target_name, getattr(self, "nodewise_alpha", None)))
        try:
            result = current(self, *args, **kwargs)
        except Exception:
            self.nodewise_alpha_ = None
            raise
        finally:
            _CV_CONTEXT.reset(token)
        estimator = getattr(self, "estimator_", None)
        self.nodewise_alpha_ = None if estimator is None else getattr(estimator, "nodewise_alpha_", None)
        return result

    setattr(wrapped, _INSTALL_MARKER, True)
    cls.fit = wrapped


def install_nodewise_alpha_inference_contract() -> None:
    for cls in (
        PenalizedGeneralizedLinearModel,
        PenalizedLinearRegression,
        Lasso,
        ElasticNet,
        LassoCV,
        ElasticNetCV,
    ):
        _add_nodewise_parameter(cls)

    _install_clear_state()

    # Existing #138 centered/weighted wrappers invoke these captured originals.
    _fifth._ORIGINAL_CPU_DEBIASED = _nodewise_cpu_debiased
    _fifth._ORIGINAL_CUPY_DEBIASED = _nodewise_cupy_debiased
    _fifth._ORIGINAL_TORCH_DEBIASED = _nodewise_torch_debiased

    _install_cpu_context()
    _wrap_gpu_fit_for_context(PenalizedGeneralizedLinearModel)
    _wrap_gpu_fit_for_context(PenalizedLinearRegression)
    _wrap_cv_reset(LassoCV)
    _wrap_cv_reset(ElasticNetCV)
    _wrap_cv_fit(LassoCV, "Lasso")
    _wrap_cv_fit(ElasticNetCV, "ElasticNet")


__all__ = ["install_nodewise_alpha_inference_contract"]
