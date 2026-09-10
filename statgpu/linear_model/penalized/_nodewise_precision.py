"""Standardized node-wise Lasso precision for debiased Gaussian inference."""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np

NODEWISE_SOLVER = "fista"
NODEWISE_STOPPING = "coef_delta"
# Iterative stopping is intentionally tighter than the publication KKT gate.
NODEWISE_TOL = 1e-8
NODEWISE_MAX_ITER = 3000
NODEWISE_KKT_TOL = 1e-5
NODEWISE_CONTRACT_VERSION = "standardized_universal_v1"
_EPS64 = np.finfo(np.float64).eps
_TAU_MIN = 64.0 * _EPS64


def validate_nodewise_alpha(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("nodewise_alpha must be None or a finite positive real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("nodewise_alpha must be None or a finite positive real scalar")
    return scalar


def resolve_effective_n(n_rows: int, sample_weight=None) -> float:
    """Response-independent Kish effective n for automatic node-wise tuning."""
    n = int(n_rows)
    if n <= 0:
        raise ValueError("node-wise precision requires at least one observation")
    if sample_weight is None:
        return float(n)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != n or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("sample_weight must be finite, non-negative, and match n_samples")
    total = float(np.sum(w))
    sumsq = float(np.sum(w * w))
    if not math.isfinite(total) or not math.isfinite(sumsq) or total <= 0.0 or sumsq <= 0.0:
        raise ValueError("sample_weight must have positive finite total weight")
    value = total * total / sumsq
    tol = 64.0 * _EPS64 * max(1.0, float(n))
    if not math.isfinite(value) or value <= 0.0 or value > float(n) + tol:
        raise FloatingPointError("invalid node-wise effective sample size")
    return float(min(value, float(n)))


def resolve_nodewise_alpha(requested, *, p: int, effective_n: float):
    requested_value = validate_nodewise_alpha(requested)
    if int(p) <= 1:
        return None, "not_applicable", "not_applicable"
    if requested_value is not None:
        return requested_value, "user", "explicit"
    n_eff = float(effective_n)
    if not math.isfinite(n_eff) or n_eff <= 0.0:
        raise FloatingPointError("node-wise automatic tuning requires positive finite effective_n")
    alpha = math.sqrt(2.0 * math.log(max(int(p), 2)) / n_eff)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise FloatingPointError("node-wise automatic tuning produced invalid alpha")
    return float(alpha), "auto", NODEWISE_CONTRACT_VERSION


def precision_metadata(*, requested, resolved, source, rule, effective_n, weighted, max_kkt, precision_method="nodewise_lasso") -> dict[str, Any]:
    metadata = {
        "precision_method": str(precision_method),
        "nodewise_alpha_requested": None if requested is None else float(requested),
        "nodewise_alpha": None if resolved is None else float(resolved),
        "nodewise_alpha_source": str(source),
        "nodewise_alpha_rule": str(rule),
        "nodewise_design_standardized": True,
        "nodewise_effective_n": float(effective_n),
        "nodewise_weighted": bool(weighted),
    }
    if precision_method == "nodewise_lasso":
        metadata.update(
            nodewise_solver=NODEWISE_SOLVER,
            nodewise_stopping=NODEWISE_STOPPING,
            nodewise_tol=float(NODEWISE_TOL),
            nodewise_max_iter=int(NODEWISE_MAX_ITER),
            nodewise_kkt_tol=float(NODEWISE_KKT_TOL),
            nodewise_max_kkt_residual=float(max_kkt),
        )
    return metadata


def _check_scales_numpy(sigma):
    diag = np.diag(np.asarray(sigma, dtype=np.float64))
    if diag.size == 0 or not np.all(np.isfinite(diag)) or np.any(diag < 0.0):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = np.sqrt(diag)
    dmax = float(np.max(d))
    if not math.isfinite(dmax) or dmax <= 0.0:
        raise FloatingPointError("node-wise design has no positive-scale feature")
    if np.any(~np.isfinite(d)) or np.any(d <= 64.0 * _EPS64 * dmax):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv = 1.0 / d
    sigma_z = sigma * inv[:, None] * inv[None, :]
    if not np.all(np.isfinite(sigma_z)):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv, sigma_z


def _check_tau(tau, cross, l1, kkt):
    if not math.isfinite(tau) or tau <= _TAU_MIN:
        raise FloatingPointError("node-wise normalizer is non-finite or numerically degenerate")
    if not math.isfinite(cross):
        raise FloatingPointError("node-wise residual cross-product is non-finite")
    bound = float(l1) * float(kkt) + 64.0 * _EPS64 * max(1.0, abs(tau))
    if abs(float(cross) - float(tau)) > bound:
        raise FloatingPointError("node-wise normalizer is inconsistent with the validated KKT system")


def build_nodewise_precision_numpy(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    X = np.asarray(X_work, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] <= 0 or X.shape[1] <= 0 or not np.all(np.isfinite(X)):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = map(int, X.shape)
    sigma = X.T @ X / float(n)
    d, inv_d, sigma_z = _check_scales_numpy(sigma)
    alpha, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = np.asarray([[1.0 / (d[0] * d[0])]], dtype=np.float64)
        return M, None, precision_metadata(
            requested=requested_alpha, resolved=None, source="not_applicable", rule="not_applicable",
            effective_n=effective_n, weighted=weighted, max_kkt=0.0, precision_method="analytic_univariate"
        )

    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_cpu_from_gram

    theta = np.zeros((p, p), dtype=np.float64)
    max_kkt = 0.0
    alpha_arr = np.asarray([alpha], dtype=np.float64)
    for j in range(p):
        cols = np.concatenate([np.arange(j), np.arange(j + 1, p)])
        smm = sigma_z[np.ix_(cols, cols)]
        smj = sigma_z[cols, j]
        path, _ = _solve_lasso_path_cpu_from_gram(
            smm * n, smj * n, n_samples=n, alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER, tol=NODEWISE_TOL,
            stopping=NODEWISE_STOPPING, cpu_solver=NODEWISE_SOLVER,
            lipschitz_L=None, cd_kkt_check_every=1,
        )
        gamma = np.asarray(path[0], dtype=np.float64)
        grad = smm @ gamma - smj
        active = gamma != 0.0
        kvals = np.where(active, np.abs(grad + alpha * np.sign(gamma)), np.maximum(np.abs(grad) - alpha, 0.0))
        kkt = float(np.max(kvals)) if kvals.size else 0.0
        if not math.isfinite(kkt) or kkt > NODEWISE_KKT_TOL:
            raise FloatingPointError(f"node-wise Lasso failed the independent KKT publication gate: {kkt:.3e} > {NODEWISE_KKT_TOL:.3e}")
        max_kkt = max(max_kkt, kkt)
        sjj = float(sigma_z[j, j])
        sg = float(smj @ gamma)
        l1 = float(np.sum(np.abs(gamma)))
        tau = sjj - 2.0 * sg + float(gamma @ (smm @ gamma)) + alpha * l1
        cross = sjj - sg
        _check_tau(tau, cross, l1, kkt)
        theta[j, j] = 1.0 / tau
        theta[j, cols] = -gamma / tau

    M = inv_d[:, None] * theta * inv_d[None, :]
    if not np.all(np.isfinite(M)):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    return M, alpha, precision_metadata(
        requested=requested_alpha, resolved=alpha, source=source, rule=rule,
        effective_n=effective_n, weighted=weighted, max_kkt=max_kkt
    )


def _cupy_scales(sigma):
    import cupy as cp
    diag = cp.diag(sigma)
    if not bool(cp.all(cp.isfinite(diag)).item()) or bool(cp.any(diag < 0).item()):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = cp.sqrt(diag)
    dmax = float(cp.max(d).item())
    if not math.isfinite(dmax) or dmax <= 0.0 or bool(cp.any(~cp.isfinite(d)).item()) or bool(cp.any(d <= 64.0 * _EPS64 * dmax).item()):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv = 1.0 / d
    sz = sigma * inv[:, None] * inv[None, :]
    if not bool(cp.all(cp.isfinite(sz)).item()):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv, sz


def build_nodewise_precision_cupy(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    import cupy as cp
    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_gpu_fista_multi_fold_from_gram

    X = cp.asarray(X_work, dtype=cp.float64)
    if X.ndim != 2 or X.shape[0] <= 0 or X.shape[1] <= 0 or not bool(cp.all(cp.isfinite(X)).item()):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = map(int, X.shape)
    sigma = X.T @ X / float(n)
    d, inv, sz = _cupy_scales(sigma)
    alpha, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = cp.ones((1, 1), dtype=X.dtype) * inv[0] * inv[0]
        return M, None, precision_metadata(
            requested=requested_alpha, resolved=None, source="not_applicable", rule="not_applicable",
            effective_n=effective_n, weighted=weighted, max_kkt=0.0, precision_method="analytic_univariate"
        )

    theta = cp.zeros((p, p), dtype=X.dtype)
    eigmax = float(cp.linalg.eigvalsh(sz)[-1].item())
    L = max(eigmax, 1e-12)
    try:
        free_mem, _ = cp.cuda.Device(int(X.device.id)).mem_info
        per = max(1, (p - 1) ** 2 * 8 * 4)
        chunk = max(1, min(p, int(free_mem * 0.6 // per)))
    except Exception:
        chunk = min(p, 16)
    max_kkt = 0.0
    alpha_arr = np.asarray([alpha], dtype=np.float64)
    for j0 in range(0, p, chunk):
        jb = cp.arange(j0, min(p, j0 + chunk), dtype=cp.int32)
        b = int(jb.size)
        base = cp.arange(p - 1, dtype=cp.int32)[None, :]
        cols = base + (base >= jb[:, None])
        smm = sz[cols[:, :, None], cols[:, None, :]]
        smj = sz[cols, jb[:, None]]
        path, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
            smm * n, smj * n, n_samples_vec=np.full(b, float(n)), alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER, tol=NODEWISE_TOL, stopping=NODEWISE_STOPPING,
            lipschitz_L=L, check_every=8,
        )
        gamma = cp.asarray(path[:, 0, :], dtype=X.dtype)
        mmg = cp.matmul(smm, gamma[:, :, None]).reshape(b, p - 1)
        grad = mmg - smj
        active = gamma != 0.0
        kvals = cp.where(active, cp.abs(grad + alpha * cp.sign(gamma)), cp.maximum(cp.abs(grad) - alpha, 0.0))
        krows = cp.max(kvals, axis=1)
        kkt = float(cp.max(krows).item())
        if not math.isfinite(kkt) or kkt > NODEWISE_KKT_TOL:
            raise FloatingPointError(f"node-wise Lasso failed the independent KKT publication gate: {kkt:.3e} > {NODEWISE_KKT_TOL:.3e}")
        max_kkt = max(max_kkt, kkt)
        sjj = sz[jb, jb]
        sg = cp.sum(smj * gamma, axis=1)
        l1 = cp.sum(cp.abs(gamma), axis=1)
        tau = sjj - 2.0 * sg + cp.sum(gamma * mmg, axis=1) + alpha * l1
        cross = sjj - sg
        bound = l1 * krows + 64.0 * _EPS64 * cp.maximum(1.0, cp.abs(tau))
        if bool(cp.any(~cp.isfinite(tau)).item()) or bool(cp.any(tau <= _TAU_MIN).item()) or bool(cp.any(~cp.isfinite(cross)).item()) or bool(cp.any(cp.abs(cross - tau) > bound).item()):
            raise FloatingPointError("node-wise normalizer failed finite/KKT consistency checks")
        itau = 1.0 / tau
        theta[jb, jb] = itau
        theta[jb[:, None], cols] = -gamma * itau[:, None]
    M = theta * inv[:, None] * inv[None, :]
    if not bool(cp.all(cp.isfinite(M)).item()):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    return M, alpha, precision_metadata(
        requested=requested_alpha, resolved=alpha, source=source, rule=rule,
        effective_n=effective_n, weighted=weighted, max_kkt=max_kkt
    )


def _torch_scales(sigma):
    import torch
    diag = torch.diag(sigma)
    if not bool(torch.all(torch.isfinite(diag)).item()) or bool(torch.any(diag < 0).item()):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = torch.sqrt(diag)
    dmax = float(torch.max(d).item())
    if not math.isfinite(dmax) or dmax <= 0.0 or bool(torch.any(~torch.isfinite(d)).item()) or bool(torch.any(d <= 64.0 * _EPS64 * dmax).item()):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv = 1.0 / d
    sz = sigma * inv[:, None] * inv[None, :]
    if not bool(torch.all(torch.isfinite(sz)).item()):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv, sz


def build_nodewise_precision_torch(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    import torch
    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch

    if not isinstance(X_work, torch.Tensor):
        raise TypeError("Torch node-wise precision requires a torch.Tensor")
    X = X_work.to(dtype=torch.float64)
    if X.ndim != 2 or X.shape[0] <= 0 or X.shape[1] <= 0 or not bool(torch.all(torch.isfinite(X)).item()):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = map(int, X.shape)
    sigma = X.T @ X / float(n)
    d, inv, sz = _torch_scales(sigma)
    alpha, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = torch.ones((1, 1), dtype=X.dtype, device=X.device) * inv[0] * inv[0]
        return M, None, precision_metadata(
            requested=requested_alpha, resolved=None, source="not_applicable", rule="not_applicable",
            effective_n=effective_n, weighted=weighted, max_kkt=0.0, precision_method="analytic_univariate"
        )

    theta = torch.zeros((p, p), dtype=X.dtype, device=X.device)
    eigmax = float(torch.linalg.eigvalsh(sz)[-1].item())
    L = max(eigmax, 1e-12)
    try:
        free_mem = torch.cuda.mem_get_info(X.device)[0] if X.is_cuda else 0
        per = max(1, (p - 1) ** 2 * 8 * 4)
        chunk = max(1, min(p, int(free_mem * 0.6 // per))) if free_mem else min(p, 16)
    except Exception:
        chunk = min(p, 16)
    max_kkt = 0.0
    alpha_arr = np.asarray([alpha], dtype=np.float64)
    for j0 in range(0, p, chunk):
        jb = torch.arange(j0, min(p, j0 + chunk), dtype=torch.long, device=X.device)
        b = int(jb.numel())
        base = torch.arange(p - 1, dtype=torch.long, device=X.device)[None, :]
        cols = base + (base >= jb[:, None]).to(torch.long)
        smm = sz[cols[:, :, None], cols[:, None, :]]
        smj = sz[cols, jb[:, None]]
        path, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
            smm * n, smj * n, n_samples_vec=np.full(b, float(n)), alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER, tol=NODEWISE_TOL, stopping=NODEWISE_STOPPING,
            lipschitz_L=L, check_every=8,
        )
        gamma = path[:, 0, :].to(dtype=X.dtype, device=X.device)
        mmg = torch.matmul(smm, gamma.unsqueeze(-1)).reshape(b, p - 1)
        grad = mmg - smj
        active = gamma != 0.0
        zero = torch.zeros((), dtype=X.dtype, device=X.device)
        kvals = torch.where(active, torch.abs(grad + alpha * torch.sign(gamma)), torch.maximum(torch.abs(grad) - alpha, zero))
        krows = torch.max(kvals, dim=1).values
        kkt = float(torch.max(krows).item())
        if not math.isfinite(kkt) or kkt > NODEWISE_KKT_TOL:
            raise FloatingPointError(f"node-wise Lasso failed the independent KKT publication gate: {kkt:.3e} > {NODEWISE_KKT_TOL:.3e}")
        max_kkt = max(max_kkt, kkt)
        sjj = sz[jb, jb]
        sg = torch.sum(smj * gamma, dim=1)
        l1 = torch.sum(torch.abs(gamma), dim=1)
        tau = sjj - 2.0 * sg + torch.sum(gamma * mmg, dim=1) + alpha * l1
        cross = sjj - sg
        bound = l1 * krows + 64.0 * _EPS64 * torch.maximum(torch.ones_like(tau), torch.abs(tau))
        if bool(torch.any(~torch.isfinite(tau)).item()) or bool(torch.any(tau <= _TAU_MIN).item()) or bool(torch.any(~torch.isfinite(cross)).item()) or bool(torch.any(torch.abs(cross - tau) > bound).item()):
            raise FloatingPointError("node-wise normalizer failed finite/KKT consistency checks")
        itau = 1.0 / tau
        theta[jb, jb] = itau
        theta[jb[:, None], cols] = -gamma * itau[:, None]
    M = theta * inv[:, None] * inv[None, :]
    if not bool(torch.all(torch.isfinite(M)).item()):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    return M, alpha, precision_metadata(
        requested=requested_alpha, resolved=alpha, source=source, rule=rule,
        effective_n=effective_n, weighted=weighted, max_kkt=max_kkt
    )


__all__ = [
    "NODEWISE_SOLVER", "NODEWISE_STOPPING", "NODEWISE_TOL", "NODEWISE_MAX_ITER",
    "NODEWISE_KKT_TOL", "NODEWISE_CONTRACT_VERSION", "validate_nodewise_alpha",
    "resolve_effective_n", "resolve_nodewise_alpha", "precision_metadata",
    "build_nodewise_precision_numpy", "build_nodewise_precision_cupy", "build_nodewise_precision_torch",
]
