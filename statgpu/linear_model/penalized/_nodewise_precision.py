"""Shared node-wise Lasso precision contract for debiased Gaussian inference.

The public tuning parameter is interpreted on a standardized copy of the
already-canonical centered/weighted working design.  Automatic tuning is
response-independent.  NumPy, CuPy, and Torch use the same statistical and
fail-closed numerical contract.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np

NODEWISE_SOLVER = "fista"
NODEWISE_STOPPING = "coef_delta"
NODEWISE_TOL = 1e-5
NODEWISE_MAX_ITER = 500
NODEWISE_KKT_TOL = 1e-5
NODEWISE_CONTRACT_VERSION = "standardized_universal_v1"
_EPS64 = np.finfo(np.float64).eps
_TAU_MIN = 64.0 * _EPS64


def validate_nodewise_alpha(value: Optional[float]) -> Optional[float]:
    """Validate a public scalar node-wise penalty without mutating caller state."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("nodewise_alpha must be None or a finite positive real scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("nodewise_alpha must be None or a finite positive real scalar")
    return scalar


def _validate_effective_n(n_eff: float, n_rows: int) -> float:
    n = int(n_rows)
    value = float(n_eff)
    tol = 64.0 * _EPS64 * max(1.0, float(n))
    if not math.isfinite(value) or value <= 0.0 or value > float(n) + tol:
        raise FloatingPointError("invalid node-wise effective sample size")
    return float(min(value, float(n)))


def resolve_effective_n(n_rows: int, sample_weight=None) -> float:
    """Return Kish-style response-independent effective n on NumPy inputs."""
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
    return _validate_effective_n(total * total / sumsq, n)


def resolve_effective_n_backend(n_rows: int, sample_weight, backend_name: str) -> float:
    """Resolve weighted effective n without a full GPU-to-host weight transfer."""
    n = int(n_rows)
    if sample_weight is None:
        return float(n)
    name = str(backend_name).strip().lower()
    if name == "cupy":
        import cupy as cp

        w = cp.asarray(sample_weight, dtype=cp.float64).reshape(-1)
        if int(w.shape[0]) != n:
            raise ValueError("sample_weight must match n_samples")
        if not bool(cp.all(cp.isfinite(w)).item()) or bool(cp.any(w < 0).item()):
            raise ValueError("sample_weight must be finite and non-negative")
        total = float(cp.sum(w).item())
        sumsq = float(cp.sum(w * w).item())
    elif name == "torch":
        import torch

        if isinstance(sample_weight, torch.Tensor):
            w = sample_weight.to(dtype=torch.float64).reshape(-1)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            w = torch.as_tensor(sample_weight, dtype=torch.float64, device=device).reshape(-1)
        if int(w.shape[0]) != n:
            raise ValueError("sample_weight must match n_samples")
        if not bool(torch.all(torch.isfinite(w)).item()) or bool(torch.any(w < 0).item()):
            raise ValueError("sample_weight must be finite and non-negative")
        total = float(torch.sum(w).item())
        sumsq = float(torch.sum(w * w).item())
    else:
        return resolve_effective_n(n, sample_weight)
    if not math.isfinite(total) or not math.isfinite(sumsq) or total <= 0.0 or sumsq <= 0.0:
        raise ValueError("sample_weight must have positive finite total weight")
    return _validate_effective_n(total * total / sumsq, n)


def resolve_nodewise_alpha(requested, *, p: int, effective_n: float) -> tuple[Optional[float], str, str]:
    """Resolve explicit/automatic node-wise alpha; p=1 has no nuisance solve."""
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


def precision_metadata(
    *,
    requested,
    resolved,
    source,
    rule,
    effective_n,
    weighted,
    max_kkt,
    precision_method="nodewise_lasso",
) -> dict[str, Any]:
    base = {
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
        base.update(
            {
                "nodewise_solver": NODEWISE_SOLVER,
                "nodewise_stopping": NODEWISE_STOPPING,
                "nodewise_tol": float(NODEWISE_TOL),
                "nodewise_max_iter": int(NODEWISE_MAX_ITER),
                "nodewise_kkt_tol": float(NODEWISE_KKT_TOL),
                "nodewise_max_kkt_residual": float(max_kkt),
            }
        )
    return base


def _check_tau_cross(tau2: float, cross: float, gamma_l1: float, kkt: float) -> None:
    if not math.isfinite(tau2) or tau2 <= _TAU_MIN:
        raise FloatingPointError("node-wise normalizer is non-finite or numerically degenerate")
    if not math.isfinite(cross):
        raise FloatingPointError("node-wise residual cross-product is non-finite")
    bound = gamma_l1 * float(kkt) + 64.0 * _EPS64 * max(1.0, abs(tau2))
    if abs(cross - tau2) > bound:
        raise FloatingPointError("node-wise normalizer is inconsistent with the validated KKT system")


def _numpy_scales(sigma: np.ndarray):
    diag = np.diag(sigma)
    if diag.ndim != 1 or diag.size == 0 or not np.all(np.isfinite(diag)) or np.any(diag < 0.0):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = np.sqrt(diag)
    d_max = float(np.max(d))
    if not math.isfinite(d_max) or d_max <= 0.0:
        raise FloatingPointError("node-wise design has no positive-scale feature")
    d_tol = 64.0 * _EPS64 * d_max
    if np.any(~np.isfinite(d)) or np.any(d <= d_tol):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv_d = 1.0 / d
    sigma_z = sigma * inv_d[:, None] * inv_d[None, :]
    if not np.all(np.isfinite(sigma_z)):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv_d, sigma_z


def build_nodewise_precision_numpy(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    """Build standardized/back-transformed precision on NumPy."""
    X = np.asarray(X_work, dtype=np.float64)
    if X.ndim != 2 or int(X.shape[0]) <= 0 or int(X.shape[1]) <= 0 or not np.all(np.isfinite(X)):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = X.shape
    sigma = X.T @ X / float(n)
    d, inv_d, sigma_z = _numpy_scales(sigma)
    resolved, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = np.asarray([[1.0 / (d[0] * d[0])]], dtype=np.float64)
        meta = precision_metadata(
            requested=requested_alpha,
            resolved=None,
            source="not_applicable",
            rule="not_applicable",
            effective_n=effective_n,
            weighted=weighted,
            max_kkt=0.0,
            precision_method="analytic_univariate",
        )
        return M, None, meta

    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_cpu_from_gram

    theta_z = np.zeros((p, p), dtype=np.float64)
    max_kkt = 0.0
    alpha_arr = np.asarray([resolved], dtype=np.float64)
    for j in range(p):
        cols = np.concatenate([np.arange(0, j), np.arange(j + 1, p)])
        sigma_mm = sigma_z[np.ix_(cols, cols)]
        sigma_mj = sigma_z[cols, j]
        coefs, _ = _solve_lasso_path_cpu_from_gram(
            sigma_mm * float(n),
            sigma_mj * float(n),
            n_samples=n,
            alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER,
            tol=NODEWISE_TOL,
            stopping=NODEWISE_STOPPING,
            cpu_solver=NODEWISE_SOLVER,
            lipschitz_L=None,
            cd_kkt_check_every=1,
        )
        gamma = np.asarray(coefs[0], dtype=np.float64)
        grad = sigma_mm @ gamma - sigma_mj
        active = gamma != 0.0
        kkt_vec = np.where(
            active,
            np.abs(grad + resolved * np.sign(gamma)),
            np.maximum(np.abs(grad) - resolved, 0.0),
        )
        kkt = float(np.max(kkt_vec)) if kkt_vec.size else 0.0
        if not math.isfinite(kkt) or kkt > NODEWISE_KKT_TOL:
            raise FloatingPointError("node-wise Lasso failed the independent KKT publication gate")
        max_kkt = max(max_kkt, kkt)
        sigma_jj = float(sigma_z[j, j])
        quad = sigma_jj - 2.0 * float(sigma_mj @ gamma) + float(gamma @ (sigma_mm @ gamma))
        gamma_l1 = float(np.sum(np.abs(gamma)))
        tau2 = quad + resolved * gamma_l1
        cross = sigma_jj - float(sigma_mj @ gamma)
        _check_tau_cross(tau2, cross, gamma_l1, kkt)
        theta_z[j, j] = 1.0 / tau2
        theta_z[j, cols] = -gamma / tau2

    M = inv_d[:, None] * theta_z * inv_d[None, :]
    if not np.all(np.isfinite(M)):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    meta = precision_metadata(
        requested=requested_alpha,
        resolved=resolved,
        source=source,
        rule=rule,
        effective_n=effective_n,
        weighted=weighted,
        max_kkt=max_kkt,
    )
    return M, resolved, meta


def _cupy_scales(sigma):
    import cupy as cp

    diag = cp.diag(sigma)
    if not bool(cp.all(cp.isfinite(diag)).item()) or bool(cp.any(diag < 0.0).item()):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = cp.sqrt(diag)
    d_max = float(cp.max(d).item())
    if not math.isfinite(d_max) or d_max <= 0.0:
        raise FloatingPointError("node-wise design has no positive-scale feature")
    d_tol = 64.0 * _EPS64 * d_max
    if bool(cp.any(~cp.isfinite(d)).item()) or bool(cp.any(d <= d_tol).item()):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv_d = 1.0 / d
    sigma_z = sigma * inv_d[:, None] * inv_d[None, :]
    if not bool(cp.all(cp.isfinite(sigma_z)).item()):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv_d, sigma_z


def build_nodewise_precision_cupy(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    """Build precision with batched Gram FISTA on the concrete CuPy device."""
    import cupy as cp
    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_gpu_fista_multi_fold_from_gram

    X = cp.asarray(X_work, dtype=cp.float64)
    if X.ndim != 2 or int(X.shape[0]) <= 0 or int(X.shape[1]) <= 0 or not bool(cp.all(cp.isfinite(X)).item()):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = map(int, X.shape)
    sigma = X.T @ X / float(n)
    d, inv_d, sigma_z = _cupy_scales(sigma)
    resolved, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = cp.asarray([[1.0]], dtype=X.dtype) * (inv_d[0] * inv_d[0])
        meta = precision_metadata(
            requested=requested_alpha,
            resolved=None,
            source="not_applicable",
            rule="not_applicable",
            effective_n=effective_n,
            weighted=weighted,
            max_kkt=0.0,
            precision_method="analytic_univariate",
        )
        return M, None, meta

    theta_z = cp.zeros((p, p), dtype=X.dtype)
    eig_max = float(cp.linalg.eigvalsh(sigma_z)[-1].item())
    L_global = max(eig_max, 1e-12)
    try:
        free_mem, _ = cp.cuda.Device(int(X.device.id)).mem_info
        bytes_per_fold = max(1, (p - 1) * (p - 1) * 8 * 4)
        chunk_size = int(max(1, min(p, free_mem * 0.6 // bytes_per_fold)))
    except Exception:
        chunk_size = min(p, 16)
    chunk_size = max(1, min(p, chunk_size))
    max_kkt = 0.0
    alpha_arr = np.asarray([resolved], dtype=np.float64)
    for j0 in range(0, p, chunk_size):
        j1 = min(p, j0 + chunk_size)
        j_batch = cp.arange(j0, j1, dtype=cp.int32)
        bsz = int(j_batch.size)
        base = cp.arange(p - 1, dtype=cp.int32).reshape(1, -1)
        cols = base + (base >= j_batch.reshape(-1, 1))
        sigma_mm = sigma_z[cols[:, :, None], cols[:, None, :]]
        sigma_mj = sigma_z[cols, j_batch.reshape(-1, 1)]
        coefs, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
            sigma_mm * float(n),
            sigma_mj * float(n),
            n_samples_vec=np.full((bsz,), float(n), dtype=np.float64),
            alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER,
            tol=NODEWISE_TOL,
            stopping=NODEWISE_STOPPING,
            lipschitz_L=L_global,
            check_every=8,
        )
        gamma = cp.asarray(coefs[:, 0, :], dtype=X.dtype)
        grad = cp.matmul(sigma_mm, gamma[:, :, None]).reshape(bsz, p - 1) - sigma_mj
        active = gamma != 0.0
        kkt_vec = cp.where(
            active,
            cp.abs(grad + resolved * cp.sign(gamma)),
            cp.maximum(cp.abs(grad) - resolved, 0.0),
        )
        kkt_rows = cp.max(kkt_vec, axis=1)
        batch_max = float(cp.max(kkt_rows).item())
        if not math.isfinite(batch_max) or batch_max > NODEWISE_KKT_TOL:
            raise FloatingPointError("node-wise Lasso failed the independent KKT publication gate")
        max_kkt = max(max_kkt, batch_max)
        sigma_jj = sigma_z[j_batch, j_batch]
        sg = cp.sum(sigma_mj * gamma, axis=1)
        quad = sigma_jj - 2.0 * sg + cp.sum(gamma * cp.matmul(sigma_mm, gamma[:, :, None]).reshape(bsz, p - 1), axis=1)
        gamma_l1 = cp.sum(cp.abs(gamma), axis=1)
        tau2 = quad + resolved * gamma_l1
        cross = sigma_jj - sg
        bound = gamma_l1 * kkt_rows + 64.0 * _EPS64 * cp.maximum(1.0, cp.abs(tau2))
        if bool(cp.any(~cp.isfinite(tau2)).item()) or bool(cp.any(tau2 <= _TAU_MIN).item()):
            raise FloatingPointError("node-wise normalizer is non-finite or numerically degenerate")
        if bool(cp.any(~cp.isfinite(cross)).item()) or bool(cp.any(cp.abs(cross - tau2) > bound).item()):
            raise FloatingPointError("node-wise normalizer is inconsistent with the validated KKT system")
        inv_tau = 1.0 / tau2
        theta_z[j_batch, j_batch] = inv_tau
        theta_z[j_batch[:, None], cols] = -gamma * inv_tau[:, None]

    M = theta_z * inv_d[:, None] * inv_d[None, :]
    if not bool(cp.all(cp.isfinite(M)).item()):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    meta = precision_metadata(
        requested=requested_alpha,
        resolved=resolved,
        source=source,
        rule=rule,
        effective_n=effective_n,
        weighted=weighted,
        max_kkt=max_kkt,
    )
    return M, resolved, meta


def _torch_scales(sigma):
    import torch

    diag = torch.diag(sigma)
    if not bool(torch.all(torch.isfinite(diag)).item()) or bool(torch.any(diag < 0.0).item()):
        raise FloatingPointError("node-wise design Gram diagonal is invalid")
    d = torch.sqrt(diag)
    d_max = float(torch.max(d).item())
    if not math.isfinite(d_max) or d_max <= 0.0:
        raise FloatingPointError("node-wise design has no positive-scale feature")
    d_tol = 64.0 * _EPS64 * d_max
    if bool(torch.any(~torch.isfinite(d)).item()) or bool(torch.any(d <= d_tol).item()):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv_d = 1.0 / d
    sigma_z = sigma * inv_d[:, None] * inv_d[None, :]
    if not bool(torch.all(torch.isfinite(sigma_z)).item()):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv_d, sigma_z


def build_nodewise_precision_torch(X_work, *, requested_alpha, effective_n: float, weighted: bool):
    """Build precision with batched Gram FISTA on the concrete Torch device."""
    import torch
    from statgpu.linear_model.wrappers._lasso import _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch

    if not isinstance(X_work, torch.Tensor):
        raise TypeError("Torch node-wise precision requires a torch.Tensor working design")
    X = X_work.to(dtype=torch.float64)
    if X.ndim != 2 or int(X.shape[0]) <= 0 or int(X.shape[1]) <= 0 or not bool(torch.all(torch.isfinite(X)).item()):
        raise ValueError("node-wise working design must be a finite non-empty 2D array")
    n, p = map(int, X.shape)
    sigma = X.T @ X / float(n)
    d, inv_d, sigma_z = _torch_scales(sigma)
    resolved, source, rule = resolve_nodewise_alpha(requested_alpha, p=p, effective_n=effective_n)
    if p == 1:
        M = torch.ones((1, 1), dtype=X.dtype, device=X.device) * (inv_d[0] * inv_d[0])
        meta = precision_metadata(
            requested=requested_alpha,
            resolved=None,
            source="not_applicable",
            rule="not_applicable",
            effective_n=effective_n,
            weighted=weighted,
            max_kkt=0.0,
            precision_method="analytic_univariate",
        )
        return M, None, meta

    theta_z = torch.zeros((p, p), dtype=X.dtype, device=X.device)
    eig_max = float(torch.linalg.eigvalsh(sigma_z)[-1].item())
    L_global = max(eig_max, 1e-12)
    try:
        free_mem = torch.cuda.mem_get_info(X.device)[0] if X.is_cuda else 0
        bytes_per_fold = max(1, (p - 1) * (p - 1) * 8 * 4)
        chunk_size = int(max(1, min(p, free_mem * 0.6 // bytes_per_fold))) if free_mem else min(p, 16)
    except Exception:
        chunk_size = min(p, 16)
    chunk_size = max(1, min(p, chunk_size))
    max_kkt = 0.0
    alpha_arr = np.asarray([resolved], dtype=np.float64)
    for j0 in range(0, p, chunk_size):
        j1 = min(p, j0 + chunk_size)
        j_batch = torch.arange(j0, j1, dtype=torch.long, device=X.device)
        bsz = int(j_batch.numel())
        base = torch.arange(p - 1, dtype=torch.long, device=X.device).reshape(1, -1)
        cols = base + (base >= j_batch.reshape(-1, 1)).to(dtype=torch.long)
        sigma_mm = sigma_z[cols[:, :, None], cols[:, None, :]]
        sigma_mj = sigma_z[cols, j_batch.reshape(-1, 1)]
        coefs, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
            sigma_mm * float(n),
            sigma_mj * float(n),
            n_samples_vec=np.full((bsz,), float(n), dtype=np.float64),
            alphas_desc=alpha_arr,
            max_iter=NODEWISE_MAX_ITER,
            tol=NODEWISE_TOL,
            stopping=NODEWISE_STOPPING,
            lipschitz_L=L_global,
            check_every=8,
        )
        gamma = coefs[:, 0, :].to(dtype=X.dtype, device=X.device)
        grad = torch.matmul(sigma_mm, gamma.unsqueeze(-1)).reshape(bsz, p - 1) - sigma_mj
        active = gamma != 0.0
        zero = torch.zeros((), dtype=X.dtype, device=X.device)
        kkt_vec = torch.where(
            active,
            torch.abs(grad + resolved * torch.sign(gamma)),
            torch.maximum(torch.abs(grad) - resolved, zero),
        )
        kkt_rows = torch.max(kkt_vec, dim=1).values
        batch_max = float(torch.max(kkt_rows).item())
        if not math.isfinite(batch_max) or batch_max > NODEWISE_KKT_TOL:
            raise FloatingPointError("node-wise Lasso failed the independent KKT publication gate")
        max_kkt = max(max_kkt, batch_max)
        sigma_jj = sigma_z[j_batch, j_batch]
        sg = torch.sum(sigma_mj * gamma, dim=1)
        mm_gamma = torch.matmul(sigma_mm, gamma.unsqueeze(-1)).reshape(bsz, p - 1)
        quad = sigma_jj - 2.0 * sg + torch.sum(gamma * mm_gamma, dim=1)
        gamma_l1 = torch.sum(torch.abs(gamma), dim=1)
        tau2 = quad + resolved * gamma_l1
        cross = sigma_jj - sg
        bound = gamma_l1 * kkt_rows + 64.0 * _EPS64 * torch.maximum(torch.ones_like(tau2), torch.abs(tau2))
        if bool(torch.any(~torch.isfinite(tau2)).item()) or bool(torch.any(tau2 <= _TAU_MIN).item()):
            raise FloatingPointError("node-wise normalizer is non-finite or numerically degenerate")
        if bool(torch.any(~torch.isfinite(cross)).item()) or bool(torch.any(torch.abs(cross - tau2) > bound).item()):
            raise FloatingPointError("node-wise normalizer is inconsistent with the validated KKT system")
        inv_tau = 1.0 / tau2
        theta_z[j_batch, j_batch] = inv_tau
        theta_z[j_batch[:, None], cols] = -gamma * inv_tau[:, None]

    M = theta_z * inv_d[:, None] * inv_d[None, :]
    if not bool(torch.all(torch.isfinite(M)).item()):
        raise FloatingPointError("node-wise back-transformed precision matrix is non-finite")
    meta = precision_metadata(
        requested=requested_alpha,
        resolved=resolved,
        source=source,
        rule=rule,
        effective_n=effective_n,
        weighted=weighted,
        max_kkt=max_kkt,
    )
    return M, resolved, meta


__all__ = [
    "NODEWISE_SOLVER",
    "NODEWISE_STOPPING",
    "NODEWISE_TOL",
    "NODEWISE_MAX_ITER",
    "NODEWISE_KKT_TOL",
    "NODEWISE_CONTRACT_VERSION",
    "validate_nodewise_alpha",
    "resolve_effective_n",
    "resolve_effective_n_backend",
    "resolve_nodewise_alpha",
    "precision_metadata",
    "build_nodewise_precision_numpy",
    "build_nodewise_precision_cupy",
    "build_nodewise_precision_torch",
]
