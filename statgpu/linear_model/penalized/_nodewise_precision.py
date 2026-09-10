"""Shared node-wise Lasso precision helpers for debiased Gaussian inference."""

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


def resolve_effective_n(n_rows: int, sample_weight=None) -> float:
    """Return response-independent effective sample size for auto node-wise tuning."""
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
    n_eff = total * total / sumsq
    tol = 64.0 * _EPS64 * max(1.0, float(n))
    if not math.isfinite(n_eff) or n_eff <= 0.0 or n_eff > float(n) + tol:
        raise FloatingPointError("invalid node-wise effective sample size")
    return float(min(n_eff, float(n)))


def resolve_nodewise_alpha(requested, *, p: int, effective_n: float) -> tuple[Optional[float], str, str]:
    """Resolve explicit/automatic node-wise alpha. p=1 has no nuisance solve."""
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


def design_scales_from_gram(sigma, xp):
    """Return column scales and standardized Gram for a float64 working design."""
    diag = xp.diag(sigma)
    d = xp.sqrt(diag)
    d_np = np.asarray(d.detach().cpu().numpy() if getattr(xp, "__name__", "") == "torch" else (d.get() if hasattr(d, "get") else d), dtype=np.float64)
    if d_np.ndim != 1 or d_np.size == 0 or not np.all(np.isfinite(d_np)):
        raise FloatingPointError("node-wise design scales must be finite")
    d_max = float(np.max(d_np))
    if d_max <= 0.0:
        raise FloatingPointError("node-wise design has no positive-scale feature")
    d_tol = 64.0 * _EPS64 * d_max
    if np.any(d_np <= d_tol):
        raise FloatingPointError("node-wise design contains a numerically degenerate feature column")
    inv_d = 1.0 / d
    sigma_z = sigma * inv_d[:, None] * inv_d[None, :]
    sigma_z_np = np.asarray(sigma_z.detach().cpu().numpy() if getattr(xp, "__name__", "") == "torch" else (sigma_z.get() if hasattr(sigma_z, "get") else sigma_z), dtype=np.float64)
    if not np.all(np.isfinite(sigma_z_np)):
        raise FloatingPointError("standardized node-wise Gram matrix is non-finite")
    return d, inv_d, sigma_z


def nodewise_kkt_residual(sigma_mm, sigma_mj, gamma, alpha, xp):
    """Return the exact Lasso KKT sup residual for one standardized node-wise solve."""
    grad = sigma_mm @ gamma - sigma_mj
    active = gamma != 0
    abs_grad = xp.abs(grad)
    residual = xp.where(
        active,
        xp.abs(grad + float(alpha) * xp.sign(gamma)),
        xp.maximum(abs_grad - float(alpha), 0.0),
    )
    max_resid = xp.max(residual) if int(residual.shape[0]) else 0.0
    if getattr(xp, "__name__", "") == "torch":
        return float(max_resid.detach().cpu().item()) if hasattr(max_resid, "detach") else float(max_resid)
    if hasattr(max_resid, "item"):
        return float(max_resid.item())
    return float(max_resid)


def tau2_from_standardized_gram(*, sigma_jj, sigma_mj, sigma_mm, gamma, alpha, xp):
    """Compute paper-style node-wise normalizer from standardized Gram quantities."""
    quad = sigma_jj - 2.0 * (sigma_mj @ gamma) + gamma @ (sigma_mm @ gamma)
    tau2 = quad + float(alpha) * xp.sum(xp.abs(gamma))
    value = float(tau2.detach().cpu().item()) if getattr(xp, "__name__", "") == "torch" else (float(tau2.item()) if hasattr(tau2, "item") else float(tau2))
    if not math.isfinite(value) or value <= 64.0 * _EPS64:
        raise FloatingPointError("node-wise normalizer is non-finite or numerically degenerate")
    return value


def precision_metadata(*, requested, resolved, source, rule, effective_n, weighted, max_kkt, precision_method="nodewise_lasso") -> dict[str, Any]:
    return {
        "precision_method": precision_method,
        "nodewise_alpha_requested": None if requested is None else float(requested),
        "nodewise_alpha": None if resolved is None else float(resolved),
        "nodewise_alpha_source": str(source),
        "nodewise_alpha_rule": str(rule),
        "nodewise_design_standardized": True,
        "nodewise_effective_n": float(effective_n),
        "nodewise_weighted": bool(weighted),
        "nodewise_solver": NODEWISE_SOLVER,
        "nodewise_stopping": NODEWISE_STOPPING,
        "nodewise_tol": float(NODEWISE_TOL),
        "nodewise_max_iter": int(NODEWISE_MAX_ITER),
        "nodewise_kkt_tol": float(NODEWISE_KKT_TOL),
        "nodewise_max_kkt_residual": float(max_kkt),
    }


__all__ = [
    "NODEWISE_SOLVER",
    "NODEWISE_STOPPING",
    "NODEWISE_TOL",
    "NODEWISE_MAX_ITER",
    "NODEWISE_KKT_TOL",
    "NODEWISE_CONTRACT_VERSION",
    "validate_nodewise_alpha",
    "resolve_effective_n",
    "resolve_nodewise_alpha",
    "design_scales_from_gram",
    "nodewise_kkt_residual",
    "tau2_from_standardized_gram",
    "precision_metadata",
]
