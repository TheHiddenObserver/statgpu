"""Breslow Hessian computation via PyTorch GPU operations (cuBLAS + vectorized).

Originally attempted a Triton serial-scan kernel, but Triton 2.0 has a compiler
bug that produces non-deterministic wrong code for kernels with runtime-bounded
loops (while/for with >= 3 iterations). The PyTorch approach is only marginally
slower since each GPU operation (matmul, outer) is highly optimized by cuBLAS.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

# Supported padded feature dimensions (next power of 2)
_SUPPORTED_P: Tuple[int, ...] = (8, 16, 32, 64, 128)


def _find_p_ce(p: int) -> Optional[int]:
    """Find the smallest supported padded feature dimension >= p."""
    for sp in _SUPPORTED_P:
        if sp >= p:
            return sp
    return None


def compute_breslow_grad_hess_triton(
    X: Any,
    beta: Any,
    time: Any,
    event: Any,
) -> Optional[Tuple[Any, Any]]:
    """Compute Breslow gradient/Hessian via PyTorch GPU operations.

    Uses the same algorithm as _cox.py Breslow path: vectorized gradient,
    then serial Python loop over unique failure times with async PyTorch
    GPU operations for the Hessian.
    """
    if not isinstance(X, torch.Tensor) or not isinstance(beta, torch.Tensor):
        return None
    if not X.is_cuda or not beta.is_cuda:
        return None

    p = int(X.shape[1])
    p_ce = _find_p_ce(p)
    if p_ce is None:
        return None

    n = int(X.shape[0])
    device = X.device

    # Linear predictor
    eta = X @ beta
    exp_eta = torch.exp(eta)
    X_exp = X * exp_eta[:, None]

    event_mask = (event == 1)
    if not torch.any(event_mask):
        return (
            torch.zeros(p, dtype=torch.float64, device=device),
            torch.zeros((p, p), dtype=torch.float64, device=device),
        )

    # Reverse cumsum for risk sets
    rev_idx = torch.arange(n - 1, -1, -1, device=device)
    risk_sum = torch.cumsum(exp_eta[rev_idx], dim=0)[rev_idx]
    risk_X_sum = torch.cumsum((X * exp_eta[:, None])[rev_idx], dim=0)[rev_idx]

    # Unique failure times
    event_times = time[event_mask]
    uft, unique_inv = torch.unique(event_times, sorted=True, return_inverse=True)
    n_uft = len(uft)
    counts = torch.bincount(unique_inv).to(torch.float64)

    sorted_times, sort_idx = torch.sort(time)
    first_in_sorted = torch.searchsorted(sorted_times, uft, side="left")
    first_idx = sort_idx[first_in_sorted]

    # Precompute risk values at unique times
    risk_at_uft = risk_sum[first_idx]
    risk_X_at_uft = risk_X_sum[first_idx]
    E_X_at_uft = risk_X_at_uft / risk_at_uft[:, None]

    # Sum X for events at each unique time
    event_indices = event_mask.nonzero(as_tuple=True)[0]
    sum_X_per_uft = torch.zeros((n_uft, p), dtype=torch.float64, device=device)
    sum_X_per_uft.index_add_(0, unique_inv, X[event_indices])

    # Gradient: Breslow closed-form
    grad = torch.sum(sum_X_per_uft - counts[:, None] * E_X_at_uft, dim=0)

    # Hessian: PyTorch GPU operations (same algorithm as _cox.py)
    risk_X2 = X_exp.T @ X
    hess = torch.zeros((p, p), dtype=torch.float64, device=device)
    pidx = 0
    for g in range(n_uft):
        idx = int(first_idx[g].item())
        if idx > pidx:
            risk_X2 -= X_exp[pidx:idx].T @ X[pidx:idx]
            pidx = idx
        rs = risk_at_uft[g]
        w = counts[g]
        ex = E_X_at_uft[g]
        hess -= risk_X2 * (w / rs)
        hess += torch.outer(ex, ex) * w

    return grad, hess
