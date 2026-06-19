"""Breslow gradient/Hessian computation via PyTorch GPU operations.

Fully vectorized: no Python loops over failure groups.
Uses prefix sums of X_exp^T @ X to compute risk_X2 at all failure times in one shot.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

_SUPPORTED_P: Tuple[int, ...] = (8, 16, 32, 64, 128)


def _find_p_ce(p: int) -> Optional[int]:
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
    """Compute Breslow gradient/Hessian via PyTorch GPU — fully vectorized.

    No Python loops over failure groups. Uses prefix sums of rank-1
    outer products to compute risk_X2 at all failure times in one batch.
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
    risk_X_sum = torch.cumsum(X_exp[rev_idx], dim=0)[rev_idx]

    # Unique failure times
    event_times = time[event_mask]
    uft, unique_inv = torch.unique(event_times, sorted=True, return_inverse=True)
    n_uft = len(uft)
    counts = torch.bincount(unique_inv).to(torch.float64)

    sorted_times, sort_idx = torch.sort(time)
    first_in_sorted = torch.searchsorted(sorted_times, uft, side="left")
    first_idx = sort_idx[first_in_sorted]

    # Risk values at failure times
    risk_at_uft = risk_sum[first_idx]
    risk_X_at_uft = risk_X_sum[first_idx]
    E_X_at_uft = risk_X_at_uft / risk_at_uft[:, None]

    # Sum X for events at each unique time
    event_indices = event_mask.nonzero(as_tuple=True)[0]
    sum_X_per_uft = torch.zeros((n_uft, p), dtype=torch.float64, device=device)
    sum_X_per_uft.index_add_(0, unique_inv, X[event_indices])

    # Gradient: Breslow closed-form
    grad = torch.sum(sum_X_per_uft - counts[:, None] * E_X_at_uft, dim=0)

    # Hessian: fully vectorized using prefix sums
    # risk_X2[g] = total_X2 - prefix_X2[first_idx[g]]
    # prefix_X2[i] = sum_{k=0}^{i-1} X_exp[k]^T @ X[k]
    # Use rank-1 outer products + cumsum on flattened (n, p*p) tensor.

    # Compute all rank-1 outer products: (n, p, p) -> (n, p*p)
    outer_flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)  # (n, p*p)
    # Prefix sum: prefix_flat[i] = sum_{k=0}^{i-1} outer_flat[k]
    prefix_flat = torch.cat([
        torch.zeros(1, p * p, dtype=torch.float64, device=device),
        torch.cumsum(outer_flat[:-1], dim=0)
    ], dim=0)  # (n, p*p)

    # Gather prefix sums at failure group boundaries
    prefix_at_fidx = prefix_flat[first_idx].reshape(n_uft, p, p)  # (n_uft, p, p)
    total_X2 = prefix_flat[-1].reshape(p, p) + outer_flat[-1].reshape(p, p)

    # risk_X2[g] = total_X2 - prefix_at_fidx[g]
    risk_X2_all = total_X2.unsqueeze(0) - prefix_at_fidx  # (n_uft, p, p)

    # Vectorized Hessian:
    # H = -sum_g (counts[g] / risk_at[g]) * risk_X2[g] + sum_g counts[g] * outer(E_X[g], E_X[g])
    weights = counts / risk_at_uft  # (n_uft,)
    hess = -torch.sum(risk_X2_all * weights[:, None, None], dim=0)
    hess += torch.einsum("g,gi,gj->ij", counts, E_X_at_uft, E_X_at_uft)

    return grad, hess
