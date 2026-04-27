"""
Triton JIT kernel for Cox PH Efron backward gradient/Hessian.

Mirrors the algorithm in `_cox_efron_cuda.py` (CuPy RawKernel serial version).

Design:
- Single Triton program (grid=(1,)) executes the entire backward scan.
- P (feature dim) is constexpr, enabling loop unrolling for small p.
- Local scalar accumulators where possible; workspace tensor for p*p matrices.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np


def _import_triton():
    """Deferred Triton import."""
    try:
        import triton
        import triton.language as tl
        return triton, tl
    except ImportError:
        return None, None


_triton, _tl = _import_triton()
HAS_TRITON_EFRON: bool = False
HAS_TRITON_BRESLOW: bool = False

if _triton is not None and _tl is not None:
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _efron_backward_scan_serial(
            # Input tensors
            X_ptr,            # [n, p] float64
            e_eta_ptr,        # [n] float64
            enter_ptr_ptr,    # [nuft+1] int32
            enter_ind_ptr,    # [n_enter_total] int32
            exit_ptr_ptr,     # [nuft+1] int32
            exit_ind_ptr,     # [n_exit_total] int32
            fail_ptr_ptr,     # [nuft+1] int32
            fail_ind_ptr,     # [n_fail_total] int32
            # Workspace (caller-allocated, zeroed)
            ws_ptr,           # [workspace_size] float64
            # Output (caller-allocated, zeroed)
            grad_ptr,         # [p] float64
            hess_ptr,         # [p*p] float64
            # Parameters
            n,
            p,
            nuft,
            # Compile-time constants
            P: tl.constexpr,
        ):
            """Single-program serial Efron backward scan kernel."""

            # Workspace layout (all offsets relative to ws_ptr):
            WS_XP0    = 0
            WS_XP1    = 1
            WS_XP2    = 1 + P
            WS_HESS   = 1 + P + P * P
            WS_XP1F   = 1 + 2 * P * P
            WS_XP2F   = 1 + 2 * P * P + P
            WS_SCRATCH = 1 + 3 * P * P + P
            WS_SIZE   = 1 + 3 * P * P + P + 1

            # ws_ptr is already zeroed by caller.

            # ---- Backward scan ----
            for ii in range(nuft - 1, -1, -1):
                # ---- Enter phase ----
                e0 = tl.load(enter_ptr_ptr + ii)
                e1 = tl.load(enter_ptr_ptr + ii + 1)
                nt = e1 - e0

                if nt > 0:
                    for t in range(0, nt, 1):
                        idx = tl.load(enter_ind_ptr + e0 + t)
                        row_off = idx * p
                        elx = tl.load(e_eta_ptr + idx)

                        # xp0 += elx
                        old = tl.load(ws_ptr + WS_XP0)
                        tl.store(ws_ptr + WS_XP0, old + elx)

                        # xp1[j] += elx * X[idx,j]
                        for j in range(0, P, 1):
                            if j < p:
                                xval = tl.load(X_ptr + row_off + j)
                                old = tl.load(ws_ptr + WS_XP1 + j)
                                tl.store(ws_ptr + WS_XP1 + j, old + elx * xval)

                        # xp2[j*P+k] += elx * X[idx,j] * X[idx,k]
                        for j in range(0, P, 1):
                            if j < p:
                                vj = tl.load(X_ptr + row_off + j)
                                for k in range(0, P, 1):
                                    if k < p:
                                        vk = tl.load(X_ptr + row_off + k)
                                        old = tl.load(ws_ptr + WS_XP2 + j * P + k)
                                        tl.store(ws_ptr + WS_XP2 + j * P + k, old + elx * vj * vk)

                # ---- Fail phase ----
                f0 = tl.load(fail_ptr_ptr + ii)
                f1 = tl.load(fail_ptr_ptr + ii + 1)
                m = f1 - f0

                if m > 0:
                    # Zero xp1f and xp2f in workspace
                    for j in range(0, P, 1):
                        if j < p:
                            tl.store(ws_ptr + WS_XP1F + j, 0.0)
                    for j in range(0, P, 1):
                        if j < p:
                            for k in range(0, P, 1):
                                if k < p:
                                    tl.store(ws_ptr + WS_XP2F + j * P + k, 0.0)

                    # Accumulate fail sums into xp1f, xp2f, xp0f
                    xp0f_acc = 0.0
                    for t in range(0, m, 1):
                        idx = tl.load(fail_ind_ptr + f0 + t)
                        row_off = idx * p
                        elx = tl.load(e_eta_ptr + idx)
                        xp0f_acc = xp0f_acc + elx

                        # grad[j] += X[idx,j]
                        for j in range(0, P, 1):
                            if j < p:
                                vj = tl.load(X_ptr + row_off + j)
                                old = tl.load(grad_ptr + j)
                                tl.store(grad_ptr + j, old + vj)

                        # xp1f[j] += elx * X[idx,j]
                        for j in range(0, P, 1):
                            if j < p:
                                vj = tl.load(X_ptr + row_off + j)
                                old = tl.load(ws_ptr + WS_XP1F + j)
                                tl.store(ws_ptr + WS_XP1F + j, old + elx * vj)

                        # xp2f[j*P+k] += elx * X[idx,j] * X[idx,k]
                        for j in range(0, P, 1):
                            if j < p:
                                vj = tl.load(X_ptr + row_off + j)
                                for k in range(0, P, 1):
                                    if k < p:
                                        vk = tl.load(X_ptr + row_off + k)
                                        old = tl.load(ws_ptr + WS_XP2F + j * P + k)
                                        tl.store(ws_ptr + WS_XP2F + j * P + k, old + elx * vj * vk)

                    # Efron correction (serial)
                    xp0v = tl.load(ws_ptr + WS_XP0)
                    sum_inv_c0 = 0.0
                    sum_J_c0 = 0.0
                    sum_aa = 0.0
                    sum_bb = 0.0
                    sum_ab = 0.0
                    for kk in range(0, m, 1):
                        Jk = (kk * 1.0) / (m * 1.0)
                        c0 = xp0v - Jk * xp0f_acc
                        if c0 < 1e-300:
                            c0 = 1e-300
                        ak = 1.0 / c0
                        bk = Jk * ak
                        sum_inv_c0 = sum_inv_c0 + ak
                        sum_J_c0 = sum_J_c0 + Jk / c0
                        sum_aa = sum_aa + ak * ak
                        sum_bb = sum_bb + bk * bk
                        sum_ab = sum_ab + ak * bk

                    # Apply to grad
                    for j in range(0, P, 1):
                        if j < p:
                            xp1j = tl.load(ws_ptr + WS_XP1 + j)
                            xp1fj = tl.load(ws_ptr + WS_XP1F + j)
                            old = tl.load(grad_ptr + j)
                            tl.store(grad_ptr + j, old - (xp1j * sum_inv_c0 - xp1fj * sum_J_c0))

                    # Apply to hess
                    for j in range(0, P, 1):
                        if j < p:
                            for k in range(0, P, 1):
                                if k < p:
                                    xp2jk = tl.load(ws_ptr + WS_XP2 + j * P + k)
                                    xp2fjk = tl.load(ws_ptr + WS_XP2F + j * P + k)
                                    hess_val = xp2jk * sum_inv_c0 - xp2fjk * sum_J_c0

                                    xp1j_v = tl.load(ws_ptr + WS_XP1 + j)
                                    xp1k_v = tl.load(ws_ptr + WS_XP1 + k)
                                    xp1fj_v = tl.load(ws_ptr + WS_XP1F + j)
                                    xp1fk_v = tl.load(ws_ptr + WS_XP1F + k)
                                    o11 = xp1j_v * xp1k_v
                                    off_v = xp1fj_v * xp1fk_v
                                    cross_v = xp1j_v * xp1fk_v + xp1fj_v * xp1k_v
                                    hsub = sum_aa * o11 + sum_bb * off_v - sum_ab * cross_v
                                    hess_val = hess_val - hsub

                                    idx2 = j * P + k
                                    old = tl.load(hess_ptr + idx2)
                                    tl.store(hess_ptr + idx2, hess_val + old)

                # ---- Exit phase ----
                x0 = tl.load(exit_ptr_ptr + ii)
                x1 = tl.load(exit_ptr_ptr + ii + 1)
                nx = x1 - x0

                if nx > 0:
                    for t in range(0, nx, 1):
                        idx = tl.load(exit_ind_ptr + x0 + t)
                        row_off = idx * p
                        elx = tl.load(e_eta_ptr + idx)

                        # xp0 -= elx
                        old = tl.load(ws_ptr + WS_XP0)
                        tl.store(ws_ptr + WS_XP0, old - elx)

                        # xp1[j] -= elx * X[idx,j]
                        for j in range(0, P, 1):
                            if j < p:
                                xval = tl.load(X_ptr + row_off + j)
                                old = tl.load(ws_ptr + WS_XP1 + j)
                                tl.store(ws_ptr + WS_XP1 + j, old - elx * xval)

                        # xp2 -= elx * X^T X
                        for j in range(0, P, 1):
                            if j < p:
                                vj = tl.load(X_ptr + row_off + j)
                                for k in range(0, P, 1):
                                    if k < p:
                                        vk = tl.load(X_ptr + row_off + k)
                                        old = tl.load(ws_ptr + WS_XP2 + j * P + k)
                                        tl.store(ws_ptr + WS_XP2 + j * P + k, old - elx * vj * vk)

        HAS_TRITON_EFRON = True

    except Exception:
        HAS_TRITON_EFRON = False
        _triton = None
        _tl = None

    # =====================================================================
    # Breslow Hessian — PyTorch GPU path (cuBLAS matmul + vectorized ops)
    # =====================================================================
    # Originally attempted a Triton serial-scan kernel, but Triton 2.0 has a
    # compiler bug producing non-deterministic wrong code for kernels with
    # runtime-bounded loops (while/for with >= 3 iterations). The PyTorch
    # approach is only marginally slower since each op is cuBLAS-optimized.
    try:
        from statgpu.survival._cox_breslow_triton_kernel import (
            compute_breslow_grad_hess_triton,
            _find_p_ce as _find_p_ce_breslow,
        )
        HAS_TRITON_BRESLOW = True
    except Exception:
        compute_breslow_grad_hess_triton = None
        HAS_TRITON_BRESLOW = False


def _triton_available() -> bool:
    return HAS_TRITON_EFRON


_SUPPORTED_P: Tuple[int, ...] = (8, 16, 32, 64, 128)


def _find_p_ce(p: int) -> Optional[int]:
    for sp in _SUPPORTED_P:
        if sp >= p:
            return sp
    return None


def compute_efron_grad_hess_triton(
    X: Any,
    beta: Any,
    efron_pre: Any,
) -> Optional[Tuple[Any, Any]]:
    """Compute Efron gradient/Hessian via Triton serial kernel."""
    if not HAS_TRITON_EFRON:
        return None

    import torch
    from statgpu.survival._cox_efron_cuda import (
        efron_indices_to_csr,
        _pick_backward_launch_params,
    )

    if len(efron_pre) == 6:
        _, uft_ix, risk_enter, risk_exit, nuft, _ = efron_pre
    else:
        _, uft_ix, risk_enter, risk_exit, nuft = efron_pre

    p = int(X.shape[1])
    p_ce = _find_p_ce(p)
    if p_ce is None:
        return None

    if nuft == 0:
        return (
            torch.zeros(p, dtype=torch.float64, device=X.device),
            torch.zeros((p, p), dtype=torch.float64, device=X.device),
        )

    n = int(X.shape[0])
    device = X.device

    # Build linear predictor
    linpred = X @ beta
    linpred = linpred - torch.max(linpred)
    e_eta = torch.exp(linpred)

    # Build CSR
    enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind = efron_indices_to_csr(
        uft_ix, risk_enter, risk_exit, nuft
    )

    enter_ptr_t = torch.as_tensor(enter_ptr, dtype=torch.int32, device=device)
    enter_ind_t = torch.as_tensor(enter_ind, dtype=torch.int32, device=device)
    exit_ptr_t = torch.as_tensor(exit_ptr, dtype=torch.int32, device=device)
    exit_ind_t = torch.as_tensor(exit_ind, dtype=torch.int32, device=device)
    fail_ptr_t = torch.as_tensor(fail_ptr, dtype=torch.int32, device=device)
    fail_ind_t = torch.as_tensor(fail_ind, dtype=torch.int32, device=device)

    seq_thresh, _ = _pick_backward_launch_params(p, nuft, n)

    # Workspace: WS_XP0(1) + WS_XP1(P) + WS_XP2(P*P) + WS_HESS(P*P) +
    #             WS_XP1F(P) + WS_XP2F(P*P) + WS_SCRATCH(1)
    ws_size = 1 + 3 * p_ce + 3 * p_ce * p_ce + 1
    ws = torch.zeros(ws_size, dtype=torch.float64, device=device)
    grad_out = torch.zeros(p, dtype=torch.float64, device=device)
    # Allocate hess_out with padded stride (p_ce) to match Triton kernel indexing
    hess_out = torch.zeros(p_ce * p_ce, dtype=torch.float64, device=device)

    try:
        _efron_backward_scan_serial[(1,)](
            X, e_eta,
            enter_ptr_t, enter_ind_t,
            exit_ptr_t, exit_ind_t,
            fail_ptr_t, fail_ind_t,
            ws, grad_out, hess_out,
            n, p, nuft,
            P=p_ce,
        )
        torch.cuda.synchronize()
    except Exception:
        return None

    return grad_out, -hess_out.view(p_ce, p_ce)[:p, :p]


# compute_breslow_grad_hess_triton and _find_p_ce are imported from
# _cox_breslow_triton_kernel.py above (in the try/except block).
