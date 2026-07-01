"""
Cox partial likelihood loss for survival analysis.

Negative log partial likelihood with Breslow/Efron tie handling.
Dispatches to GPU-optimized kernels (CuPy CUDA / PyTorch) when available;
explicit GPU inputs raise RuntimeError if GPU path is unavailable.
CPU inputs use numpy implementation.

Matches R's survival::coxph() interface.
"""

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp, _xp_zeros, _xp_asarray
from statgpu.backends._utils import _to_float_scalar, _to_numpy
from ._base import LossBase
from ._registry import register_loss


# ── Build efron_pre from sorted time/event (numpy) ──────────────────

def _build_efron_pre_numpy(time_np, event_np):
    """Build efron_pre structure as numpy arrays (for kernel dispatch)."""
    event_mask = event_np == 1
    event_idx = np.where(event_mask)[0]
    event_times = time_np[event_idx]
    uft, inv = np.unique(event_times, return_inverse=True)
    nuft = len(uft)
    uft_ix = [event_idx[inv == g].astype(np.int32) for g in range(nuft)]
    first_idx_uft = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    risk_enter = [[np.int64(np.searchsorted(time_np, t, side="left"))] for t in uft]
    risk_exit = [[np.int64(np.searchsorted(time_np, t, side="right"))] for t in uft]
    return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft


def _build_breslow_pre_numpy(time_np, event_np):
    """Build Breslow tie groups as numpy arrays."""
    event_mask = event_np == 1
    event_times = time_np[event_mask]
    uft, _, counts = np.unique(event_times, return_inverse=True, return_counts=True)
    first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int32)
    return first_idx, counts.astype(np.float64)


@register_loss('cox_ph')
class CoxPartialLikelihoodLoss(LossBase):
    """Cox proportional hazards negative log partial likelihood.

    Dispatches to GPU-optimized kernels when input is CuPy/Torch-CUDA.
    CPU inputs use numpy implementation; explicit GPU inputs raise
    RuntimeError if GPU path is unavailable.

    Note: This loss does NOT support ``sample_weight``. All methods raise
    ``NotImplementedError`` if ``sample_weight is not None``.

    Note: ``preprocess()`` returns ``(X_sorted, zeros)`` — the second element
    is a placeholder (not ``y``). The loss sorts data by time and precomputes
    risk-set structures; ``value()``/``gradient()`` use ``_ensure_sorted()``
    internally.

    Parameters
    ----------
    ties : str, default='breslow'
        Method for handling ties: 'breslow' or 'efron'.
    """

    name = "cox_ph"
    y_type = "survival"
    smooth_gradient = True
    has_hessian = True

    _lipschitz_safety = 1.0
    _has_constant_hessian = False

    def __init__(self, ties: str = 'breslow'):
        ties = ties.lower()
        if ties not in ('breslow', 'efron'):
            raise ValueError("ties must be 'breslow' or 'efron'")
        self.ties = ties

        self._sorted = False
        self._X_sorted = None
        self._time_sorted = None
        self._event_sorted = None
        self._order = None
        self._time_np = None
        self._event_np = None
        self._efron_pre_np = None
        self._breslow_pre_np = None
        self._efron_csr = None
        self._n_events = 0

    def _ensure_sorted(self, X, y):
        """Ensure data is preprocessed. Call at start of every public method."""
        if self._sorted and X is self._X_sorted:
            return
        self._sorted = False
        self.preprocess(X, y)

    def preprocess(self, X, y):
        """Sort data by time and precompute risk-set structures."""
        xp = _get_xp(X)

        if isinstance(y, dict):
            time = _xp_asarray(y['time'], dtype=xp.float64, ref_arr=X)
            event = _xp_asarray(y['event'], dtype=xp.float64, ref_arr=X)
        else:
            y_arr = _xp_asarray(y, dtype=xp.float64, ref_arr=X)
            if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
                time, event = y_arr[:, 0], y_arr[:, 1]
            else:
                raise ValueError("y must be dict or (n, 2) array")

        X_arr = _xp_asarray(X, dtype=xp.float64, ref_arr=X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        order = xp.argsort(time, stable=True) if xp.__name__ == "torch" else xp.argsort(time)
        self._X_sorted = X_arr[order]
        self._time_sorted = time[order]
        self._event_sorted = event[order]
        self._order = order
        self._sorted = True
        self._n_events = int(_to_float_scalar(xp.sum(self._event_sorted)))

        # Numpy copies for kernel dispatch
        time_np = _to_numpy(self._time_sorted).astype(np.float64)
        event_np = _to_numpy(self._event_sorted).astype(np.float64)
        self._time_np = time_np
        self._event_np = event_np

        if self.ties == 'efron':
            self._efron_pre_np = _build_efron_pre_numpy(time_np, event_np)
            self._breslow_pre_np = None
            try:
                from statgpu.survival._cox_efron_cuda import efron_indices_to_csr
                _, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = self._efron_pre_np
                csr6 = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
                # Pack as 8-tuple for compute_efron_grad_hess_raw
                self._efron_csr = csr6 + (first_idx_uft.astype(np.int32), int(nuft))
            except ImportError:
                self._efron_csr = None
        else:
            self._breslow_pre_np = _build_breslow_pre_numpy(time_np, event_np)
            self._efron_pre_np = None
            self._efron_csr = None

        return self._X_sorted, _xp_zeros(X_arr.shape[0], dtype=xp.float64, ref_arr=X_arr)

    # ── Public API ───────────────────────────────────────────────────

    def value(self, X, y, coef, sample_weight=None) -> float:
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]

        # GPU path: both Efron and Breslow
        is_gpu = xp.__name__ in ("cupy",) or (xp.__name__ == "torch" and X_s.is_cuda)
        if is_gpu:
            loglik = self._gpu_loglik(coef_dev, X_s)
            if loglik is not None:
                return -_to_float_scalar(loglik) / n
            raise RuntimeError(
                "CoxPH GPU loglik path failed. "
                "Explicit GPU devices do not fall back to CPU. "
                "Use device='cpu' to run the CPU implementation."
            )

        # CPU path (numpy)
        eta = X_s @ coef_dev
        loglik = self._cpu_loglik(_to_numpy(eta), self._time_np, self._event_np)
        return -loglik / n

    def gradient(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]
        grad, _ = self._compute_grad_hess(coef_dev, X_s)
        return -grad / n

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]

        xp = _get_xp(X_s)
        is_gpu = xp.__name__ == "cupy" or (xp.__name__ == "torch" and X_s.is_cuda)

        if is_gpu:
            # GPU path: _loglik_from_eta raises if GPU path unavailable
            grad, _ = self._compute_grad_hess(coef_dev, X_s)
            eta = X_s @ coef_dev
            loglik = self._loglik_from_eta(eta, X_s)
            return -_to_float_scalar(loglik) / n, -grad / n

        # CPU path: fused loglik + gradient + hessian in one pass
        X_np = _to_numpy(X_s)
        eta_np = _to_numpy(X_s @ coef_dev)
        if self.ties == 'efron' and self._efron_pre_np is not None:
            loglik, grad_np, hess_np = self._cpu_fused_loglik_grad_hess(eta_np, X_np, self._time_np, self._event_np)
            # Cache hessian for Newton solver
            if hess_np is not None:
                self._cached_hess = hess_np
            return -loglik / n, _xp_asarray(-grad_np / n, dtype=xp.float64, ref_arr=X_s)
        else:
            loglik = self._cpu_loglik(eta_np, self._time_np, self._event_np)
            grad_np, _ = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
            return -loglik / n, _xp_asarray(-grad_np / n, dtype=xp.float64, ref_arr=X_s)

    def hessian(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        n = X_s.shape[0]

        # Use cached Hessian from fused_value_and_gradient if available
        if hasattr(self, '_cached_hess') and self._cached_hess is not None:
            hess = self._cached_hess
            self._cached_hess = None  # Clear cache
            return _xp_asarray(-hess / n, dtype=xp.float64, ref_arr=X_s)

        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        _, hess = self._compute_grad_hess(coef_dev, X_s)
        return -hess / n

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        from statgpu.backends._array_ops import _max_eigval_power
        self._ensure_sorted(X, y)
        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s) if coef is not None else _xp_zeros(X_s.shape[1], dtype=xp.float64, ref_arr=X_s)
        _, hess = self._compute_grad_hess(coef_dev, X_s)
        return _max_eigval_power(-hess / X_s.shape[0])

    # ── GPU dispatch ─────────────────────────────────────────────────

    def _is_gpu(self, arr):
        xp = _get_xp(arr)
        return xp.__name__ == "cupy" or (xp.__name__ == "torch" and arr.is_cuda)

    def _compute_grad_hess(self, coef_dev, X_s):
        """Compute gradient and Hessian, dispatching to GPU kernel if available."""
        xp = _get_xp(X_s)
        is_cupy = xp.__name__ == "cupy"
        is_torch_cuda = xp.__name__ == "torch" and X_s.is_cuda

        # Efron: try CuPy kernel (works for both cupy and torch-CUDA via DLPack)
        if (is_cupy or is_torch_cuda) and self.ties == 'efron':
            if is_torch_cuda:
                import cupy as cp
                import torch
                X_cp = cp.from_dlpack(X_s.__dlpack__())
                coef_cp = cp.from_dlpack(coef_dev.__dlpack__())
                result = self._cupy_grad_hess(coef_cp, X_cp)
                if result is not None:
                    return (
                        torch.from_dlpack(result[0].__dlpack__()),
                        torch.from_dlpack(result[1].__dlpack__()),
                    )
                # Fallback: Triton kernel
                result = self._triton_grad_hess(coef_dev, X_s)
                if result is not None:
                    return result
            else:
                result = self._cupy_grad_hess(coef_dev, X_s)
                if result is not None:
                    return result

        if is_torch_cuda and self.ties == 'breslow':
            result = self._torch_breslow_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        # Backend-aware Efron fallback (stays on device, no GPU→CPU transfer)
        if self.ties == 'efron' and self._efron_pre_np is not None:
            eta = X_s @ coef_dev
            eta_shifted = eta - (xp.max(eta) if xp.__name__ != "torch" else xp.max(eta))
            try:
                grad, hess = self._efron_grad_hess_backend(eta_shifted, X_s, xp)
                return grad, hess
            except Exception:
                pass

        # CPU-only (numpy). CuPy/Torch CUDA must NOT silently fall back.
        if is_cupy or is_torch_cuda:
            raise RuntimeError(
                "CoxPH GPU gradient/Hessian path failed. "
                "Explicit GPU devices do not fall back to CPU. "
                "Use device='cpu' to run the CPU implementation."
            )
        eta_np = _to_numpy(X_s @ coef_dev)
        grad_np, hess_np = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
        return (
            _xp_asarray(grad_np, dtype=xp.float64, ref_arr=X_s),
            _xp_asarray(hess_np, dtype=xp.float64, ref_arr=X_s),
        )

    def _loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from eta, dispatching to GPU if available."""
        xp = _get_xp(X_s)
        is_cupy = xp.__name__ == "cupy"
        is_torch_cuda = xp.__name__ == "torch" and X_s.is_cuda

        if (is_cupy or is_torch_cuda):
            result = self._gpu_loglik_from_eta(eta, X_s)
            if result is not None:
                return result
            # Explicit GPU must not silently fall back to CPU
            raise RuntimeError(
                "CoxPH GPU loglik path failed. "
                "Explicit GPU devices do not fall back to CPU. "
                "Use device='cpu' to run the CPU implementation."
            )
        return self._cpu_loglik(_to_numpy(eta), self._time_np, self._event_np)

    def _grad_from_eta(self, eta, X_s):
        """Compute gradient from eta via CPU (eta is already computed)."""
        xp = _get_xp(X_s)
        grad_np, _ = self._cpu_grad_hess(_to_numpy(eta), self._time_np, self._event_np)
        return _xp_asarray(grad_np, dtype=xp.float64, ref_arr=X_s)

    # ── CuPy CUDA kernel path ────────────────────────────────────────

    def _cupy_grad_hess(self, coef_dev, X_s):
        """Efron gradient/Hessian on CuPy.

        Tries existing CUDA kernel (nuft<=512), falls back to prefix-sum
        loop for larger nuft.
        """
        try:
            import cupy as cp
        except ImportError:
            return None

        if self._efron_pre_np is None:
            return None

        _, _, _, _, nuft, _ = self._efron_pre_np

        # Try multi-block CUDA kernel (works for any nuft)
        try:
            from statgpu.survival._cox_efron_cuda import efron_indices_to_csr
            from statgpu.survival._cox_efron_grad_hess_kernel import compute_efron_grad_hess_multiblock

            _, uft_ix, risk_enter, risk_exit, _, first_idx_uft = self._efron_pre_np
            if self._efron_csr is None:
                csr6 = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
                self._efron_csr = csr6 + (first_idx_uft.astype(np.int32), int(nuft))
            _, _, _, _, fail_ptr, fail_ind, _, _ = self._efron_csr

            # Prepare arrays (must be contiguous for CUDA kernels)
            n, p = int(X_s.shape[0]), int(X_s.shape[1])
            eta = X_s @ coef_dev
            eta = eta - cp.max(eta)
            exp_eta = cp.exp(eta)
            X_exp = X_s * exp_eta[:, None]

            risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
            risk_X_sum = cp.cumsum(X_exp[::-1], axis=0)[::-1]
            outer_flat = (X_exp[:, :, None] * X_s[:, None, :]).reshape(n, p * p)
            prefix_flat = cp.concatenate([
                cp.zeros((1, p * p), dtype=cp.float64),
                cp.cumsum(outer_flat[:-1], axis=0)
            ], axis=0)
            total_X2 = prefix_flat[-1].reshape(p, p) + outer_flat[-1].reshape(p, p)

            result = compute_efron_grad_hess_multiblock(
                X_s, exp_eta, risk_sum, risk_X_sum, prefix_flat, total_X2,
                cp.asarray(fail_ptr, dtype=cp.int32),
                cp.asarray(fail_ind, dtype=cp.int32),
                cp.asarray(first_idx_uft.astype(np.int32), dtype=cp.int32),
                nuft, p, cupy_module=cp,
            )
            if result is not None:
                return result
        except Exception:
            pass

        # Fallback: Python loop (CuPy backend-aware, no CPU round-trip)
        _, uft_ix, risk_enter, _, _, _ = self._efron_pre_np
        n, p = int(X_s.shape[0]), int(X_s.shape[1])

        eta = X_s @ coef_dev
        eta = eta - cp.max(eta)
        exp_eta = cp.exp(eta)
        X_exp = X_s * exp_eta[:, None]

        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = cp.cumsum(X_exp[::-1], axis=0)[::-1]

        outer_flat = (X_exp[:, :, None] * X_s[:, None, :]).reshape(n, p * p)
        prefix_flat = cp.concatenate([
            cp.zeros((1, p * p), dtype=cp.float64),
            cp.cumsum(outer_flat[:-1], axis=0)
        ], axis=0)
        total_X2 = prefix_flat[-1].reshape(p, p) + outer_flat[-1].reshape(p, p)

        grad = cp.zeros(p, dtype=cp.float64)
        hess = cp.zeros((p, p), dtype=cp.float64)

        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue
            re_val = risk_enter[g]
            re = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            s0 = float(risk_sum[re])
            s1 = risk_X_sum[re]  # (p,)

            # Tied failure quantities — ALL failures in group
            v = X_s[ix_ev]  # (d, p)
            elx = exp_eta[ix_ev]  # (d,)
            xp0f = float(cp.sum(elx))
            xp1f = v.T @ elx  # (p,)
            xp2f = (v * elx[:, None]).T @ v  # (p, p)

            # Efron correction: for k=0..d-1, denominator = s0 - (k/d)*xp0f
            k_vals = cp.arange(d, dtype=cp.float64)
            J = k_vals / d  # (d,)
            c0 = s0 - J * xp0f  # (d,)
            safe_denom = cp.maximum(c0, 1e-300)
            inv = 1.0 / safe_denom  # (d,)
            J_inv = J * inv  # (d,)
            sum_inv = float(cp.sum(inv))
            sum_J = float(cp.sum(J_inv))
            sum_aa = float(cp.dot(inv, inv))
            sum_bb = float(cp.dot(J_inv, J_inv))
            sum_ab = float(cp.dot(inv, J_inv))

            # Gradient: sum of ALL failure X's minus Efron-corrected risk term
            grad += cp.sum(v, axis=0)  # sum_{i in D_g} X_i
            grad -= s1 * sum_inv - xp1f * sum_J

            # Hessian: Efron-corrected second moment
            risk_X2 = total_X2 - prefix_flat[re].reshape(p, p)
            hess -= risk_X2 * sum_inv
            hess += xp2f * sum_J
            hess += sum_aa * cp.outer(s1, s1)
            hess += sum_bb * cp.outer(xp1f, xp1f)
            hess -= sum_ab * (cp.outer(s1, xp1f) + cp.outer(xp1f, s1))

        return grad, -hess

    def _gpu_loglik(self, coef_dev, X_s):
        """Compute log-likelihood via GPU kernel."""
        xp = _get_xp(X_s)
        eta = X_s @ coef_dev
        return self._gpu_loglik_from_eta(eta, X_s)

    def _gpu_loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from precomputed eta on GPU.

        Supports cupy and torch-CUDA (via DLPack conversion).
        """
        xp = _get_xp(X_s)
        is_cupy = xp.__name__ == "cupy"
        is_torch_cuda = xp.__name__ == "torch" and X_s.is_cuda

        if self.ties == 'efron' and self._efron_pre_np is not None:
            try:
                if is_cupy or is_torch_cuda:
                    import cupy as cp
                    from statgpu.survival._cox_efron_cuda import compute_efron_loglik_raw_csr

                    if is_torch_cuda:
                        eta_cp = cp.from_dlpack(eta.__dlpack__())
                    else:
                        eta_cp = eta

                    exp_eta = cp.exp(eta_cp)
                    risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
                    _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
                    first_idx_uft_dev = cp.asarray(first_idx_uft, dtype=cp.int32)
                    if self._efron_csr is not None:
                        result = compute_efron_loglik_raw_csr(
                            eta_cp, exp_eta, risk_sum,
                            self._efron_csr[4], self._efron_csr[5],
                            first_idx_uft_dev, nuft, cupy_module=cp
                        )
                        return result
            except (ImportError, RuntimeError):
                pass

        # Breslow: can compute directly on any backend
        if self.ties == 'breslow' and self._breslow_pre_np is not None:
            exp_eta = xp.exp(eta)
            if xp.__name__ == "torch":
                risk_sum = xp.cumsum(exp_eta.flip(0), dim=0).flip(0)
            else:
                risk_sum = xp.cumsum(exp_eta[::-1])[::-1]
            first_idx, counts_np = self._breslow_pre_np
            if xp.__name__ == "torch":
                import torch
                first_idx_dev = torch.from_numpy(first_idx).long().to(eta.device)
                counts = torch.from_numpy(counts_np).to(eta.device)
            elif xp.__name__ == "cupy":
                import cupy
                first_idx_dev = cupy.asarray(first_idx)
                counts = cupy.asarray(counts_np)
            else:
                first_idx_dev = first_idx
                counts = counts_np
            risk_at = risk_sum[first_idx_dev]
            event_mask = (self._event_sorted == 1) if hasattr(self, '_event_sorted') else (self._event_np == 1)
            if xp.__name__ == "torch":
                event_mask_dev = torch.from_numpy(self._event_np).bool().to(eta.device) if hasattr(self, '_event_np') else event_mask
            else:
                event_mask_dev = event_mask
            event_eta = eta[event_mask_dev]
            if xp.__name__ == "torch":
                return xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at))
            return float(xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at)))

        return None

    # ── Triton/Torch kernel paths ────────────────────────────────────

    def _triton_grad_hess(self, coef_dev, X_s):
        try:
            from statgpu.survival._cox_efron_triton import compute_efron_grad_hess_triton
            if self._efron_pre_np is None:
                return None
            return compute_efron_grad_hess_triton(X_s, coef_dev, self._efron_pre_np)
        except (ImportError, RuntimeError):
            return None

    def _torch_breslow_grad_hess(self, coef_dev, X_s):
        try:
            from statgpu.survival._cox_breslow_triton_kernel import compute_breslow_grad_hess_triton
            return compute_breslow_grad_hess_triton(X_s, coef_dev, self._time_sorted, self._event_sorted)
        except (ImportError, RuntimeError):
            return None

    # ── CPU fallback (numpy) ─────────────────────────────────────────

    def _cpu_loglik_cached(self, eta_np, X_np):
        """Compute loglik using cached suffix sums from fused_value_and_gradient.

        Reuses risk_sum, risk_X_sum, suffix_outer from the previous
        fused computation. Much faster than recomputing from scratch.
        """
        if not hasattr(self, '_cached_suffix') or self._cached_suffix is None:
            return None

        # Note: cached suffix sums are from the PREVIOUS eta, not current.
        # For line search, eta changes slightly (beta + step*direction).
        # The suffix sums depend on exp(eta) which changes with eta.
        # So we CANNOT reuse them for a different eta.
        # Instead, compute loglik from scratch but share the uft_ix structure.
        efron_pre = self._efron_pre_np
        _, uft_ix, risk_enter, _, nuft, _ = efron_pre

        exp_eta = np.exp(eta_np)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        ll = 0.0
        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue
            re_val = risk_enter[g]
            re = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            s0 = risk_sum[re]
            se = float(np.sum(exp_eta[ix_ev]))
            k = np.arange(d, dtype=np.float64)
            denom = s0 - (k / d) * se
            safe = np.maximum(denom, 1e-300)
            ll += float(np.sum(eta_np[ix_ev])) - float(np.sum(np.log(safe)))
        return ll

    def _cpu_loglik(self, eta_np, time_np, event_np):
        """Compute log partial likelihood in numpy."""
        exp_eta = np.exp(eta_np)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        event_mask = event_np == 1
        if not np.any(event_mask):
            return 0.0

        if self.ties == 'breslow':
            pre = self._breslow_pre_np
            if pre is not None and pre[0].size > 0:
                first_idx, counts = pre
            else:
                event_times = time_np[event_mask]
                uft, _, counts = np.unique(event_times, return_inverse=True, return_counts=True)
                first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
            return float(np.sum(eta_np[event_mask]) - np.sum(counts * np.log(risk_sum[first_idx])))

        # Efron
        efron_pre = self._efron_pre_np
        if efron_pre is not None:
            _, uft_ix, _, _, nuft, first_idx_uft = efron_pre
            all_eta_sum = 0.0
            all_log_denom_sum = 0.0
            for g in range(nuft):
                ix_ev = uft_ix[g]
                d = len(ix_ev)
                if d == 0:
                    continue
                idx = int(first_idx_uft[g])
                risk_at_t = risk_sum[idx]
                sum_events = float(np.sum(exp_eta[ix_ev]))
                all_eta_sum += float(np.sum(eta_np[ix_ev]))
                k_vals = np.arange(d, dtype=np.float64)
                denom = risk_at_t - (k_vals / d) * sum_events
                all_log_denom_sum += float(np.sum(np.log(np.maximum(denom, 1e-300))))
            return float(all_eta_sum - all_log_denom_sum)

        return 0.0

    def _efron_grad_hess_backend(self, eta, X, xp):
        """Efron gradient/Hessian — backend-aware (works with cupy/torch/numpy).

        Uses incremental accumulator backward scan (O(p²) memory) for all backends.
        For numpy: delegates to _efron_grad_hess_np.
        For cupy/torch: incremental accumulators, no GPU→CPU transfer.
        """
        n, p = int(X.shape[0]), int(X.shape[1])

        # Numpy: delegate to optimized _efron_grad_hess_np
        if xp.__name__ == "numpy":
            eta_np = _to_numpy(eta) if not isinstance(eta, np.ndarray) else eta
            X_np = _to_numpy(X) if not isinstance(X, np.ndarray) else X
            return self._efron_grad_hess_np(eta_np, X_np, self._efron_pre_np)

        exp_eta = xp.exp(eta)
        X_exp = X * exp_eta[:, None]

        _, uft_ix, _, _, nuft, first_idx_uft = self._efron_pre_np

        if nuft == 0:
            return _xp_zeros(p, dtype=xp.float64, ref_arr=X), _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        # Suffix sums with sentinel zero at end
        if xp.__name__ == "torch":
            risk_sum = xp.zeros(n + 1, dtype=xp.float64, device=X.device)
            risk_sum[:n] = xp.cumsum(exp_eta.flip(0), dim=0).flip(0)
            risk_X_sum = xp.zeros((n + 1, p), dtype=xp.float64, device=X.device)
            risk_X_sum[:n] = xp.cumsum(X_exp.flip(0), dim=0).flip(0)
        else:
            risk_sum = xp.zeros(n + 1, dtype=xp.float64)
            risk_sum[:n] = xp.cumsum(exp_eta[::-1])[::-1]
            risk_X_sum = xp.zeros((n + 1, p), dtype=xp.float64)
            risk_X_sum[:n] = xp.cumsum(X_exp[::-1], axis=0)[::-1]

        # Running accumulators (backward scan)
        xp0 = 0.0
        xp1 = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        xp2 = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        grad = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        hess = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        for g in range(nuft - 1, -1, -1):
            # ── Enter phase: add samples with time in [uft[g], uft[g+1]) ──
            enter_start = int(first_idx_uft[g])
            enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
            if enter_end > enter_start:
                xp0 += float(risk_sum[enter_start] - risk_sum[enter_end])
                xp1 = xp1 + (risk_X_sum[enter_start] - risk_X_sum[enter_end])
                blk = X_exp[enter_start:enter_end]
                xp2 = xp2 + (blk.T @ X[enter_start:enter_end])

            # ── Fail phase: Efron correction ──
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue

            v = X[ix_ev]
            elx = exp_eta[ix_ev]
            xp0f = float(xp.sum(elx))
            xp1f = v.T @ elx
            xp2f = (v * elx[:, None]).T @ v

            # Vectorized Efron correction over tie size d
            if xp.__name__ == "torch":
                J = xp.arange(d, dtype=xp.float64, device=X.device) / d
            else:
                J = xp.arange(d, dtype=xp.float64) / d
            c0 = xp0 - J * xp0f
            c0 = xp.maximum(c0, xp.float64(1e-300)) if xp.__name__ == "torch" else xp.maximum(c0, 1e-300)
            inv = 1.0 / c0
            sum_inv = float(xp.sum(inv))
            sum_J = float(xp.sum(J * inv))
            sum_aa = float(xp.sum(inv * inv))
            sum_bb = float(xp.sum((J * inv) * (J * inv)))
            sum_ab = float(xp.sum(inv * (J * inv)))

            grad = grad + xp.sum(v, axis=0) - (xp1 * sum_inv - xp1f * sum_J)

            hess = hess - xp2 * sum_inv + xp2f * sum_J
            hess = hess + (
                sum_aa * xp.outer(xp1, xp1)
                + sum_bb * xp.outer(xp1f, xp1f)
                - sum_ab * (xp.outer(xp1, xp1f) + xp.outer(xp1f, xp1))
            )

        return grad, -hess

    def _cpu_grad_hess(self, eta_np, time_np, event_np):
        """Compute gradient and Hessian in numpy."""
        X_np = _to_numpy(self._X_sorted)
        p = X_np.shape[1]
        exp_eta = np.exp(eta_np)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        X_exp_eta = X_np * exp_eta[:, None]
        risk_X_sum = np.cumsum(X_exp_eta[::-1], axis=0)[::-1]
        event_mask = event_np == 1

        if self.ties == 'breslow':
            grad = np.zeros(p, dtype=np.float64)
            pre = self._breslow_pre_np
            has_events = bool(np.any(event_mask))
            if has_events and pre is not None and pre[0].size > 0:
                first_idx, counts = pre
                sum_X_events = np.sum(X_np[event_mask], axis=0)
                E_X = risk_X_sum[first_idx] / risk_sum[first_idx][:, None]
                grad = sum_X_events - np.sum(E_X * counts[:, None], axis=0)

            if not has_events:
                hess = np.zeros((p, p), dtype=np.float64)
            elif pre is not None and pre[0].size > 0:
                first_idx, counts = pre
                x2_weighted = np.einsum("ni,nj,n->nij", X_np, X_np, exp_eta)
                risk_X2_sum = np.cumsum(x2_weighted[::-1], axis=0)[::-1]
                risk_sum_at = risk_sum[first_idx]
                E_X = risk_X_sum[first_idx] / risk_sum_at[:, None]
                E_XX = risk_X2_sum[first_idx] / risk_sum_at[:, None, None]
                centered = E_XX - np.einsum("ni,nj->nij", E_X, E_X)
                hess = -np.sum(centered * counts[:, None, None], axis=0)
            else:
                hess = np.zeros((p, p), dtype=np.float64)
        else:
            eta_shift = eta_np - np.max(eta_np)
            efron_pre = self._efron_pre_np
            if efron_pre is not None:
                grad, hess = self._efron_grad_hess_np(eta_shift, X_np, efron_pre)
            else:
                grad, hess = np.zeros(p), np.zeros((p, p))

        return grad, hess

    @staticmethod
    def _efron_grad_hess_np(eta, X, efron_pre):
        """Efron gradient/Hessian — incremental accumulator backward scan.

        Uses the same algorithm as statsmodels PHReg: maintain running
        xp0/xp1/xp2 accumulators, update incrementally at each failure time.
        O(nuft·p²) time, O(p²) memory — no O(n·p²) suffix outer product.
        """
        n, p = X.shape
        exp_eta = np.exp(eta)
        X_exp = X * exp_eta[:, None]

        _, uft_ix, _, _, nuft, first_idx_uft = efron_pre

        if nuft == 0:
            return np.zeros(p, dtype=np.float64), np.zeros((p, p), dtype=np.float64)

        # Suffix sums with sentinel zero at end so that
        # risk_sum[i] - risk_sum[j] = sum(exp_eta[i:j]) for any i < j.
        risk_sum = np.zeros(n + 1, dtype=np.float64)
        risk_sum[:n] = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.zeros((n + 1, p), dtype=np.float64)
        risk_X_sum[:n] = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Running accumulators (backward scan)
        xp0 = 0.0
        xp1 = np.zeros(p, dtype=np.float64)
        xp2 = np.zeros((p, p), dtype=np.float64)

        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)

        for g in range(nuft - 1, -1, -1):
            # ── Enter phase: add samples with time in [uft[g], uft[g+1]) ──
            enter_start = int(first_idx_uft[g])
            enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
            if enter_end > enter_start:
                xp0 += risk_sum[enter_start] - risk_sum[enter_end]
                xp1 += risk_X_sum[enter_start] - risk_X_sum[enter_end]
                xp2 += X_exp[enter_start:enter_end].T @ X[enter_start:enter_end]

            # ── Fail phase: Efron correction ──
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue

            v = X[ix_ev]
            elx = exp_eta[ix_ev]
            xp0f = float(elx.sum())
            xp1f = v.T @ elx
            xp2f = (v * elx[:, None]).T @ v

            # Vectorized Efron correction over tie size d
            J = np.arange(d, dtype=np.float64) / d
            c0 = xp0 - J * xp0f
            np.maximum(c0, 1e-300, out=c0)
            inv = 1.0 / c0
            J_inv = J * inv
            sum_inv = inv.sum()
            sum_J = J_inv.sum()
            sum_aa = np.dot(inv, inv)
            sum_bb = np.dot(J_inv, J_inv)
            sum_ab = np.dot(inv, J_inv)

            grad += v.sum(axis=0)
            grad -= xp1 * sum_inv - xp1f * sum_J

            hess -= xp2 * sum_inv
            hess += xp2f * sum_J
            hess += sum_aa * np.outer(xp1, xp1)
            hess += sum_bb * np.outer(xp1f, xp1f)
            hess -= sum_ab * (np.outer(xp1, xp1f) + np.outer(xp1f, xp1))

        return grad, -hess

    def _cpu_fused_loglik_grad(self, eta_np, X_np, time_np, event_np):
        """Fused loglik + gradient for Efron — single pass.

        Shares suffix sums across loglik and gradient computation.
        """
        n, p = X_np.shape
        exp_eta = np.exp(eta_np)
        X_exp = X_np * exp_eta[:, None]

        efron_pre = self._efron_pre_np
        _, uft_ix, risk_enter, _, nuft, _ = efron_pre

        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.cumsum(X_exp[::-1], axis=0)[::-1]

        ll = 0.0
        grad = np.zeros(p, dtype=np.float64)

        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue
            re_val = risk_enter[g]
            re = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            s0 = risk_sum[re]
            s1 = risk_X_sum[re]
            se = float(np.sum(exp_eta[ix_ev]))
            sx = np.sum(X_np[ix_ev], axis=0)
            k = np.arange(d, dtype=np.float64)
            denom = s0 - (k / d) * se
            safe = np.maximum(denom, 1e-300)
            si = np.sum(1.0 / safe)

            ll += float(np.sum(eta_np[ix_ev])) - float(np.sum(np.log(safe)))
            grad += sx - s1 * si * d

        return ll, grad, None

    def _cpu_fused_loglik_grad_hess(self, eta_np, X_np, time_np, event_np):
        """Fused loglik + gradient + Hessian for Efron — incremental accumulator.

        Uses the same backward-scan algorithm as statsmodels PHReg:
        maintain running xp0/xp1/xp2 accumulators, update incrementally
        at each failure time.  O(nuft·p²) time, O(p²) memory.
        """
        n, p = X_np.shape
        # Numerical stability: shift eta to prevent exp overflow
        eta_shift = eta_np - np.max(eta_np)
        exp_eta = np.exp(eta_shift)
        X_exp = X_np * exp_eta[:, None]

        efron_pre = self._efron_pre_np
        _, uft_ix, _, _, nuft, first_idx_uft = efron_pre

        if nuft == 0:
            return 0.0, np.zeros(p, dtype=np.float64), np.zeros((p, p), dtype=np.float64)

        # Suffix sums with sentinel zero at end so that
        # risk_sum[i] - risk_sum[j] = sum(exp_eta[i:j]) for any i < j.
        risk_sum = np.zeros(n + 1, dtype=np.float64)
        risk_sum[:n] = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.zeros((n + 1, p), dtype=np.float64)
        risk_X_sum[:n] = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Running accumulators (backward scan)
        xp0 = 0.0
        xp1 = np.zeros(p, dtype=np.float64)
        xp2 = np.zeros((p, p), dtype=np.float64)

        ll = 0.0
        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)

        for g in range(nuft - 1, -1, -1):
            # ── Enter phase: add samples with time in [uft[g], uft[g+1]) ──
            enter_start = int(first_idx_uft[g])
            enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
            if enter_end > enter_start:
                xp0 += risk_sum[enter_start] - risk_sum[enter_end]
                xp1 += risk_X_sum[enter_start] - risk_X_sum[enter_end]
                xp2 += X_exp[enter_start:enter_end].T @ X_np[enter_start:enter_end]

            # ── Fail phase: Efron correction ──
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue

            v = X_np[ix_ev]
            elx = exp_eta[ix_ev]
            xp0f = float(elx.sum())
            xp1f = v.T @ elx
            xp2f = (v * elx[:, None]).T @ v

            # Vectorized Efron correction over tie size d
            J = np.arange(d, dtype=np.float64) / d
            c0 = xp0 - J * xp0f
            np.maximum(c0, 1e-300, out=c0)
            inv = 1.0 / c0
            J_inv = J * inv
            sum_inv = inv.sum()
            sum_J = J_inv.sum()
            sum_aa = np.dot(inv, inv)
            sum_bb = np.dot(J_inv, J_inv)
            sum_ab = np.dot(inv, J_inv)

            # Loglik (use shifted eta for numerical stability)
            ll += float(np.sum(eta_shift[ix_ev])) - float(np.sum(np.log(c0)))

            grad += v.sum(axis=0)
            grad -= xp1 * sum_inv - xp1f * sum_J

            hess -= xp2 * sum_inv
            hess += xp2f * sum_J
            hess += sum_aa * np.outer(xp1, xp1)
            hess += sum_bb * np.outer(xp1f, xp1f)
            hess -= sum_ab * (np.outer(xp1, xp1f) + np.outer(xp1f, xp1))

        return ll, grad, -hess

