"""
Cox partial likelihood loss for survival analysis.

Negative log partial likelihood with Breslow/Efron tie handling.
Dispatches to GPU-optimized kernels (CuPy CUDA / PyTorch) when available,
falls back to numpy Python loops on CPU.

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
    Falls back to numpy Python loops on CPU.

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

        # CPU fallback
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
            # GPU path: use existing dispatch
            grad, _ = self._compute_grad_hess(coef_dev, X_s)
            eta = X_s @ coef_dev
            loglik = self._loglik_from_eta(eta, X_s)
            if loglik is None:
                loglik = self._cpu_loglik(_to_numpy(eta), self._time_np, self._event_np)
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

        # CPU fallback (numpy)
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

        # Fallback: Python loop
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
            s1 = risk_X_sum[re]
            sum_ev_exp = float(cp.sum(exp_eta[ix_ev]))
            sum_ev_X = cp.sum(X_s[ix_ev], axis=0)
            risk_X2 = total_X2 - prefix_flat[re].reshape(p, p)

            k_vals = cp.arange(d, dtype=cp.float64)
            denom = s0 - (k_vals / d) * sum_ev_exp
            safe_denom = cp.maximum(denom, 1e-300)
            sum_inv = float(cp.sum(1.0 / safe_denom))
            sum_inv2 = float(cp.sum(1.0 / (safe_denom * safe_denom)))

            grad = grad + sum_ev_X - s1 * sum_inv * d
            hess = hess - (risk_X2 * sum_inv - cp.outer(s1, s1) * sum_inv2 * d)

        return grad, hess

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

        For numpy: uses Python loop with prefix sums.
        For cupy/torch: uses prefix sum with Python loop (no GPU→CPU transfer).
        """
        n, p = int(X.shape[0]), int(X.shape[1])

        # Numpy: delegate to _efron_grad_hess_np
        if xp.__name__ == "numpy":
            eta_np = _to_numpy(eta) if not isinstance(eta, np.ndarray) else eta
            X_np = _to_numpy(X) if not isinstance(X, np.ndarray) else X
            return self._efron_grad_hess_np(eta_np, X_np, self._efron_pre_np)
        exp_eta = xp.exp(eta)
        X_exp = X * exp_eta[:, None]

        _, uft_ix, risk_enter, _, nuft, _ = self._efron_pre_np

        # Suffix sums
        if xp.__name__ == "torch":
            risk_sum = xp.cumsum(exp_eta.flip(0), dim=0).flip(0)
            risk_X_sum = xp.cumsum(X_exp.flip(0), dim=0).flip(0)
        else:
            risk_sum = xp.cumsum(exp_eta[::-1])[::-1]
            risk_X_sum = xp.cumsum(X_exp[::-1], axis=0)[::-1]

        # Prefix sum of rank-1 outer products
        outer_flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
        if xp.__name__ == "torch":
            prefix_flat = xp.cat([
                xp.zeros(1, p * p, dtype=xp.float64, device=X.device),
                xp.cumsum(outer_flat[:-1], dim=0)
            ], dim=0)
        else:
            prefix_flat = xp.concatenate([
                xp.zeros((1, p * p), dtype=xp.float64),
                xp.cumsum(outer_flat[:-1], axis=0)
            ], axis=0)
        total_X2 = prefix_flat[-1].reshape(p, p) + outer_flat[-1].reshape(p, p)

        # Collect per-group quantities
        s0_list, s1_list, sum_ev_exp_list, sum_ev_X_list, risk_X2_list, d_list = [], [], [], [], [], []
        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            d_list.append(d)
            if d == 0:
                continue
            re_val = risk_enter[g]
            re = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            s0_list.append(float(risk_sum[re]))
            s1_list.append(risk_X_sum[re])
            sum_ev_exp_list.append(float(xp.sum(exp_eta[ix_ev])))
            sum_ev_X_list.append(xp.sum(X[ix_ev], axis=0))
            risk_X2_list.append(total_X2 - prefix_flat[re].reshape(p, p))

        # Vectorized gradient and Hessian
        grad = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        hess = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        for i in range(len(s0_list)):
            d = d_list[i] if i < len(d_list) else 0
            if d == 0:
                continue
            s0 = float(s0_list[i])
            s1 = s1_list[i]
            sum_ev_exp = float(sum_ev_exp_list[i])
            sum_ev_X = sum_ev_X_list[i]
            risk_X2 = risk_X2_list[i]

            if xp.__name__ == "torch":
                k_vals = xp.arange(d, dtype=xp.float64, device=X.device)
            else:
                k_vals = xp.arange(d, dtype=xp.float64)
            denom = s0 - (k_vals / d) * sum_ev_exp
            safe_denom = xp.maximum(denom, xp.float64(1e-300)) if xp.__name__ == "torch" else xp.maximum(denom, 1e-300)
            sum_inv = float(xp.sum(1.0 / safe_denom))
            sum_inv2 = float(xp.sum(1.0 / (safe_denom * safe_denom)))

            grad = grad + sum_ev_X - s1 * sum_inv * d
            hess = hess - (risk_X2 * sum_inv - xp.outer(s1, s1) * sum_inv2 * d)

        return grad, hess

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
        """Efron gradient/Hessian — vectorized for d=1 groups, loop for ties.

        Most groups have d=1 (no ties) and are processed via vectorized ops.
        Only groups with d>1 (tied events) use a Python loop.
        """
        n, p = X.shape
        exp_eta = np.exp(eta)
        X_exp = X * exp_eta[:, None]

        _, uft_ix, risk_enter, _, nuft, _ = efron_pre

        # Suffix sums
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Collect per-group data
        re_arr = np.zeros(nuft, dtype=np.int64)
        d_arr = np.zeros(nuft, dtype=np.int64)
        for g in range(nuft):
            re_val = risk_enter[g]
            re_arr[g] = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            d_arr[g] = len(uft_ix[g])

        # Separate d=1 groups (vectorizable) from d>1 groups (loop)
        d1_mask = d_arr == 1
        d1_indices = np.where(d1_mask)[0]
        dgt1_indices = np.where(~d1_mask)[0]

        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)

        # ── Vectorized path for d=1 groups (no ties) ──
        if len(d1_indices) > 0:
            re_d1 = re_arr[d1_indices]
            s0_d1 = risk_sum[re_d1]                    # (G1,)
            s1_d1 = risk_X_sum[re_d1]                  # (G1, p)

            # For d=1: sum_ev_exp = exp_eta[event_idx], denom = s0 - sum_ev_exp
            # Need event indices for each group
            ev_indices = np.array([uft_ix[g][0] for g in d1_indices], dtype=np.int64)
            se_d1 = exp_eta[ev_indices]                # (G1,)
            sx_d1 = X[ev_indices]                      # (G1, p)

            denom_d1 = s0_d1 - se_d1                   # (G1,)
            safe_d1 = np.maximum(denom_d1, 1e-300)
            inv_d1 = 1.0 / safe_d1                     # (G1,)

            # grad += sum(sx_d1 - s1_d1 * inv_d1, axis=0)
            grad_contrib = sx_d1 - s1_d1 * inv_d1[:, None]  # (G1, p)
            grad += np.sum(grad_contrib, axis=0)

            # hess -= sum(risk_X2[g] * inv_d1[g] - outer(s1[g], s1[g]) * inv_d1[g]^2)
            # risk_X2[g] = suffix_outer[re_d1[g]]
            # Compute suffix outer at re_d1 indices
            outer_flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
            suffix_flat = np.cumsum(outer_flat[::-1], axis=0)[::-1]
            risk_X2_d1 = suffix_flat[re_d1].reshape(-1, p, p)  # (G1, p, p)

            hess -= np.sum(risk_X2_d1 * inv_d1[:, None, None], axis=0)
            hess += np.einsum("g,gi,gj->ij", inv_d1_sq, s1_d1, s1_d1)

        # ── Loop path for d>1 groups (tied events) ──
        for g in dgt1_indices:
            ix_ev = uft_ix[g]
            d = int(d_arr[g])
            re = int(re_arr[g])
            s0 = risk_sum[re]
            s1 = risk_X_sum[re]
            se = float(np.sum(exp_eta[ix_ev]))
            sx = np.sum(X[ix_ev], axis=0)
            k = np.arange(d, dtype=np.float64)
            denom = s0 - (k / d) * se
            safe = np.maximum(denom, 1e-300)
            si = np.sum(1.0 / safe)
            si2 = np.sum(1.0 / (safe * safe))

            # risk_X2 via suffix outer
            outer_flat_g = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
            suffix_flat_g = np.cumsum(outer_flat_g[::-1], axis=0)[::-1]
            risk_X2 = suffix_flat_g[re].reshape(p, p)

            grad += sx - s1 * si * d
            hess -= risk_X2 * si - np.outer(s1, s1) * si2 * d

        return grad, hess

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
        """Fused loglik + gradient + Hessian for Efron — vectorized for d=1.

        Most groups have d=1 (no ties) and are processed via vectorized ops.
        Only groups with d>1 use a Python loop.
        """
        n, p = X_np.shape
        exp_eta = np.exp(eta_np)
        X_exp = X_np * exp_eta[:, None]

        efron_pre = self._efron_pre_np
        _, uft_ix, risk_enter, _, nuft, _ = efron_pre

        # Suffix sums
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Collect per-group data
        re_arr = np.zeros(nuft, dtype=np.int64)
        d_arr = np.zeros(nuft, dtype=np.int64)
        for g in range(nuft):
            re_val = risk_enter[g]
            re_arr[g] = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
            d_arr[g] = len(uft_ix[g])

        # Separate d=1 from d>1
        d1_indices = np.where(d_arr == 1)[0]
        dgt1_indices = np.where(d_arr > 1)[0]

        ll = 0.0
        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)

        # ── Vectorized path for d=1 groups ──
        if len(d1_indices) > 0:
            re_d1 = re_arr[d1_indices]
            s0_d1 = risk_sum[re_d1]
            s1_d1 = risk_X_sum[re_d1]
            ev_indices = np.array([uft_ix[g][0] for g in d1_indices], dtype=np.int64)
            se_d1 = exp_eta[ev_indices]
            sx_d1 = X_np[ev_indices]

            denom_d1 = s0_d1 - se_d1
            safe_d1 = np.maximum(denom_d1, 1e-300)
            inv_d1 = 1.0 / safe_d1

            # Loglik
            ll += float(np.sum(eta_np[ev_indices])) - float(np.sum(np.log(safe_d1)))
            # Gradient
            grad += np.sum(sx_d1 - s1_d1 * inv_d1[:, None], axis=0)
            # Hessian via suffix outer
            outer_flat = (X_exp[:, :, None] * X_np[:, None, :]).reshape(n, p * p)
            suffix_flat = np.cumsum(outer_flat[::-1], axis=0)[::-1]
            risk_X2_d1 = suffix_flat[re_d1].reshape(-1, p, p)
            hess -= np.sum(risk_X2_d1 * inv_d1[:, None, None], axis=0)
            inv_d1_sq = np.minimum(inv_d1 * inv_d1, 1e30)
            hess += np.einsum("g,gi,gj->ij", inv_d1_sq, s1_d1, s1_d1)

        # ── Loop for d>1 groups (tied events) ──
        for g in dgt1_indices:
            ix_ev = uft_ix[g]
            d = int(d_arr[g])
            re = int(re_arr[g])
            s0 = risk_sum[re]
            s1 = risk_X_sum[re]
            se = float(np.sum(exp_eta[ix_ev]))
            sx = np.sum(X_np[ix_ev], axis=0)
            k = np.arange(d, dtype=np.float64)
            denom = s0 - (k / d) * se
            safe = np.maximum(denom, 1e-300)
            si = np.sum(1.0 / safe)
            si2 = np.sum(1.0 / (safe * safe))

            outer_flat_g = (X_exp[:, :, None] * X_np[:, None, :]).reshape(n, p * p)
            suffix_flat_g = np.cumsum(outer_flat_g[::-1], axis=0)[::-1]
            risk_X2 = suffix_flat_g[re].reshape(p, p)

            ll += float(np.sum(eta_np[ix_ev])) - float(np.sum(np.log(safe)))
            grad += sx - s1 * si * d
            hess -= risk_X2 * si - np.outer(s1, s1) * si2 * d

        return ll, grad, hess

