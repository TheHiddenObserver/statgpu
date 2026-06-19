"""
Cox partial likelihood loss for survival analysis.

Negative log partial likelihood with Breslow/Efron tie handling.
Dispatches to GPU-optimized kernels (CuPy CUDA / Triton) when available,
falls back to numpy Python loops on CPU.

Matches R's survival::coxph() interface.
"""

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp, _xp_zeros, _xp_asarray
from statgpu.backends._utils import _to_float_scalar
from ._base import LossBase
from ._registry import register_loss


# ── Backend detection ────────────────────────────────────────────────

def _is_cupy(arr):
    return _get_xp(arr).__name__ == "cupy"


def _is_torch(arr):
    return _get_xp(arr).__name__ == "torch"


def _is_torch_cuda(arr):
    return _is_torch(arr) and arr.is_cuda


# ── Cross-backend helpers ───────────────────────────────────────────

def _xp_argsort(x, xp):
    if xp.__name__ == "torch":
        return xp.argsort(x, stable=True)
    return xp.argsort(x)


def _xp_unique(x, xp):
    return xp.unique(x, return_inverse=True, return_counts=True)


def _xp_searchsorted(sorted_arr, values, side, xp):
    if xp.__name__ == "torch":
        return xp.searchsorted(sorted_arr, values, right=(side == "right"))
    return xp.searchsorted(sorted_arr, values, side=side)


def _xp_bincount(x, weights, minlength, xp):
    if xp.__name__ == "torch":
        w = weights.to(x.device) if weights is not None else None
        return xp.bincount(x.long(), weights=w, minlength=minlength).to(x.dtype)
    return xp.bincount(x, weights=weights, minlength=minlength).astype(np.float64)


def _xp_cumsum_flip(x, xp):
    """cumsum(flipped(x)) — suffix sum across backends."""
    if xp.__name__ == "torch":
        return xp.cumsum(x.flip(0), dim=0).flip(0)
    return xp.cumsum(x[::-1])[::-1]


def _xp_cumsum_flip_2d(x, xp):
    """cumsum(flipped(x), axis=0) for 2D arrays."""
    if xp.__name__ == "torch":
        return xp.cumsum(x.flip(0), dim=0).flip(0)
    return xp.cumsum(x[::-1], axis=0)[::-1]


def _to_numpy(arr):
    """Convert any array to numpy."""
    if hasattr(arr, 'cpu'):
        return arr.cpu().numpy()
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


# ── Build efron_pre from sorted time/event (numpy arrays for kernel dispatch) ──

def _build_efron_pre_numpy(time_np, event_np):
    """Build efron_pre structure as numpy arrays (for kernel dispatch)."""
    event_mask = event_np == 1
    event_idx = np.where(event_mask)[0]
    event_times = time_np[event_idx]
    uft, inv = np.unique(event_times, return_inverse=True)
    nuft = len(uft)
    uft_ix = [event_idx[inv == g].astype(np.int32) for g in range(nuft)]
    first_idx_uft = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    risk_enter = [np.int64(np.searchsorted(time_np, t, side="left")) for t in uft]
    risk_exit = [np.int64(np.searchsorted(time_np, t, side="right")) for t in uft]
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

        # Internal state set by preprocess()
        self._sorted = False
        self._X_sorted = None
        self._time_sorted = None
        self._event_sorted = None
        self._order = None
        # Numpy copies for kernel dispatch (always available)
        self._time_np = None
        self._event_np = None
        self._efron_pre_np = None
        self._breslow_pre_np = None
        self._efron_csr = None  # CSR packed indices for CuPy kernel
        self._n_events = 0

    def preprocess(self, X, y):
        """Sort data by time and precompute risk-set structures."""
        xp = _get_xp(X)

        # Extract time and event, move to same device
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

        # Sort by time ascending
        order = _xp_argsort(time, xp)
        self._X_sorted = X_arr[order]
        self._time_sorted = time[order]
        self._event_sorted = event[order]
        self._order = order
        self._sorted = True

        if xp.__name__ == "torch":
            self._n_events = int(xp.sum(self._event_sorted).item())
        else:
            self._n_events = int(xp.sum(self._event_sorted))

        # Build numpy copies for kernel dispatch
        time_np = _to_numpy(self._time_sorted).astype(np.float64)
        event_np = _to_numpy(self._event_sorted).astype(np.float64)
        self._time_np = time_np
        self._event_np = event_np

        if self.ties == 'efron':
            self._efron_pre_np = _build_efron_pre_numpy(time_np, event_np)
            self._breslow_pre_np = None
            # Pre-pack CSR for CuPy kernel
            try:
                from statgpu.survival._cox_efron_cuda import efron_indices_to_csr
                _, uft_ix, risk_enter, risk_exit, nuft, _ = self._efron_pre_np
                self._efron_csr = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
            except Exception:
                self._efron_csr = None
        else:
            self._breslow_pre_np = _build_breslow_pre_numpy(time_np, event_np)
            self._efron_pre_np = None
            self._efron_csr = None

        y_sorted = _xp_zeros(X_arr.shape[0], dtype=xp.float64, ref_arr=X_arr)
        return self._X_sorted, y_sorted

    def _check_stale(self, X):
        if self._sorted and X is not self._X_sorted:
            self._sorted = False

    def value(self, X, y, coef, sample_weight=None) -> float:
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]

        # Try GPU kernel for Efron
        if self.ties == 'efron' and (_is_cupy(X_s) or _is_torch_cuda(X_s)):
            loglik = self._gpu_loglik(coef_dev, X_s)
            return -_to_float_scalar(loglik) / n

        # Fallback: numpy Python loops
        eta = X_s @ coef_dev
        loglik = self._cpu_loglik(_to_numpy(eta), self._time_np, self._event_np)
        return -loglik / n

    def gradient(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]
        grad, _ = self._compute_grad_hess(coef_dev, X_s)
        return -grad / n

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]

        # Fused: compute eta once
        eta = X_s @ coef_dev
        loglik = self._loglik_from_eta(eta, X_s)
        grad = self._grad_from_eta(eta, X_s)
        return -_to_float_scalar(loglik) / n, -grad / n

    def hessian(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]
        _, hess = self._compute_grad_hess(coef_dev, X_s)
        return -hess / n

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        from statgpu.backends._array_ops import _max_eigval_power
        if not self._sorted:
            self.preprocess(X, y)
        X_s = self._X_sorted
        xp = _get_xp(X_s)
        if coef is not None:
            coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        else:
            coef_dev = _xp_zeros(X_s.shape[1], dtype=xp.float64, ref_arr=X_s)
        _, hess = self._compute_grad_hess(coef_dev, X_s)
        n = X_s.shape[0]
        return _max_eigval_power(-hess / n)

    # ── Dispatch: GPU vs CPU ─────────────────────────────────────────

    def _compute_grad_hess(self, coef_dev, X_s):
        """Compute gradient and Hessian, dispatching to GPU kernel if available."""
        xp = _get_xp(X_s)

        # Try GPU kernel
        if _is_cupy(X_s) and self.ties == 'efron':
            result = self._cupy_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        if _is_torch_cuda(X_s) and self.ties == 'efron':
            result = self._triton_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        if _is_torch_cuda(X_s) and self.ties == 'breslow':
            result = self._torch_breslow_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        # CPU fallback: numpy Python loops
        eta_np = _to_numpy(X_s @ coef_dev)
        grad_np, hess_np = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
        return (
            _xp_asarray(grad_np, dtype=xp.float64, ref_arr=X_s),
            _xp_asarray(hess_np, dtype=xp.float64, ref_arr=X_s),
        )

    def _loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from eta, dispatching to GPU if available."""
        if _is_cupy(X_s) and self.ties == 'efron':
            result = self._gpu_loglik_from_eta(eta, X_s)
            if result is not None:
                return result
        # CPU fallback
        eta_np = _to_numpy(eta)
        return self._cpu_loglik(eta_np, self._time_np, self._event_np)

    def _grad_from_eta(self, eta, X_s):
        """Compute gradient from eta, dispatching to GPU if available."""
        xp = _get_xp(X_s)
        if _is_cupy(X_s) and self.ties == 'efron':
            result = self._cupy_grad_hess(X_s @ (eta @ X_s.T @ X_s * 0 + 1), X_s)  # dummy coef
            if result is not None:
                return result[0]
        # CPU fallback
        eta_np = _to_numpy(eta)
        grad_np, _ = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
        return _xp_asarray(grad_np, dtype=xp.float64, ref_arr=X_s)

    # ── CuPy CUDA kernel path ────────────────────────────────────────

    def _cupy_grad_hess(self, coef_dev, X_s):
        """Try CuPy CUDA kernel for Efron gradient/Hessian."""
        try:
            import cupy as cp
            from statgpu.survival._cox_efron_cuda import (
                compute_efron_grad_hess_raw,
                compute_efron_loglik_raw_csr,
                efron_indices_to_csr,
            )

            if self._efron_pre_np is None:
                return None

            efron_csr = self._efron_csr
            if efron_csr is None:
                _, uft_ix, risk_enter, risk_exit, nuft, _ = self._efron_pre_np
                efron_csr = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)

            grad, hess = compute_efron_grad_hess_raw(
                X_s, coef_dev, self._efron_pre_np, cupy_module=cp, efron_csr=efron_csr
            )
            if grad is None:
                return None
            return grad, hess
        except Exception:
            return None

    def _gpu_loglik(self, coef_dev, X_s):
        """Compute log-likelihood via GPU kernel (CuPy)."""
        try:
            import cupy as cp
            from statgpu.survival._cox_efron_cuda import compute_efron_loglik_raw_csr

            if self._efron_pre_np is None:
                return self._gpu_loglik_fallback(coef_dev, X_s)

            eta = X_s @ coef_dev
            exp_eta = cp.exp(eta)
            risk_sum = cp.cumsum(exp_eta[::-1])[::-1]

            _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
            first_idx_uft_cp = cp.asarray(first_idx_uft, dtype=cp.int32)

            if self._efron_csr is not None:
                fail_ptr, fail_ind = self._efron_csr[4], self._efron_csr[5]
                return compute_efron_loglik_raw_csr(
                    eta, exp_eta, risk_sum, fail_ptr, fail_ind,
                    first_idx_uft_cp, nuft, cupy_module=cp
                )
            return self._gpu_loglik_fallback(coef_dev, X_s)
        except Exception:
            return self._gpu_loglik_fallback(coef_dev, X_s)

    def _gpu_loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from precomputed eta (CuPy)."""
        try:
            import cupy as cp
            from statgpu.survival._cox_efron_cuda import compute_efron_loglik_raw_csr

            if self._efron_pre_np is None:
                return None

            exp_eta = cp.exp(eta)
            risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
            _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
            first_idx_uft_cp = cp.asarray(first_idx_uft, dtype=cp.int32)

            if self._efron_csr is not None:
                fail_ptr, fail_ind = self._efron_csr[4], self._efron_csr[5]
                return compute_efron_loglik_raw_csr(
                    eta, exp_eta, risk_sum, fail_ptr, fail_ind,
                    first_idx_uft_cp, nuft, cupy_module=cp
                )
            return None
        except Exception:
            return None

    def _gpu_loglik_fallback(self, coef_dev, X_s):
        """Fallback GPU log-likelihood (no CSR kernel)."""
        xp = _get_xp(X_s)
        eta = X_s @ coef_dev
        exp_eta = xp.exp(eta)
        return self._cpu_loglik(_to_numpy(eta), self._time_np, self._event_np)

    # ── Triton kernel path ───────────────────────────────────────────

    def _triton_grad_hess(self, coef_dev, X_s):
        """Try Triton kernel for Efron gradient/Hessian."""
        try:
            from statgpu.survival._cox_efron_triton import compute_efron_grad_hess_triton
            if self._efron_pre_np is None:
                return None
            result = compute_efron_grad_hess_triton(X_s, coef_dev, self._efron_pre_np)
            return result  # None if Triton not available
        except Exception:
            return None

    def _torch_breslow_grad_hess(self, coef_dev, X_s):
        """Try PyTorch GPU path for Breslow gradient/Hessian."""
        try:
            from statgpu.survival._cox_breslow_triton_kernel import compute_breslow_grad_hess_triton
            result = compute_breslow_grad_hess_triton(
                X_s, coef_dev, self._time_sorted, self._event_sorted
            )
            return result
        except Exception:
            return None

    # ── CPU fallback (numpy Python loops) ────────────────────────────

    def _cpu_loglik(self, eta_np, time_np, event_np):
        """Compute log partial likelihood in numpy."""
        xp = np
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

        # No precomputation
        event_idx = np.where(event_mask)[0]
        event_times = time_np[event_idx]
        uft, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
        risk_at = risk_sum[first_idx]
        sum_events = np.bincount(inv, weights=exp_eta[event_idx], minlength=len(uft)).astype(np.float64)
        ll = float(np.sum(np.bincount(inv, weights=eta_np[event_idx], minlength=len(uft))))
        for g in range(len(uft)):
            d = int(counts[g])
            if d == 0:
                continue
            k = np.arange(d, dtype=np.float64) / d
            denom = risk_at[g] - k * sum_events[g]
            ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))
        return ll

    def _cpu_grad_hess(self, eta_np, time_np, event_np):
        """Compute gradient and Hessian in numpy."""
        n, p = len(eta_np), self._X_sorted.shape[1] if self._X_sorted is not None else 0
        X_np = _to_numpy(self._X_sorted)
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
            # Efron
            eta_shift = eta_np - np.max(eta_np)
            efron_pre = self._efron_pre_np
            if efron_pre is not None:
                grad, hess = self._efron_grad_hess_np(eta_shift, X_np, efron_pre)
            else:
                grad, hess = np.zeros(p), np.zeros((p, p))

        return grad, hess

    @staticmethod
    def _efron_grad_hess_np(eta, X, efron_pre):
        """Efron gradient/Hessian in numpy."""
        n, p = X.shape
        exp_eta = np.exp(eta)
        grad = np.zeros(p, dtype=np.float64)
        hess = np.zeros((p, p), dtype=np.float64)
        X_exp = X * exp_eta[:, None]
        _, uft_ix, risk_enter, risk_exit, nuft, _ = efron_pre

        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = len(ix_ev)
            if d == 0:
                continue
            re = int(risk_enter[g])
            s0 = float(np.sum(exp_eta[re:]))
            s1 = np.sum(X_exp[re:], axis=0)
            sum_ev_exp = float(np.sum(exp_eta[ix_ev]))
            sum_ev_X = np.sum(X[ix_ev], axis=0)
            for k in range(d):
                denom = s0 - (float(k) / float(d)) * sum_ev_exp
                if denom <= 1e-300:
                    continue
                E_X = s1 / denom
                grad += (sum_ev_X / d) - E_X
                E_XX = X_exp[re:].T @ X[re:] / denom
                hess -= E_XX - np.outer(E_X, E_X)

        return grad, hess
