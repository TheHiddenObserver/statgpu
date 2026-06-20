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
                _, uft_ix, risk_enter, risk_exit, nuft, _ = self._efron_pre_np
                self._efron_csr = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
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

        eta = X_s @ coef_dev
        loglik = self._loglik_from_eta(eta, X_s)
        grad = self._grad_from_eta(eta, X_s)
        return -_to_float_scalar(loglik) / n, -grad / n

    def hessian(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]
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

        if is_cupy and self.ties == 'efron':
            result = self._cupy_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        if is_torch_cuda and self.ties == 'efron':
            result = self._triton_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        if is_torch_cuda and self.ties == 'breslow':
            result = self._torch_breslow_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        # CPU fallback
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
        try:
            import cupy as cp
            from statgpu.survival._cox_efron_cuda import compute_efron_grad_hess_raw, efron_indices_to_csr
        except ImportError:
            return None

        if self._efron_pre_np is None:
            return None

        efron_csr = self._efron_csr
        if efron_csr is None:
            _, uft_ix, risk_enter, risk_exit, nuft, _ = self._efron_pre_np
            efron_csr = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)

        grad, hess = compute_efron_grad_hess_raw(
            X_s, coef_dev, self._efron_pre_np, cupy_module=cp, efron_csr=efron_csr
        )
        return (grad, hess) if grad is not None else None

    def _gpu_loglik(self, coef_dev, X_s):
        """Compute log-likelihood via GPU kernel."""
        xp = _get_xp(X_s)
        eta = X_s @ coef_dev
        return self._gpu_loglik_from_eta(eta, X_s)

    def _gpu_loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from precomputed eta on GPU."""
        xp = _get_xp(X_s)
        is_cupy = xp.__name__ == "cupy"

        if self.ties == 'efron' and self._efron_pre_np is not None:
            try:
                if is_cupy:
                    import cupy as cp
                    from statgpu.survival._cox_efron_cuda import compute_efron_loglik_raw_csr
                    exp_eta = cp.exp(eta)
                    risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
                    _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
                    first_idx_uft_dev = cp.asarray(first_idx_uft, dtype=cp.int32)
                    if self._efron_csr is not None:
                        return compute_efron_loglik_raw_csr(
                            eta, exp_eta, risk_sum, self._efron_csr[4], self._efron_csr[5],
                            first_idx_uft_dev, nuft, cupy_module=cp
                        )
                # Torch-CUDA Efron: fall through to CPU
            except (ImportError, RuntimeError):
                pass

        # CPU fallback (no GPU→CPU transfer of eta; recompute from coef)
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
            re_val = risk_enter[g]
            re = int(re_val[0]) if isinstance(re_val, (list, np.ndarray)) else int(re_val)
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
