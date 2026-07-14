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
        self._efron_backend_index_cache = {}
        self._n_events = 0
        self._x_reference = None

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
        if time.ndim != 1 or event.ndim != 1:
            raise ValueError("time and event must have shape (n_samples,)")
        if time.shape[0] != X_arr.shape[0] or event.shape[0] != X_arr.shape[0]:
            raise ValueError("X, time, and event must contain the same number of rows")
        if _to_float_scalar(xp.sum(~xp.isfinite(X_arr))) > 0 or _to_float_scalar(
            xp.sum(~xp.isfinite(time))
        ) > 0:
            raise ValueError("X and time must contain only finite values")
        if _to_float_scalar(xp.sum(~xp.isfinite(event))) > 0 or _to_float_scalar(
            xp.sum((event != 0) & (event != 1))
        ) > 0:
            raise ValueError("event must contain only 0/1 finite values")
        if _to_float_scalar(xp.sum(time <= 0)) > 0:
            raise ValueError("time must contain only positive values")
        if xp.__name__ == "torch":
            self._x_reference = xp.mean(X_arr, dim=0)
        else:
            self._x_reference = xp.mean(X_arr, axis=0)
        # Cox partial likelihood derivatives are invariant to a common column
        # shift.  Center once on the active backend to prevent raw-moment
        # cancellation and eta under/overflow for X = z + a large constant.
        X_arr = X_arr - self._x_reference.reshape(1, -1)
        order = xp.argsort(time, stable=True) if xp.__name__ == "torch" else xp.argsort(time)
        self._X_sorted = X_arr[order]
        self._time_sorted = time[order]
        self._event_sorted = event[order]
        self._order = order
        self._sorted = True
        self._n_events = int(_to_float_scalar(xp.sum(self._event_sorted)))
        self._efron_backend_index_cache = {}

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
        is_gpu = xp.__name__ == "cupy" or (
            xp.__name__ == "torch" and X_s.is_cuda
        )
        if is_gpu:
            grad, _ = self._compute_grad_hess(coef_dev, X_s)
            return -grad / n
        eta_np = _to_numpy(X_s @ coef_dev)
        _, grad_np = self._cpu_loglik_grad(eta_np, _to_numpy(X_s))
        return _xp_asarray(-grad_np / n, dtype=xp.float64, ref_arr=X_s)

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

        # CPU first-order solvers do not need the O(p^2) Hessian.  Compute
        # value and score together using O(n p) storage.
        X_np = _to_numpy(X_s)
        eta_np = _to_numpy(X_s @ coef_dev)
        loglik, grad_np = self._cpu_loglik_grad(eta_np, X_np)
        return -loglik / n, _xp_asarray(
            -grad_np / n, dtype=xp.float64, ref_arr=X_s
        )

    def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
        """Return loss gradient and Hessian from one derivative evaluation."""
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )
        self._ensure_sorted(X, y)
        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        grad, hess = self._compute_grad_hess(coef_dev, X_s)
        n = X_s.shape[0]
        return -grad / n, -hess / n

    def hessian(self, X, y, coef, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError("CoxPartialLikelihoodLoss does not support sample_weight")
        self._ensure_sorted(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        n = X_s.shape[0]

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

        # Efron dispatch is backend-native.  Torch must not require CuPy (or a
        # DLPack round trip through CuPy) merely to evaluate a Torch model.
        if self.ties == 'efron':
            if is_torch_cuda:
                result = self._triton_grad_hess(coef_dev, X_s)
                if result is not None:
                    return result
            elif is_cupy:
                result = self._cupy_grad_hess(coef_dev, X_s)
                if result is not None:
                    return result

        if is_torch_cuda and self.ties == 'breslow':
            result = self._torch_breslow_grad_hess(coef_dev, X_s)
            if result is not None:
                return result

        if is_cupy and self.ties == 'breslow':
            from statgpu.survival._risk_sets import (
                cox_counting_process_objective,
            )

            result = cox_counting_process_objective(
                coef_dev,
                X_s,
                self._time_sorted,
                self._event_sorted,
                ties="breslow",
            )
            # Loss helpers expose derivatives of log partial likelihood;
            # the shared engine exposes positive observed information.
            return result["score"], -result["information"]

        # Backend-aware Efron fallback (stays on device, no GPU→CPU transfer)
        if self.ties == 'efron' and self._efron_pre_np is not None:
            eta = X_s @ coef_dev
            eta_shifted = eta - xp.max(eta)
            try:
                grad, hess = self._efron_grad_hess_backend(eta_shifted, X_s, xp)
                return grad, hess
            except Exception as exc:
                if is_cupy or is_torch_cuda:
                    raise RuntimeError(
                        f"CoxPH {xp.__name__} Efron gradient/Hessian path failed; "
                        "no CPU fallback is performed for an explicit GPU backend."
                    ) from exc

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
        """Correct backend-native Efron gradient/Hessian on CuPy.

        The historical multiblock kernel omits tied-failure E1/E2 terms for
        ``d > 1``.  Route through the audited shared counting-process engine
        until that specialized kernel has a complete Efron implementation.
        """
        from statgpu.survival._risk_sets import cox_counting_process_objective

        result = cox_counting_process_objective(
            coef_dev,
            X_s,
            self._time_sorted,
            self._event_sorted,
            ties="efron",
        )
        return result["score"], -result["information"]

    def _gpu_loglik(self, coef_dev, X_s):
        """Compute log-likelihood via GPU kernel."""
        eta = X_s @ coef_dev
        return self._gpu_loglik_from_eta(eta, X_s)

    def _gpu_loglik_from_eta(self, eta, X_s):
        """Compute log-likelihood from precomputed eta on GPU.

        CuPy uses its CUDA kernel when available.  Torch CUDA uses only Torch
        tensor operations and therefore does not require CuPy.
        """
        xp = _get_xp(X_s)
        is_cupy = xp.__name__ == "cupy"
        is_torch_cuda = xp.__name__ == "torch" and X_s.is_cuda

        if self.ties == 'efron' and self._efron_pre_np is not None:
            if is_torch_cuda:
                return self._efron_loglik_backend(eta, X_s, xp)

            if is_cupy:
                try:
                    import cupy as cp
                    from statgpu.survival._cox_efron_cuda import compute_efron_loglik_raw_csr

                    eta_shifted = eta - cp.max(eta)
                    exp_eta = cp.exp(eta_shifted)
                    risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
                    _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
                    first_idx_uft_dev = cp.asarray(first_idx_uft, dtype=cp.int32)
                    if self._efron_csr is not None:
                        result = compute_efron_loglik_raw_csr(
                            eta_shifted, exp_eta, risk_sum,
                            self._efron_csr[4], self._efron_csr[5],
                            first_idx_uft_dev, nuft, cupy_module=cp
                        )
                        return result
                except (ImportError, RuntimeError):
                    # The backend-native implementation remains on CuPy and
                    # is the explicit fallback when the custom kernel is not
                    # available.
                    pass
                return self._efron_loglik_backend(eta, X_s, xp)

        # Breslow: can compute directly on any backend
        if self.ties == 'breslow' and self._breslow_pre_np is not None:
            eta_shift = xp.max(eta)
            eta_shifted = eta - eta_shift
            exp_eta = xp.exp(eta_shifted)
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
            event_eta = eta_shifted[event_mask_dev]
            if xp.__name__ == "torch":
                return xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at))
            return float(xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at)))

        return None

    def _efron_event_indices_backend(self, X, xp):
        """Return cached event-index tensors for the active GPU backend."""
        _, uft_ix, _, _, _, _ = self._efron_pre_np
        if xp.__name__ == "numpy":
            return uft_ix

        if xp.__name__ == "torch":
            key = ("torch", str(X.device))
        else:
            key = ("cupy", int(X.device.id))

        cached = self._efron_backend_index_cache.get(key)
        if cached is not None:
            return cached

        if xp.__name__ == "torch":
            indices = tuple(
                xp.as_tensor(ix, dtype=xp.long, device=X.device) for ix in uft_ix
            )
        else:
            indices = tuple(xp.asarray(ix, dtype=xp.int64) for ix in uft_ix)
        self._efron_backend_index_cache[key] = indices
        return indices

    def _efron_loglik_backend(self, eta, X, xp):
        """Efron log partial likelihood using only the active array backend."""
        _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
        if nuft == 0:
            return _xp_zeros((), dtype=xp.float64, ref_arr=eta)

        # Shifting eta is exactly invariant for a Cox partial likelihood and
        # avoids overflow in exp() for every backend.
        eta_shifted = eta - xp.max(eta)
        exp_eta = xp.exp(eta_shifted)
        if xp.__name__ == "torch":
            risk_sum = xp.cumsum(exp_eta.flip(0), dim=0).flip(0)
        else:
            risk_sum = xp.cumsum(exp_eta[::-1])[::-1]

        event_indices = self._efron_event_indices_backend(X, xp)
        loglik = _xp_zeros((), dtype=xp.float64, ref_arr=eta)
        for g in range(nuft):
            ix_ev = event_indices[g]
            d = int(ix_ev.shape[0])
            if d == 0:
                continue
            risk_at_t = risk_sum[int(first_idx_uft[g])]
            sum_events = xp.sum(exp_eta[ix_ev])
            if xp.__name__ == "torch":
                k_vals = xp.arange(d, dtype=xp.float64, device=X.device)
                denom = xp.clamp(
                    risk_at_t - (k_vals / d) * sum_events, min=1e-300
                )
            else:
                k_vals = xp.arange(d, dtype=xp.float64)
                denom = xp.maximum(
                    risk_at_t - (k_vals / d) * sum_events, 1e-300
                )
            loglik = loglik + xp.sum(eta_shifted[ix_ev]) - xp.sum(xp.log(denom))
        return loglik

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

        eta_np = eta_np - np.max(eta_np)
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
        eta_np = eta_np - np.max(eta_np)
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

        _, _, _, _, nuft, first_idx_uft = self._efron_pre_np
        event_indices = self._efron_event_indices_backend(X, xp)

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
        xp0 = _xp_zeros((), dtype=xp.float64, ref_arr=X)
        xp1 = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        xp2 = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        grad = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        hess = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)

        for g in range(nuft - 1, -1, -1):
            # ── Enter phase: add samples with time in [uft[g], uft[g+1]) ──
            enter_start = int(first_idx_uft[g])
            enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
            if enter_end > enter_start:
                xp0 = xp0 + (risk_sum[enter_start] - risk_sum[enter_end])
                xp1 = xp1 + (risk_X_sum[enter_start] - risk_X_sum[enter_end])
                blk = X_exp[enter_start:enter_end]
                xp2 = xp2 + (blk.T @ X[enter_start:enter_end])

            # ── Fail phase: Efron correction ──
            ix_ev = event_indices[g]
            d = int(ix_ev.shape[0])
            if d == 0:
                continue

            v = X[ix_ev]
            elx = exp_eta[ix_ev]
            xp0f = xp.sum(elx)
            xp1f = v.T @ elx
            xp2f = (v * elx[:, None]).T @ v

            # Vectorized Efron correction over tie size d
            if xp.__name__ == "torch":
                J = xp.arange(d, dtype=xp.float64, device=X.device) / d
            else:
                J = xp.arange(d, dtype=xp.float64) / d
            c0 = xp0 - J * xp0f
            if xp.__name__ == "torch":
                c0 = xp.clamp(c0, min=1e-300)
            else:
                c0 = xp.maximum(c0, 1e-300)
            inv = 1.0 / c0
            sum_inv = xp.sum(inv)
            sum_J = xp.sum(J * inv)
            sum_aa = xp.sum(inv * inv)
            sum_bb = xp.sum((J * inv) * (J * inv))
            sum_ab = xp.sum(inv * (J * inv))

            grad = grad + xp.sum(v, axis=0) - (xp1 * sum_inv - xp1f * sum_J)

            hess = hess - xp2 * sum_inv + xp2f * sum_J
            hess = hess + (
                sum_aa * xp.outer(xp1, xp1)
                + sum_bb * xp.outer(xp1f, xp1f)
                - sum_ab * (xp.outer(xp1, xp1f) + xp.outer(xp1f, xp1))
            )

        return grad, hess

    def _cpu_grad_hess(self, eta_np, time_np, event_np):
        """Compute gradient and Hessian in numpy."""
        X_np = _to_numpy(self._X_sorted)
        p = X_np.shape[1]
        eta_np = eta_np - np.max(eta_np)
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

        return grad, hess

    def _cpu_fused_loglik_grad(self, eta_np, X_np, time_np, event_np):
        """Fused loglik + gradient for Efron — single pass.

        Shares suffix sums across loglik and gradient computation.
        """
        n, p = X_np.shape
        eta_np = eta_np - np.max(eta_np)
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

    def _cpu_loglik_grad(self, eta_np, X_np):
        """Compute CPU log likelihood and score without allocating a Hessian."""
        n, p = X_np.shape
        eta_shift = eta_np - np.max(eta_np)
        exp_eta = np.exp(eta_shift)
        X_exp = X_np * exp_eta[:, None]
        risk_sum = np.zeros(n + 1, dtype=np.float64)
        risk_sum[:n] = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.zeros((n + 1, p), dtype=np.float64)
        risk_X_sum[:n] = np.cumsum(X_exp[::-1], axis=0)[::-1]
        event_mask = self._event_np == 1
        if not np.any(event_mask):
            return 0.0, np.zeros(p, dtype=np.float64)

        if self.ties == "breslow":
            first_idx, counts = self._breslow_pre_np
            risk_at = np.maximum(risk_sum[first_idx], 1e-300)
            mean_x = risk_X_sum[first_idx] / risk_at[:, None]
            loglik = float(
                np.sum(eta_shift[event_mask])
                - np.sum(counts * np.log(risk_at))
            )
            grad = np.sum(X_np[event_mask], axis=0) - np.sum(
                counts[:, None] * mean_x, axis=0
            )
            return loglik, grad

        _, uft_ix, _, _, nuft, first_idx_uft = self._efron_pre_np
        loglik = 0.0
        grad = np.zeros(p, dtype=np.float64)
        for group in range(nuft):
            event_idx = uft_ix[group]
            d = int(event_idx.shape[0])
            if d == 0:
                continue
            first_idx = int(first_idx_uft[group])
            s0 = risk_sum[first_idx]
            s1 = risk_X_sum[first_idx]
            event_exp = exp_eta[event_idx]
            event_x = X_np[event_idx]
            e0 = float(np.sum(event_exp))
            e1 = event_x.T @ event_exp
            fractions = np.arange(d, dtype=np.float64) / d
            denominators = np.maximum(s0 - fractions * e0, 1e-300)
            loglik += float(np.sum(eta_shift[event_idx])) - float(
                np.sum(np.log(denominators))
            )
            grad += np.sum(event_x, axis=0)
            grad -= np.sum(
                (s1[None, :] - fractions[:, None] * e1[None, :])
                / denominators[:, None],
                axis=0,
            )
        return loglik, grad

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

        return ll, grad, hess
