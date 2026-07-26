"""Cox partial-likelihood loss for survival analysis.

The public loss API delegates to the shared counting-process risk-set engine so
Breslow and Efron likelihoods use the same numerical definition as
``statgpu.survival.CoxPH`` on NumPy, CuPy, and Torch.  In particular, every
failure time is normalized inside its own risk set; a global linear-predictor
shift is not sufficient once the sample attaining the maximum has left a later
risk set.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends._array_ops import (
    _max_eigval_power,
    _xp as _get_xp,
    _xp_asarray,
    _xp_zeros,
)
from statgpu.backends._utils import _to_float_scalar, _to_numpy
from statgpu.survival._risk_sets import cox_counting_process_objective

from ._base import LossBase
from ._registry import register_loss


def _build_efron_pre_numpy(time_np, event_np):
    """Build deterministic Efron failure groups for compatibility helpers."""
    event_mask = event_np == 1
    event_idx = np.where(event_mask)[0]
    event_times = time_np[event_idx]
    uft, inv = np.unique(event_times, return_inverse=True)
    nuft = len(uft)
    uft_ix = [event_idx[inv == group].astype(np.int64) for group in range(nuft)]
    first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    risk_enter = [[np.int64(index)] for index in first_idx]
    risk_exit = [
        [np.int64(np.searchsorted(time_np, value, side="right"))] for value in uft
    ]
    return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx


def _build_breslow_pre_numpy(time_np, event_np):
    """Build the historical two-array Breslow preprocessing tuple."""
    event_times = time_np[event_np == 1]
    uft, counts = np.unique(event_times, return_counts=True)
    first_idx = np.searchsorted(time_np, uft, side="left").astype(np.int64)
    return first_idx, counts.astype(np.float64)


def _build_breslow_event_indices_numpy(time_np, event_np):
    """Return event-row indices grouped in the same order as Breslow predata."""
    event_idx = np.flatnonzero(event_np == 1)
    event_times = time_np[event_idx]
    _, inverse = np.unique(event_times, return_inverse=True)
    return [
        event_idx[inverse == group].astype(np.int64)
        for group in range(int(inverse.max()) + 1)
    ]


def _backend_index(values, xp, reference):
    """Create integer indices on the backend/device of ``reference``."""
    if xp.__name__ == "torch":
        return xp.as_tensor(values, dtype=xp.long, device=reference.device)
    return xp.asarray(values, dtype=xp.int64)


def _backend_zeros(shape, xp, reference):
    if xp.__name__ == "torch":
        return xp.zeros(shape, dtype=reference.dtype, device=reference.device)
    return xp.zeros(shape, dtype=reference.dtype)


def _sum(value, xp, axis=None):
    if xp.__name__ == "torch":
        return xp.sum(value) if axis is None else xp.sum(value, dim=axis)
    return xp.sum(value, axis=axis)


def _transpose2d(value, xp):
    return value.transpose(0, 1) if xp.__name__ == "torch" else value.T


def _is_nonpositive(value) -> bool:
    return _to_float_scalar(value) <= 0.0


@register_loss("cox_ph")
class CoxPartialLikelihoodLoss(LossBase):
    """Negative Cox partial likelihood with Breslow or Efron ties.

    The response is either a ``{"time": ..., "event": ...}`` dictionary or an
    ``(n, 2)`` array.  ``sample_weight`` is intentionally unsupported because
    case weights require a separate, explicitly documented survival contract.
    """

    name = "cox_ph"
    y_type = "survival"
    smooth_gradient = True
    has_hessian = True

    _lipschitz_safety = 1.0
    _has_constant_hessian = False

    def __init__(self, ties: str = "breslow"):
        ties = str(ties).lower()
        if ties not in {"breslow", "efron"}:
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
        self._breslow_event_indices_np = None
        self._efron_csr = None
        self._efron_backend_index_cache = {}
        self._n_events = 0
        self._x_reference = None

    def _ensure_sorted(self, X, y):
        if self._sorted and X is self._X_sorted:
            return
        self._sorted = False
        self.preprocess(X, y)

    def preprocess(self, X, y):
        """Validate, center, and stably sort right-censored survival data."""
        xp = _get_xp(X)
        if isinstance(y, dict):
            if "time" not in y or "event" not in y:
                raise ValueError("survival y dict must contain time and event")
            time = _xp_asarray(y["time"], dtype=xp.float64, ref_arr=X)
            event = _xp_asarray(y["event"], dtype=xp.float64, ref_arr=X)
        else:
            y_arr = _xp_asarray(y, dtype=xp.float64, ref_arr=X)
            if y_arr.ndim != 2 or y_arr.shape[1] < 2:
                raise ValueError("y must be dict or (n, 2) array")
            time, event = y_arr[:, 0], y_arr[:, 1]

        X_arr = _xp_asarray(X, dtype=xp.float64, ref_arr=X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if time.ndim != 1 or event.ndim != 1:
            raise ValueError("time and event must have shape (n_samples,)")
        if int(time.shape[0]) != int(X_arr.shape[0]) or int(event.shape[0]) != int(
            X_arr.shape[0]
        ):
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
        if _to_float_scalar(xp.sum(event)) <= 0:
            raise ValueError("at least one observed event is required")

        self._x_reference = (
            xp.mean(X_arr, dim=0)
            if xp.__name__ == "torch"
            else xp.mean(X_arr, axis=0)
        )
        X_arr = X_arr - self._x_reference.reshape(1, -1)
        order = (
            xp.argsort(time, stable=True)
            if xp.__name__ == "torch"
            else xp.argsort(time, kind="stable")
            if xp.__name__ == "numpy"
            else xp.argsort(time)
        )
        self._X_sorted = X_arr[order]
        self._time_sorted = time[order]
        self._event_sorted = event[order]
        self._order = order
        self._sorted = True
        self._n_events = int(_to_float_scalar(xp.sum(self._event_sorted)))
        self._efron_backend_index_cache = {}

        self._time_np = np.asarray(_to_numpy(self._time_sorted), dtype=np.float64)
        self._event_np = np.asarray(_to_numpy(self._event_sorted), dtype=np.float64)
        if self.ties == "efron":
            self._efron_pre_np = _build_efron_pre_numpy(
                self._time_np, self._event_np
            )
            self._breslow_pre_np = None
        else:
            self._breslow_pre_np = _build_breslow_pre_numpy(
                self._time_np, self._event_np
            )
            self._breslow_event_indices_np = _build_breslow_event_indices_numpy(
                self._time_np, self._event_np
            )
            self._efron_pre_np = None
        self._efron_csr = None
        return self._X_sorted, _xp_zeros(
            X_arr.shape[0], dtype=xp.float64, ref_arr=X_arr
        )

    @staticmethod
    def _reject_sample_weight(sample_weight):
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )

    def _shared_objective(self, coef_dev, *, compute_derivatives: bool):
        """Use the audited three-backend risk-set implementation."""
        return cox_counting_process_objective(
            coef_dev,
            self._X_sorted,
            self._time_sorted,
            self._event_sorted,
            ties=self.ties,
            compute_derivatives=compute_derivatives,
        )

    def value(self, X, y, coef, sample_weight=None) -> float:
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = _xp_asarray(
            coef, dtype=xp.float64, ref_arr=self._X_sorted
        ).reshape(-1)
        eta = self._X_sorted @ coef_dev
        loglik, _, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        return -_to_float_scalar(loglik) / self._X_sorted.shape[0]

    def gradient(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = _xp_asarray(
            coef, dtype=xp.float64, ref_arr=self._X_sorted
        ).reshape(-1)
        eta = self._X_sorted @ coef_dev
        _, score, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        return -score / self._X_sorted.shape[0]

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = _xp_asarray(
            coef, dtype=xp.float64, ref_arr=self._X_sorted
        ).reshape(-1)
        eta = self._X_sorted @ coef_dev
        loglik, score, _ = self._objective_from_eta_backend(
            eta, self._X_sorted, xp, self.ties, compute_information=False
        )
        n = self._X_sorted.shape[0]
        return -_to_float_scalar(loglik) / n, -score / n

    def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = _xp_asarray(
            coef, dtype=xp.float64, ref_arr=self._X_sorted
        ).reshape(-1)
        result = self._shared_objective(coef_dev, compute_derivatives=True)
        n = self._X_sorted.shape[0]
        return -result["score"] / n, result["information"] / n

    def hessian(self, X, y, coef, sample_weight=None):
        return self.fused_gradient_and_hessian(
            X, y, coef, sample_weight=sample_weight
        )[1]

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        self._reject_sample_weight(sample_weight)
        self._ensure_sorted(X, y)
        xp = _get_xp(self._X_sorted)
        coef_dev = (
            _xp_asarray(coef, dtype=xp.float64, ref_arr=self._X_sorted).reshape(-1)
            if coef is not None
            else _xp_zeros(
                self._X_sorted.shape[1],
                dtype=xp.float64,
                ref_arr=self._X_sorted,
            )
        )
        result = self._shared_objective(coef_dev, compute_derivatives=True)
        return _max_eigval_power(result["information"] / self._X_sorted.shape[0])

    # ------------------------------------------------------------------
    # Compatibility helpers used by focused kernel tests.  They share one
    # failure-time-local normalization routine and are not used by the public
    # optimization path.
    # ------------------------------------------------------------------

    def _objective_from_eta_backend(
        self, eta, X, xp, ties, *, compute_information=True
    ):
        n, p = int(X.shape[0]), int(X.shape[1])
        loglik = _backend_zeros((), xp, X)
        score = _backend_zeros((p,), xp, X)
        information = (
            _backend_zeros((p, p), xp, X) if compute_information else None
        )

        if ties == "breslow":
            if self._breslow_pre_np is None:
                first_indices, counts = _build_breslow_pre_numpy(
                    self._time_np, self._event_np
                )
            else:
                first_indices, counts = self._breslow_pre_np
            grouped_event_idx = (
                self._breslow_event_indices_np
                if self._breslow_event_indices_np is not None
                else _build_breslow_event_indices_numpy(
                    self._time_np, self._event_np
                )
            )
        else:
            if self._efron_pre_np is None:
                efron_pre = _build_efron_pre_numpy(self._time_np, self._event_np)
            else:
                efron_pre = self._efron_pre_np
            _, grouped_event_idx, _, _, _, first_indices = efron_pre
            counts = np.asarray(
                [len(indices) for indices in grouped_event_idx], dtype=np.float64
            )

        for first_index, count, event_indices_np in zip(
            first_indices, counts, grouped_event_idx
        ):
            first_index = int(first_index)
            d = int(count)
            if d <= 0:
                continue
            risk_X = X[first_index:n]
            risk_eta = eta[first_index:n]
            shift = xp.max(risk_eta)
            risk_weights = xp.exp(risk_eta - shift)
            s0 = _sum(risk_weights, xp)
            if _is_nonpositive(s0):
                raise FloatingPointError("non-positive Cox risk-set denominator")
            s1 = _transpose2d(risk_X, xp) @ risk_weights
            s2 = (
                _transpose2d(risk_X, xp)
                @ (risk_X * risk_weights.reshape(-1, 1))
                if compute_information
                else None
            )

            event_indices = _backend_index(event_indices_np, xp, X)
            event_X = X[event_indices]
            event_eta = eta[event_indices]
            event_weights = xp.exp(event_eta - shift)
            e0 = _sum(event_weights, xp)
            e1 = _transpose2d(event_X, xp) @ event_weights
            e2 = (
                _transpose2d(event_X, xp)
                @ (event_X * event_weights.reshape(-1, 1))
                if compute_information
                else None
            )

            loglik = loglik + _sum(event_eta, xp)
            score = score + _sum(event_X, xp, axis=0)
            substeps = 1 if ties == "breslow" else d
            for substep in range(substeps):
                frac = 0.0 if ties == "breslow" else float(substep) / float(d)
                denom = s0 - frac * e0
                if _is_nonpositive(denom):
                    raise FloatingPointError("non-positive Cox risk-set denominator")
                a1 = s1 - frac * e1
                mean = a1 / denom
                loglik = loglik - (xp.log(denom) + shift)
                score = score - mean
                if compute_information:
                    a2 = s2 - frac * e2
                    information = information + a2 / denom - xp.outer(mean, mean)
            if ties == "breslow" and d > 1:
                # The loop above consumed one denominator; Breslow repeats that
                # same denominator and moment contribution d times.
                mean = s1 / s0
                loglik = loglik - float(d - 1) * (xp.log(s0) + shift)
                score = score - float(d - 1) * mean
                if compute_information:
                    covariance = s2 / s0 - xp.outer(mean, mean)
                    information = information + float(d - 1) * covariance

        return loglik, score, None if information is None else -information

    def _is_gpu(self, arr):
        xp = _get_xp(arr)
        return xp.__name__ == "cupy" or (
            xp.__name__ == "torch" and bool(arr.is_cuda)
        )

    def _compute_grad_hess(self, coef_dev, X_s):
        result = self._shared_objective(coef_dev, compute_derivatives=True)
        return result["score"], -result["information"]

    def _gpu_loglik(self, coef_dev, X_s):
        result = self._shared_objective(coef_dev, compute_derivatives=False)
        return result["log_likelihood"]

    def _loglik_from_eta(self, eta, X_s):
        xp = _get_xp(X_s)
        return self._objective_from_eta_backend(eta, X_s, xp, self.ties)[0]

    def _gpu_loglik_from_eta(self, eta, X_s):
        return self._loglik_from_eta(eta, X_s)

    def _grad_from_eta(self, eta, X_s):
        xp = _get_xp(X_s)
        return self._objective_from_eta_backend(eta, X_s, xp, self.ties)[1]

    def _cupy_grad_hess(self, coef_dev, X_s):
        return self._compute_grad_hess(coef_dev, X_s)

    def _triton_grad_hess(self, coef_dev, X_s):
        return None

    def _torch_breslow_grad_hess(self, coef_dev, X_s):
        return None

    def _efron_loglik_backend(self, eta, X, xp):
        return self._objective_from_eta_backend(eta, X, xp, "efron")[0]

    def _efron_grad_hess_backend(self, eta, X, xp):
        _, score, hessian = self._objective_from_eta_backend(eta, X, xp, "efron")
        return score, hessian

    def _cpu_loglik_cached(self, eta_np, X_np):
        return self._cpu_loglik(eta_np, self._time_np, self._event_np)

    def _cpu_loglik(self, eta_np, time_np, event_np):
        X_np = np.asarray(_to_numpy(self._X_sorted), dtype=np.float64)
        eta_np = np.asarray(eta_np, dtype=np.float64)
        return float(
            self._objective_from_eta_backend(
                eta_np, X_np, np, self.ties
            )[0]
        )

    def _cpu_grad_hess(self, eta_np, time_np, event_np):
        X_np = np.asarray(_to_numpy(self._X_sorted), dtype=np.float64)
        eta_np = np.asarray(eta_np, dtype=np.float64)
        _, score, hessian = self._objective_from_eta_backend(
            eta_np, X_np, np, self.ties
        )
        return np.asarray(score, dtype=np.float64), np.asarray(
            hessian, dtype=np.float64
        )

    @staticmethod
    def _efron_grad_hess_np(eta, X, efron_pre):
        # Kept only for compatibility with external private-method probes.  Use
        # a temporary lightweight loss so the same stable implementation is used.
        temp = CoxPartialLikelihoodLoss(ties="efron")
        temp._X_sorted = np.asarray(X, dtype=np.float64)
        temp._time_np = np.asarray(efron_pre[0], dtype=np.float64)
        temp._event_np = np.zeros(X.shape[0], dtype=np.float64)
        temp._efron_pre_np = efron_pre
        _, score, hessian = temp._objective_from_eta_backend(
            np.asarray(eta, dtype=np.float64), temp._X_sorted, np, "efron"
        )
        return np.asarray(score), np.asarray(hessian)

    def _cpu_fused_loglik_grad(self, eta_np, X_np, time_np, event_np):
        loglik = self._cpu_loglik(eta_np, time_np, event_np)
        score, _ = self._cpu_grad_hess(eta_np, time_np, event_np)
        return loglik, score, None

    def _cpu_loglik_grad(self, eta_np, X_np):
        loglik = self._cpu_loglik(eta_np, self._time_np, self._event_np)
        score, _ = self._cpu_grad_hess(eta_np, self._time_np, self._event_np)
        return loglik, score

    def _cpu_fused_loglik_grad_hess(self, eta_np, X_np, time_np, event_np):
        loglik = self._cpu_loglik(eta_np, time_np, event_np)
        score, hessian = self._cpu_grad_hess(eta_np, time_np, event_np)
        return loglik, score, hessian
