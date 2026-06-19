"""
Cox partial likelihood loss for survival analysis.

Negative log partial likelihood with Breslow/Efron tie handling.
Self-contained implementation as a LossBase subclass so it can use
the generic penalty/solver infrastructure (Newton, L-BFGS, FISTA).

Supports numpy / cupy / torch backends via _xp dispatch.
All computation stays on the input device — no implicit CPU transfers.

Matches R's survival::coxph() interface.
"""

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp, _xp_zeros, _xp_asarray
from statgpu.backends._utils import _to_float_scalar
from ._base import LossBase
from ._registry import register_loss


def _xp_argsort(x, xp):
    """argsort that works across numpy/cupy/torch."""
    if xp.__name__ == "torch":
        return xp.argsort(x, stable=True)
    return xp.argsort(x)


def _xp_unique(x, xp):
    """unique returning (values, inverse_indices, counts) across backends."""
    if xp.__name__ == "torch":
        vals, inv, counts = xp.unique(x, return_inverse=True, return_counts=True)
        return vals, inv, counts
    vals, inv, counts = xp.unique(x, return_inverse=True, return_counts=True)
    return vals, inv, counts


def _xp_searchsorted(sorted_arr, values, side, xp):
    """searchsorted across numpy/cupy/torch."""
    if xp.__name__ == "torch":
        right = side == "right"
        return xp.searchsorted(sorted_arr, values, right=right)
    return xp.searchsorted(sorted_arr, values, side=side)


def _xp_bincount(x, weights, minlength, xp):
    """bincount across numpy/cupy/torch."""
    if xp.__name__ == "torch":
        if weights is not None:
            return xp.bincount(x.long(), weights=weights, minlength=minlength).to(x.dtype)
        return xp.bincount(x.long(), minlength=minlength).to(x.dtype)
    return xp.bincount(x, weights=weights, minlength=minlength).astype(np.float64)


def _xp_zeros_int(n, xp, ref):
    """Create int64 zeros on same device as ref."""
    if xp.__name__ == "torch":
        return xp.zeros(n, dtype=xp.int64, device=ref.device)
    return xp.zeros(n, dtype=np.int64)


@register_loss('cox_ph')
class CoxPartialLikelihoodLoss(LossBase):
    """Cox proportional hazards negative log partial likelihood.

    This loss is a self-contained LossBase subclass for Cox PH, enabling
    use with the generic penalty/solver infrastructure (Newton, FISTA,
    L-BFGS, etc.).

    Parameters
    ----------
    ties : str, default='breslow'
        Method for handling ties: 'breslow' or 'efron'.

    Notes
    -----
    - Supports numpy/cupy/torch backends. All computation stays on the
      input device — no implicit CPU transfers.
    - ``preprocess()`` sorts data by time and caches risk-set structures.
    - ``value()`` returns **negative** log partial likelihood (for minimization).
    - ``gradient()`` returns negative gradient of log-lik (for minimization).
    - ``hessian()`` returns negative Hessian of log-lik (for minimization).
    - Data ``y`` passed to the loss should be a dict with keys ``'time'``
      and ``'event'``, or a (n, 2) array where column 0 is time and column 1
      is event.
    - ``sample_weight`` is **not supported** for Cox PH.
    """

    name = "cox_ph"
    y_type = "survival"
    smooth_gradient = True
    has_hessian = True

    # Optimization hints
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
        self._efron_pre = None
        self._breslow_pre = None
        self._n_events = 0

    def preprocess(self, X, y):
        """Sort data by time and precompute risk-set structures.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariate matrix.
        y : array-like
            Survival response. Either:
            - dict with 'time' and 'event' keys
            - (n, 2) array: column 0 = time, column 1 is event

        Returns
        -------
        X_sorted : array
            Covariate matrix sorted by time ascending (same backend as input).
        y_sorted : array
            Pseudo-response (zeros) sorted by time ascending.
        """
        xp = _get_xp(X)

        # Extract time and event from y, move to same device
        if isinstance(y, dict):
            time = _xp_asarray(y['time'], dtype=xp.float64, ref_arr=X)
            event = _xp_asarray(y['event'], dtype=xp.float64, ref_arr=X)
        else:
            y_arr = _xp_asarray(y, dtype=xp.float64, ref_arr=X)
            if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
                time = y_arr[:, 0]
                event = y_arr[:, 1]
            else:
                raise ValueError(
                    "y must be a dict with 'time'/'event' keys or (n, 2) array"
                )

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

        # Count events using backend-aware method
        if xp.__name__ == "torch":
            self._n_events = int(xp.sum(self._event_sorted).item())
        else:
            self._n_events = int(xp.sum(self._event_sorted))

        # Precompute tie structure
        if self.ties == 'efron':
            self._efron_pre = self._build_efron_pre(
                self._time_sorted, self._event_sorted, xp
            )
            self._breslow_pre = None
        else:
            self._breslow_pre = self._build_breslow_pre(
                self._time_sorted, self._event_sorted, xp
            )
            self._efron_pre = None

        y_sorted = _xp_zeros(X_arr.shape[0], dtype=xp.float64, ref_arr=X_arr)
        return self._X_sorted, y_sorted

    @staticmethod
    def _build_breslow_pre(time, event, xp):
        """Build Breslow tie group structure."""
        event_mask = event == 1
        event_times = time[event_mask]
        uft, _, counts = _xp_unique(event_times, xp)
        first_idx = _xp_searchsorted(time, uft, side="left", xp=xp)
        return first_idx, counts

    @staticmethod
    def _build_efron_pre(time, event, xp):
        """Build Efron tie group structure."""
        event_mask = event == 1
        if xp.__name__ == "torch":
            event_idx = xp.where(event_mask)[0]
        else:
            event_idx = xp.where(event_mask)[0]
        event_times = time[event_idx]
        uft, inv, _ = _xp_unique(event_times, xp)
        nuft = int(uft.shape[0])

        uft_ix = []
        for g in range(nuft):
            mask = inv == g
            uft_ix.append(event_idx[mask])

        first_idx_uft = _xp_searchsorted(time, uft, side="left", xp=xp)
        risk_enter = [_xp_searchsorted(time, uft[g], side="left", xp=xp) for g in range(nuft)]
        risk_exit = [_xp_searchsorted(time, uft[g], side="right", xp=xp) for g in range(nuft)]

        return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft

    def _check_stale(self, X, y):
        """Warn if X/y differ from cached sorted data."""
        if not self._sorted:
            return
        if X is not self._X_sorted:
            self._sorted = False

    def value(self, X, y, coef, sample_weight=None) -> float:
        """Negative log partial likelihood (for minimization)."""
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X, y)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        loglik = self._compute_log_likelihood(
            coef_dev, X_s, self._time_sorted, self._event_sorted
        )
        n = X_s.shape[0]
        return -_to_float_scalar(loglik) / n

    def gradient(self, X, y, coef, sample_weight=None):
        """Negative gradient of log partial likelihood (for minimization)."""
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X, y)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        grad, _ = self._compute_gradient_hessian(
            coef_dev, X_s, self._time_sorted, self._event_sorted
        )
        n = X_s.shape[0]
        return -grad / n

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Compute negative loglik and negative gradient in one pass."""
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X, y)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        n = X_s.shape[0]

        # Fused: compute eta, exp_eta once
        eta = X_s @ coef_dev
        exp_eta = xp.exp(eta)

        loglik = self._compute_loglik_from_stats(eta, exp_eta, X_s, self._time_sorted, self._event_sorted)
        grad = self._compute_grad_from_stats(eta, exp_eta, X_s, self._time_sorted, self._event_sorted)
        return -_to_float_scalar(loglik) / n, -grad / n

    def hessian(self, X, y, coef, sample_weight=None):
        """Negative Hessian of log partial likelihood (for minimization)."""
        if sample_weight is not None:
            raise NotImplementedError(
                "CoxPartialLikelihoodLoss does not support sample_weight"
            )
        if not self._sorted:
            self.preprocess(X, y)
        else:
            self._check_stale(X, y)
            if not self._sorted:
                self.preprocess(X, y)

        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s)
        _, hess = self._compute_gradient_hessian(
            coef_dev, X_s, self._time_sorted, self._event_sorted
        )
        n = X_s.shape[0]
        return -hess / n

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        """Lipschitz constant via max eigenvalue of observed information / n."""
        from statgpu.backends._array_ops import _max_eigval_power
        if not self._sorted:
            self.preprocess(X, y)
        X_s = self._X_sorted
        xp = _get_xp(X_s)
        coef_dev = _xp_asarray(coef, dtype=xp.float64, ref_arr=X_s) if coef is not None else _xp_zeros(X_s.shape[1], dtype=xp.float64, ref_arr=X_s)
        _, hess = self._compute_gradient_hessian(
            coef_dev, X_s, self._time_sorted, self._event_sorted
        )
        n = X_s.shape[0]
        neg_hess = -hess / n
        return _max_eigval_power(neg_hess)

    # ── Internal Cox computations (backend-aware) ────────────────────

    def _compute_log_likelihood(self, beta, X, time, event):
        """Compute log partial likelihood."""
        xp = _get_xp(X)
        eta = X @ beta
        exp_eta = xp.exp(eta)
        return self._compute_loglik_from_stats(eta, exp_eta, X, time, event)

    def _compute_loglik_from_stats(self, eta, exp_eta, X, time, event):
        """Compute log partial likelihood from precomputed eta/exp_eta."""
        xp = _get_xp(X)
        risk_sum = xp.cumsum(exp_eta.flip(0), dim=0).flip(0) if xp.__name__ == "torch" else xp.cumsum(exp_eta[::-1])[::-1]
        event_mask = event == 1

        if not xp.any(event_mask):
            return _xp_zeros(1, dtype=xp.float64, ref_arr=X)[0] if xp.__name__ == "torch" else xp.float64(0.0)

        if self.ties == 'breslow':
            breslow_pre = self._breslow_pre
            if breslow_pre is not None and breslow_pre[0].shape[0] > 0:
                first_idx = breslow_pre[0]
                counts = breslow_pre[1]
            else:
                event_times = time[event_mask]
                uft, _, counts = _xp_unique(event_times, xp)
                first_idx = _xp_searchsorted(time, uft, side="left", xp=xp)
            risk_at = risk_sum[first_idx]
            event_eta = eta[event_mask]
            if xp.__name__ == "torch":
                return xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at))
            return float(xp.sum(event_eta) - xp.sum(counts * xp.log(risk_at)))

        # Efron
        efron_pre = self._efron_pre
        if efron_pre is not None:
            uft, uft_ix, _, _, nuft, first_idx_uft = efron_pre
            all_eta_sum = _xp_zeros(1, dtype=xp.float64, ref_arr=X)[0] if xp.__name__ == "torch" else 0.0
            all_log_denom_sum = _xp_zeros(1, dtype=xp.float64, ref_arr=X)[0] if xp.__name__ == "torch" else 0.0
            for g in range(nuft):
                ix_ev = uft_ix[g]
                d = int(ix_ev.shape[0])
                if d == 0:
                    continue
                idx = int(first_idx_uft[g])
                risk_at_t = risk_sum[idx]
                sum_events = xp.sum(exp_eta[ix_ev])
                all_eta_sum = all_eta_sum + xp.sum(eta[ix_ev])
                k_vals = xp.arange(d, dtype=xp.float64, device=X.device) if xp.__name__ == "torch" else xp.arange(d, dtype=xp.float64)
                denom = risk_at_t - (k_vals / d) * sum_events
                denom = xp.maximum(denom, xp.float64(1e-300)) if xp.__name__ == "torch" else xp.maximum(denom, 1e-300)
                all_log_denom_sum = all_log_denom_sum + xp.sum(xp.log(denom))
            result = all_eta_sum - all_log_denom_sum
            return result if xp.__name__ == "torch" else float(result)

        # No precomputation
        if xp.__name__ == "torch":
            event_idx = xp.where(event_mask)[0]
        else:
            event_idx = xp.where(event_mask)[0]
        event_times = time[event_idx]
        uft, inv, counts = _xp_unique(event_times, xp)
        first_idx = _xp_searchsorted(time, uft, side="left", xp=xp)
        risk_at = risk_sum[first_idx]
        nuft = int(uft.shape[0])

        # bincount equivalent
        sum_events = _xp_bincount(inv, weights=exp_eta[event_idx], minlength=nuft, xp=xp)
        sum_eta_events = _xp_bincount(inv, weights=eta[event_idx], minlength=nuft, xp=xp)

        ll = xp.sum(sum_eta_events)
        for g in range(nuft):
            d = int(counts[g])
            if d == 0:
                continue
            k_vals = xp.arange(d, dtype=xp.float64, device=X.device) if xp.__name__ == "torch" else xp.arange(d, dtype=xp.float64)
            denom = risk_at[g] - (k_vals / d) * sum_events[g]
            denom = xp.maximum(denom, xp.float64(1e-300)) if xp.__name__ == "torch" else xp.maximum(denom, 1e-300)
            ll = ll - xp.sum(xp.log(denom))
        return ll if xp.__name__ == "torch" else float(ll)

    def _compute_grad_from_stats(self, eta, exp_eta, X, time, event):
        """Compute gradient from precomputed eta/exp_eta."""
        grad, _ = self._compute_grad_hess_from_stats(eta, exp_eta, X, time, event)
        return grad

    def _compute_gradient_hessian(self, beta, X, time, event):
        """Compute gradient and Hessian of log partial likelihood."""
        xp = _get_xp(X)
        eta = X @ beta
        exp_eta = xp.exp(eta)
        return self._compute_grad_hess_from_stats(eta, exp_eta, X, time, event)

    def _compute_grad_hess_from_stats(self, eta, exp_eta, X, time, event):
        """Compute gradient and Hessian from precomputed eta/exp_eta."""
        xp = _get_xp(X)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        risk_sum = xp.cumsum(exp_eta.flip(0), dim=0).flip(0) if xp.__name__ == "torch" else xp.cumsum(exp_eta[::-1])[::-1]
        X_exp_eta = X * exp_eta[:, None]
        risk_X_sum = xp.cumsum(X_exp_eta.flip(0), dim=0).flip(0) if xp.__name__ == "torch" else xp.cumsum(X_exp_eta[::-1], axis=0)[::-1]

        event_mask = event == 1
        zero_vec = _xp_zeros(n_features, dtype=xp.float64, ref_arr=X)
        zero_mat = _xp_zeros((n_features, n_features), dtype=xp.float64, ref_arr=X)

        if self.ties == 'breslow':
            grad = zero_vec.copy() if hasattr(zero_vec, 'copy') else zero_vec + 0
            breslow_pre = self._breslow_pre
            has_events = bool(xp.any(event_mask)) if xp.__name__ != "torch" else bool(xp.any(event_mask).item())

            if has_events and breslow_pre is not None and breslow_pre[0].shape[0] > 0:
                first_idx = breslow_pre[0]
                counts = breslow_pre[1]
                sum_X_events = xp.sum(X[event_mask], axis=0)
                risk_sum_at = risk_sum[first_idx]
                E_X = risk_X_sum[first_idx] / risk_sum_at[:, None]
                grad = sum_X_events - xp.sum(E_X * counts[:, None], axis=0)

            # Hessian
            if not has_events:
                hess = zero_mat.copy() if hasattr(zero_mat, 'copy') else zero_mat + 0
            else:
                if breslow_pre is not None and breslow_pre[0].shape[0] > 0:
                    first_idx = breslow_pre[0]
                    counts = breslow_pre[1]
                else:
                    event_times = time[event_mask]
                    uft, _, counts = _xp_unique(event_times, xp)
                    first_idx = _xp_searchsorted(time, uft, side="left", xp=xp)

                # Tensor grouped path for small p
                if n_features <= 24 and int(first_idx.shape[0]) <= 512:
                    x2_weighted = xp.einsum("ni,nj,n->nij", X, X, exp_eta)
                    if xp.__name__ == "torch":
                        risk_X2_sum = xp.cumsum(x2_weighted.flip(0), dim=0).flip(0)
                    else:
                        risk_X2_sum = xp.cumsum(x2_weighted[::-1], axis=0)[::-1]
                    risk_sum_at = risk_sum[first_idx]
                    E_X = risk_X_sum[first_idx] / risk_sum_at[:, None]
                    E_XX = risk_X2_sum[first_idx] / risk_sum_at[:, None, None]
                    centered = E_XX - xp.einsum("ni,nj->nij", E_X, E_X)
                    hess = -xp.sum(centered * counts[:, None, None], axis=0)
                else:
                    # Incremental grouped path
                    X_exp = X * exp_eta[:, None]
                    risk_X2 = X_exp.T @ X
                    hess = zero_mat.copy() if hasattr(zero_mat, 'copy') else zero_mat + 0
                    prev_idx = 0
                    for g in range(int(first_idx.shape[0])):
                        idx = int(first_idx[g])
                        if idx > prev_idx:
                            blk = slice(prev_idx, idx)
                            risk_X2 = risk_X2 - X_exp[blk].T @ X[blk]
                            prev_idx = idx
                        s0 = float(risk_sum[idx])
                        if s0 <= 1e-300:
                            continue
                        s1 = risk_X_sum[idx]
                        ex = s1 / s0
                        hess = hess - float(counts[g]) * (risk_X2 / s0 - xp.outer(ex, ex))
        else:
            # Efron
            eta_efron = eta - xp.max(eta)
            efron_pre = self._efron_pre
            if efron_pre is not None:
                uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = efron_pre
                grad, hess = self._efron_grad_hess(eta_efron, X, risk_enter, risk_exit, uft_ix, nuft, xp)
            else:
                grad, hess = self._efron_grad_hess_no_pre(eta, X, time, event, xp)

        return grad, hess

    @staticmethod
    def _efron_grad_hess(eta, X, risk_enter, risk_exit, uft_ix, nuft, xp):
        """Efron gradient and Hessian (backend-aware)."""
        n, p = int(X.shape[0]), int(X.shape[1])
        exp_eta = xp.exp(eta)
        grad = _xp_zeros(p, dtype=xp.float64, ref_arr=X)
        hess = _xp_zeros((p, p), dtype=xp.float64, ref_arr=X)
        X_exp = X * exp_eta[:, None]

        for g in range(nuft):
            ix_ev = uft_ix[g]
            d = int(ix_ev.shape[0])
            if d == 0:
                continue

            re = int(risk_enter[g]) if not hasattr(risk_enter[g], 'item') else int(risk_enter[g].item())
            s0 = xp.sum(exp_eta[re:])
            s1 = xp.sum(X_exp[re:], axis=0)
            sum_ev_exp = xp.sum(exp_eta[ix_ev])
            sum_ev_X = xp.sum(X[ix_ev], axis=0)

            for k in range(d):
                frac = float(k) / float(d)
                denom = s0 - frac * sum_ev_exp
                if _to_float_scalar(denom) <= 1e-300:
                    continue
                E_X = s1 / denom
                grad = grad + (sum_ev_X / d) - E_X
                X_exp_re = X_exp[re:]
                E_XX = X_exp_re.T @ X[re:] / denom
                hess = hess - (E_XX - xp.outer(E_X, E_X))

        return grad, hess

    @staticmethod
    def _efron_grad_hess_no_pre(eta, X, time, event, xp):
        """Efron gradient/Hessian without precomputation."""
        event_mask = event == 1
        if xp.__name__ == "torch":
            event_idx = xp.where(event_mask)[0]
        else:
            event_idx = xp.where(event_mask)[0]
        event_times = time[event_idx]
        uft, inv, _ = _xp_unique(event_times, xp)
        nuft = int(uft.shape[0])
        uft_ix = [event_idx[inv == g] for g in range(nuft)]
        risk_enter = [_xp_searchsorted(time, uft[g], side="left", xp=xp) for g in range(nuft)]
        risk_exit = [_xp_searchsorted(time, uft[g], side="right", xp=xp) for g in range(nuft)]
        return CoxPartialLikelihoodLoss._efron_grad_hess(eta, X, risk_enter, risk_exit, uft_ix, nuft, xp)
