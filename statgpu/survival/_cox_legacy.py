"""Inactive Cox reference kernels retained for regression comparisons.

The public :class:`statgpu.survival.CoxPH` estimator never dispatches to this
mixin. Keeping the historical CPU, CuPy, and Torch implementations here makes
the canonical estimator path auditable while preserving private numerical
reference entry points used by regression tests.
"""

from __future__ import annotations

import os

import numpy as np

from statgpu._config import Device
from statgpu.inference._distributions_backend import chi2, norm
from statgpu.survival._cox_counting import (
    _is_singular_linalg_error,
    _solve as _solve_counting_information,
)
from statgpu.survival._cox_inference import (
    _invert_information_cupy,
    _invert_information_numpy,
    _invert_information_torch,
)


_DEFAULT_BRESLOW_HESSIAN_MAX_BYTES = 512 * 1024 * 1024


def _breslow_hessian_max_bytes():
    """Return the configured ceiling for explicit ``(n, p, p)`` moments."""
    raw = os.environ.get("STATGPU_BRESLOW_HESSIAN_MAX_BYTES")
    if raw is None:
        return _DEFAULT_BRESLOW_HESSIAN_MAX_BYTES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_BRESLOW_HESSIAN_MAX_BYTES


def _estimate_breslow_tensor_bytes(n, p, n_groups, itemsize=8):
    """Conservatively estimate simultaneously live grouped moment buffers."""
    elements = (
        2 * int(n) * int(p) * int(p)
        + int(n) * int(p)
        + 3 * int(n_groups) * int(p) * int(p)
        + 2 * int(n_groups) * int(p)
    )
    return int(elements) * int(itemsize)

# Optional Cython import for faster Efron gradient/Hessian computation
try:
    from ._cox_efron_cy import efron_grad_hess as _efron_grad_hess_cython
    HAS_CYTHON_EFRON = True
except ImportError:
    HAS_CYTHON_EFRON = False
    _efron_grad_hess_cython = None

try:
    from statgpu.survival._cox_efron_triton import _find_p_ce
    HAS_TRITON_EFRON = True
except ImportError:
    HAS_TRITON_EFRON = False
    _find_p_ce = None


def _unpack_efron_pre6(efron_pre):
    """``(uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft)`` — supports legacy 5-tuple in tests only."""
    if len(efron_pre) == 6:
        return efron_pre
    if len(efron_pre) == 5:
        uft, uft_ix, re, rx, nuft = efron_pre
        return uft, uft_ix, re, rx, nuft, None
    raise ValueError(f"invalid efron_pre length {len(efron_pre)}")


# ── Numba JIT-compiled Efron backward scan (opt-in via env var) ─────
_USE_NUMBA = (
    os.environ.get("STATGPU_USE_NUMBA", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_HAS_NUMBA_EFRON = False
if _USE_NUMBA:
    try:
        from numba import njit

        @njit(cache=True)
        def _efron_backward_scan_numba(
            X, e_linpred, risk_sum, risk_X_sum,
            first_idx_uft, fail_ptr, fail_ind,
            nuft, n, p,
        ):
            """Numba-compiled Efron backward scan — eliminates Python loop overhead."""
            xp0 = 0.0
            xp1 = np.zeros(p)
            xp2 = np.zeros((p, p))
            grad = np.zeros(p)
            hess = np.zeros((p, p))

            for g in range(nuft - 1, -1, -1):
                enter_start = first_idx_uft[g]
                enter_end = n if g == nuft - 1 else first_idx_uft[g + 1]
                if enter_end > enter_start:
                    xp0 += risk_sum[enter_start] - risk_sum[enter_end]
                    for j in range(p):
                        xp1[j] += risk_X_sum[enter_start, j] - risk_X_sum[enter_end, j]
                    for r in range(enter_start, enter_end):
                        elx = e_linpred[r]
                        for j in range(p):
                            for k in range(p):
                                xp2[j, k] += elx * X[r, j] * X[r, k]

                fs = fail_ptr[g]
                fe = fail_ptr[g + 1]
                d = fe - fs
                if d == 0:
                    continue

                xp0f = 0.0
                xp1f = np.zeros(p)
                xp2f = np.zeros((p, p))
                for idx in range(fs, fe):
                    r = fail_ind[idx]
                    elx = e_linpred[r]
                    xp0f += elx
                    for j in range(p):
                        xp1f[j] += elx * X[r, j]
                        for k in range(p):
                            xp2f[j, k] += elx * X[r, j] * X[r, k]

                sum_inv = 0.0
                sum_J = 0.0
                sum_aa = 0.0
                sum_bb = 0.0
                sum_ab = 0.0
                for k in range(d):
                    c0 = xp0 - (float(k) / float(d)) * xp0f
                    if c0 < 1e-300:
                        c0 = 1e-300
                    inv_k = 1.0 / c0
                    J_k = float(k) / float(d) * inv_k
                    sum_inv += inv_k
                    sum_J += J_k
                    sum_aa += inv_k * inv_k
                    sum_bb += J_k * J_k
                    sum_ab += inv_k * J_k

                for idx in range(fs, fe):
                    r = fail_ind[idx]
                    for j in range(p):
                        grad[j] += X[r, j]
                for j in range(p):
                    grad[j] -= xp1[j] * sum_inv - xp1f[j] * sum_J

                for j in range(p):
                    for k in range(p):
                        hess[j, k] -= xp2[j, k] * sum_inv
                        hess[j, k] += xp2f[j, k] * sum_J
                        hess[j, k] += sum_aa * xp1[j] * xp1[k]
                        hess[j, k] += sum_bb * xp1f[j] * xp1f[k]
                        hess[j, k] -= sum_ab * (xp1[j] * xp1f[k] + xp1f[j] * xp1[k])

            return grad, -hess

        _HAS_NUMBA_EFRON = True
    except ImportError:
        pass


def _efron_backward_scan_python(
    X, e_linpred, risk_sum, risk_X_sum,
    first_idx_uft, uft_ix, nuft, n, p,
):
    """Pure Python fallback — same algorithm, no Numba required."""
    xp0 = 0.0
    xp1 = np.zeros(p, dtype=np.float64)
    xp2 = np.zeros((p, p), dtype=np.float64)
    grad = np.zeros(p, dtype=np.float64)
    hess = np.zeros((p, p), dtype=np.float64)

    for g in range(nuft - 1, -1, -1):
        enter_start = int(first_idx_uft[g])
        enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
        if enter_end > enter_start:
            xp0 += risk_sum[enter_start] - risk_sum[enter_end]
            xp1 += risk_X_sum[enter_start] - risk_X_sum[enter_end]
            xp2 += X[enter_start:enter_end].T @ (
                X[enter_start:enter_end] * e_linpred[enter_start:enter_end, None]
            )

        ix_ev = uft_ix[g]
        d = len(ix_ev)
        if d == 0:
            continue

        v = X[ix_ev]
        elx = e_linpred[ix_ev]
        xp0f = float(elx.sum())
        xp1f = v.T @ elx
        xp2f = (v * elx[:, None]).T @ v

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


def _efron_backward_scan_vectorized(
    X, e_linpred, risk_sum, risk_X_sum,
    first_idx_uft, uft_ix, nuft, n, p,
):
    """Vectorized Efron gradient/Hessian via suffix outer products.

    Properly handles tied failures with Efron's k/d correction.
    O(n·p²) memory for suffix outer products; O(nuft·d·p) for Efron loop.
    """
    X_exp = X * e_linpred[:, None]
    total = X_exp.T @ X  # (p, p)

    # Suffix outer products: risk_X2[g] = sum_{i >= first_idx[g]} X_i exp(eta_i) X_i'
    fi = first_idx_uft.astype(np.int64)
    flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
    prefix_flat = np.cumsum(flat, axis=0)  # (n, p*p)

    prefix_at_g = np.zeros((nuft, p, p), dtype=np.float64)
    mask = fi > 0
    if mask.any():
        prefix_at_g[mask] = prefix_flat[fi[mask] - 1].reshape(-1, p, p)
    risk_X2 = total[None, :, :] - prefix_at_g  # (nuft, p, p)

    # Efron gradient/Hessian with proper tied-event correction
    grad = np.zeros(p, dtype=np.float64)
    hess = np.zeros((p, p), dtype=np.float64)

    for g in range(nuft):
        ix_ev = uft_ix[g]
        d = len(ix_ev)
        if d == 0:
            continue

        # Risk set quantities at this failure time
        s0 = float(risk_sum[fi[g]])
        s1 = risk_X_sum[fi[g]]  # (p,)

        # Tied failure quantities
        v = X[ix_ev]  # (d, p) — ALL failures, not just first
        elx = e_linpred[ix_ev]  # (d,)
        xp0f = float(elx.sum())
        xp1f = v.T @ elx  # (p,) — weighted sum of failure covariates

        # Efron correction: for k=0..d-1, denominator = s0 - (k/d)*xp0f
        J = np.arange(d, dtype=np.float64) / d  # (d,)
        c0 = s0 - J * xp0f  # (d,)
        np.maximum(c0, 1e-300, out=c0)
        inv = 1.0 / c0  # (d,)
        J_inv = J * inv  # (d,)
        sum_inv = inv.sum()
        sum_J = J_inv.sum()
        sum_aa = np.dot(inv, inv)
        sum_bb = np.dot(J_inv, J_inv)
        sum_ab = np.dot(inv, J_inv)

        # Gradient: sum of ALL failure X's minus Efron-corrected risk term
        grad += v.sum(axis=0)  # sum_{i in D_g} X_i
        grad -= s1 * sum_inv - xp1f * sum_J

        # Hessian: Efron-corrected second moment
        hess -= risk_X2[g] * sum_inv
        hess += (v * elx[:, None]).T @ v * sum_J  # xp2f * sum_J
        hess += sum_aa * np.outer(s1, s1)
        hess += sum_bb * np.outer(xp1f, xp1f)
        hess -= sum_ab * (np.outer(s1, xp1f) + np.outer(xp1f, s1))

    return grad, -hess


class _LegacyCoxReferenceMixin:
    # Legacy reference implementations below are retained only for targeted
    # regression comparisons. Public ``fit`` never dispatches to this block;
    # the canonical path is ``_fit_counting_process_dispatch`` above.
    def _fit_cpu(self, X, time, event, entry=None, cluster=None, init_coef=None):
        """Fit using CPU (NumPy)."""
        if entry is not None:
            self._fit_counting_process_dispatch(
                X,
                time,
                event,
                entry=np.asarray(entry, dtype=np.float64),
                strata=None,
                cluster=cluster,
                subject_id=None,
                init_coef=init_coef,
                device=Device.CPU,
            )
            return
        n_samples, n_features = X.shape

        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = np.argsort(time, kind='stable')
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else np.asarray(entry, dtype=np.float64)[order]
        cluster_sorted = None if cluster is None else np.asarray(cluster)[order]

        self._efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            self._efron_pre = self._efron_unique_failure_indices(time_sorted, event_sorted)
            try:
                uft, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(self._efron_pre)
                self._efron_all_singletons = bool(nuft > 0) and all(
                    len(ix) == 1 for ix in uft_ix
                )
            except Exception:
                self._efron_all_singletons = False
        else:
            self._efron_all_singletons = False
            self._breslow_pre = self._breslow_unique_failure_groups(
                time_sorted, event_sorted
            )
            if entry_sorted is not None:
                event_idx_np = np.flatnonzero(event_sorted.astype(np.int32) == 1)
                event_times_np = time_sorted[event_idx_np].astype(np.float64, copy=False)
                uft_np, inv_np = np.unique(event_times_np, return_inverse=True)
                self._entry_fail_groups_np = [
                    event_idx_np[inv_np == g].astype(np.int64, copy=False)
                    for g in range(len(uft_np))
                ]
                self._entry_fail_times_np = uft_np.astype(np.float64, copy=False)
                self._entry_order_np = np.argsort(entry_sorted).astype(np.int64, copy=False)
                self._entry_add_end_np = np.searchsorted(
                    entry_sorted, uft_np, side="left"
                ).astype(np.int64, copy=False)
                self._entry_rem_end_np = np.searchsorted(
                    time_sorted, uft_np, side="left"
                ).astype(np.int64, copy=False)
            else:
                self._entry_fail_groups_np = None
                self._entry_fail_times_np = None
                self._entry_order_np = None
                self._entry_add_end_np = None
                self._entry_rem_end_np = None

        # Initialize coefficients (supports warm-start path in CV)
        if init_coef is None:
            beta = np.zeros(n_features, dtype=np.float64)
        else:
            beta = np.asarray(init_coef, dtype=np.float64).reshape(-1)
            if beta.shape[0] != n_features:
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood (beta = 0)
        self._log_likelihood_null = self._compute_log_likelihood(
            np.zeros(n_features), X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
        )

        # Newton-Raphson optimization with a backend-neutral KKT and
        # line-search contract. The observed-information helper normalizes the
        # historical Breslow/Efron Hessian sign difference before solving.
        penalty = float(self.penalty)
        use_penalty = penalty > 0.0
        identity = np.eye(n_features, dtype=np.float64)
        kkt_tol = max(self.tol * 1e-3, 1e-9)
        objective_tol = 1e-10
        self._termination_reason = 'max_iter'
        iteration = -1
        current_obj = self._compute_log_likelihood(
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre
        ) - penalty * float(beta @ beta)
        self._objective_history = [float(current_obj)]

        for iteration in range(self.max_iter):
            grad_data, hess_data = self._compute_gradient_hessian(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )
            penalized_grad = grad_data - 2.0 * penalty * beta
            kkt_inf = float(np.linalg.norm(penalized_grad, ord=np.inf))
            kkt_norm = kkt_inf / (
                1.0
                + float(np.linalg.norm(grad_data, ord=np.inf))
                + 2.0 * penalty * float(np.linalg.norm(beta, ord=np.inf))
            )
            if kkt_norm <= kkt_tol:
                self._converged = True
                self._termination_reason = 'kkt_converged'
                self._final_kkt_inf = kkt_inf
                self._final_kkt_normalized = kkt_norm
                break

            information = self._observed_information(hess_data)
            if use_penalty:
                information = information + 2.0 * penalty * identity
            try:
                delta = np.linalg.solve(information, penalized_grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(information, penalized_grad, rcond=None)[0]

            accepted = False
            accepted_beta = beta
            accepted_obj = current_obj
            for direction in (1.0, -1.0):
                step = 1.0
                for _ in range(21):
                    trial_beta = beta + direction * step * delta
                    trial_obj = self._compute_log_likelihood(
                        trial_beta, X_sorted, time_sorted, event_sorted, self._efron_pre
                    ) - penalty * float(trial_beta @ trial_beta)
                    if np.isfinite(trial_obj) and trial_obj >= current_obj - objective_tol:
                        accepted = True
                        accepted_beta = trial_beta
                        accepted_obj = float(trial_obj)
                        break
                    step *= 0.5
                if accepted:
                    break

            if not accepted:
                self._converged = False
                self._termination_reason = 'line_search_failed'
                break

            update_norm = float(np.linalg.norm(accepted_beta - beta))
            beta = accepted_beta
            current_obj = accepted_obj
            self._objective_history.append(current_obj)

            if update_norm < max(self.tol * (1.0 + float(np.linalg.norm(beta))), 1e-8):
                trial_grad, _ = self._compute_gradient_hessian(
                    beta, X_sorted, time_sorted, event_sorted, self._efron_pre
                )
                trial_pen_grad = trial_grad - 2.0 * penalty * beta
                trial_kkt_inf = float(np.linalg.norm(trial_pen_grad, ord=np.inf))
                trial_kkt_norm = trial_kkt_inf / (
                    1.0
                    + float(np.linalg.norm(trial_grad, ord=np.inf))
                    + 2.0 * penalty * float(np.linalg.norm(beta, ord=np.inf))
                )
                self._final_kkt_inf = trial_kkt_inf
                self._final_kkt_normalized = trial_kkt_norm
                if trial_kkt_norm <= kkt_tol:
                    self._converged = True
                    self._termination_reason = 'kkt_converged'
                else:
                    self._converged = False
                    self._termination_reason = 'stalled_with_large_kkt'
                break

        final_grad, final_hess = self._compute_gradient_hessian(
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre
        )
        final_pen_grad = final_grad - 2.0 * penalty * beta
        self._final_kkt_inf = float(np.linalg.norm(final_pen_grad, ord=np.inf))
        self._final_kkt_normalized = self._final_kkt_inf / (
            1.0
            + float(np.linalg.norm(final_grad, ord=np.inf))
            + 2.0 * penalty * float(np.linalg.norm(beta, ord=np.inf))
        )
        if self._final_kkt_normalized <= kkt_tol:
            self._converged = True
            self._termination_reason = 'kkt_converged'
        elif self._converged:
            self._converged = False
            self._termination_reason = 'stalled_with_large_kkt'

        self._iterations = iteration + 1
        self.coef_ = beta
        self.hazard_ratios_ = np.exp(beta)

        # Compute final log-likelihood
        self._log_likelihood = self._compute_log_likelihood(
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
        )
        self._penalized_objective = self._log_likelihood - penalty * float(beta @ beta)

        # Compute optional inference statistics
        if self.compute_inference:
            self._compute_inference_cpu(X_sorted, time_sorted, event_sorted, cluster_sorted)
            self._compute_baseline_hazard(X_sorted, time_sorted, event_sorted, entry=entry_sorted)
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

        if self.compute_cindex:
            self._compute_cindex()
        else:
            self._cindex = None

    def _fit_gpu(self, X, time, event, entry=None, cluster=None, init_coef=None):
        """Fit using GPU with full GPU computation."""
        import cupy as cp
        from statgpu.inference._distributions_backend import norm

        n_samples, n_features = X.shape

        # Transfer to GPU once
        X = cp.asarray(X, dtype=cp.float64)
        time = cp.asarray(time, dtype=cp.float64)
        event = cp.asarray(event, dtype=cp.int32)

        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = cp.argsort(time, kind="stable")
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else entry[order]
        cluster_sorted = None if cluster is None else cluster[order]
        event_idx_sorted = cp.where(event_sorted == 1)[0]
        self._event_idx_gpu = event_idx_sorted
        self._event_X_sum_gpu = (
            cp.sum(X_sorted[event_idx_sorted], axis=0)
            if int(event_idx_sorted.size) > 0
            else cp.zeros(n_features, dtype=cp.float64)
        )

        # Precompute Efron tie structure once (depends only on time/event order).
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            if entry_sorted is None:
                efron_pre = self._efron_unique_failure_indices(
                    cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
                )
                self._efron_pre = efron_pre
                try:
                    _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                    self._efron_all_singletons = bool(nuft > 0) and all(
                        len(ix) == 1 for ix in uft_ix
                    )
                except Exception:
                    self._efron_all_singletons = False
                # Pack enter/exit/fail indices once; reuse across Newton steps on GPU.
                try:
                    from ._cox_efron_cuda import efron_indices_to_csr

                    uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = _unpack_efron_pre6(
                        efron_pre
                    )
                    (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                    ) = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
                    self._efron_pre_csr = (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                        first_idx_uft,
                        nuft,
                    )
                    self._efron_pre_csr_gpu = (
                        cp.asarray(enter_ptr, dtype=cp.int32),
                        cp.asarray(enter_ind, dtype=cp.int32),
                        cp.asarray(exit_ptr, dtype=cp.int32),
                        cp.asarray(exit_ind, dtype=cp.int32),
                        cp.asarray(fail_ptr, dtype=cp.int32),
                        cp.asarray(fail_ind, dtype=cp.int32),
                        cp.asarray(first_idx_uft, dtype=cp.int32),
                        int(nuft),
                    )
                except Exception:
                    self._efron_pre_csr = None
                    self._efron_pre_csr_gpu = None
            else:
                self._efron_pre = None
                self._efron_pre_csr = None
                self._efron_pre_csr_gpu = None
        else:
            self._efron_pre = None
            self._efron_all_singletons = False
            self._efron_pre_csr = None
            self._efron_pre_csr_gpu = None
            first_idx_uft, counts_uft = self._breslow_unique_failure_groups(
                cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
            )
            self._breslow_pre = (first_idx_uft, counts_uft)
            self._breslow_pre_gpu = (
                cp.asarray(first_idx_uft, dtype=cp.int32),
                cp.asarray(counts_uft, dtype=cp.int32),
            )
            self._breslow_counts_f_gpu = cp.asarray(counts_uft, dtype=cp.float64)
            self._breslow_first_idx_np = np.asarray(first_idx_uft, dtype=np.int64)
            self._breslow_counts_np = np.asarray(counts_uft, dtype=np.float64)
            if entry_sorted is not None:
                # Entry path: avoid stale index cache drift across different sort permutations.
                self._entry_fail_groups_gpu = None
                self._entry_fail_times_gpu = None
                self._entry_order_gpu = None
                self._entry_add_end_np_gpu = None
                self._entry_rem_end_np_gpu = None
            else:
                self._entry_fail_groups_gpu = None
                self._entry_fail_times_gpu = None
                self._entry_order_gpu = None
                self._entry_add_end_np_gpu = None
                self._entry_rem_end_np_gpu = None

        # Initialize coefficients on GPU (supports warm-start path in CV)
        if init_coef is None:
            beta = cp.zeros(n_features, dtype=cp.float64)
        else:
            beta = cp.asarray(np.asarray(init_coef, dtype=np.float64), dtype=cp.float64).reshape(-1)
            if int(beta.shape[0]) != int(n_features):
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood on GPU
        entry_ctx_gpu = None
        if entry_sorted is not None:
            _ctx = self._build_entry_ctx_gpu(time_sorted, event_sorted, entry_sorted, cp)
            event_idx_ctx = _ctx[5]
            entry_ctx_gpu = (
                _ctx[0], _ctx[1], _ctx[2], _ctx[3],
                cp.ascontiguousarray(X_sorted[_ctx[0]]),
                cp.ascontiguousarray(X_sorted),
                event_idx_ctx,
                cp.sum(X_sorted[event_idx_ctx], axis=0),
                _ctx[6],
            )
        loglik_null_gpu = self._compute_log_likelihood_gpu(
            cp.zeros(n_features, dtype=cp.float64),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
            entry=entry_sorted,
            entry_ctx=entry_ctx_gpu,
        )

        # Newton-Raphson optimization on GPU with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        diag_idx = cp.arange(n_features, dtype=cp.int64) if use_penalty else None
        eye_cache = (
            cp.eye(n_features, dtype=cp.float64)
            if (self.compute_inference or use_penalty)
            else None
        )

        # Newton-Raphson optimization on GPU with KKT-based convergence
        loglik_gpu = None
        current_obj = None
        iteration = -1
        kkt_tol = max(self.tol * 1e-3, 1e-9)  # KKT threshold
        objective_tol = 1e-10
        self._termination_reason = "max_iter"
        self._final_kkt_inf = None
        self._final_kkt_normalized = None

        for iteration in range(self.max_iter):
            # Compute gradient and Hessian at CURRENT beta_k
            grad, hess, aux_stats = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_gpu
            )

            # Check KKT at current beta BEFORE taking the step.
            if use_penalty:
                pen_grad = grad - 2 * penalty * beta
            else:
                pen_grad = grad
            kkt_inf = float(cp.linalg.norm(pen_grad, ord=cp.inf).item())
            grad_inf = float(cp.linalg.norm(grad, ord=cp.inf).item())
            beta_inf = float(cp.linalg.norm(beta, ord=cp.inf).item())
            kkt_norm = kkt_inf / (1.0 + grad_inf + 2.0 * penalty * beta_inf)

            if kkt_norm <= kkt_tol:
                self._converged = True
                self._termination_reason = "kkt_converged"
                self._final_kkt_inf = kkt_inf
                self._final_kkt_normalized = kkt_norm
                break

            # Add penalty terms for Newton step
            if use_penalty:
                grad = pen_grad
                hess[diag_idx, diag_idx] -= 2 * penalty

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_gpu(hess, grad, cp, eye_cache=eye_cache)
            if current_obj is None:
                current_obj = self._compute_log_likelihood_gpu_from_stats(
                    aux_stats[0], aux_stats[1], aux_stats[2],
                    time_sorted, event_sorted, efron_pre,
                    entry=entry_sorted, entry_ctx=entry_ctx_gpu,
                )
                if use_penalty:
                    current_obj = current_obj - penalty * cp.sum(beta * beta)
                self._objective_history = [float(current_obj.item())]

            accepted_step = False
            accepted_beta = beta
            accepted_obj = current_obj
            accepted_step_size = 0.0
            for direction in (-1.0, 1.0):
                step = 1.0
                for _ in range(21):
                    trial_beta = beta + direction * step * delta
                    trial_obj = self._compute_log_likelihood_gpu(
                        trial_beta, X_sorted, time_sorted, event_sorted, efron_pre,
                        entry=entry_sorted, entry_ctx=entry_ctx_gpu,
                    )
                    if use_penalty:
                        trial_obj = trial_obj - penalty * cp.sum(trial_beta * trial_beta)
                    if float((trial_obj - current_obj).item()) >= -objective_tol:
                        accepted_step = True
                        accepted_beta = trial_beta
                        accepted_obj = trial_obj
                        accepted_step_size = step
                        break
                    step *= 0.5
                if accepted_step:
                    break

            if accepted_step:
                beta = accepted_beta
                current_obj = accepted_obj
                self._objective_history.append(float(current_obj.item()))

            # Step-norm check: must verify KKT before declaring convergence.
            if not accepted_step:
                self._termination_reason = "line_search_failed"
                self._converged = False
                break

            delta_norm = float(cp.linalg.norm(delta).item())
            step_norm = delta_norm * accepted_step_size
            if step_norm < max(self.tol * (1.0 + float(cp.linalg.norm(beta).item())), 1e-8):
                # Step is small — check if KKT is actually satisfied.
                grad_check, hess_check, _aux_check = self._compute_gradient_hessian_gpu(
                    beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True,
                    entry=entry_sorted, entry_ctx=entry_ctx_gpu,
                )
                if use_penalty:
                    pg = grad_check - 2 * penalty * beta
                else:
                    pg = grad_check
                kkt_check = float(cp.linalg.norm(pg, ord=cp.inf).item())
                kkt_n_check = kkt_check / (
                    1.0 + float(cp.linalg.norm(grad_check, ord=cp.inf).item())
                    + 2.0 * penalty * float(cp.linalg.norm(beta, ord=cp.inf).item())
                )
                if kkt_n_check <= kkt_tol:
                    self._converged = True
                    self._termination_reason = "kkt_converged"
                    self._final_kkt_inf = kkt_check
                    self._final_kkt_normalized = kkt_n_check
                else:
                    self._converged = False
                    self._termination_reason = "stalled_with_large_kkt"
                    self._final_kkt_inf = kkt_check
                    self._final_kkt_normalized = kkt_n_check
                break

        # Compute final KKT at exit point if not done yet.
        if self._final_kkt_inf is None:
            grad_final, hess_final, _aux_final = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True,
                entry=entry_sorted, entry_ctx=entry_ctx_gpu,
            )
            if use_penalty:
                pen_grad_final = grad_final - 2 * penalty * beta
            else:
                pen_grad_final = grad_final
            self._final_kkt_inf = float(cp.linalg.norm(pen_grad_final, ord=cp.inf).item())
            self._final_kkt_normalized = self._final_kkt_inf / (
                1.0 + float(cp.linalg.norm(grad_final, ord=cp.inf).item())
                + 2.0 * penalty * float(cp.linalg.norm(beta, ord=cp.inf).item())
            )

        # Override _converged if final KKT is too large.
        if (self._final_kkt_normalized is not None
                and self._final_kkt_normalized > kkt_tol):
            if self._converged:
                self._termination_reason = "stalled_with_large_kkt"
            self._converged = False

        # Recompute gradient, Hessian, and log-likelihood at final beta
        # so that coef_, _log_likelihood, and _var_matrix are all anchored
        # at the same parameter point, regardless of convergence path.
        final_hess = None
        if self.compute_inference:
            _, final_hess, final_aux = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre,
                return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_gpu,
            )
            if use_penalty:
                final_hess[diag_idx, diag_idx] -= 2.0 * penalty
            loglik_gpu = self._compute_log_likelihood_gpu_from_stats(
                final_aux[0], final_aux[1], final_aux[2],
                time_sorted, event_sorted, efron_pre, entry=entry_sorted,
            )
        else:
            loglik_gpu = self._compute_log_likelihood_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre,
                entry=entry_sorted, entry_ctx=entry_ctx_gpu,
            )

        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = cp.asnumpy(beta)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(cp.asnumpy(loglik_null_gpu))
        self._log_likelihood = float(cp.asnumpy(loglik_gpu))
        self._penalized_objective = (
            self._log_likelihood - penalty * float(np.dot(self.coef_, self.coef_))
        )
        if not self._objective_history:
            self._objective_history = [self._penalized_objective]
        if self.compute_cindex:
            cindex_gpu = self._compute_cindex_gpu(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cp.asnumpy(cindex_gpu))
        else:
            self._cindex = None

        # Inference stays on the selected GPU backend.  Recompute curvature at
        # the final coefficient vector; the loop-local Hessian precedes the
        # last accepted Newton update and may be stale (or undefined when
        # max_iter=0).
        if self.compute_inference:
            _, inference_hess = self._compute_gradient_hessian_gpu(
                beta,
                X_sorted,
                time_sorted,
                event_sorted,
                efron_pre,
                entry=entry_sorted,
                entry_ctx=entry_ctx_gpu,
            )
            if use_penalty:
                inference_hess[diag_idx, diag_idx] -= 2 * penalty
            info = self._observed_information_cupy(inference_hess)
            if self.cov_type == "nonrobust":
                var_gpu = _invert_information_cupy(info)
                var_gpu = 0.5 * (var_gpu + var_gpu.T)
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_gpu)))
                z_crit = norm.ppf(0.975)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._var_matrix = cp.asnumpy(var_gpu)
                self.inference_method_ = (
                    'penalized_observed_information'
                    if self.penalty > 0 else 'observed_information'
                )
                self.inference_backend_ = 'cupy'
                self.inference_approximate_ = False
                self._var_matrix = 0.5 * (self._var_matrix + self._var_matrix.T)  # numerical symmetrization
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = float(chi2.sf(self._lr_test_stat, df=n_features))
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = float(chi2.sf(self._wald_test_stat, df=n_features))
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
            else:
                score_resid_gpu = self._compute_robust_score_residuals_gpu(X_sorted, time_sorted, event_sorted)
                bread = _invert_information_cupy(info)

                if self.cov_type == "cluster":
                    if cluster_sorted is None:
                        raise ValueError("cov_type='cluster' requires cluster ids in fit(..., cluster=...)")
                    unique_clusters = cp.unique(cluster_sorted)
                    meat = cp.zeros((n_features, n_features), dtype=cp.float64)
                    for g in unique_clusters:
                        u_g = cp.sum(score_resid_gpu[cluster_sorted == g], axis=0)
                        meat += cp.outer(u_g, u_g)
                else:
                    meat = score_resid_gpu.T @ score_resid_gpu
                    if self.cov_type == "hc1":
                        n = X_sorted.shape[0]
                        k = X_sorted.shape[1]
                        if n > k:
                            meat = meat * (n / (n - k))

                var_gpu = bread @ meat @ bread
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_gpu)))
                z_crit = norm.ppf(0.975)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._var_matrix = cp.asnumpy(var_gpu)
                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = float(chi2.sf(self._lr_test_stat, df=n_features))
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = float(chi2.sf(self._wald_test_stat, df=n_features))
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan

            # Baseline hazard is part of the inference contract for every
            # covariance type, including the nonrobust fast path.
            self._compute_baseline_hazard_gpu(
                X_sorted, time_sorted, event_sorted, beta, entry=entry_sorted
            )
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

    def _fit_torch(self, X, time, event, entry=None, cluster=None, torch_device="cuda", init_coef=None):
        """Fit using Torch with full GPU computation."""
        import torch
        from statgpu.inference._distributions_backend import norm

        n_samples, n_features = X.shape

        # Sort by time ascending so risk-set terms are suffix sums
        order = torch.argsort(time, stable=True)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else entry[order]
        cluster_sorted = None if cluster is None else cluster[order]

        # Precompute Efron tie structure once (depends only on time/event order)
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_torch = None
        if self.ties == "efron":
            if entry_sorted is None:
                efron_pre = self._efron_unique_failure_indices(
                    time_sorted.cpu().numpy(), event_sorted.cpu().numpy()
                )
                self._efron_pre = efron_pre
                try:
                    _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                    self._efron_all_singletons = bool(nuft > 0) and all(
                        len(ix) == 1 for ix in uft_ix
                    )
                except Exception:
                    self._efron_all_singletons = False
                # Torch Efron stays native: no CuPy dependency or numerical
                # fallback is needed for the grouped Torch implementation.
                self._efron_pre_csr = None
                self._efron_pre_csr_gpu = None
            else:
                self._efron_pre = None
                self._efron_pre_csr = None
                self._efron_pre_csr_gpu = None
        else:
            self._efron_pre = None
            self._efron_all_singletons = False
            self._efron_pre_csr = None
            self._efron_pre_csr_gpu = None
            first_idx_uft, counts_uft = self._breslow_unique_failure_groups(
                time_sorted.cpu().numpy(), event_sorted.cpu().numpy()
            )
            self._breslow_pre = (first_idx_uft, counts_uft)
            self._breslow_pre_torch = (
                torch.tensor(first_idx_uft, dtype=torch.int32, device=torch_device),
                torch.tensor(counts_uft, dtype=torch.int32, device=torch_device),
            )
            if entry_sorted is not None:
                # Entry path: avoid stale index cache drift across different sort permutations.
                self._entry_fail_groups_torch = None
                self._entry_fail_times_torch = None
                self._entry_order_torch = None
                self._entry_add_end_np_torch = None
                self._entry_rem_end_np_torch = None
            else:
                self._entry_fail_groups_torch = None
                self._entry_fail_times_torch = None
                self._entry_order_torch = None
                self._entry_add_end_np_torch = None
                self._entry_rem_end_np_torch = None

        # Initialize coefficients on Torch device (supports warm-start path in CV)
        if init_coef is None:
            beta = torch.zeros(n_features, dtype=torch.float64, device=torch_device)
        else:
            beta = torch.as_tensor(init_coef, dtype=torch.float64, device=torch_device).reshape(-1)
            if int(beta.shape[0]) != int(n_features):
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood on Torch
        entry_ctx_torch = None
        if entry_sorted is not None:
            _ctx = self._build_entry_ctx_torch(time_sorted, event_sorted, entry_sorted, torch_device)
            event_idx_ctx = _ctx[5]
            entry_ctx_torch = (
                _ctx[0],
                _ctx[1],
                _ctx[2],
                _ctx[3],
                X_sorted.index_select(0, _ctx[0]).contiguous(),
                X_sorted.contiguous(),
                event_idx_ctx,
                torch.sum(X_sorted.index_select(0, event_idx_ctx), dim=0),
                _ctx[6],
            )
        loglik_null_torch = self._compute_log_likelihood_torch(
            torch.zeros(n_features, dtype=torch.float64, device=torch_device),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
            entry=entry_sorted,
            entry_ctx=entry_ctx_torch,
        )

        # Newton-Raphson optimization on Torch with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        diag_idx = torch.arange(n_features, dtype=torch.long, device=torch_device) if use_penalty else None

        # Newton-Raphson optimization on Torch with KKT-based convergence
        iteration = -1
        loglik_torch = None
        current_obj = None
        kkt_tol = max(self.tol * 1e-3, 1e-9)
        objective_tol = 1e-10
        self._termination_reason = "max_iter"
        self._final_kkt_inf = None
        self._final_kkt_normalized = None

        for iteration in range(self.max_iter):
            # Compute gradient and Hessian at CURRENT beta_k
            grad, hess, aux_stats = self._compute_gradient_hessian_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_torch
            )

            # Check KKT at current beta BEFORE taking the step.
            if use_penalty:
                pen_grad = grad - 2 * penalty * beta
            else:
                pen_grad = grad
            kkt_inf = float(torch.linalg.norm(pen_grad, ord=float('inf')).item())
            grad_inf = float(torch.linalg.norm(grad, ord=float('inf')).item())
            beta_inf = float(torch.linalg.norm(beta, ord=float('inf')).item())
            kkt_norm = kkt_inf / (1.0 + grad_inf + 2.0 * penalty * beta_inf)

            if kkt_norm <= kkt_tol:
                self._converged = True
                self._termination_reason = "kkt_converged"
                self._final_kkt_inf = kkt_inf
                self._final_kkt_normalized = kkt_norm
                break

            # Add penalty terms for Newton step
            if use_penalty:
                grad = pen_grad
                hess[diag_idx, diag_idx] -= 2 * penalty

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_torch(hess, grad)
            if current_obj is None:
                current_obj = self._compute_log_likelihood_torch_from_stats(
                    aux_stats[0], aux_stats[1], aux_stats[2],
                    time_sorted, event_sorted, efron_pre,
                    entry=entry_sorted, entry_ctx=entry_ctx_torch,
                )
                if use_penalty:
                    current_obj = current_obj - penalty * torch.sum(beta * beta)
                self._objective_history = [float(current_obj.item())]

            accepted_step = False
            accepted_beta = beta
            accepted_obj = current_obj
            accepted_step_size = 0.0
            for direction in (-1.0, 1.0):
                step = 1.0
                for _ in range(21):
                    trial_beta = beta + direction * step * delta
                    trial_obj = self._compute_log_likelihood_torch(
                        trial_beta, X_sorted, time_sorted, event_sorted, efron_pre,
                        entry=entry_sorted, entry_ctx=entry_ctx_torch,
                    )
                    if use_penalty:
                        trial_obj = trial_obj - penalty * torch.sum(trial_beta * trial_beta)
                    if float((trial_obj - current_obj).item()) >= -objective_tol:
                        accepted_step = True
                        accepted_beta = trial_beta
                        accepted_obj = trial_obj
                        accepted_step_size = step
                        break
                    step *= 0.5
                if accepted_step:
                    break

            if accepted_step:
                beta = accepted_beta
                current_obj = accepted_obj
                self._objective_history.append(float(current_obj.item()))

            # Step-norm check: must verify KKT before declaring convergence.
            if not accepted_step:
                self._termination_reason = "line_search_failed"
                self._converged = False
                break

            delta_norm = float(torch.linalg.norm(delta).item())
            step_norm = delta_norm * accepted_step_size
            if step_norm < max(self.tol * (1.0 + float(torch.linalg.norm(beta).item())), 1e-8):
                grad_check, hess_check, _aux_check = self._compute_gradient_hessian_torch(
                    beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True,
                    entry=entry_sorted, entry_ctx=entry_ctx_torch,
                )
                if use_penalty:
                    pg = grad_check - 2 * penalty * beta
                else:
                    pg = grad_check
                kkt_check = float(torch.linalg.norm(pg, ord=float('inf')).item())
                kkt_n_check = kkt_check / (
                    1.0 + float(torch.linalg.norm(grad_check, ord=float('inf')).item())
                    + 2.0 * penalty * float(torch.linalg.norm(beta, ord=float('inf')).item())
                )
                if kkt_n_check <= kkt_tol:
                    self._converged = True
                    self._termination_reason = "kkt_converged"
                    self._final_kkt_inf = kkt_check
                    self._final_kkt_normalized = kkt_n_check
                else:
                    self._converged = False
                    self._termination_reason = "stalled_with_large_kkt"
                    self._final_kkt_inf = kkt_check
                    self._final_kkt_normalized = kkt_n_check
                break

        # Compute final KKT at exit point if not done yet.
        if self._final_kkt_inf is None:
            grad_final, hess_final, _aux_final = self._compute_gradient_hessian_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True,
                entry=entry_sorted, entry_ctx=entry_ctx_torch,
            )
            if use_penalty:
                pen_grad_final = grad_final - 2 * penalty * beta
            else:
                pen_grad_final = grad_final
            self._final_kkt_inf = float(torch.linalg.norm(pen_grad_final, ord=float('inf')).item())
            self._final_kkt_normalized = self._final_kkt_inf / (
                1.0 + float(torch.linalg.norm(grad_final, ord=float('inf')).item())
                + 2.0 * penalty * float(torch.linalg.norm(beta, ord=float('inf')).item())
            )

        # Override _converged if final KKT is too large.
        if (self._final_kkt_normalized is not None
                and self._final_kkt_normalized > kkt_tol):
            if self._converged:
                self._termination_reason = "stalled_with_large_kkt"
            self._converged = False

        # Recompute gradient, Hessian, and log-likelihood at final beta
        # for consistent inference regardless of convergence path.
        final_hess = None
        if self.compute_inference:
            _, final_hess, final_aux = self._compute_gradient_hessian_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre,
                return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_torch,
            )
            if use_penalty:
                final_hess[diag_idx, diag_idx] -= 2.0 * penalty
            loglik_torch = self._compute_log_likelihood_torch_from_stats(
                final_aux[0], final_aux[1], final_aux[2],
                time_sorted, event_sorted, efron_pre, entry=entry_sorted,
            )
        else:
            loglik_torch = self._compute_log_likelihood_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre,
                entry=entry_sorted, entry_ctx=entry_ctx_torch,
            )

        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = beta.cpu().numpy()
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(loglik_null_torch.item())
        self._log_likelihood = float(loglik_torch.item())
        self._penalized_objective = (
            self._log_likelihood - penalty * float(np.dot(self.coef_, self.coef_))
        )
        if not self._objective_history:
            self._objective_history = [self._penalized_objective]
        if self.compute_cindex:
            cindex_torch = self._compute_cindex_torch(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cindex_torch.item())
        else:
            self._cindex = None

        # Recompute the final curvature natively on Torch for nonrobust
        # inference.  Robust score residuals still use the established CPU
        # implementation, but baseline-hazard estimation remains on Torch.
        if self.compute_inference:
            hess = final_hess  # use final-beta Hessian
            if self.cov_type == "nonrobust":
                _, inference_hess = self._compute_gradient_hessian_torch(
                    beta,
                    X_sorted,
                    time_sorted,
                    event_sorted,
                    efron_pre,
                    entry=entry_sorted,
                    entry_ctx=entry_ctx_torch,
                )
                if use_penalty:
                    inference_hess[diag_idx, diag_idx] -= 2 * penalty
                info = self._observed_information_torch(inference_hess)
                var_torch = _invert_information_torch(info)
                var_torch = 0.5 * (var_torch + var_torch.transpose(0, 1))
                bse_torch = torch.sqrt(torch.maximum(torch.diag(var_torch), torch.tensor(0.0, dtype=torch.float64, device=torch_device)))
                z_torch = beta / (bse_torch + 1e-30)
                p_torch = torch.minimum(torch.tensor(1.0, device=torch_device), 2.0 * norm.sf(torch.abs(z_torch)))
                z_crit = norm.ppf(0.975)
                ci_torch = torch.stack([beta - z_crit * bse_torch, beta + z_crit * bse_torch], dim=1)

                self._bse = bse_torch.cpu().numpy()
                self._zvalues = z_torch.cpu().numpy()
                self._pvalues = p_torch.cpu().numpy()
                self._conf_int = ci_torch.cpu().numpy()
                self._var_matrix = var_torch.cpu().numpy()
                self.inference_method_ = (
                    'penalized_observed_information'
                    if self.penalty > 0 else 'observed_information'
                )
                self.inference_backend_ = 'torch'
                self.inference_approximate_ = False
                self._var_matrix = 0.5 * (self._var_matrix + self._var_matrix.T)  # numerical symmetrization
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = float(chi2.sf(self._lr_test_stat, df=n_features))
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = float(chi2.sf(self._wald_test_stat, df=n_features))
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
            else:
                # For hc0/hc1/cluster, use CPU inference path
                self.full_host_transfer_performed_ = True
                self._compute_inference_cpu(X_sorted.cpu().numpy(), time_sorted.cpu().numpy(), event_sorted.cpu().numpy(),
                                           cluster_sorted.cpu().numpy() if cluster_sorted is not None else None)
            # Compute baseline hazard on Torch for all covariance types
            if self.compute_inference:
                self._compute_baseline_hazard_torch(X_sorted, time_sorted, event_sorted, beta, entry=entry_sorted)
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None
        self._cleanup_torch_memory()

    def _compute_log_likelihood(self, beta, X, time, event, efron_pre=None, entry=None):
        """Compute log partial likelihood (Breslow/Efron tie handling)."""
        eta = X @ beta
        eta_eff = eta
        if entry is not None and self.ties == "breslow":
            eta_eff = eta - np.max(eta)
        # Note: We do NOT center eta here. While centering prevents exp overflow,
        # it introduces a beta-dependent shift that complicates numeric gradient verification.
        # In practice, exp(eta) overflow is rare when beta is near convergence.
        exp_eta = np.exp(eta_eff)

        # Risk set suffix sums for standard (no-entry) path.
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] if entry is None else None

        event_mask = event == 1
        if not np.any(event_mask):
            return 0.0

        if self.ties == "breslow":
            if entry is not None:
                fail_groups = getattr(self, "_entry_fail_groups_np", None)
                add_end_np = getattr(self, "_entry_add_end_np", None)
                rem_end_np = getattr(self, "_entry_rem_end_np", None)
                order_np = getattr(self, "_entry_order_np", None)
                if (
                    fail_groups is None
                    or add_end_np is None
                    or rem_end_np is None
                    or order_np is None
                ):
                    event_idx = np.flatnonzero(event_mask)
                    event_times = time[event_idx]
                    uft_np, inv_np = np.unique(event_times, return_inverse=True)
                    fail_groups = [
                        event_idx[inv_np == g].astype(np.int64, copy=False)
                        for g in range(len(uft_np))
                    ]
                    order_np = np.argsort(np.asarray(entry, dtype=np.float64)).astype(np.int64, copy=False)
                    add_end_np = np.searchsorted(
                        np.asarray(entry, dtype=np.float64)[order_np], uft_np, side="left"
                    ).astype(np.int64, copy=False)
                    rem_end_np = np.searchsorted(time, uft_np, side="left").astype(np.int64, copy=False)

                s0 = 0.0
                add_ptr = 0
                rem_ptr = 0
                ll = 0.0
                for g, fail_idx in enumerate(fail_groups):
                    add_end = int(add_end_np[g])
                    if add_end > add_ptr:
                        idx_add = order_np[add_ptr:add_end]
                        s0 += float(np.sum(exp_eta[idx_add]))
                        add_ptr = add_end
                    rem_end = int(rem_end_np[g])
                    if rem_end > rem_ptr:
                        s0 -= float(np.sum(exp_eta[rem_ptr:rem_end]))
                        rem_ptr = rem_end
                    d_t = int(fail_idx.shape[0])
                    if d_t <= 0:
                        continue
                    s0_safe = max(s0, 1e-300)
                    ll += float(np.sum(eta_eff[fail_idx]) - d_t * np.log(s0_safe))
                return float(ll)

            # l(β) = sum_i(eta_i) - sum_t(d_t * log(S0(t)))
            breslow_pre = getattr(self, "_breslow_pre", None)
            if (
                breslow_pre is not None
                and len(breslow_pre) == 2
                and breslow_pre[0].size > 0
            ):
                first_idx = breslow_pre[0].astype(np.int64, copy=False)
                counts = breslow_pre[1].astype(np.float64, copy=False)
            else:
                event_times = time[event_mask]
                uft, counts_i = np.unique(event_times, return_counts=True)
                first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                counts = counts_i.astype(np.float64)
            risk_at = risk_sum[first_idx]
            # With centering: ll = sum(eta_i - eta_max) - sum(d_t * log(S0(t) * exp(-eta_max)))
            #              = sum(eta_i) - n_events*eta_max - sum(d_t * (log(S0(t)) - eta_max))
            #              = sum(eta_i) - n_events*eta_max - sum(d_t * log(S0(t))) + n_events*eta_max
            #              = sum(eta_i) - sum(d_t * log(S0(t)))  [eta_max cancels]
            return float(np.sum(eta_eff[event_mask]) - np.sum(counts * np.log(risk_at)))

        # ---- Efron ----
        ll = 0.0
        if efron_pre is not None:
            uft, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)

            # Sum of eta for all events (centering cancels out, use original eta)
            all_eta_sum = 0.0
            all_log_denom_sum = 0.0

            for g in range(nuft):
                ix_ev = uft_ix[g]
                d = len(ix_ev)
                if d == 0:
                    continue
                first_idx = (
                    int(first_idx_uft[g])
                    if first_idx_uft is not None
                    else int(np.searchsorted(time, uft[g], side="left"))
                )
                risk_at_t = risk_sum[first_idx]
                sum_events = float(np.sum(exp_eta[ix_ev]))
                all_eta_sum += float(np.sum(eta[ix_ev]))

                # Vectorized log denominator sum
                # Pre-compute k/d values to avoid repeated division
                k_vals = np.arange(d, dtype=np.float64)
                denom = risk_at_t - (k_vals / d) * sum_events
                all_log_denom_sum += float(np.sum(np.log(np.maximum(denom, 1e-300))))

            return float(all_eta_sum - all_log_denom_sum)

        # No precomputation: group event rows by unique failure times (vectorized).
        event_idx = np.flatnonzero(event_mask)
        event_times = time[event_idx]
        uft, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
        risk_at = risk_sum[first_idx]

        sum_events = np.bincount(inv, weights=exp_eta[event_idx], minlength=len(uft)).astype(np.float64)
        sum_eta_events = np.bincount(inv, weights=eta[event_idx], minlength=len(uft)).astype(np.float64)

        # Vectorized log-likelihood computation
        ll = float(np.sum(sum_eta_events))

        # For each unique failure time, compute sum of log denominators
        max_d = int(np.max(counts)) if len(counts) > 0 else 0
        if max_d > 0:
            # Create k matrix: (n_uft, max_d) where each row has [0/d, 1/d, ..., (d-1)/d]
            # Use broadcasting with careful masking for different d values
            # Tie sizes differ by group; a short loop is clearer and avoids a
            # padded temporary matrix whose unused entries would need masking.
            for g in range(len(uft)):
                d = int(counts[g])
                if d == 0:
                    continue
                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at[g] - k * sum_events[g]
                ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))
        else:
            for g in range(len(uft)):
                d = int(counts[g])
                if d == 0:
                    continue
                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at[g] - k * sum_events[g]
                ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))

        return float(ll)

    def _solve_newton_delta_gpu(self, hess, grad, cp, eye_cache=None):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        p = int(hess.shape[0])
        H = -hess
        eps = 1e-11 * (cp.max(cp.abs(cp.diag(H))) + 1.0)
        jitter_eye = eye_cache if eye_cache is not None else cp.eye(p, dtype=cp.float64)
        H = H + eps * jitter_eye
        # Fast path: SPD solve via Cholesky is usually faster than generic solve.
        try:
            L = cp.linalg.cholesky(H)
            y = cp.linalg.solve(L, grad)
            x = cp.linalg.solve(L.T, y)
            return -x
        except Exception as exc:
            if not _is_singular_linalg_error(exc):
                raise
        try:
            return -cp.linalg.solve(H, grad)
        except Exception as exc:
            if not _is_singular_linalg_error(exc):
                raise
        return _solve_counting_information(hess, grad, "cupy", cp)

    def _compute_log_likelihood_gpu(self, beta, X, time, event, efron_pre=None, entry=None, entry_ctx=None):
        """Compute log partial likelihood on GPU."""
        import cupy as cp

        eta = X @ beta
        exp_eta = cp.exp(eta)
        # Entry+breslow path does not consume risk_sum; skip the cumsum to
        # reduce per-evaluation overhead during line-search probes.
        risk_sum = None if entry is not None else cp.cumsum(exp_eta[::-1])[::-1]
        return self._compute_log_likelihood_gpu_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre, entry=entry, entry_ctx=entry_ctx
        )

    def _build_entry_ctx_gpu(self, time, event, entry, cp):
        """Build entry-time grouped indexing context for a specific sorted GPU view."""
        event_mask = event == 1
        event_idx = cp.where(event_mask)[0]
        evt_t = cp.asnumpy(time[event_idx])
        if evt_t.size == 0:
            return (
                cp.zeros((0,), dtype=cp.int64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                cp.zeros((0,), dtype=cp.int64),
                cp.zeros((0,), dtype=cp.int64),
                np.zeros((1,), dtype=np.int64),
            )
        uft_np, d_counts = np.unique(evt_t, return_counts=True)
        d_counts = d_counts.astype(np.float64, copy=False)
        entry_order = cp.argsort(entry)
        entry_sorted_np = cp.asnumpy(entry[entry_order])
        time_np = cp.asnumpy(time)
        add_end_np = np.searchsorted(entry_sorted_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_end_np = np.searchsorted(time_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_order = cp.arange(int(time.shape[0]), dtype=cp.int64)
        event_idx = event_idx.astype(cp.int64, copy=False)
        fail_ptr = np.empty(d_counts.shape[0] + 1, dtype=np.int64)
        fail_ptr[0] = 0
        fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
        return (entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr)

    def _compute_log_likelihood_gpu_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None, entry=None, entry_ctx=None
    ):
        """Compute log partial likelihood on GPU with precomputed Efron stats."""
        import cupy as cp

        ll = cp.array(0.0, dtype=cp.float64)
        event_mask = event == 1

        if not cp.any(event_mask):
            return ll

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, _rem_order, event_idx, fail_ptr = self._build_entry_ctx_gpu(
                    time, event, entry, cp
                )
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else cp.where(event_mask)[0]
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                return cp.array(0.0, dtype=cp.float64)
            if fail_ptr is None:
                fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                fail_ptr[0] = 0
                fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)

            exp_entry = exp_eta[entry_order]
            exp_rem = exp_eta
            add_pref = cp.cumsum(exp_entry, axis=0)
            rem_pref = cp.cumsum(exp_rem, axis=0)
            s0_add = cp.zeros(n_groups, dtype=cp.float64)
            s0_rem = cp.zeros(n_groups, dtype=cp.float64)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = cp.asarray(add_end_np[mask_add] - 1, dtype=cp.int64)
                s0_add[cp.asarray(mask_add)] = add_pref[idx_add]
            if np.any(mask_rem):
                idx_rem = cp.asarray(rem_end_np[mask_rem] - 1, dtype=cp.int64)
                s0_rem[cp.asarray(mask_rem)] = rem_pref[idx_rem]
            s0_vec = cp.maximum(s0_add - s0_rem, 1e-300)
            event_eta = eta[event_idx]

            if self.ties == "breslow":
                d_vec = cp.asarray(d_counts, dtype=cp.float64)
                return cp.sum(event_eta) - cp.sum(d_vec * cp.log(s0_vec))

            ll = cp.sum(event_eta)
            event_exp = exp_eta[event_idx]
            for g in range(n_groups):
                d = int(d_counts[g])
                if d <= 0:
                    continue
                st = int(fail_ptr[g])
                ed = int(fail_ptr[g + 1])
                ef = cp.sum(event_exp[st:ed])
                base = s0_vec[g]
                for k in range(d):
                    denom = cp.maximum(base - (float(k) / float(d)) * ef, 1e-300)
                    ll = ll - cp.log(denom)
            return ll

        if self.ties == 'breslow':
            # Vectorized Breslow using cached failure groups to avoid
            # Python loops and host-device sync in GPU hot path.
            breslow_pre_gpu = getattr(self, "_breslow_pre_gpu", None)
            if (
                breslow_pre_gpu is not None
                and len(breslow_pre_gpu) == 2
                and int(breslow_pre_gpu[0].size) > 0
            ):
                first_idx_uft, counts_uft = breslow_pre_gpu
            else:
                uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
                first_idx_uft = cp.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.astype(cp.int32, copy=False)
            risk_at = risk_sum[first_idx_uft]
            return cp.sum(eta[event_mask]) - cp.sum(
                counts_uft.astype(cp.float64) * cp.log(risk_at)
            )

        # Efron: if all groups are singleton failures, Efron == Breslow.
        if getattr(self, "_efron_all_singletons", False):
            ep = efron_pre if efron_pre is not None else getattr(self, "_efron_pre", None)
            if ep is not None:
                _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(ep)
                first_idx_uft = cp.asarray(first_idx_uft, dtype=cp.int32)
                counts_uft = cp.ones(int(nuft), dtype=cp.int32)
            else:
                uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
                first_idx_uft = cp.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.astype(cp.int32, copy=False)
            risk_at = risk_sum[first_idx_uft]
            return cp.sum(eta[event_mask]) - cp.sum(
                counts_uft.astype(cp.float64) * cp.log(risk_at)
            )

        # Efron: loop over cached failure groups (see `_cox_efron_cuda.compute_efron_loglik_raw`)
        if efron_pre is not None:
            try:
                csr_gpu = getattr(self, "_efron_pre_csr_gpu", None)
                if csr_gpu is not None:
                    from ._cox_efron_cuda import compute_efron_loglik_raw_csr

                    _, _, _, _, fail_ptr, fail_ind, first_idx_uft, nuft = csr_gpu
                    return compute_efron_loglik_raw_csr(
                        eta,
                        exp_eta,
                        risk_sum,
                        fail_ptr,
                        fail_ind,
                        first_idx_uft,
                        nuft,
                        cupy_module=cp,
                    )
            except Exception:
                pass

            from ._cox_efron_cuda import compute_efron_loglik_raw

            return compute_efron_loglik_raw(
                eta, exp_eta, risk_sum, time, efron_pre, cupy_module=cp
            )

        unique_times = cp.unique(time[event_mask])
        for t in unique_times:
            at_time_t = time == t
            events_at_t = at_time_t & event_mask
            d = int(cp.sum(events_at_t).item())

            if d == 0:
                continue

            risk_indices = cp.where(time >= t)[0]
            if risk_indices.size == 0:
                continue

            first_idx = risk_indices[0]
            risk_at_t = risk_sum[first_idx]
            sum_events = cp.sum(exp_eta[events_at_t])

            ll += cp.sum(eta[events_at_t])
            for k in range(d):
                ll -= cp.log(cp.maximum(risk_at_t - (k / d) * sum_events, 1e-300))

        return ll

    def _compute_gradient_hessian(self, beta, X, time, event, efron_pre=None, entry=None):
        """
        Gradient and Hessian of the log partial likelihood (same sign convention as statsmodels).

        Parameters
        ----------
        efron_pre : optional
            Output of `_efron_unique_failure_indices`; if None and ties='efron', it is recomputed.
            Pass the cached structure from `fit` to avoid O(n) Python work every Newton step.
        """
        n_samples, n_features = X.shape

        # Linear predictor
        eta = X @ beta
        eta_eff = eta
        if entry is not None and self.ties == "breslow":
            eta_eff = eta - np.max(eta)
        exp_eta = np.exp(eta_eff)

        risk_sum = np.cumsum(exp_eta[::-1])[::-1] if entry is None else None
        X_exp_eta = X * exp_eta[:, np.newaxis]
        risk_X_sum = np.cumsum(X_exp_eta[::-1], axis=0)[::-1] if entry is None else None

        if self.ties == 'breslow':
            event_mask = event == 1
            grad = np.zeros(n_features, dtype=np.float64)
            if entry is not None:
                fail_groups = getattr(self, "_entry_fail_groups_np", None)
                add_end_np = getattr(self, "_entry_add_end_np", None)
                rem_end_np = getattr(self, "_entry_rem_end_np", None)
                order_np = getattr(self, "_entry_order_np", None)
                if (
                    fail_groups is None
                    or add_end_np is None
                    or rem_end_np is None
                    or order_np is None
                ):
                    event_idx = np.flatnonzero(event_mask)
                    event_times = time[event_idx]
                    uft_np, inv_np = np.unique(event_times, return_inverse=True)
                    fail_groups = [
                        event_idx[inv_np == g].astype(np.int64, copy=False)
                        for g in range(len(uft_np))
                    ]
                    order_np = np.argsort(np.asarray(entry, dtype=np.float64)).astype(np.int64, copy=False)
                    add_end_np = np.searchsorted(
                        np.asarray(entry, dtype=np.float64)[order_np], uft_np, side="left"
                    ).astype(np.int64, copy=False)
                    rem_end_np = np.searchsorted(time, uft_np, side="left").astype(np.int64, copy=False)

                hess = np.zeros((n_features, n_features), dtype=np.float64)
                s0 = 0.0
                s1 = np.zeros(n_features, dtype=np.float64)
                s2 = np.zeros((n_features, n_features), dtype=np.float64)
                add_ptr = 0
                rem_ptr = 0
                for g, fail_idx in enumerate(fail_groups):
                    add_end = int(add_end_np[g])
                    if add_end > add_ptr:
                        idx_add = order_np[add_ptr:add_end]
                        x_add = X[idx_add]
                        w_add = exp_eta[idx_add]
                        wx_add = x_add * w_add[:, np.newaxis]
                        s0 += float(np.sum(w_add))
                        s1 += np.sum(wx_add, axis=0)
                        s2 += wx_add.T @ x_add
                        add_ptr = add_end
                    rem_end = int(rem_end_np[g])
                    if rem_end > rem_ptr:
                        x_rem = X[rem_ptr:rem_end]
                        w_rem = exp_eta[rem_ptr:rem_end]
                        wx_rem = x_rem * w_rem[:, np.newaxis]
                        s0 -= float(np.sum(w_rem))
                        s1 -= np.sum(wx_rem, axis=0)
                        s2 -= wx_rem.T @ x_rem
                        rem_ptr = rem_end
                    d_t = int(fail_idx.shape[0])
                    if d_t <= 0:
                        continue
                    d_t_f = float(d_t)
                    grad += np.sum(X[fail_idx], axis=0)
                    s0_safe = max(s0, 1e-300)
                    if s0 <= 1e-15:
                        continue
                    ex = s1 / s0_safe
                    grad -= d_t_f * ex
                    hess -= d_t_f * (s2 / s0_safe - np.outer(ex, ex))
                return grad, hess

            first_idx = np.array([], dtype=np.int64)
            counts = np.array([], dtype=np.float64)
            if np.any(event_mask):
                breslow_pre = getattr(self, "_breslow_pre", None)
                if (
                    breslow_pre is not None
                    and len(breslow_pre) == 2
                    and breslow_pre[0].size > 0
                ):
                    first_idx = breslow_pre[0].astype(np.int64, copy=False)
                    counts = breslow_pre[1].astype(np.float64, copy=False)
                else:
                    event_times = time[event_mask]
                    uft, counts_i = np.unique(event_times, return_counts=True)
                    first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                    counts = counts_i.astype(np.float64)

                sum_X_events = np.sum(X[event_mask], axis=0)
                E_X = risk_X_sum[first_idx] / risk_sum[first_idx][:, np.newaxis]
                grad = sum_X_events - np.sum(E_X * counts[:, np.newaxis], axis=0)

            hess = self._compute_hessian_breslow_fast(
                X, time, event, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
        else:
            # Efron: prefer Cython core if available; fall back to Python implementation
            # for environments without compiled extension or unexpected runtime issues.
            # Shift eta by a constant for numerical stability in exp(eta). This does not
            # change Efron gradient/Hessian because terms are scale-invariant.
            eta_efron = eta - np.max(eta)
            if HAS_CYTHON_EFRON and efron_pre is not None:
                try:
                    uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
                    grad, hess = _efron_grad_hess_cython(
                        eta_efron, X, risk_enter, risk_exit, uft_ix, nuft
                    )
                    # Align sign convention with existing CPU Efron backward path.
                    hess = -hess
                    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
                        raise FloatingPointError("non-finite Cython Efron grad/hess")
                except Exception:
                    from ._cox_efron_cy import efron_grad_hess_python
                    uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
                    grad, hess = efron_grad_hess_python(
                        eta_efron, X, risk_enter, risk_exit, uft_ix, nuft
                    )
                    hess = -hess
                    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
                        grad, hess = self._compute_gradient_hessian_efron_backward(
                            beta, X, time, event, efron_pre
                        )
            else:
                grad, hess = self._compute_gradient_hessian_efron_backward(
                    beta, X, time, event, efron_pre
                )

        return grad, hess

    def _compute_hessian_breslow_fast(
        self,
        X,
        time,
        event,
        risk_sum,
        risk_X_sum,
        exp_eta,
        first_idx=None,
        counts=None,
    ):
        """Compute Breslow Hessian with an auto-selected CPU strategy."""
        event_mask = event == 1
        if not np.any(event_mask):
            return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

        # Group tied events by unique failure times to share the same R(t)
        # denominator across all events at time t (Breslow ties).
        if first_idx is None or counts is None or len(first_idx) == 0:
            breslow_pre = getattr(self, "_breslow_pre", None)
            if (
                breslow_pre is not None
                and len(breslow_pre) == 2
                and breslow_pre[0].size > 0
            ):
                first_idx = breslow_pre[0].astype(np.int64, copy=False)
                counts = breslow_pre[1].astype(np.float64, copy=False)
            else:
                event_times = time[event_mask]
                uft, counts_i = np.unique(event_times, return_counts=True)
                first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                counts = counts_i.astype(np.float64)

        # Two CPU kernels are kept intentionally:
        # 1) Tensor path: higher memory, but can be faster for small p / few groups.
        # 2) Incremental path: lower memory traffic for larger (n, p).
        p = int(X.shape[1])
        n_groups = int(len(first_idx))
        estimated_bytes = _estimate_breslow_tensor_bytes(
            int(X.shape[0]), p, n_groups, int(X.dtype.itemsize)
        )
        max_bytes = _breslow_hessian_max_bytes()
        self._last_breslow_hessian_workspace_estimate_ = estimated_bytes
        self._last_breslow_hessian_workspace_limit_ = max_bytes
        if p <= 24 and n_groups <= 512 and estimated_bytes <= max_bytes:
            self._last_breslow_hessian_strategy_ = "tensor"
            return self._compute_hessian_breslow_tensor_grouped(
                X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
        self._last_breslow_hessian_strategy_ = "incremental"
        return self._compute_hessian_breslow_incremental_grouped(
            X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
        )

    def _compute_hessian_breslow_tensor_grouped(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """Grouped Breslow Hessian using explicit (n, p, p) tensor moments."""
        x2_weighted = np.einsum("ni,nj,n->nij", X, X, exp_eta)
        risk_X2_sum = np.cumsum(x2_weighted[::-1], axis=0)[::-1]
        risk_sum_at = risk_sum[first_idx]
        E_X = risk_X_sum[first_idx] / risk_sum_at[:, np.newaxis]
        E_XX = risk_X2_sum[first_idx] / risk_sum_at[:, np.newaxis, np.newaxis]
        centered = E_XX - np.einsum("ni,nj->nij", E_X, E_X)
        return -np.sum(centered * counts[:, np.newaxis, np.newaxis], axis=0)

    def _compute_hessian_breslow_incremental_grouped(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """Grouped Breslow Hessian with incremental risk-set second moments."""
        # risk_X2 tracks sum_{j in current risk set} exp_eta[j] * x_j x_j^T.
        X_exp = X * exp_eta[:, np.newaxis]
        risk_X2 = X_exp.T @ X

        hess = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        prev_idx = 0
        for g in range(len(first_idx)):
            idx = int(first_idx[g])
            if idx > prev_idx:
                blk = slice(prev_idx, idx)
                # Remove rows that are no longer in risk set.
                risk_X2 -= X_exp[blk].T @ X[blk]
                prev_idx = idx

            rs = float(risk_sum[idx])
            if rs <= 0.0:
                continue
            ex = risk_X_sum[idx] / rs
            exx = risk_X2 / rs
            hess -= counts[g] * (exx - np.outer(ex, ex))

        return hess

    def _compute_hessian_breslow_incremental_grouped_cupy(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """CuPy Breslow Hessian — vectorized via cumsum of outer products.

        O(n·p²) memory (acceptable on 16GB P100), zero Python loop over groups.
        """
        import cupy as cp

        n, p = int(X.shape[0]), int(X.shape[1])
        nuft = int(first_idx.shape[0])
        if nuft == 0:
            return cp.zeros((p, p), dtype=cp.float64)
        estimated_bytes = _estimate_breslow_tensor_bytes(
            n, p, nuft, int(X.dtype.itemsize)
        )
        max_bytes = _breslow_hessian_max_bytes()
        self._last_breslow_hessian_workspace_estimate_ = estimated_bytes
        self._last_breslow_hessian_workspace_limit_ = max_bytes
        if estimated_bytes > max_bytes:
            self._last_breslow_hessian_strategy_ = "cupy_streaming"
            return self._compute_hessian_breslow_streaming_grouped_cupy(
                X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
        self._last_breslow_hessian_strategy_ = "cupy_vectorized"

        X_exp = X * exp_eta[:, cp.newaxis]
        total = X_exp.T @ X  # (p, p)

        risk_at = risk_sum[first_idx]
        E_X = risk_X_sum[first_idx] / risk_at[:, None]
        sc = counts / risk_at  # (nuft,)

        # Cumsum of outer products → prefix at each failure time
        flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
        prefix_flat = cp.cumsum(flat, axis=0)  # (n, p*p)

        # prefix_at_g[g] = prefix_flat[first_idx[g] - 1] if first_idx[g] > 0 else 0
        fi = first_idx.astype(cp.int64)
        prefix_at_g = cp.zeros((nuft, p, p), dtype=cp.float64)
        mask = fi > 0
        if mask.any():
            prefix_at_g[mask] = prefix_flat[fi[mask] - 1].reshape(-1, p, p)

        # risk_X2[g] = total - prefix[g]
        risk_X2 = total[None, :, :] - prefix_at_g  # (nuft, p, p)

        # hess = -sum_g sc[g] * risk_X2[g] + sum_g counts[g] * outer(E_X[g], E_X[g])
        hess = -cp.einsum("g,gij->ij", sc, risk_X2)
        hess += cp.einsum("g,gi,gj->ij", counts, E_X, E_X)

        return hess

    def _compute_hessian_breslow_streaming_grouped_cupy(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """Bounded-memory CuPy Breslow Hessian using grouped GEMM updates."""
        import cupy as cp

        p = int(X.shape[1])
        first_idx_host = cp.asnumpy(first_idx).astype(np.int64, copy=False)
        X_exp = X * exp_eta[:, cp.newaxis]
        risk_X2 = X_exp.T @ X
        hess = cp.zeros((p, p), dtype=X.dtype)
        prev_idx = 0
        for group, idx_value in enumerate(first_idx_host):
            idx = int(idx_value)
            if idx > prev_idx:
                block = slice(prev_idx, idx)
                risk_X2 -= X_exp[block].T @ X[block]
                prev_idx = idx
            rs = risk_sum[idx]
            ex = risk_X_sum[idx] / rs
            hess -= counts[group] * (risk_X2 / rs - cp.outer(ex, ex))
        return hess

    def _compute_hessian_breslow_fused_cupy(self, X, first_idx, counts, exp_eta):
        """Run the bounded fused RawKernel; only import absence may fall back."""
        import cupy as cp
        try:
            from ._cox_efron_cuda import compute_breslow_hess_raw
        except ImportError:
            return None
        return compute_breslow_hess_raw(
            X,
            first_idx,
            counts,
            cupy_module=cp,
            exp_eta=exp_eta,
        )

    def _compute_hessian_breslow(self, beta, X, time, event, risk_sum, risk_X_sum, exp_eta):
        """
        Compute Hessian for Breslow approximation.

        Uses an incremental suffix-scan so total cost is O(n·p²) instead of
        the previous O(n_events × n × p²) triple-loop.

        Algorithm:
          1. Compute the full second-moment matrix M = (X * exp_eta).T @ X  -- O(n·p²).
          2. Walk through sorted event positions left-to-right, subtracting the
             contribution of rows that fall *before* the current event (and are
             therefore not in its risk set) from M incrementally.
             Each row is subtracted exactly once, so total subtraction work = O(n·p²).
        """
        n_samples, n_features = X.shape
        hess = np.zeros((n_features, n_features), dtype=np.float64)

        X_exp = X * exp_eta[:, np.newaxis]                  # (n, p)
        risk_X2_sum = X_exp.T @ X                           # (p, p), O(n·p²)

        event_positions = np.where(event)[0]                # sorted ascending
        prev_pos = 0

        for ev_i in event_positions:
            # Remove rows [prev_pos, ev_i) from risk_X2_sum;
            # they have t < t[ev_i] and are no longer in R(t[ev_i]).
            if ev_i > prev_pos:
                blk = slice(prev_pos, ev_i)
                risk_X2_sum -= X_exp[blk].T @ X[blk]       # O(k·p²), k = ev_i - prev_pos
            prev_pos = ev_i  # next event will subtract starting from here

            E_X = risk_X_sum[ev_i] / risk_sum[ev_i]        # (p,)
            E_XX = risk_X2_sum / risk_sum[ev_i]             # (p, p)
            hess -= E_XX - np.outer(E_X, E_X)

        return hess

    def _efron_unique_failure_indices(self, time: np.ndarray, event: np.ndarray):
        """
        Unique failure-time bookkeeping (single stratum), matching statsmodels PHSurvivalTime.
        `time` must be sorted ascending (as in fit).
        """
        ift = np.flatnonzero(event == 1)
        if ift.size == 0:
            return np.array([], dtype=np.float64), [], [], [], 0, np.array([], dtype=np.int32)
        ft = time[ift]
        uft = np.unique(ft)
        nuft = int(uft.size)

        # First row index at each unique failure time (sorted time); avoids searchsorted in log-likelihood loops.
        first_idx_uft = np.searchsorted(time, uft, side="left").astype(np.int32)

        # uft_ix: group indices of event rows by unique failure time.
        group_ids = np.searchsorted(uft, ft, side="left").astype(np.int32)  # shape: (n_events,)
        order_ev = np.argsort(group_ids, kind="stable")
        ift_sorted = ift[order_ev]
        group_sorted = group_ids[order_ev]
        counts_ev = np.bincount(group_sorted, minlength=nuft)
        ptr_ev = np.empty(nuft + 1, dtype=np.int32)
        ptr_ev[0] = 0
        ptr_ev[1:] = np.cumsum(counts_ev, dtype=np.int32)
        uft_ix = [ift_sorted[ptr_ev[i] : ptr_ev[i + 1]].tolist() for i in range(nuft)]

        # risk_enter: for each unique failure time i, indices of samples with
        # uft[i-1] <= time < uft[i] (samples entering risk set as we scan backward).
        # For i=0, includes all samples with time >= uft[0].
        j_enter = np.searchsorted(uft, time, side="right").astype(np.int32) - 1
        mask_enter = j_enter >= 0
        idx_enter = np.nonzero(mask_enter)[0]
        j_enter_m = j_enter[mask_enter]
        order_en = np.argsort(j_enter_m, kind="stable")
        idx_enter_sorted = idx_enter[order_en]
        j_enter_sorted = j_enter_m[order_en]
        counts_en = np.bincount(j_enter_sorted, minlength=nuft)
        ptr_en = np.empty(nuft + 1, dtype=np.int32)
        ptr_en[0] = 0
        ptr_en[1:] = np.cumsum(counts_en, dtype=np.int32)
        risk_enter = [
            idx_enter_sorted[ptr_en[i] : ptr_en[i + 1]].tolist() for i in range(nuft)
        ]

        # risk_exit: for backward scan, this is NOT used in the standard Efron algorithm.
        # The original code had a placeholder that put all samples at index 0, which was wrong.
        # For proper backward scan, we don't need risk_exit - we only add samples via risk_enter.
        # Set risk_exit to empty lists for all indices.
        risk_exit = [[] for _ in range(nuft)]

        return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft

    @staticmethod
    def _use_heavy_ties_cpu_fallback() -> bool:
        """Opt-in adaptive CPU fallback for heavy-ties GPU/Torch runs."""
        v = os.environ.get("STATGPU_HEAVY_TIES_CPU_FALLBACK", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _should_cpu_fallback_heavy_ties(self, n_samples, n_features, avg_tie_size):
        """Heuristic: small/medium problems with dense ties are often CPU-faster."""
        if not self._use_heavy_ties_cpu_fallback():
            return False
        if self.ties not in ("efron", "breslow"):
            return False
        if avg_tie_size < 8.0:
            return False
        return int(n_samples) <= 20000 and int(n_features) <= 64

    def _breslow_unique_failure_groups(self, time: np.ndarray, event: np.ndarray):
        """
        Breslow tie groups for sorted time/event.
        Returns (first_idx_uft, counts_uft), both int32 arrays.
        """
        ift = np.flatnonzero(event == 1)
        if ift.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        ft = time[ift]
        uft, counts = np.unique(ft, return_counts=True)
        first_idx_uft = np.searchsorted(time, uft, side="left").astype(np.int32)
        return first_idx_uft, counts.astype(np.int32)

    def _compute_gradient_hessian_efron_backward(self, beta, X, time, event, efron_pre=None):
        """
        Efron gradient and Hessian — incremental accumulator backward scan.

        Uses the same algorithm as statsmodels PHReg and the Cython path:
        maintain running xp0/xp1/xp2 accumulators, update incrementally at each
        failure time.  O(nuft·p²) time, O(p²) memory.

        Note: X and time are already sorted by time (caller guarantees this).
        """
        n_features = X.shape[1]
        linpred = X @ beta
        e_linpred = np.exp(linpred)

        # Build Efron precomputed structure if not provided
        if efron_pre is not None:
            uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
        else:
            event_mask = event == 1
            event_idx = np.where(event_mask)[0]
            if len(event_idx) == 0:
                return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)
            uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = self._efron_unique_failure_indices(time, event)

        if nuft == 0:
            return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)

        # first_idx_uft[g] = first row index in sorted data with time == uft[g]
        # Suffix sums with sentinel zero at end so that
        # risk_sum[i] - risk_sum[j] = sum(exp_eta[i:j]) for any i < j.
        n = X.shape[0]
        X_exp = X * e_linpred[:, None]
        risk_sum = np.zeros(n + 1, dtype=np.float64)
        risk_sum[:n] = np.cumsum(e_linpred[::-1])[::-1]
        risk_X_sum = np.zeros((n + 1, n_features), dtype=np.float64)
        risk_X_sum[:n] = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Dispatch: Numba > Vectorized cumsum > Python incremental
        # Vectorized cumsum: O(n·p²) memory, no Python loop — fast for p <= ~100.
        _VEC_MAX_P = int(os.environ.get("STATGPU_EFRON_VEC_MAX_P", "30"))

        if _HAS_NUMBA_EFRON:
            # Numba JIT — best for all sizes
            fail_ptr = np.zeros(nuft + 1, dtype=np.int64)
            for g in range(nuft):
                fail_ptr[g + 1] = fail_ptr[g] + len(uft_ix[g])
            n_fail = int(fail_ptr[nuft])
            fail_ind = np.empty(n_fail, dtype=np.int64)
            for g in range(nuft):
                ix = uft_ix[g]
                for j in range(len(ix)):
                    fail_ind[fail_ptr[g] + j] = int(ix[j])
            grad, hess = _efron_backward_scan_numba(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft.astype(np.int64),
                fail_ptr, fail_ind,
                nuft, n, n_features,
            )
        elif n_features <= _VEC_MAX_P:
            # Vectorized cumsum — eliminates Python loop, O(n·p²) memory
            grad, hess = _efron_backward_scan_vectorized(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft, uft_ix, nuft, n, n_features,
            )
        else:
            # Python incremental — O(p²) memory, Python loop over groups
            grad, hess = _efron_backward_scan_python(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft, uft_ix, nuft, n, n_features,
            )

        return grad, hess

    def _compute_gradient_hessian_gpu(
        self, beta, X, time, event, efron_pre=None, return_aux=False, entry=None, entry_ctx=None
    ):
        """Compute gradient and Hessian on GPU."""
        import cupy as cp
        import time as _time

        n_samples, n_features = X.shape

        profile_breslow = (
            os.environ.get("STATGPU_PROFILE_BRESLOW_CUDA", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        _t0_all = _time.perf_counter() if profile_breslow else None
        eta = X @ beta
        exp_eta = cp.exp(eta)
        event_mask = event == 1

        # Risk sets (entry-aware path uses dynamic masks below).
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1] if entry is None else None
        X_exp_eta = X * exp_eta[:, cp.newaxis]
        risk_X_sum = cp.cumsum(X_exp_eta[::-1], axis=0)[::-1] if entry is None else None
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_pre = _time.perf_counter()

        # Efron: when no ties, use Breslow vectorized path.
        if self.ties == "efron" and entry is None:
            if getattr(self, "_efron_all_singletons", False):
                ep = efron_pre if efron_pre is not None else getattr(self, "_efron_pre", None)
                if ep is not None:
                    _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(ep)
                    first_idx_uft = cp.asarray(first_idx_uft, dtype=cp.int32)
                    counts_uft = cp.ones(int(nuft), dtype=cp.int32)
                else:
                    uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
                    first_idx_uft = cp.searchsorted(time, uft, side="left")
                    counts_uft = counts_uft.astype(cp.int32, copy=False)
                counts_f = counts_uft.astype(cp.float64)
                grad_pre = getattr(self, "_event_X_sum_gpu", None)
                grad = (
                    grad_pre.copy()
                    if grad_pre is not None and int(grad_pre.shape[0]) == int(n_features)
                    else cp.sum(X[event_mask], axis=0)
                )
                E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
                grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
                use_fused_breslow = (
                    os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "0").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                hess = None
                if use_fused_breslow:
                    hess = self._compute_hessian_breslow_fused_cupy(
                        X, first_idx_uft, counts_f, exp_eta
                    )
                if hess is None:
                    hess = self._compute_hessian_breslow_incremental_grouped_cupy(
                        X, risk_sum, risk_X_sum, exp_eta, first_idx_uft, counts_f
                    )
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            if efron_pre is None:
                efron_pre = self._efron_unique_failure_indices(
                    cp.asnumpy(time), cp.asnumpy(event)
                )
            out = self._compute_gradient_hessian_efron_backward_gpu(
                beta, X, efron_pre
            )
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        # Breslow gradient/Hessian (entry-aware path).
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)

        if not cp.any(event_mask):
            out = (grad, cp.zeros((n_features, n_features), dtype=cp.float64))
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr = self._build_entry_ctx_gpu(
                    time, event, entry, cp
                )
                X_entry = cp.ascontiguousarray(X[entry_order])
                X_rem = cp.ascontiguousarray(X[rem_order])
                grad += cp.sum(X[event_idx], axis=0)
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                X_entry = entry_ctx[4] if len(entry_ctx) > 4 else X[entry_order]
                X_rem = entry_ctx[5] if len(entry_ctx) > 5 else X
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else cp.where(event_mask)[0]
                grad += entry_ctx[7] if len(entry_ctx) > 7 else cp.sum(X[event_mask], axis=0)
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            hess = cp.zeros((n_features, n_features), dtype=cp.float64)
            exp_entry = exp_eta[entry_order]
            exp_rem = exp_eta
            wx_entry = X_entry * exp_entry[:, cp.newaxis]
            wx_rem = X_rem * exp_rem[:, cp.newaxis]
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            s0_add_pref = cp.cumsum(exp_entry, axis=0)
            s0_rem_pref = cp.cumsum(exp_rem, axis=0)
            s1_add_pref = cp.cumsum(wx_entry, axis=0)
            s1_rem_pref = cp.cumsum(wx_rem, axis=0)
            s0_add = cp.zeros(n_groups, dtype=cp.float64)
            s0_rem = cp.zeros(n_groups, dtype=cp.float64)
            s1_add = cp.zeros((n_groups, n_features), dtype=cp.float64)
            s1_rem = cp.zeros((n_groups, n_features), dtype=cp.float64)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = cp.asarray(add_end_np[mask_add] - 1, dtype=cp.int64)
                mask_add_cp = cp.asarray(mask_add)
                s0_add[mask_add_cp] = s0_add_pref[idx_add]
                s1_add[mask_add_cp] = s1_add_pref[idx_add]
            if np.any(mask_rem):
                idx_rem = cp.asarray(rem_end_np[mask_rem] - 1, dtype=cp.int64)
                mask_rem_cp = cp.asarray(mask_rem)
                s0_rem[mask_rem_cp] = s0_rem_pref[idx_rem]
                s1_rem[mask_rem_cp] = s1_rem_pref[idx_rem]
            s0_vec = s0_add - s0_rem
            s1_vec = s1_add - s1_rem
            d_vec = cp.asarray(d_counts, dtype=cp.float64)
            s0_safe_vec = cp.maximum(s0_vec, 1e-15)
            use_efron_entry = (self.ties == "efron")
            ex_vec = s1_vec / s0_safe_vec[:, cp.newaxis]
            if not use_efron_entry:
                grad -= cp.sum(d_vec[:, cp.newaxis] * ex_vec, axis=0)
            if use_efron_entry:
                if fail_ptr is None:
                    fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                    fail_ptr[0] = 0
                    fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
                event_exp = exp_eta[event_idx]
                X_fail = X[event_idx]
            add_ptr = 0
            rem_ptr = 0
            s2 = cp.zeros((n_features, n_features), dtype=cp.float64)
            s2_block_size = int(os.environ.get("STATGPU_ENTRY_S2_BLOCK_SIZE", "8192"))
            if s2_block_size <= 0:
                s2_block_size = 10**18
            use_s2_fused = (
                os.environ.get("STATGPU_ENTRY_S2_FUSED_CUPY", "0").strip().lower()
                in ("1", "true", "yes", "on")
            )
            s2_fused_min_rows = int(os.environ.get("STATGPU_ENTRY_S2_FUSED_MIN_ROWS", "512"))
            if s2_fused_min_rows < 1:
                s2_fused_min_rows = 1
            for g in range(n_groups):
                add_end = int(add_end_np[g])
                if add_end > add_ptr:
                    x_add = X_entry[add_ptr:add_end]
                    w_add = exp_entry[add_ptr:add_end]
                    n_add = int(add_end - add_ptr)
                    if use_s2_fused and n_add >= s2_fused_min_rows:
                        s2 = self._s2_weighted_update_cupy_fused(s2, x_add, w_add, sign=1.0)
                    elif n_add <= s2_block_size:
                        s2 = s2 + (x_add.T @ (x_add * w_add[:, cp.newaxis]))
                    else:
                        s2 = self._s2_weighted_update_cupy_blocked(
                            s2, x_add, w_add, s2_block_size, sign=1.0
                        )
                    add_ptr = add_end

                rem_end = int(rem_end_np[g])
                if rem_end > rem_ptr:
                    x_rem = X_rem[rem_ptr:rem_end]
                    w_rem = exp_eta[rem_ptr:rem_end]
                    n_rem = int(rem_end - rem_ptr)
                    if use_s2_fused and n_rem >= s2_fused_min_rows:
                        s2 = self._s2_weighted_update_cupy_fused(s2, x_rem, w_rem, sign=-1.0)
                    elif n_rem <= s2_block_size:
                        s2 = s2 - (x_rem.T @ (x_rem * w_rem[:, cp.newaxis]))
                    else:
                        s2 = self._s2_weighted_update_cupy_blocked(
                            s2, x_rem, w_rem, s2_block_size, sign=-1.0
                        )
                    rem_ptr = rem_end

                d_t_f = float(d_counts[g])
                if d_t_f <= 0:
                    continue
                if use_efron_entry:
                    st = int(fail_ptr[g])
                    ed = int(fail_ptr[g + 1])
                    ef = event_exp[st:ed]
                    xf = X_fail[st:ed]
                    ef_sum = cp.sum(ef)
                    ef_x_sum = cp.sum(xf * ef[:, cp.newaxis], axis=0)
                    ef_x2_sum = (xf.T @ (xf * ef[:, cp.newaxis]))
                    s0_g = cp.maximum(s0_vec[g], 1e-15)
                    s1_g = s1_vec[g]
                    d_i = int(d_t_f)
                    for k in range(d_i):
                        frac = float(k) / float(d_i)
                        denom = cp.maximum(s0_g - frac * ef_sum, 1e-15)
                        s1_k = s1_g - frac * ef_x_sum
                        s2_k = s2 - frac * ef_x2_sum
                        ex_k = s1_k / denom
                        grad -= ex_k
                        hess -= s2_k / denom
                        hess += cp.outer(ex_k, ex_k)
                else:
                    s0_safe = s0_safe_vec[g]
                    hess -= (d_t_f / s0_safe) * s2
            if not use_efron_entry:
                hess += ex_vec.T @ (d_vec[:, cp.newaxis] * ex_vec)
            if return_aux:
                return grad, hess, (eta, exp_eta, risk_sum)
            return grad, hess

        # For Breslow ties, all events at the same failure time share the
        # same risk set R(t); grouping is required for correctness.
        breslow_pre_gpu = getattr(self, "_breslow_pre_gpu", None)
        if (
            breslow_pre_gpu is not None
            and len(breslow_pre_gpu) == 2
            and int(breslow_pre_gpu[0].size) > 0
        ):
            first_idx_uft, counts_uft = breslow_pre_gpu
        else:
            uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
            first_idx_uft = cp.searchsorted(time, uft, side="left")
            counts_uft = counts_uft.astype(cp.int32, copy=False)

        counts_f = getattr(self, "_breslow_counts_f_gpu", None)
        if counts_f is None or int(counts_f.shape[0]) != int(counts_uft.shape[0]):
            counts_f = counts_uft.astype(cp.float64)
        grad_pre = getattr(self, "_event_X_sum_gpu", None)
        grad = (
            grad_pre.copy()
            if grad_pre is not None and int(grad_pre.shape[0]) == int(n_features)
            else cp.sum(X[event_mask], axis=0)
        )
        E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
        grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_grad = _time.perf_counter()
        use_fused_breslow = (
            os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        hess = None
        if use_fused_breslow:
            hess = self._compute_hessian_breslow_fused_cupy(
                X, first_idx_uft, counts_f, exp_eta
            )
        if hess is None:
            hess = self._compute_hessian_breslow_incremental_grouped_cupy(
                X, risk_sum, risk_X_sum, exp_eta, first_idx_uft, counts_f
            )
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_hess = _time.perf_counter()
            print(
                f"[CUDA Breslow profile] pre={(_t_pre - _t0_all):.4f}s "
                f"grad={(_t_grad - _t_pre):.4f}s "
                f"hess={(_t_hess - _t_grad):.4f}s "
                f"total={(_t_hess - _t0_all):.4f}s"
            )
        if return_aux:
            return grad, hess, (eta, exp_eta, risk_sum)
        return grad, hess

    def _s2_weighted_update_cupy_blocked(self, s2, x, w, block_size, sign=1.0):
        """Blocked update for large slices: s2 += sign * X^T (X * w)."""
        import cupy as cp

        n = int(x.shape[0])
        if n <= 0:
            return s2
        for st in range(0, n, block_size):
            ed = min(st + block_size, n)
            xb = x[st:ed]
            wb = w[st:ed]
            s2 = s2 + sign * (xb.T @ (xb * wb[:, cp.newaxis]))
        return s2

    def _get_entry_s2_fused_kernel_cupy(self):
        """Build/cache CuPy RawKernel for fused weighted X^T X update."""
        k = getattr(self, "_entry_s2_fused_kernel_cupy", None)
        if k is not None:
            return k
        import cupy as cp

        src = r"""
        extern "C" __global__
        void entry_s2_outer_f64(const double* x, const double* w, double* out, int n, int p) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (i >= p || j >= p) return;
            double acc = 0.0;
            for (int r = 0; r < n; ++r) {
                double wr = w[r];
                double xi = x[(size_t)r * (size_t)p + (size_t)i];
                double xj = x[(size_t)r * (size_t)p + (size_t)j];
                acc += wr * xi * xj;
            }
            out[(size_t)i * (size_t)p + (size_t)j] = acc;
        }
        """
        k = cp.RawKernel(src, "entry_s2_outer_f64")
        self._entry_s2_fused_kernel_cupy = k
        return k

    def _s2_weighted_update_cupy_fused(self, s2, x, w, sign=1.0):
        """CuPy fused kernel update for s2 += sign * X^T (X * w)."""
        import cupy as cp

        n = int(x.shape[0])
        if n <= 0:
            return s2
        x = cp.ascontiguousarray(x, dtype=cp.float64)
        w = cp.ascontiguousarray(w, dtype=cp.float64)
        p = int(x.shape[1])
        out = cp.empty((p, p), dtype=cp.float64)
        threads = (16, 16, 1)
        blocks = ((p + 15) // 16, (p + 15) // 16, 1)
        ker = self._get_entry_s2_fused_kernel_cupy()
        ker(blocks, threads, (x, w, out, np.int32(n), np.int32(p)))
        if sign > 0:
            return s2 + out
        return s2 - out

    def _compute_gradient_hessian_efron_backward_gpu(self, beta, X, efron_pre):
        """CuPy Efron grad/Hessian: prefer single CUDA RawKernel scan, else Python-loop fallback."""
        import cupy as cp

        uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = X.shape[1]
        if nuft == 0:
            return cp.zeros(n_features, dtype=cp.float64), cp.zeros(
                (n_features, n_features), dtype=cp.float64
            )

        n_samples = int(X.shape[0])
        avg_tie = float(n_samples) / max(1.0, float(nuft))
        use_grouped_gemm = (
            os.environ.get("STATGPU_EFRON_GROUPED_GEMM", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if use_grouped_gemm and n_features <= 192 and avg_tie >= 24.0:
            return self._compute_gradient_hessian_efron_grouped_gemm_cupy(
                beta, X, efron_pre
            )

        try:
            from ._cox_efron_cuda import compute_efron_grad_hess_raw

            csr_gpu = getattr(self, "_efron_pre_csr_gpu", None)
            if csr_gpu is not None:
                out = compute_efron_grad_hess_raw(
                    X,
                    beta,
                    efron_pre,
                    efron_csr=csr_gpu,
                    cupy_module=cp,
                )
            else:
                out = compute_efron_grad_hess_raw(X, beta, efron_pre, cupy_module=cp)
            if out is not None:
                return out[0], out[1]
        except Exception:
            pass

        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        e_linpred = cp.exp(linpred)

        grad = cp.zeros(n_features, dtype=cp.float64)
        hess_inner = cp.zeros((n_features, n_features), dtype=cp.float64)
        xp0 = cp.zeros((), dtype=cp.float64)
        xp1 = cp.zeros(n_features, dtype=cp.float64)
        xp2 = cp.zeros((n_features, n_features), dtype=cp.float64)
        for i in range(nuft)[::-1]:
            ix = risk_enter[i]
            if len(ix) > 0:
                ix = cp.array(ix, dtype=cp.int32)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 = xp0 + elx.sum()
                xp1 = xp1 + (elx[:, None] * v).sum(axis=0)
                xp2 = xp2 + cp.einsum("ij,ik,i->jk", v, v, elx)
            ixf = uft_ix[i]
            if len(ixf) > 0:
                ixf = cp.array(ixf, dtype=cp.int32)
                v = X[ixf]
                elx = e_linpred[ixf]
                xp0f = elx.sum()
                xp1f = (elx[:, None] * v).sum(axis=0)
                xp2f = cp.einsum("ij,ik,i->jk", v, v, elx)
                m = len(ixf)
                J = cp.arange(m, dtype=cp.float64) / max(m, 1)
                c0 = xp0 - J * xp0f
                c0 = cp.maximum(c0, 1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = cp.sum(ak)
                sum_J_c0 = cp.sum(bk)
                sum_aa = cp.sum(ak * ak)
                sum_bb = cp.sum(bk * bk)
                sum_ab = cp.sum(ak * bk)
                grad = grad + v.sum(axis=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * cp.outer(xp1, xp1)
                    + sum_bb * cp.outer(xp1f, xp1f)
                    - sum_ab * (cp.outer(xp1, xp1f) + cp.outer(xp1f, xp1))
                )
            ix = risk_exit[i]
            if len(ix) > 0:
                ix = cp.array(ix, dtype=cp.int32)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 = xp0 - elx.sum()
                xp1 = xp1 - (elx[:, None] * v).sum(axis=0)
                xp2 = xp2 - cp.einsum("ij,ik,i->jk", v, v, elx)

        hess = -hess_inner
        return grad, hess

    @staticmethod
    def _efron_cumulative_workspace_fits(
        efron_pre,
        n_samples,
        n_features,
        itemsize,
        *,
        include_second_moments,
    ):
        """Return whether the dense Efron workspace fits its configured cap."""
        _, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
        if (
            nuft == 0
            or first_idx_uft is None
            or float(n_samples) / float(max(nuft, 1)) < 24.0
        ):
            return False

        max_tie = max((len(ix) for ix in uft_ix), default=0)
        # ``frac``, denominators, masks, inverse weights, and reduction
        # temporaries coexist at the group-by-substep boundary. Eight dense
        # values per substep is a conservative estimate across CuPy and Torch.
        substep_bytes = 8 * nuft * max_tie * itemsize
        moment_bytes = 0
        if include_second_moments:
            moment_bytes = 2 * n_samples * n_features * n_features * itemsize
        estimated_bytes = moment_bytes + substep_bytes
        max_bytes = max(
            0,
            int(
                os.environ.get(
                    "STATGPU_EFRON_CUMULATIVE_MAX_BYTES", 512 * 1024 * 1024
                )
            ),
        )
        return estimated_bytes <= max_bytes

    def _compute_gradient_hessian_efron_grouped_gemm_cupy(self, beta, X, efron_pre):
        """Vectorized CuPy Efron moments from cumulative risk-set statistics.

        Dense ties previously launched several small kernels for every failure
        group.  For memory-safe shapes, form all risk/failure moments once and
        evaluate every Efron substep as one group-by-substep matrix.  Wide or
        very large shapes retain the bounded grouped-GEMM fallback.
        """
        import cupy as cp

        _, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        if not self._efron_cumulative_workspace_fits(
            efron_pre,
            n_samples,
            n_features,
            int(X.dtype.itemsize),
            include_second_moments=True,
        ):
            return self._compute_gradient_hessian_efron_grouped_gemm_loop_cupy(
                beta, X, efron_pre
            )

        csr_gpu = getattr(self, "_efron_pre_csr_gpu", None)
        if csr_gpu is not None:
            _, _, _, _, fail_ptr, fail_ind, first_idx, _ = csr_gpu
        else:
            counts_np = np.fromiter((len(ix) for ix in uft_ix), dtype=np.int64)
            fail_ptr_np = np.empty(nuft + 1, dtype=np.int64)
            fail_ptr_np[0] = 0
            fail_ptr_np[1:] = np.cumsum(counts_np, dtype=np.int64)
            fail_ind_np = np.asarray(
                [row for group in uft_ix for row in group], dtype=np.int64
            )
            fail_ptr = cp.asarray(fail_ptr_np)
            fail_ind = cp.asarray(fail_ind_np)
            first_idx = cp.asarray(first_idx_uft, dtype=cp.int64)

        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        weights = cp.exp(linpred)
        first_idx = first_idx.astype(cp.int64, copy=False)
        fail_ptr = fail_ptr.astype(cp.int64, copy=False)
        fail_ind = fail_ind.astype(cp.int64, copy=False)

        risk0_all = cp.cumsum(weights[::-1], axis=0)[::-1]
        weighted_X = weights[:, None] * X
        risk1_all = cp.cumsum(weighted_X[::-1], axis=0)[::-1]
        row_second = weighted_X[:, :, None] * X[:, None, :]
        risk2_all = cp.cumsum(row_second[::-1], axis=0)[::-1]
        risk0 = risk0_all[first_idx]
        risk1 = risk1_all[first_idx]
        risk2 = risk2_all[first_idx].copy()
        del risk0_all, risk1_all, risk2_all, row_second, weighted_X

        fail_X = X[fail_ind]
        fail_weights = weights[fail_ind]
        fail_weighted_X = fail_weights[:, None] * fail_X

        def segment_sum(values):
            zero = cp.zeros((1,) + tuple(values.shape[1:]), dtype=values.dtype)
            prefix = cp.concatenate((zero, cp.cumsum(values, axis=0)), axis=0)
            return prefix[fail_ptr[1:]] - prefix[fail_ptr[:-1]]

        fail0 = segment_sum(fail_weights)
        fail1 = segment_sum(fail_weighted_X)
        fail2 = segment_sum(
            fail_weighted_X[:, :, None] * fail_X[:, None, :]
        )
        fail_X_sum = segment_sum(fail_X)
        counts = (fail_ptr[1:] - fail_ptr[:-1]).astype(X.dtype, copy=False)
        max_tie = max(len(ix) for ix in uft_ix)
        steps = cp.arange(max_tie, dtype=X.dtype).reshape(1, -1)
        active = steps < counts.reshape(-1, 1)
        frac = steps / counts.reshape(-1, 1)
        denominator = cp.maximum(
            risk0.reshape(-1, 1) - frac * fail0.reshape(-1, 1), 1e-300
        )
        inv = cp.where(active, 1.0 / denominator, 0.0)
        frac_inv = frac * inv
        sum_inv = cp.sum(inv, axis=1)
        sum_frac_inv = cp.sum(frac_inv, axis=1)
        sum_inv2 = cp.sum(inv * inv, axis=1)
        sum_frac_inv2 = cp.sum(frac_inv * frac_inv, axis=1)
        sum_cross = cp.sum(inv * frac_inv, axis=1)

        grad = cp.sum(
            fail_X_sum
            - risk1 * sum_inv[:, None]
            + fail1 * sum_frac_inv[:, None],
            axis=0,
        )
        risk_outer = risk1[:, :, None] * risk1[:, None, :]
        fail_outer = fail1[:, :, None] * fail1[:, None, :]
        cross_outer = (
            risk1[:, :, None] * fail1[:, None, :]
            + fail1[:, :, None] * risk1[:, None, :]
        )
        hess_inner = cp.sum(
            risk2 * sum_inv[:, None, None]
            - fail2 * sum_frac_inv[:, None, None]
            - risk_outer * sum_inv2[:, None, None]
            - fail_outer * sum_frac_inv2[:, None, None]
            + cross_outer * sum_cross[:, None, None],
            axis=0,
        )
        return grad, -hess_inner

    def _compute_gradient_hessian_efron_grouped_gemm_loop_cupy(self, beta, X, efron_pre):
        """Memory-bounded grouped-GEMM fallback for CuPy Efron moments."""
        import cupy as cp

        _, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = int(X.shape[1])
        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        e_linpred = cp.exp(linpred)

        grad = cp.zeros(n_features, dtype=cp.float64)
        hess_inner = cp.zeros((n_features, n_features), dtype=cp.float64)
        xp0 = cp.zeros((), dtype=cp.float64)
        xp1 = cp.zeros(n_features, dtype=cp.float64)
        xp2 = cp.zeros((n_features, n_features), dtype=cp.float64)
        j_cache = {}

        for i in range(nuft - 1, -1, -1):
            ix = risk_enter[i]
            if len(ix) > 0:
                idx = cp.asarray(ix, dtype=cp.int32)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 + cp.sum(elx)
                xp1 = xp1 + cp.sum(wv, axis=0)
                xp2 = xp2 + (wv.T @ v)

            ixf = uft_ix[i]
            if len(ixf) > 0:
                idxf = cp.asarray(ixf, dtype=cp.int32)
                v = X[idxf]
                elx = e_linpred[idxf]
                wv = v * elx[:, None]
                xp0f = cp.sum(elx)
                xp1f = cp.sum(wv, axis=0)
                xp2f = wv.T @ v
                m = len(ixf)
                if m not in j_cache:
                    j_cache[m] = cp.arange(m, dtype=cp.float64) / float(max(m, 1))
                J = j_cache[m]
                c0 = cp.maximum(xp0 - J * xp0f, 1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = cp.sum(ak)
                sum_J_c0 = cp.sum(bk)
                sum_aa = cp.sum(ak * ak)
                sum_bb = cp.sum(bk * bk)
                sum_ab = cp.sum(ak * bk)
                grad = grad + cp.sum(v, axis=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * cp.outer(xp1, xp1)
                    + sum_bb * cp.outer(xp1f, xp1f)
                    - sum_ab * (cp.outer(xp1, xp1f) + cp.outer(xp1f, xp1))
                )

            ix = risk_exit[i]
            if len(ix) > 0:
                idx = cp.asarray(ix, dtype=cp.int32)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 - cp.sum(elx)
                xp1 = xp1 - cp.sum(wv, axis=0)
                xp2 = xp2 - (wv.T @ v)

        return grad, -hess_inner

    def _solve_newton_delta_torch(self, hess, grad):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        import torch

        p = int(hess.shape[0])
        H = -hess
        eps = 1e-11 * (torch.max(torch.abs(torch.diag(H))) + 1.0)
        H = H + eps * torch.eye(p, dtype=torch.float64, device=hess.device)
        try:
            return -torch.linalg.solve(H, grad)
        except Exception as exc:
            if not _is_singular_linalg_error(exc):
                raise
        return _solve_counting_information(hess, grad, "torch", torch)

    def _efron_cumulative_indices_torch(self, efron_pre, device):
        """Cache grouped Efron indices on the active Torch device."""
        import torch

        cache = getattr(self, "_efron_cumulative_torch_cache", None)
        if cache is not None and cache[0] is efron_pre and cache[1] == device:
            return cache[2:]
        _, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
        counts_np = np.fromiter((len(ix) for ix in uft_ix), dtype=np.int64)
        fail_ptr_np = np.empty(nuft + 1, dtype=np.int64)
        fail_ptr_np[0] = 0
        fail_ptr_np[1:] = np.cumsum(counts_np, dtype=np.int64)
        fail_ind_np = np.asarray(
            [row for group in uft_ix for row in group], dtype=np.int64
        )
        first_idx = torch.as_tensor(first_idx_uft, dtype=torch.long, device=device)
        fail_ptr = torch.as_tensor(fail_ptr_np, dtype=torch.long, device=device)
        fail_ind = torch.as_tensor(fail_ind_np, dtype=torch.long, device=device)
        counts = torch.as_tensor(counts_np, dtype=torch.long, device=device)
        max_tie = int(np.max(counts_np)) if counts_np.size else 0
        cache = (
            efron_pre,
            device,
            first_idx,
            fail_ptr,
            fail_ind,
            counts,
            max_tie,
        )
        self._efron_cumulative_torch_cache = cache
        return cache[2:]

    def _compute_gradient_hessian_efron_grouped_gemm_torch(self, beta, X, efron_pre):
        """Vectorized Torch Efron moments from cumulative risk-set statistics."""
        import torch

        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        if not self._efron_cumulative_workspace_fits(
            efron_pre,
            n_samples,
            n_features,
            X.element_size(),
            include_second_moments=True,
        ):
            return self._compute_gradient_hessian_efron_grouped_gemm_loop_torch(
                beta, X, efron_pre
            )

        first_idx, fail_ptr, fail_ind, counts_int, max_tie = (
            self._efron_cumulative_indices_torch(efron_pre, beta.device)
        )

        linpred = X @ beta
        linpred = linpred - torch.max(linpred)
        weights = torch.exp(linpred)
        risk0_all = torch.flip(torch.cumsum(torch.flip(weights, (0,)), dim=0), (0,))
        weighted_X = weights[:, None] * X
        risk1_all = torch.flip(
            torch.cumsum(torch.flip(weighted_X, (0,)), dim=0), (0,)
        )
        row_second = weighted_X[:, :, None] * X[:, None, :]
        risk2_all = torch.flip(
            torch.cumsum(torch.flip(row_second, (0,)), dim=0), (0,)
        )
        risk0 = risk0_all[first_idx]
        risk1 = risk1_all[first_idx]
        risk2 = risk2_all[first_idx].clone()
        del risk0_all, risk1_all, risk2_all, row_second, weighted_X

        fail_X = X[fail_ind]
        fail_weights = weights[fail_ind]
        fail_weighted_X = fail_weights[:, None] * fail_X

        def segment_sum(values):
            zero = torch.zeros(
                (1,) + tuple(values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
            prefix = torch.cat((zero, torch.cumsum(values, dim=0)), dim=0)
            return prefix[fail_ptr[1:]] - prefix[fail_ptr[:-1]]

        fail0 = segment_sum(fail_weights)
        fail1 = segment_sum(fail_weighted_X)
        fail2 = segment_sum(
            fail_weighted_X[:, :, None] * fail_X[:, None, :]
        )
        fail_X_sum = segment_sum(fail_X)
        counts = counts_int.to(dtype=X.dtype)
        max_tie = int(max_tie)
        steps = torch.arange(max_tie, dtype=X.dtype, device=X.device).reshape(1, -1)
        active = steps < counts.reshape(-1, 1)
        frac = steps / counts.reshape(-1, 1)
        denominator = torch.clamp(
            risk0.reshape(-1, 1) - frac * fail0.reshape(-1, 1), min=1e-300
        )
        inv = torch.where(active, 1.0 / denominator, torch.zeros_like(denominator))
        frac_inv = frac * inv
        sum_inv = torch.sum(inv, dim=1)
        sum_frac_inv = torch.sum(frac_inv, dim=1)
        sum_inv2 = torch.sum(inv * inv, dim=1)
        sum_frac_inv2 = torch.sum(frac_inv * frac_inv, dim=1)
        sum_cross = torch.sum(inv * frac_inv, dim=1)

        grad = torch.sum(
            fail_X_sum
            - risk1 * sum_inv[:, None]
            + fail1 * sum_frac_inv[:, None],
            dim=0,
        )
        risk_outer = risk1[:, :, None] * risk1[:, None, :]
        fail_outer = fail1[:, :, None] * fail1[:, None, :]
        cross_outer = (
            risk1[:, :, None] * fail1[:, None, :]
            + fail1[:, :, None] * risk1[:, None, :]
        )
        hess_inner = torch.sum(
            risk2 * sum_inv[:, None, None]
            - fail2 * sum_frac_inv[:, None, None]
            - risk_outer * sum_inv2[:, None, None]
            - fail_outer * sum_frac_inv2[:, None, None]
            + cross_outer * sum_cross[:, None, None],
            dim=0,
        )
        return grad, -hess_inner

    def _compute_gradient_hessian_efron_grouped_gemm_loop_torch(self, beta, X, efron_pre):
        """Memory-bounded grouped-GEMM fallback for Torch Efron moments."""
        import torch

        _, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = int(X.shape[1])
        linpred = X @ beta
        linpred = linpred - torch.max(linpred)
        e_linpred = torch.exp(linpred)

        grad = torch.zeros(n_features, dtype=torch.float64, device=beta.device)
        hess_inner = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
        xp0 = torch.zeros((), dtype=torch.float64, device=beta.device)
        xp1 = torch.zeros(n_features, dtype=torch.float64, device=beta.device)
        xp2 = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
        j_cache = {}

        for i in range(nuft - 1, -1, -1):
            ix = risk_enter[i]
            if len(ix) > 0:
                idx = torch.as_tensor(ix, dtype=torch.long, device=beta.device)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 + torch.sum(elx)
                xp1 = xp1 + torch.sum(wv, dim=0)
                xp2 = xp2 + (wv.transpose(0, 1) @ v)

            ixf = uft_ix[i]
            if len(ixf) > 0:
                idxf = torch.as_tensor(ixf, dtype=torch.long, device=beta.device)
                v = X[idxf]
                elx = e_linpred[idxf]
                wv = v * elx[:, None]
                xp0f = torch.sum(elx)
                xp1f = torch.sum(wv, dim=0)
                xp2f = wv.transpose(0, 1) @ v
                m = len(ixf)
                if m not in j_cache:
                    j_cache[m] = torch.arange(m, dtype=torch.float64, device=beta.device) / float(max(m, 1))
                J = j_cache[m]
                c0 = torch.clamp(xp0 - J * xp0f, min=1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = torch.sum(ak)
                sum_J_c0 = torch.sum(bk)
                sum_aa = torch.sum(ak * ak)
                sum_bb = torch.sum(bk * bk)
                sum_ab = torch.sum(ak * bk)
                grad = grad + torch.sum(v, dim=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * torch.outer(xp1, xp1)
                    + sum_bb * torch.outer(xp1f, xp1f)
                    - sum_ab * (torch.outer(xp1, xp1f) + torch.outer(xp1f, xp1))
                )

            ix = risk_exit[i]
            if len(ix) > 0:
                idx = torch.as_tensor(ix, dtype=torch.long, device=beta.device)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 - torch.sum(elx)
                xp1 = xp1 - torch.sum(wv, dim=0)
                xp2 = xp2 - (wv.transpose(0, 1) @ v)

        return grad, -hess_inner

    def _compute_log_likelihood_torch(self, beta, X, time, event, efron_pre=None, entry=None, entry_ctx=None):
        """Compute log partial likelihood on Torch."""
        import torch

        eta = X @ beta
        exp_eta = torch.exp(eta)
        # Entry+breslow path does not consume risk_sum; skip the cumsum to
        # reduce per-evaluation overhead during line-search probes.
        risk_sum = None if entry is not None else torch.cumsum(exp_eta.flip(0), dim=0).flip(0)
        return self._compute_log_likelihood_torch_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre, entry=entry, entry_ctx=entry_ctx
        )

    def _build_entry_ctx_torch(self, time, event, entry, device):
        """Build entry-time grouped indexing context for a specific sorted Torch view."""
        import torch

        event_mask = event == 1
        event_idx = torch.where(event_mask)[0]
        evt_t = time[event_idx].detach().cpu().numpy()
        if evt_t.size == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                np.zeros((1,), dtype=np.int64),
            )
        uft_np, d_counts = np.unique(evt_t, return_counts=True)
        d_counts = d_counts.astype(np.float64, copy=False)
        entry_order = torch.argsort(entry, stable=True)
        entry_sorted_np = entry.index_select(0, entry_order).detach().cpu().numpy()
        time_np = time.detach().cpu().numpy()
        add_end_np = np.searchsorted(entry_sorted_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_end_np = np.searchsorted(time_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_order = torch.arange(int(time.shape[0]), dtype=torch.long, device=device)
        event_idx = event_idx.to(torch.long)
        fail_ptr = np.empty(d_counts.shape[0] + 1, dtype=np.int64)
        fail_ptr[0] = 0
        fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
        return (entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr)

    def _compute_log_likelihood_torch_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None, entry=None, entry_ctx=None
    ):
        """Compute log partial likelihood on Torch with precomputed stats."""
        import torch

        ll = torch.tensor(0.0, dtype=torch.float64, device=eta.device)
        event_mask = event == 1

        if not torch.any(event_mask):
            return ll

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, _rem_order, event_idx, fail_ptr = self._build_entry_ctx_torch(
                    time, event, entry, eta.device
                )
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else torch.where(event_mask)[0]
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None

            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                return torch.tensor(0.0, dtype=torch.float64, device=eta.device)
            if fail_ptr is None:
                fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                fail_ptr[0] = 0
                fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)

            exp_entry = exp_eta.index_select(0, entry_order)
            exp_rem = exp_eta
            s0_add_pref = torch.cumsum(exp_entry, dim=0)
            s0_rem_pref = torch.cumsum(exp_rem, dim=0)
            s0_add = torch.zeros(n_groups, dtype=torch.float64, device=eta.device)
            s0_rem = torch.zeros(n_groups, dtype=torch.float64, device=eta.device)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = torch.as_tensor(add_end_np[mask_add] - 1, dtype=torch.long, device=eta.device)
                s0_add[torch.as_tensor(mask_add, dtype=torch.bool, device=eta.device)] = s0_add_pref.index_select(0, idx_add)
            if np.any(mask_rem):
                idx_rem = torch.as_tensor(rem_end_np[mask_rem] - 1, dtype=torch.long, device=eta.device)
                s0_rem[torch.as_tensor(mask_rem, dtype=torch.bool, device=eta.device)] = s0_rem_pref.index_select(0, idx_rem)
            s0_vec = torch.clamp(s0_add - s0_rem, min=1e-300)
            event_eta = eta.index_select(0, event_idx)

            if self.ties == "breslow":
                d_vec = torch.as_tensor(d_counts, dtype=torch.float64, device=eta.device)
                return torch.sum(event_eta) - torch.sum(d_vec * torch.log(s0_vec))

            ll = torch.sum(event_eta)
            event_exp = exp_eta.index_select(0, event_idx)
            for g in range(n_groups):
                d = int(d_counts[g])
                if d <= 0:
                    continue
                st = int(fail_ptr[g])
                ed = int(fail_ptr[g + 1])
                ef = torch.sum(event_exp[st:ed])
                base = s0_vec[g]
                for k in range(d):
                    denom = torch.clamp(base - (float(k) / float(d)) * ef, min=1e-300)
                    ll = ll - torch.log(denom)
            return ll

        if self.ties == "breslow":
            # Vectorized Breslow using cached failure groups
            breslow_pre_torch = getattr(self, "_breslow_pre_torch", None)
            if (
                breslow_pre_torch is not None
                and len(breslow_pre_torch) == 2
                and int(breslow_pre_torch[0].numel()) > 0
            ):
                first_idx_uft, counts_uft = breslow_pre_torch
            else:
                uft, counts_uft = torch.unique(time[event_mask], return_counts=True)
                first_idx_uft = torch.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.to(torch.int32)
            risk_at = risk_sum[first_idx_uft]
            return torch.sum(eta[event_mask]) - torch.sum(
                counts_uft.to(torch.float64) * torch.log(risk_at)
            )

        # Efron: keep computation fully on torch backend.
        if efron_pre is not None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)
            # No-tie Efron equals Breslow; keep computation on torch device.
            if not needs_exact_ties:
                _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
                first_idx_t = torch.as_tensor(first_idx_uft, dtype=torch.int64, device=eta.device)
                counts_t = torch.ones(int(nuft), dtype=torch.float64, device=eta.device)
                risk_at = risk_sum[first_idx_t]
                return torch.sum(eta[event_mask]) - torch.sum(counts_t * torch.log(risk_at))

        if (
            efron_pre is not None
            and needs_exact_ties
            and self._efron_cumulative_workspace_fits(
                efron_pre,
                int(eta.shape[0]),
                0,
                eta.element_size(),
                include_second_moments=False,
            )
        ):
            first_idx, fail_ptr, fail_ind, counts_int, max_tie = (
                self._efron_cumulative_indices_torch(efron_pre, eta.device)
            )
            risk_at = risk_sum[first_idx]
            fail_weights = exp_eta[fail_ind]
            zero = torch.zeros(1, dtype=exp_eta.dtype, device=eta.device)
            fail_prefix = torch.cat((zero, torch.cumsum(fail_weights, dim=0)))
            fail_sum = fail_prefix[fail_ptr[1:]] - fail_prefix[fail_ptr[:-1]]
            counts = counts_int.to(dtype=eta.dtype)
            steps = torch.arange(
                int(max_tie), dtype=eta.dtype, device=eta.device
            ).reshape(1, -1)
            active = steps < counts.reshape(-1, 1)
            frac = steps / counts.reshape(-1, 1)
            denominator = torch.clamp(
                risk_at.reshape(-1, 1) - frac * fail_sum.reshape(-1, 1),
                min=1e-300,
            )
            log_terms = torch.where(
                active, torch.log(denominator), torch.zeros_like(denominator)
            )
            return torch.sum(eta[fail_ind]) - torch.sum(log_terms)

        # Memory-bounded fallback for sparse ties or oversized cumulative moments.
        unique_times = torch.unique(time[event_mask])
        for t in unique_times:
            at_time_t = time == t
            events_at_t = at_time_t & event_mask
            d = int(torch.sum(events_at_t).item())

            if d == 0:
                continue

            risk_indices = torch.where(time >= t)[0]
            if risk_indices.numel() == 0:
                continue

            first_idx = risk_indices[0]
            risk_at_t = risk_sum[first_idx]
            sum_events = torch.sum(exp_eta[events_at_t])

            ll += torch.sum(eta[events_at_t])
            for k in range(d):
                ll -= torch.log(torch.maximum(risk_at_t - (k / d) * sum_events, torch.tensor(1e-300, dtype=torch.float64, device=eta.device)))

        return ll

    def _compute_gradient_hessian_torch(
        self, beta, X, time, event, efron_pre=None, return_aux=False, entry=None, entry_ctx=None
    ):
        """Fully vectorized gradient/Hessian for Torch - Efron and Breslow."""
        import torch
        n_samples, n_features = X.shape
        eta = X @ beta
        exp_eta = torch.exp(eta)
        rev_idx = torch.arange(n_samples - 1, -1, -1, device=beta.device)
        risk_sum = torch.cumsum(exp_eta[rev_idx], dim=0)[rev_idx] if entry is None else None

        if self.ties == "efron" and efron_pre is not None and entry is None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)

            if needs_exact_ties:
                # Triton as optional fast path.
                if (
                    os.environ.get("STATGPU_EFRON_TRITON", "0").strip().lower()
                    in ("1", "true", "yes", "on")
                    and beta.is_cuda
                ):
                    from statgpu.survival._cox_efron_triton import (
                        compute_efron_grad_hess_triton,
                    )
                    triton_out = compute_efron_grad_hess_triton(X, beta, efron_pre)
                    if triton_out is not None:
                        grad, hess = triton_out
                        if return_aux:
                            return grad, hess, (eta, exp_eta, risk_sum)
                        return grad, hess

                # Mandatory exact fallback: grouped-GEMM for all real ties.
                out = self._compute_gradient_hessian_efron_grouped_gemm_torch(
                    beta, X, efron_pre
                )
                if return_aux:
                    return out[0], out[1], (eta, exp_eta, risk_sum)
                return out

            # ---- Triton Efron path ----
            if (
                os.environ.get("STATGPU_EFRON_TRITON", "0").strip().lower()
                in ("1", "true", "yes", "on")
                and beta.is_cuda
            ):
                from statgpu.survival._cox_efron_triton import compute_efron_grad_hess_triton
                triton_out = compute_efron_grad_hess_triton(X, beta, efron_pre)
                if triton_out is not None:
                    grad, hess = triton_out
                    if return_aux:
                        return grad, hess, (eta, exp_eta, risk_sum)
                    return grad, hess

            if needs_exact_ties:
                out = self._compute_gradient_hessian_efron_grouped_gemm_torch(
                    beta, X, efron_pre
                )
                if return_aux:
                    return out[0], out[1], (eta, exp_eta, risk_sum)
                return out

        # Reverse cumsum for risk sets (vectorized)
        risk_X_sum = torch.cumsum((X * exp_eta[:, None])[rev_idx], dim=0)[rev_idx] if entry is None else None

        event_mask = event == 1
        if not torch.any(event_mask):
            out = (
                torch.zeros(n_features, dtype=torch.float64, device=beta.device),
                torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device),
            )
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr = self._build_entry_ctx_torch(
                    time, event, entry, beta.device
                )
                X_entry = X.index_select(0, entry_order).contiguous()
                X_rem = X.index_select(0, rem_order).contiguous()
                grad = torch.sum(X.index_select(0, event_idx), dim=0)
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                X_entry = entry_ctx[4] if len(entry_ctx) > 4 else X.index_select(0, entry_order)
                X_rem = entry_ctx[5] if len(entry_ctx) > 5 else X
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else torch.where(event_mask)[0]
                grad = entry_ctx[7] if len(entry_ctx) > 7 else torch.sum(X[event_mask], dim=0)
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            hess = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
            exp_entry = exp_eta.index_select(0, entry_order)
            exp_rem = exp_eta
            wx_entry = X_entry * exp_entry.unsqueeze(1)
            wx_rem = X_rem * exp_rem.unsqueeze(1)
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            s0_add_pref = torch.cumsum(exp_entry, dim=0)
            s0_rem_pref = torch.cumsum(exp_rem, dim=0)
            s1_add_pref = torch.cumsum(wx_entry, dim=0)
            s1_rem_pref = torch.cumsum(wx_rem, dim=0)
            s0_add = torch.zeros(n_groups, dtype=torch.float64, device=beta.device)
            s0_rem = torch.zeros(n_groups, dtype=torch.float64, device=beta.device)
            s1_add = torch.zeros((n_groups, n_features), dtype=torch.float64, device=beta.device)
            s1_rem = torch.zeros((n_groups, n_features), dtype=torch.float64, device=beta.device)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = torch.as_tensor(add_end_np[mask_add] - 1, dtype=torch.long, device=beta.device)
                mask_add_t = torch.as_tensor(mask_add, dtype=torch.bool, device=beta.device)
                s0_add[mask_add_t] = s0_add_pref.index_select(0, idx_add)
                s1_add[mask_add_t] = s1_add_pref.index_select(0, idx_add)
            if np.any(mask_rem):
                idx_rem = torch.as_tensor(rem_end_np[mask_rem] - 1, dtype=torch.long, device=beta.device)
                mask_rem_t = torch.as_tensor(mask_rem, dtype=torch.bool, device=beta.device)
                s0_rem[mask_rem_t] = s0_rem_pref.index_select(0, idx_rem)
                s1_rem[mask_rem_t] = s1_rem_pref.index_select(0, idx_rem)
            s0_vec = s0_add - s0_rem
            s1_vec = s1_add - s1_rem
            d_vec = torch.as_tensor(d_counts, dtype=torch.float64, device=beta.device)
            s0_safe_vec = torch.clamp(s0_vec, min=1e-15)
            use_efron_entry = (self.ties == "efron")
            ex_vec = s1_vec / s0_safe_vec.unsqueeze(1)
            if not use_efron_entry:
                grad = grad - torch.sum(d_vec.unsqueeze(1) * ex_vec, dim=0)
            if use_efron_entry:
                if fail_ptr is None:
                    fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                    fail_ptr[0] = 0
                    fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
                event_exp = exp_eta.index_select(0, event_idx)
                X_fail = X.index_select(0, event_idx)
            add_ptr = 0
            rem_ptr = 0
            s2 = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
            s2_block_size = int(os.environ.get("STATGPU_ENTRY_S2_BLOCK_SIZE", "8192"))
            if s2_block_size <= 0:
                s2_block_size = 10**18
            s2_fn = self._get_entry_s2_torch_fn()
            for g in range(n_groups):
                add_end = int(add_end_np[g])
                if add_end > add_ptr:
                    x_add = X_entry[add_ptr:add_end]
                    w_add = exp_entry[add_ptr:add_end]
                    n_add = int(add_end - add_ptr)
                    if n_add <= s2_block_size:
                        s2 = s2 + s2_fn(x_add, w_add)
                    else:
                        s2 = self._s2_weighted_update_torch_blocked(
                            s2, x_add, w_add, s2_block_size, sign=1.0
                        )
                    add_ptr = add_end

                rem_end = int(rem_end_np[g])
                if rem_end > rem_ptr:
                    x_rem = X_rem[rem_ptr:rem_end]
                    w_rem = exp_eta[rem_ptr:rem_end]
                    n_rem = int(rem_end - rem_ptr)
                    if n_rem <= s2_block_size:
                        s2 = s2 - s2_fn(x_rem, w_rem)
                    else:
                        s2 = self._s2_weighted_update_torch_blocked(
                            s2, x_rem, w_rem, s2_block_size, sign=-1.0
                        )
                    rem_ptr = rem_end

                d_t_f = float(d_counts[g])
                if d_t_f <= 0:
                    continue
                if use_efron_entry:
                    st = int(fail_ptr[g])
                    ed = int(fail_ptr[g + 1])
                    ef = event_exp[st:ed]
                    xf = X_fail[st:ed]
                    ef_sum = torch.sum(ef)
                    ef_x_sum = torch.sum(xf * ef.unsqueeze(1), dim=0)
                    ef_x2_sum = xf.transpose(0, 1) @ (xf * ef.unsqueeze(1))
                    s0_g = torch.clamp(s0_vec[g], min=1e-15)
                    s1_g = s1_vec[g]
                    d_i = int(d_t_f)
                    for k in range(d_i):
                        frac = float(k) / float(d_i)
                        denom = torch.clamp(s0_g - frac * ef_sum, min=1e-15)
                        s1_k = s1_g - frac * ef_x_sum
                        s2_k = s2 - frac * ef_x2_sum
                        ex_k = s1_k / denom
                        grad = grad - ex_k
                        hess = hess - (s2_k / denom)
                        hess = hess + torch.outer(ex_k, ex_k)
                else:
                    s0_safe = s0_safe_vec[g]
                    hess = hess - (d_t_f / s0_safe) * s2
            if not use_efron_entry:
                hess = hess + ex_vec.transpose(0, 1) @ (d_vec.unsqueeze(1) * ex_vec)
            if return_aux:
                return grad, hess, (eta, exp_eta, risk_sum)
            return grad, hess

        # Get event data
        event_times = time[event_mask]

        # Unique failure times with inverse mapping
        uft, unique_inv = torch.unique(event_times, sorted=True, return_inverse=True)
        n_uft = len(uft)
        counts = torch.bincount(unique_inv).to(torch.float64)

        # The optimizer contract supplies a stable time-ascending array, so the
        # left boundary is the complete tied risk set for each failure time.
        first_idx = torch.searchsorted(time, uft, side="left")

        # Risk values at unique times
        risk_at_uft = risk_sum[first_idx]
        risk_X_at_uft = risk_X_sum[first_idx]
        E_X_at_uft = risk_X_at_uft / risk_at_uft[:, None]

        # Sum X and exp(eta) for events at each unique time
        event_indices = event_mask.nonzero(as_tuple=True)[0]
        sum_X_per_uft = torch.zeros((n_uft, n_features), dtype=torch.float64, device=beta.device)
        sum_X_per_uft.index_add_(0, unique_inv, X[event_indices])

        # ============= GRADIENT =============
        if self.ties == "efron":
            # Efron closed-form: (d+1)/2 * E[X|R]
            efron_weight = (counts + 1) / 2.0
            grad = torch.sum(sum_X_per_uft - efron_weight[:, None] * E_X_at_uft, dim=0)
        else:
            # Breslow: d * E[X|R]
            grad = torch.sum(sum_X_per_uft - counts[:, None] * E_X_at_uft, dim=0)

        # Hessian
        # Use incremental risk-set second moments to avoid materializing
        # a (n_samples, n_features, n_features) tensor on GPU (can OOM at 50k x 100).
        X_exp = X * exp_eta[:, None]
        risk_X2 = X_exp.transpose(0, 1) @ X

        # Weight by counts (Breslow) or Efron-adjusted weights
        if self.ties == "efron":
            weights = efron_weight
        else:
            weights = counts

        # ---- Triton Breslow path ----
        if (
            self.ties != "efron"
            and os.environ.get("STATGPU_BRESLOW_TRITON", "0").strip().lower()
            in ("1", "true", "yes", "on")
            and beta.is_cuda
        ):
            from statgpu.survival._cox_efron_triton import compute_breslow_grad_hess_triton
            triton_out = compute_breslow_grad_hess_triton(X, beta, time, event)
            if triton_out is not None:
                grad, hess = triton_out
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess

        # ---- Vectorized Hessian via cumsum of outer products ----
        # hess = -sum_g (counts[g]/s0[g]) * risk_X2[g] + sum_g counts[g] * outer(E_X[g], E_X[g])
        # where risk_X2[g] = total - prefix[g], prefix = cumsum of outer products.
        total = risk_X2  # X_exp.T @ X
        # Stream risk-set second moments. This keeps peak memory at O(p^2)
        # instead of materializing an O(n*p^2) prefix tensor.
        hess = self._compute_hessian_grouped_streaming_torch(
            X, X_exp, total, risk_at_uft, risk_X_sum,
            first_idx, weights,
        )
        if return_aux:
            return grad, hess, (eta, exp_eta, risk_sum)
        return grad, hess

    def _compute_hessian_grouped_streaming_torch(
        self, X, X_exp, total, risk_at, risk_X_sum, first_idx, weights
    ):
        '''Grouped Torch Hessian with O(p^2) working memory.'''
        import torch

        risk_x2 = total.clone()
        hess = torch.zeros_like(total)
        previous = 0
        first_idx_host = first_idx.detach().cpu().tolist()
        self._last_torch_hessian_peak_shape_ = tuple(total.shape)
        for group, index_value in enumerate(first_idx_host):
            index = int(index_value)
            if index > previous:
                block = slice(previous, index)
                risk_x2 = risk_x2 - X_exp[block].transpose(0, 1) @ X[block]
                previous = index
            denominator = torch.clamp(risk_at[group], min=1e-300)
            expected_x = risk_X_sum[index] / denominator
            centered = risk_x2 / denominator - torch.outer(expected_x, expected_x)
            hess = hess - weights[group] * centered
        return hess

    def _s2_weighted_update_torch_blocked(self, s2, x, w, block_size, sign=1.0):
        """Blocked update for large slices: s2 += sign * X^T (X * w)."""
        s2_fn = self._get_entry_s2_torch_fn()

        n = int(x.shape[0])
        if n <= 0:
            return s2
        for st in range(0, n, block_size):
            ed = min(st + block_size, n)
            xb = x[st:ed]
            wb = w[st:ed]
            s2 = s2 + sign * s2_fn(xb, wb)
        return s2

    def _get_entry_s2_torch_fn(self):
        """Build/cache torch or torch.compile function for weighted X^T X."""
        fn = getattr(self, "_entry_s2_torch_fn", None)
        if fn is not None:
            return fn
        import torch

        def _s2_core(x, w):
            return x.transpose(0, 1) @ (x * w.unsqueeze(1))

        use_compile = (
            os.environ.get("STATGPU_ENTRY_S2_COMPILE_TORCH", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if use_compile and hasattr(torch, "compile"):
            mode = os.environ.get("STATGPU_ENTRY_S2_COMPILE_MODE", "default")
            try:
                fn = torch.compile(_s2_core, dynamic=True, fullgraph=False, mode=mode)
            except Exception:
                fn = _s2_core
        else:
            fn = _s2_core
        self._entry_s2_torch_fn = fn
        return fn

    def _compute_cindex_torch(self, X, time, event, beta):
        """Compute concordance index (C-index) on Torch."""
        import torch

        # Linear predictor (risk score)
        risk_score = X @ beta

        n = len(time)
        event_mask = (event == 1)

        if torch.sum(event_mask) == 0:
            return torch.tensor(0.5, dtype=torch.float64, device=beta.device)

        # Use chunked vectorized approach for memory efficiency
        event_idx = torch.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return torch.tensor(float("nan"), dtype=torch.float64, device=beta.device)

        concordant = torch.tensor(0, dtype=torch.int64, device=beta.device)
        permissible = torch.tensor(0, dtype=torch.int64, device=beta.device)
        tied_risk = torch.tensor(0, dtype=torch.int64, device=beta.device)

        # Chunk size for memory efficiency (~128 MB per batch matrix)
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk][:, None]
            risk_i = risk_score[idx_chunk][:, None]
            time_j = time[None, :]
            risk_j = risk_score[None, :]
            event_j = event[None, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons
            chunk_indices = torch.arange(end - start, device=beta.device)
            perm[chunk_indices, idx_chunk] = False

            concordant += torch.sum(perm & (risk_i > risk_j))
            tied_risk += torch.sum(perm & (risk_i == risk_j))
            permissible += torch.sum(perm)

        if permissible > 0:
            return (concordant.to(torch.float64) + 0.5 * tied_risk.to(torch.float64)) / permissible.to(torch.float64)
        else:
            return torch.tensor(float("nan"), dtype=torch.float64, device=beta.device)

    @staticmethod
    def _observed_information(hess):
        """Return a symmetric, positive-oriented observed information matrix.

        Legacy Efron kernels expose observed information directly, whereas
        Breslow and native GPU kernels expose the Hessian of the log partial
        likelihood.  Normalize that historical sign difference at the
        inference boundary by choosing the orientation with greater positive
        spectral mass.
        """
        hess_arr = np.asarray(hess, dtype=np.float64)
        sym = 0.5 * (hess_arr + hess_arr.T)
        eigvals = np.linalg.eigvalsh(sym)
        positive_mass = float(np.sum(np.clip(eigvals, 0.0, None)))
        negative_mass = float(np.sum(np.clip(-eigvals, 0.0, None)))
        return sym if positive_mass >= negative_mass else -sym

    @staticmethod
    def _observed_information_cupy(hess):
        """CuPy-native counterpart of :meth:`_observed_information`."""
        import cupy as cp

        sym = 0.5 * (hess + hess.T)
        eigvals = cp.linalg.eigvalsh(sym)
        positive_mass = cp.sum(cp.maximum(eigvals, 0.0))
        negative_mass = cp.sum(cp.maximum(-eigvals, 0.0))
        return sym if bool((positive_mass >= negative_mass).item()) else -sym

    @staticmethod
    def _observed_information_torch(hess):
        """Torch-native counterpart of :meth:`_observed_information`."""
        import torch

        sym = 0.5 * (hess + hess.transpose(0, 1))
        eigvals = torch.linalg.eigvalsh(sym)
        positive_mass = torch.sum(torch.clamp(eigvals, min=0.0))
        negative_mass = torch.sum(torch.clamp(-eigvals, min=0.0))
        return sym if bool((positive_mass >= negative_mass).item()) else -sym

    def _compute_inference_cpu(self, X, time, event, cluster=None):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]

        # Keep inference self-contained (no nested external model fitting),
        # so runtime reflects this implementation directly.

        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(
            self.coef_, X, time, event, getattr(self, "_efron_pre", None), entry=getattr(self, "_entry", None)
        )

        # Bread matrix from observed information.
        information = self._observed_information(hess)
        if self.penalty > 0:
            information = information + 2.0 * self.penalty * np.eye(
                n_features, dtype=np.float64
            )
        bread = _invert_information_numpy(information)

        if self.cov_type == "nonrobust":
            self._var_matrix = bread
            self.inference_method_ = (
                'penalized_observed_information'
                if self.penalty > 0 else 'observed_information'
            )
            self.inference_backend_ = 'numpy'
            self.inference_approximate_ = False
        elif self.cov_type == "cluster":
            if cluster is None:
                raise ValueError("cov_type='cluster' requires cluster ids in fit(..., cluster=...)")
            cluster = np.asarray(cluster)
            score_resid = self._compute_robust_score_residuals(X, time, event)
            uniq = np.unique(cluster)
            meat = np.zeros((n_features, n_features), dtype=np.float64)
            for g in uniq:
                u_g = np.sum(score_resid[cluster == g], axis=0)
                meat += np.outer(u_g, u_g)
            self._var_matrix = bread @ meat @ bread
        else:
            score_resid = self._compute_robust_score_residuals(X, time, event)
            meat = score_resid.T @ score_resid
            self._var_matrix = bread @ meat @ bread
            if self.cov_type == "hc1":
                n = X.shape[0]
                k = X.shape[1]
                if n > k:
                    self._var_matrix = self._var_matrix * (n / (n - k))

        # Standard errors
        self._bse = np.sqrt(np.maximum(np.diag(self._var_matrix), 0.0))

        # z-values (add epsilon to avoid division by zero)
        self._zvalues = self.coef_ / (self._bse + 1e-30)

        # p-values (two-sided)
        self._pvalues = 2 * (1 - norm.cdf(np.abs(self._zvalues)))

        # 95% confidence intervals
        alpha = 0.05
        z_crit = norm.ppf(1 - alpha / 2)
        self._conf_int = np.column_stack([
            self.coef_ - z_crit * self._bse,
            self.coef_ + z_crit * self._bse
        ])

        # Wald test (global test that all coefficients are 0)
        try:
            var_inv = np.linalg.solve(self._var_matrix, np.eye(n_features))
            self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
        except np.linalg.LinAlgError:
            self._wald_test_stat = np.nan
        self._wald_test_pvalue = float(chi2.sf(self._wald_test_stat, df=n_features))

        # Likelihood ratio test
        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = float(chi2.sf(self._lr_test_stat, df=n_features))

        # Score test (Rao's test) - computed at beta = 0.  Compute the
        # gradient and Hessian in one call because Efron paths can be expensive.
        ep = getattr(self, "_efron_pre", None)
        try:
            grad_0, hess_0 = self._compute_gradient_hessian(
                np.zeros(n_features),
                X,
                time,
                event,
                ep,
                entry=getattr(self, "_entry", None),
            )
            info_0 = self._observed_information(hess_0)
            info_0_inv = np.linalg.solve(info_0, np.eye(n_features))
            self._score_test_stat = float(grad_0 @ info_0_inv @ grad_0)
            self.score_test_available_ = True
            self.score_test_failure_reason_ = None
        except np.linalg.LinAlgError as exc:
            self._score_test_stat = np.nan
            self.score_test_available_ = False
            self.score_test_failure_reason_ = (
                f"numpy null information is singular: {exc}"
            )
        self._score_test_pvalue = float(chi2.sf(self._score_test_stat, df=n_features))

    def _score_residuals_via_statsmodels_if_available(self, X, time, event):
        """Compatibility helper for callers that explicitly probe PHReg."""
        try:
            import statsmodels.duration.api as smd
            model = smd.PHReg(time, X, status=event, ties=self.ties)
            residuals = model.score_residuals(self.coef_)
            return np.nan_to_num(
                np.asarray(residuals, dtype=np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        except Exception:
            return None

    def _compute_robust_score_residuals(self, X, time, event):
        """Return exact or explicitly opted-in approximate score residuals."""
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=np.int64)
        if self.inference_mode == "approx":
            eta = X @ self.coef_
            exp_eta = np.exp(eta)
            risk_sum = np.cumsum(exp_eta[::-1])[::-1] + 1e-30
            risk_x = np.cumsum((X * exp_eta[:, None])[::-1], axis=0)[::-1]
            residuals = np.zeros_like(X)
            mask = event == 1
            residuals[mask] = X[mask] - risk_x[mask] / risk_sum[mask, None]
            self.inference_method_ = "event_row_score_sandwich"
            self.inference_backend_ = "numpy"
            self.inference_approximate_ = True
            self.inference_fallback_reason_ = "inference_mode=approx"
            return residuals
        from statgpu.survival._risk_sets import cox_counting_process_objective
        result = cox_counting_process_objective(
            self.coef_, X, time, event,
            start=getattr(self, "_entry", None),
            strata=getattr(self, "_strata", None),
            ties=self.ties,
            score_residuals=True,
        )
        self.inference_method_ = "counting_process_score_sandwich"
        self.inference_backend_ = "numpy"
        self.inference_approximate_ = False
        self.inference_fallback_reason_ = None
        return np.asarray(result["score_residuals"], dtype=np.float64)

    def _compute_robust_score_residuals_gpu(self, X, time, event):
        """Return exact or explicitly opted-in CuPy score residuals."""
        import cupy as cp
        if self.inference_mode == "approx":
            eta = X @ cp.asarray(self.coef_, dtype=cp.float64)
            exp_eta = cp.exp(eta)
            risk_sum = cp.cumsum(exp_eta[::-1])[::-1] + 1e-30
            risk_x = cp.cumsum((X * exp_eta[:, None])[::-1], axis=0)[::-1]
            residuals = cp.zeros_like(X)
            mask = event == 1
            residuals[mask] = X[mask] - risk_x[mask] / risk_sum[mask, None]
            self.inference_method_ = "event_row_score_sandwich"
            self.inference_backend_ = "cupy"
            self.inference_approximate_ = True
            self.inference_fallback_reason_ = "inference_mode=approx"
            return residuals
        from statgpu.survival._risk_sets import cox_counting_process_objective
        result = cox_counting_process_objective(
            cp.asarray(self.coef_, dtype=cp.float64), X, time, event,
            ties=self.ties,
            score_residuals=True,
        )
        self.inference_method_ = "counting_process_score_sandwich"
        self.inference_backend_ = "cupy"
        self.inference_approximate_ = False
        self.inference_fallback_reason_ = None
        self.full_host_transfer_performed_ = False
        return result["score_residuals"]

    def _compute_baseline_hazard(self, X, time, event, entry=None):
        """Compute Breslow estimator of baseline hazard and survival function."""
        # Get unique event times
        event_mask = event == 1
        if not np.any(event_mask):
            self._unique_times = np.array([])
            self._baseline_hazard = np.array([])
            self._baseline_cumulative_hazard = np.array([])
            return

        unique_times, event_counts = np.unique(time[event_mask], return_counts=True)
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ self.coef_
        exp_eta = np.exp(eta)

        if entry is None:
            suffix_risk = np.cumsum(exp_eta[::-1])[::-1]
            first_idx = np.searchsorted(time, unique_times, side='left')
            risk_at = suffix_risk[first_idx]
        else:
            entry_order = np.argsort(entry, kind='stable')
            entry_sorted = np.asarray(entry)[entry_order]
            entry_prefix = np.cumsum(exp_eta[entry_order])
            time_prefix = np.cumsum(exp_eta)
            add_end = np.searchsorted(entry_sorted, unique_times, side='left')
            remove_end = np.searchsorted(time, unique_times, side='left')
            add_sum = np.where(add_end > 0, entry_prefix[np.maximum(add_end - 1, 0)], 0.0)
            remove_sum = np.where(
                remove_end > 0, time_prefix[np.maximum(remove_end - 1, 0)], 0.0
            )
            risk_at = add_sum - remove_sum
        self._baseline_hazard = event_counts / np.maximum(risk_at, 1e-300)
        self._baseline_cumulative_hazard = np.cumsum(self._baseline_hazard)

    def _compute_baseline_hazard_gpu(self, X, time, event, beta, entry=None):
        """Compute Breslow estimator of baseline hazard and survival function on GPU."""
        import cupy as cp

        event_mask = event == 1
        if not cp.any(event_mask):
            self._unique_times = np.array([], dtype=np.float64)
            self._baseline_hazard = np.array([], dtype=np.float64)
            self._baseline_cumulative_hazard = np.array([], dtype=np.float64)
            return

        unique_times, event_counts = cp.unique(time[event_mask], return_counts=True)
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = cp.exp(eta)

        if entry is None:
            suffix_risk = cp.cumsum(exp_eta[::-1])[::-1]
            first_idx = cp.searchsorted(time, unique_times, side='left')
            risk_at = suffix_risk[first_idx]
        else:
            entry_order = cp.argsort(entry)
            entry_sorted = entry[entry_order]
            entry_prefix = cp.cumsum(exp_eta[entry_order])
            time_prefix = cp.cumsum(exp_eta)
            add_end = cp.searchsorted(entry_sorted, unique_times, side='left')
            remove_end = cp.searchsorted(time, unique_times, side='left')
            add_sum = cp.where(
                add_end > 0, entry_prefix[cp.maximum(add_end - 1, 0)], 0.0
            )
            remove_sum = cp.where(
                remove_end > 0, time_prefix[cp.maximum(remove_end - 1, 0)], 0.0
            )
            risk_at = add_sum - remove_sum
        hazard = event_counts.astype(cp.float64) / cp.maximum(risk_at, 1e-300)
        cumulative_hazard = cp.cumsum(hazard)

        self._unique_times = cp.asnumpy(unique_times)
        self._baseline_hazard = cp.asnumpy(hazard)
        self._baseline_cumulative_hazard = cp.asnumpy(cumulative_hazard)

    def _compute_baseline_hazard_torch(self, X, time, event, beta, entry=None):
        """Compute Breslow estimator of baseline hazard and survival function on Torch."""
        import torch

        event_mask = event == 1
        if not torch.any(event_mask):
            self._unique_times = np.array([], dtype=np.float64)
            self._baseline_hazard = np.array([], dtype=np.float64)
            self._baseline_cumulative_hazard = np.array([], dtype=np.float64)
            return

        unique_times, event_counts = torch.unique(
            time[event_mask], sorted=True, return_counts=True
        )
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = torch.exp(eta)

        if entry is None:
            suffix_risk = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)
            first_idx = torch.searchsorted(time, unique_times, side='left')
            risk_at = suffix_risk[first_idx]
        else:
            entry_order = torch.argsort(entry, stable=True)
            entry_sorted = entry[entry_order]
            entry_prefix = torch.cumsum(exp_eta[entry_order], dim=0)
            time_prefix = torch.cumsum(exp_eta, dim=0)
            add_end = torch.searchsorted(entry_sorted, unique_times, side='left')
            remove_end = torch.searchsorted(time, unique_times, side='left')
            add_sum = torch.where(
                add_end > 0,
                entry_prefix[torch.clamp(add_end - 1, min=0)],
                torch.zeros_like(unique_times),
            )
            remove_sum = torch.where(
                remove_end > 0,
                time_prefix[torch.clamp(remove_end - 1, min=0)],
                torch.zeros_like(unique_times),
            )
            risk_at = add_sum - remove_sum
        hazard = event_counts.to(torch.float64) / torch.clamp(
            risk_at, min=1e-300
        )
        cumulative_hazard = torch.cumsum(hazard, dim=0)

        self._unique_times = unique_times.detach().cpu().numpy()
        self._baseline_hazard = hazard.detach().cpu().numpy()
        self._baseline_cumulative_hazard = cumulative_hazard.detach().cpu().numpy()

    def _compute_cindex_gpu(self, X, time, event, beta):
        """Compute concordance index (C-index) on GPU using chunked vectorized approach."""
        import cupy as cp

        # Linear predictor (risk score) on GPU
        risk_score = X @ beta

        n = len(time)
        event_mask = (event == 1)

        if cp.sum(event_mask) == 0:
            return cp.array(0.5, dtype=cp.float64)

        # Use chunked vectorized approach for memory efficiency
        event_idx = cp.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return cp.array(float("nan"), dtype=cp.float64)

        concordant = cp.int64(0)
        permissible = cp.int64(0)
        tied_risk = cp.int64(0)

        # Chunk size for memory efficiency (~128 MB per batch matrix)
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk][:, None]
            risk_i = risk_score[idx_chunk][:, None]
            time_j = time[None, :]
            risk_j = risk_score[None, :]
            event_j = event[None, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons
            chunk_indices = cp.arange(end - start, dtype=cp.int64)
            perm[chunk_indices, idx_chunk] = False

            concordant += cp.sum(perm & (risk_i > risk_j))
            tied_risk += cp.sum(perm & (risk_i == risk_j))
            permissible += cp.sum(perm)

        if permissible > 0:
            return (concordant.astype(cp.float64) + 0.5 * tied_risk.astype(cp.float64)) / permissible.astype(cp.float64)
        else:
            return cp.array(float("nan"), dtype=cp.float64)

    def _compute_cindex(self):
        """
        Compute concordance index (C-index) using chunked vectorized NumPy.

        Replaces the O(n²) double Python loop with batched boolean matrix ops.
        Chunk size is chosen so each batch matrix stays within ~128 MB.
        """
        if self._X is None or self.coef_ is None:
            self._cindex = None
            return

        risk_score = self._X @ self.coef_
        time = self._time
        event = self._event
        n = len(time)

        event_idx = np.where(event == 1)[0]
        n_events = len(event_idx)

        if n_events == 0:
            self._cindex = np.nan
            return

        concordant = np.int64(0)
        permissible = np.int64(0)
        tied_risk   = np.int64(0)

        # Chunk so each (chunk × n) bool matrix is ≤ 128 MB.
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]          # (c,)

            time_i  = time[idx_chunk, np.newaxis]     # (c, 1)
            risk_i  = risk_score[idx_chunk, np.newaxis]
            time_j  = time[np.newaxis, :]             # (1, n)
            risk_j  = risk_score[np.newaxis, :]
            event_j = event[np.newaxis, :]

            # Permissible pairs: earlier time OR same time with j censored.
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons.
            perm[np.arange(end - start), idx_chunk] = False

            concordant  += int(np.sum(perm & (risk_i > risk_j)))
            tied_risk   += int(np.sum(perm & (risk_i == risk_j)))
            permissible += int(np.sum(perm))

        if permissible > 0:
            self._cindex = (concordant + 0.5 * tied_risk) / permissible
        else:
            self._cindex = np.nan



class _LegacyCoxReference(_LegacyCoxReferenceMixin):
    """Test-only composition adapter around a canonical Cox estimator.

    Historical numerical methods execute on this adapter while all fitted
    state remains owned by ``estimator``. Method overrides stay local to the
    adapter, keeping regression tests explicit without polluting the public
    estimator MRO.
    """

    def __init__(self, estimator):
        object.__setattr__(self, "_estimator", estimator)

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    def __setattr__(self, name, value):
        if name == "_estimator" or any(
            name in cls.__dict__ for cls in type(self).__mro__
        ):
            object.__setattr__(self, name, value)
            return
        setattr(self._estimator, name, value)

    def __delattr__(self, name):
        if name in self.__dict__:
            object.__delattr__(self, name)
            return
        delattr(self._estimator, name)


__all__ = [
    "_LegacyCoxReference",
    "_LegacyCoxReferenceMixin",
    "_estimate_breslow_tensor_bytes",
]
