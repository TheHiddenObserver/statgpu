"""
Cox Proportional Hazards regression with GPU acceleration.

Implements Cox PH model using Breslow and Efron approximations for ties with
Newton-Raphson optimization. Matches R's survival::coxph() API.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import os
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device

# Optional Cython import for faster Efron gradient/Hessian computation
try:
    from ._cox_efron_cy import efron_grad_hess as _efron_grad_hess_cython
    HAS_CYTHON_EFRON = True
except ImportError:
    HAS_CYTHON_EFRON = False
    _efron_grad_hess_cython = None


def _unpack_efron_pre6(efron_pre):
    """``(uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft)`` — supports legacy 5-tuple in tests only."""
    if len(efron_pre) == 6:
        return efron_pre
    if len(efron_pre) == 5:
        uft, uft_ix, re, rx, nuft = efron_pre
        return uft, uft_ix, re, rx, nuft, None
    raise ValueError(f"invalid efron_pre length {len(efron_pre)}")


class CoxPH(BaseEstimator):
    """
    Cox Proportional Hazards regression with GPU acceleration.
    
    Parameters
    ----------
    ties : str, default='breslow'
        Method for handling ties: 'breslow' or 'efron'.
    tol : float, default=1e-9
        Convergence tolerance for Newton-Raphson.
    max_iter : int, default=100
        Maximum number of iterations.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    compute_inference : bool, default=True
        If True, compute standard errors/tests/baseline hazard on CPU after fitting.
        Set to False to reduce CPU-GPU data transfers in CUDA mode.
    compute_cindex : bool, default=True
        If True, compute training-set C-index during fit. Disabling this can
        significantly reduce fit time, especially on CUDA/Torch for moderate n.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (log hazard ratios).
    hazard_ratios_ : ndarray of shape (n_features,)
        exp(coef) = hazard ratios.
    """
    
    def __init__(
        self,
        ties: str = 'breslow',
        tol: float = 1e-9,
        max_iter: int = 100,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        compute_cindex: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        penalty: float = 0.0,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.ties = ties.lower()
        self.tol = tol
        self.max_iter = max_iter
        self.compute_inference = compute_inference
        self.compute_cindex = bool(compute_cindex)
        self.cov_type = cov_type.lower()
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.penalty = float(penalty)

        if self.ties not in ('breslow', 'efron'):
            raise ValueError("ties must be 'breslow' or 'efron'")
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "cluster"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'")
        if self.penalty < 0:
            raise ValueError("penalty must be non-negative")
        
        # Fitted attributes
        self.coef_ = None
        self.hazard_ratios_ = None
        
        # Internal storage for inference
        self._time = None
        self._event = None
        self._X = None
        self._entry = None
        self._nobs = None
        self._nevents = None
        self._bse = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._log_likelihood = None
        self._log_likelihood_null = None
        self._iterations = 0
        self._converged = False
        self._var_matrix = None
        self._score_test_stat = None
        self._baseline_hazard = None
        self._baseline_cumulative_hazard = None
        self._unique_times = None
        self._cindex = None
        self._feature_names = None
        self._wald_test_stat = None
        self._wald_test_pvalue = None
        self._lr_test_stat = None
        self._lr_test_pvalue = None
        self._score_test_pvalue = None
        # Efron only: cached (uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft); depends only on sorted time/event.
        self._efron_pre = None
        # Efron optimization: True when all failure groups are singletons (no ties),
        # in which case Efron equals Breslow and we can use faster vectorized paths.
        self._efron_all_singletons = False
        # Efron only: cached CSR packed indices for GPU kernels.
        # (enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind, first_idx_uft, nuft)
        self._efron_pre_csr = None
        # Breslow only: cached (first_idx_uft, counts_uft) on CPU.
        self._breslow_pre = None
        # Breslow only: cached (first_idx_uft_gpu, counts_uft_gpu) on GPU.
        self._breslow_pre_gpu = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    @staticmethod
    def _extract_convergence_status(result):
        """Best-effort convergence extraction from statsmodels results."""
        conv_attr = getattr(result, "converged", None)
        if conv_attr is not None:
            return bool(conv_attr)

        mle_retvals = getattr(result, "mle_retvals", None)
        if isinstance(mle_retvals, dict):
            conv_attr = mle_retvals.get("converged")
            if conv_attr is not None:
                return bool(conv_attr)
        elif mle_retvals is not None:
            conv_attr = getattr(mle_retvals, "converged", None)
            if conv_attr is not None:
                return bool(conv_attr)
        return None
        
    def fit(self, X, time, event, entry=None, cluster=None):
        """
        Fit Cox Proportional Hazards model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like of shape (n_samples,)
            Time to event or censoring.
        event : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored).
        entry : array-like of shape (n_samples,), optional
            Entry time for delayed entry (left truncation).
        cluster : array-like of shape (n_samples,), optional
            Cluster ids for cluster-robust covariance when `cov_type='cluster'`.
        
        Returns
        -------
        self : CoxPH
            Fitted estimator.
        """
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            import cupy as cp
            
            X_gpu = cp.asarray(self._to_array(X), dtype=cp.float64)
            time_gpu = cp.asarray(self._to_array(time), dtype=cp.float64)
            event_gpu = cp.asarray(self._to_array(event), dtype=cp.int32)
            entry_gpu = None if entry is None else cp.asarray(self._to_array(entry), dtype=cp.float64)
            
            if X_gpu.ndim == 1:
                X_gpu = X_gpu.reshape(-1, 1)
            if entry_gpu is not None and entry_gpu.shape[0] != X_gpu.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")
            
            self._nobs = int(X_gpu.shape[0])
            self._nevents = int(cp.sum(event_gpu).item())
            self._feature_names = [f'x{i+1}' for i in range(int(X_gpu.shape[1]))]
            
            # Keep CPU copies only when CPU-side inference/baseline stats are requested.
            if self.compute_inference:
                self._X = cp.asnumpy(X_gpu)
                self._time = cp.asnumpy(time_gpu)
                self._event = cp.asnumpy(event_gpu)
                self._entry = None if entry_gpu is None else cp.asnumpy(entry_gpu)
            else:
                self._X = None
                self._time = None
                self._event = None
                self._entry = None
            
            cluster_gpu = None if cluster is None else cp.asarray(self._to_array(cluster), dtype=cp.int64)
            if entry_gpu is not None:
                self._fit_cpu_with_entry(
                    cp.asnumpy(X_gpu),
                    cp.asnumpy(time_gpu),
                    cp.asnumpy(event_gpu),
                    cp.asnumpy(entry_gpu),
                    None if cluster_gpu is None else cp.asnumpy(cluster_gpu),
                )
                self._cleanup_cuda_memory()
            else:
                self._fit_gpu(X_gpu, time_gpu, event_gpu, entry_gpu, cluster_gpu)
        elif device == Device.TORCH:
            import torch

            # Determine torch device (cuda if available, else cpu)
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

            X_torch = torch.tensor(self._to_array(X), dtype=torch.float64, device=torch_device)
            time_torch = torch.tensor(self._to_array(time), dtype=torch.float64, device=torch_device)
            event_torch = torch.tensor(self._to_array(event), dtype=torch.int32, device=torch_device)
            entry_torch = None if entry is None else torch.tensor(self._to_array(entry), dtype=torch.float64, device=torch_device)

            if X_torch.ndim == 1:
                X_torch = X_torch.reshape(-1, 1)
            if entry_torch is not None and entry_torch.shape[0] != X_torch.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")

            self._nobs = int(X_torch.shape[0])
            self._nevents = int(torch.sum(event_torch).item())
            self._feature_names = [f'x{i+1}' for i in range(int(X_torch.shape[1]))]

            # Keep CPU copies only when CPU-side inference/baseline stats are requested.
            if self.compute_inference:
                self._X = X_torch.cpu().numpy()
                self._time = time_torch.cpu().numpy()
                self._event = event_torch.cpu().numpy()
                self._entry = None if entry_torch is None else entry_torch.cpu().numpy()
            else:
                self._X = None
                self._time = None
                self._event = None
                self._entry = None

            cluster_torch = None if cluster is None else torch.tensor(self._to_array(cluster), dtype=torch.int64, device=torch_device)
            if entry_torch is not None:
                # Fall back to CPU with statsmodels for delayed entry
                self._fit_cpu_with_entry(
                    X_torch.cpu().numpy(),
                    time_torch.cpu().numpy(),
                    event_torch.cpu().numpy(),
                    entry_torch.cpu().numpy(),
                    None if cluster_torch is None else cluster_torch.cpu().numpy(),
                )
            else:
                self._fit_torch(X_torch, time_torch, event_torch, entry_torch, cluster_torch, torch_device)
        else:
            X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
            time_np = np.asarray(self._to_array(time, Device.CPU), dtype=np.float64)
            event_np = np.asarray(self._to_array(event, Device.CPU), dtype=np.int32)
            entry_np = None if entry is None else np.asarray(self._to_array(entry, Device.CPU), dtype=np.float64)
            
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            if entry_np is not None and entry_np.shape[0] != X_np.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")
            
            self._nobs = X_np.shape[0]
            self._nevents = np.sum(event_np)
            
            # Store original data (CPU mode is CPU-only)
            self._time = time_np.copy()
            self._event = event_np.copy()
            self._X = X_np.copy()
            self._entry = None if entry_np is None else entry_np.copy()
            self._feature_names = [f'x{i+1}' for i in range(X_np.shape[1])]
            
            cluster_np = None if cluster is None else np.asarray(self._to_array(cluster, Device.CPU))
            self._fit_cpu(X_np, time_np, event_np, entry_np, cluster_np)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, time, event, entry=None, cluster=None):
        """Fit using CPU (NumPy)."""
        if entry is not None:
            self._fit_cpu_with_entry(X, time, event, np.asarray(entry, dtype=np.float64), cluster)
            return
        n_samples, n_features = X.shape
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = np.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
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
        
        # Initialize coefficients
        beta = np.zeros(n_features, dtype=np.float64)

        # Compute null log-likelihood (beta = 0)
        self._log_likelihood_null = self._compute_log_likelihood(
            np.zeros(n_features), X_sorted, time_sorted, event_sorted, self._efron_pre
        )

        # Newton-Raphson optimization with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        # Preferred Newton direction for CPU path; updated adaptively.
        preferred_direction = -1.0

        for iteration in range(self.max_iter):
            # Compute gradient and Hessian
            grad, hess = self._compute_gradient_hessian(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                hess = hess - 2 * penalty * np.eye(n_features, dtype=np.float64)

            # Solve a Newton-like step on (-hess). In practice, different tie paths
            # may expose Hessian with different sign conventions, so we choose the
            # ascent direction adaptively below using objective evaluation.
            try:
                delta = np.linalg.solve(-hess, grad)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                delta = np.linalg.lstsq(-hess, grad, rcond=None)[0]

            # Line search with step halving
            # Compute log-likelihood at current point
            old_ll = self._compute_log_likelihood(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )
            if use_penalty:
                old_ll = old_ll - penalty * np.sum(beta ** 2)

            # Fast path: try preferred direction first, only test opposite
            # when the preferred full step does not improve.
            direction = preferred_direction
            new_beta = beta + direction * delta
            new_ll = self._compute_log_likelihood(
                new_beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )
            if use_penalty:
                new_ll = new_ll - penalty * np.sum(new_beta ** 2)

            if new_ll <= old_ll - 1e-8:
                # Probe the opposite direction only when needed.
                alt_direction = -direction
                alt_beta = beta + alt_direction * delta
                alt_ll = self._compute_log_likelihood(
                    alt_beta, X_sorted, time_sorted, event_sorted, self._efron_pre
                )
                if use_penalty:
                    alt_ll = alt_ll - penalty * np.sum(alt_beta ** 2)
                if alt_ll > new_ll:
                    direction = alt_direction
                    preferred_direction = alt_direction
                    new_beta = alt_beta
                    new_ll = alt_ll

                # Backtracking line search from step=0.5; step=1 was already evaluated.
                if new_ll <= old_ll - 1e-8:
                    step = 0.5
                    for _ in range(20):
                        trial_beta = beta + direction * step * delta
                        trial_ll = self._compute_log_likelihood(
                            trial_beta, X_sorted, time_sorted, event_sorted, self._efron_pre
                        )
                        if use_penalty:
                            trial_ll = trial_ll - penalty * np.sum(trial_beta ** 2)
                        if trial_ll > old_ll - 1e-8:
                            new_beta = trial_beta
                            new_ll = trial_ll
                            break
                        step *= 0.5
                else:
                    step = 1.0
            else:
                # Keep successful direction for the next iteration.
                preferred_direction = direction
                step = 1.0

            # Check convergence
            if np.linalg.norm(delta) * step < self.tol:
                self._converged = True
                beta = new_beta
                break

            beta = new_beta
        
        self._iterations = iteration + 1
        self.coef_ = beta
        self.hazard_ratios_ = np.exp(beta)
        
        # Compute final log-likelihood
        self._log_likelihood = self._compute_log_likelihood(
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre
        )
        
        # Compute optional inference statistics
        if self.compute_inference:
            self._compute_inference_cpu(X_sorted, time_sorted, event_sorted, cluster_sorted)
            self._compute_baseline_hazard(X_sorted, time_sorted, event_sorted)
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

        # Release large temporary GPU tensors early.
        try:
            del X_sorted
        except Exception:
            pass
        try:
            del time_sorted
        except Exception:
            pass
        try:
            del event_sorted
        except Exception:
            pass
        try:
            del grad
        except Exception:
            pass
        try:
            del hess
        except Exception:
            pass
        try:
            del delta
        except Exception:
            pass
        self._cleanup_cuda_memory()
        if self.compute_cindex:
            self._compute_cindex()
        else:
            self._cindex = None

    def _fit_cpu_with_entry(self, X, time, event, entry, cluster=None):
        """Fit using statsmodels PHReg when delayed entry is provided."""
        import statsmodels.duration.api as smd

        n_samples, n_features = X.shape
        model = smd.PHReg(time, X, status=event, entry=entry, ties=self.ties)
        res = model.fit(disp=0)

        self._iterations = int(getattr(res, "iterations", 0) or 0)
        conv_attr = self._extract_convergence_status(res)
        self._converged = bool(conv_attr) if conv_attr is not None else False
        self.coef_ = np.asarray(res.params, dtype=np.float64)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood = float(res.llf)

        try:
            null_model = smd.PHReg(time, np.zeros((n_samples, 1), dtype=np.float64), status=event, entry=entry, ties=self.ties)
            null_res = null_model.fit(disp=0)
            self._log_likelihood_null = float(null_res.llf)
        except Exception:
            self._log_likelihood_null = np.nan

        cov = np.asarray(res.cov_params(), dtype=np.float64)
        if cov.shape != (n_features, n_features):
            cov = np.full((n_features, n_features), np.nan, dtype=np.float64)
        self._var_matrix = cov
        self._bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
        self._zvalues = self.coef_ / (self._bse + 1e-30)
        self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._zvalues)))
        self._conf_int = np.asarray(res.conf_int(), dtype=np.float64)

        # Delayed-entry robust covariance override is intentionally skipped:
        # current internal robust score/hessian helpers do not account for entry.

        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
        try:
            var_inv = np.linalg.solve(self._var_matrix, np.eye(n_features))
            self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
        except np.linalg.LinAlgError:
            self._wald_test_stat = np.nan
        self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
        self._score_test_stat = np.nan
        self._score_test_pvalue = np.nan

        # Baseline hazard from PHReg output.
        try:
            base = res.baseline_cumulative_hazard[0]
            self._unique_times = np.asarray(base[0], dtype=np.float64)
            self._baseline_cumulative_hazard = np.asarray(base[1], dtype=np.float64)
            if self._baseline_cumulative_hazard.size > 0:
                self._baseline_hazard = np.diff(
                    np.concatenate([[0.0], self._baseline_cumulative_hazard])
                )
            else:
                self._baseline_hazard = np.array([], dtype=np.float64)
        except Exception:
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

        if self.compute_cindex:
            self._compute_cindex()
        else:
            self._cindex = None
    
    def _fit_gpu(self, X, time, event, entry=None, cluster=None):
        """Fit using GPU with full GPU computation."""
        import cupy as cp
        from ..inference._distributions_gpu import norm
        
        n_samples, n_features = X.shape

        # Optional fast bridge: for large Breslow CUDA cases, Torch backend is
        # significantly faster and numerically aligned in this project.
        use_torch_bridge = (
            self.ties == "breslow"
            and n_samples >= 30000
            and n_features >= 80
            and os.environ.get("STATGPU_CUDA_BRESLOW_TORCH_BRIDGE", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if use_torch_bridge:
            try:
                import torch
                if torch.cuda.is_available():
                    X_t = torch.tensor(self._to_array(X), dtype=torch.float64, device="cuda")
                    t_t = torch.tensor(self._to_array(time), dtype=torch.float64, device="cuda")
                    e_t = torch.tensor(self._to_array(event), dtype=torch.int32, device="cuda")
                    entry_t = None if entry is None else torch.tensor(
                        self._to_array(entry), dtype=torch.float64, device="cuda"
                    )
                    cluster_t = None if cluster is None else torch.tensor(
                        self._to_array(cluster), dtype=torch.int64, device="cuda"
                    )
                    self._fit_torch(X_t, t_t, e_t, entry_t, cluster_t, torch_device="cuda")
                    return
            except Exception:
                pass
        
        # Transfer to GPU once
        X = cp.asarray(X, dtype=cp.float64)
        time = cp.asarray(time, dtype=cp.float64)
        event = cp.asarray(event, dtype=cp.int32)
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = cp.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        cluster_sorted = None if cluster is None else cluster[order]
        
        # Precompute Efron tie structure once (depends only on time/event order).
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
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
            try:
                _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                n_events = int(cp.asnumpy(cp.sum(event_sorted)))
                avg_tie = float(n_events / max(1, int(nuft)))
            except Exception:
                avg_tie = 1.0
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
            n_events = int(cp.asnumpy(cp.sum(event_sorted)))
            avg_tie = float(n_events / max(1, int(len(counts_uft))))

        # Initialize coefficients on GPU
        beta = cp.zeros(n_features, dtype=cp.float64)

        # Compute null log-likelihood on GPU
        loglik_null_gpu = self._compute_log_likelihood_gpu(
            cp.zeros(n_features, dtype=cp.float64),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
        )

        # Newton-Raphson optimization on GPU with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0

        # Newton-Raphson optimization on GPU
        loglik_gpu = None
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on GPU
            grad, hess, aux_stats = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                hess = hess - 2 * penalty * cp.eye(n_features, dtype=cp.float64)

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_gpu(hess, grad, cp)

            # Check convergence on GPU
            if cp.linalg.norm(delta) < self.tol:
                self._converged = True
                # Reuse current iteration statistics to avoid an extra
                # Efron log-likelihood setup pass when converged.
                eta_cur, exp_eta_cur, risk_sum_cur = aux_stats
                loglik_gpu = self._compute_log_likelihood_gpu_from_stats(
                    eta_cur, exp_eta_cur, risk_sum_cur, time_sorted, event_sorted, efron_pre
                )
                break

            beta = beta - delta
        
        # Compute final log-likelihood on GPU unless already obtained on convergence.
        if loglik_gpu is None:
            loglik_gpu = self._compute_log_likelihood_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre
            )
        
        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = cp.asnumpy(beta)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(cp.asnumpy(loglik_null_gpu))
        self._log_likelihood = float(cp.asnumpy(loglik_gpu))
        if self.compute_cindex:
            cindex_gpu = self._compute_cindex_gpu(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cp.asnumpy(cindex_gpu))
        else:
            self._cindex = None
        
        # Inference:
        # - nonrobust: stay on GPU to avoid expensive host transfers/recompute
        # - hc0/hc1/cluster: use CPU inference path (current implementation)
        if self.compute_inference:
            if self.cov_type == "nonrobust":
                try:
                    info = -hess
                    var_gpu = cp.linalg.solve(info, cp.eye(info.shape[0], dtype=info.dtype))
                except Exception:
                    var_gpu = cp.linalg.pinv(-hess)
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_gpu)))
                z_crit = norm.ppf(0.975)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._var_matrix = np.diag(np.square(self._bse))
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Keep baseline hazard optional in CUDA fast path to reduce transfer overhead.
                self._baseline_hazard = None
                self._baseline_cumulative_hazard = None
                self._unique_times = None
            else:
                score_resid_gpu = self._compute_robust_score_residuals_gpu(X_sorted, time_sorted, event_sorted)
                try:
                    info = -hess
                    bread = cp.linalg.solve(info, cp.eye(info.shape[0], dtype=info.dtype))
                except Exception:
                    bread = cp.linalg.pinv(-hess)

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
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Compute baseline hazard on GPU
                self._compute_baseline_hazard_gpu(X_sorted, time_sorted, event_sorted, beta)
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

    def _fit_torch(self, X, time, event, entry=None, cluster=None, torch_device="cuda"):
        """Fit using Torch with full GPU computation."""
        import torch
        from ..inference._distributions_torch import norm

        n_samples, n_features = X.shape

        # Sort by time ascending so risk-set terms are suffix sums
        order = torch.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        cluster_sorted = None if cluster is None else cluster[order]

        # Precompute Efron tie structure once (depends only on time/event order)
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_torch = None
        if self.ties == "efron":
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
            # Reuse CUDA CSR packing for Torch-CUDA fused kernels when available.
            try:
                import cupy as cp
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
            try:
                _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                n_events = int(torch.sum(event_sorted).item())
                avg_tie = float(n_events / max(1, int(nuft)))
            except Exception:
                avg_tie = 1.0
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
            n_events = int(torch.sum(event_sorted).item())
            avg_tie = float(n_events / max(1, int(len(counts_uft))))

        # Initialize coefficients on Torch device
        beta = torch.zeros(n_features, dtype=torch.float64, device=torch_device)

        # Compute null log-likelihood on Torch
        loglik_null_torch = self._compute_log_likelihood_torch(
            torch.zeros(n_features, dtype=torch.float64, device=torch_device),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
        )

        # Newton-Raphson optimization on Torch with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0

        # Newton-Raphson optimization on Torch
        iteration = 0
        loglik_torch = None
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on Torch
            grad, hess, aux_stats = self._compute_gradient_hessian_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                hess = hess - 2 * penalty * torch.eye(n_features, dtype=torch.float64, device=torch_device)

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_torch(hess, grad)

            # Check convergence
            if torch.linalg.norm(delta) < self.tol:
                self._converged = True
                eta_cur, exp_eta_cur, risk_sum_cur = aux_stats
                loglik_torch = self._compute_log_likelihood_torch_from_stats(
                    eta_cur, exp_eta_cur, risk_sum_cur, time_sorted, event_sorted, efron_pre
                )
                break

            beta = beta - delta

        # Compute final log-likelihood on Torch unless already obtained.
        if loglik_torch is None:
            loglik_torch = self._compute_log_likelihood_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre
            )

        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = beta.cpu().numpy()
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(loglik_null_torch.item())
        self._log_likelihood = float(loglik_torch.item())
        if self.compute_cindex:
            cindex_torch = self._compute_cindex_torch(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cindex_torch.item())
        else:
            self._cindex = None

        # Inference: nonrobust on Torch, other types fall back to CPU
        if self.compute_inference:
            if self.cov_type == "nonrobust":
                try:
                    info = -hess
                    var_torch = torch.linalg.solve(info, torch.eye(info.shape[0], dtype=info.dtype, device=torch_device))
                except Exception:
                    var_torch = torch.linalg.pinv(-hess)
                bse_torch = torch.sqrt(torch.maximum(torch.diag(var_torch), torch.tensor(0.0, dtype=torch.float64, device=torch_device)))
                z_torch = beta / (bse_torch + 1e-30)
                p_torch = torch.minimum(torch.tensor(1.0, device=torch_device), 2.0 * norm.sf(torch.abs(z_torch)))
                z_crit = norm.ppf(0.975)
                ci_torch = torch.stack([beta - z_crit * bse_torch, beta + z_crit * bse_torch], dim=1)

                self._bse = bse_torch.cpu().numpy()
                self._zvalues = z_torch.cpu().numpy()
                self._pvalues = p_torch.cpu().numpy()
                self._conf_int = ci_torch.cpu().numpy()
                self._var_matrix = np.diag(np.square(self._bse))
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Compute baseline hazard on Torch
                self._compute_baseline_hazard_torch(X_sorted, time_sorted, event_sorted, beta)
            else:
                # For hc0/hc1/cluster, use CPU inference path
                self._compute_inference_cpu(X_sorted.cpu().numpy(), time_sorted.cpu().numpy(), event_sorted.cpu().numpy(),
                                           cluster_sorted.cpu().numpy() if cluster_sorted is not None else None)
                self._baseline_hazard = None
                self._baseline_cumulative_hazard = None
                self._unique_times = None
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

    def _compute_log_likelihood(self, beta, X, time, event, efron_pre=None):
        """Compute log partial likelihood (Breslow/Efron tie handling)."""
        eta = X @ beta
        # Note: We do NOT center eta here. While centering prevents exp overflow,
        # it introduces a beta-dependent shift that complicates numeric gradient verification.
        # In practice, exp(eta) overflow is rare when beta is near convergence.
        exp_eta = np.exp(eta)

        # Risk set suffix sums:
        #   risk_sum[i] = sum_{j: time[j] >= time[i]} exp_eta[j]
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        event_mask = event == 1
        if not np.any(event_mask):
            return 0.0

        if self.ties == "breslow":
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
            return float(np.sum(eta[event_mask]) - np.sum(counts * np.log(risk_at)))

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
            k_matrix = np.arange(max_d, dtype=np.float64) / np.arange(1, max_d + 1, dtype=np.float64)[:, np.newaxis]
            # This is complex; fall back to loop for correctness
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
    
    def _solve_newton_delta_gpu(self, hess, grad, cp):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        p = int(hess.shape[0])
        try:
            H = -hess
            eps = 1e-11 * (cp.max(cp.abs(cp.diag(H))) + 1.0)
            H = H + eps * cp.eye(p, dtype=cp.float64)
            return -cp.linalg.solve(H, grad)
        except Exception:
            try:
                return cp.linalg.solve(hess, grad)
            except Exception:
                return cp.linalg.lstsq(hess, grad, rcond=None)[0].flatten()

    def _compute_log_likelihood_gpu(self, beta, X, time, event, efron_pre=None):
        """Compute log partial likelihood on GPU."""
        import cupy as cp

        eta = X @ beta
        exp_eta = cp.exp(eta)
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        return self._compute_log_likelihood_gpu_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre
        )

    def _compute_log_likelihood_gpu_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None
    ):
        """Compute log partial likelihood on GPU with precomputed Efron stats."""
        import cupy as cp

        ll = cp.array(0.0, dtype=cp.float64)
        event_mask = event == 1

        if not cp.any(event_mask):
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
    
    def _compute_gradient_hessian(self, beta, X, time, event, efron_pre=None):
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
        exp_eta = np.exp(eta)

        # Risk sets: cumulative sum of exp(eta) for all at risk
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        # Weighted risk sets for gradient
        X_exp_eta = X * exp_eta[:, np.newaxis]
        risk_X_sum = np.cumsum(X_exp_eta[::-1], axis=0)[::-1]

        if self.ties == 'breslow':
            event_mask = event == 1
            grad = np.zeros(n_features, dtype=np.float64)
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
        if p <= 24 and n_groups <= 512:
            return self._compute_hessian_breslow_tensor_grouped(
                X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
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
        """CuPy grouped Breslow Hessian with incremental risk-set second moments."""
        import cupy as cp

        X_exp = X * exp_eta[:, cp.newaxis]
        risk_X2 = X_exp.T @ X

        p = int(X.shape[1])
        hess = cp.zeros((p, p), dtype=cp.float64)
        prev_idx = 0
        n_groups = int(first_idx.size)
        for g in range(n_groups):
            idx = int(first_idx[g].item())
            if idx > prev_idx:
                blk = slice(prev_idx, idx)
                risk_X2 = risk_X2 - (X_exp[blk].T @ X[blk])
                prev_idx = idx

            rs = float(risk_sum[idx].item())
            if rs <= 0.0:
                continue
            ex = risk_X_sum[idx] / rs
            exx = risk_X2 / rs
            hess = hess - counts[g] * (exx - cp.outer(ex, ex))

        return hess

    def _compute_hessian_breslow_fused_cupy(self, X, first_idx, counts, exp_eta):
        """Try fused RawKernel Hessian for Breslow; return None on failure."""
        import cupy as cp
        try:
            from ._cox_efron_cuda import compute_breslow_hess_raw
            return compute_breslow_hess_raw(
                X,
                first_idx,
                counts,
                cupy_module=cp,
                exp_eta=exp_eta,
            )
        except Exception:
            return None

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
        n = time.shape[0]
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
        Efron gradient and Hessian using direct computation (O(n*d) per unique failure time).

        The gradient is: sum_events(X_i) - sum_events(E[X|R(t_i)])
        where E[X|R(t_i)] uses the Efron approximation.

        Note: We do NOT center eta for consistency with _compute_log_likelihood.
        The Efron ratio formula is scale-invariant, so centering is not needed for
        numerical stability in typical use cases.
        """
        n_samples, n_features = X.shape
        linpred = X @ beta
        # No centering - matches _compute_log_likelihood
        e_linpred = np.exp(linpred)

        event_mask = event == 1
        event_idx = np.where(event_mask)[0]

        if len(event_idx) == 0:
            return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)

        # Get unique failure times and their counts
        event_times = time[event_mask]
        uft, counts = np.unique(event_times, return_counts=True)
        nuft = len(uft)

        grad = np.zeros(n_features, dtype=np.float64)
        hess_inner = np.zeros((n_features, n_features), dtype=np.float64)

        # Pre-compute suffix sums for risk sets
        # risk_sum[i] = sum of exp(lp) for all j with time[j] >= time[i]
        order = np.argsort(time)
        time_sorted = time[order]
        e_lp_sorted = e_linpred[order]
        X_sorted = X[order]

        # Suffix sum: risk_sum_sorted[i] = sum of e_lp_sorted[j] for j >= i
        risk_sum_sorted = np.cumsum(e_lp_sorted[::-1])[::-1]
        # risk_X_sum_sorted[i] = sum of e_lp_sorted[j] * X_sorted[j] for j >= i
        risk_X_sum_sorted = np.cumsum((X_sorted * e_lp_sorted[:, np.newaxis])[::-1], axis=0)[::-1]
        # risk_XX_sum_sorted[i] = sum of e_lp_sorted[j] * X_sorted[j] @ X_sorted[j]^T for j >= i
        # Use matrix multiplication trick: (X^T diag(e) X) but we need per-row cumulative
        # Direct einsum is clearest but slow; alternative is loop-based accumulation
        # For now, use einsum - it's O(n*p^2) but vectorized
        XX_outer = np.einsum('ni,nj,n->nij', X_sorted, X_sorted, e_lp_sorted)
        risk_XX_sum_sorted = np.cumsum(XX_outer[::-1], axis=0)[::-1]

        # For each unique failure time, compute the Efron-adjusted expectation
        for g in range(nuft):
            t_g = uft[g]
            d_g = counts[g]

            # Find first index in sorted array with time >= t_g
            first_idx = np.searchsorted(time_sorted, t_g, side='left')

            # Risk set sums at t_g
            S0 = risk_sum_sorted[first_idx]
            S1 = risk_X_sum_sorted[first_idx]  # sum of e^lp * X for risk set

            # Events at this time
            events_at_g = event_idx[event_times == t_g]
            X_events = X[events_at_g]
            sum_X_events = X_events.sum(axis=0)

            # Efron approximation: E[X|R(t)] ≈ (1/d) * sum_{k=0}^{d-1} S1(t - k*S0/d) / (S0 - k*S0/d)
            # Simplified: for each k, weight = 1/(S0 * (1 - k/d)) = 1/(S0 - k*S0/d)
            # But we need to handle the case where some observations are the events themselves

            # Direct Efron formula for gradient contribution:
            # sum_{j in events} X_j - sum_{k=0}^{d-1} S1 / (S0 - k*S0/d)

            # Actually, the correct Efron gradient is:
            # sum_events(X) - sum_{k=0}^{d-1} [S1 / (S0 - (k/d)*sum_events(e^lp))]

            # sum of e^lp for events at this time
            sum_e_events = e_linpred[events_at_g].sum()

            # Efron adjustment: for k in 0..d-1, compute gradient and Hessian contributions
            for k in range(d_g):
                frac = k / d_g
                denom = S0 - frac * sum_e_events
                if denom < 1e-300:
                    denom = 1e-300

                # Gradient: S1 / denom (subtracted from sum_X_events later)
                grad_contrib = S1 / denom
                grad -= grad_contrib

                # Hessian: -risk_XX_sum/denom + outer(S1,S1)/denom^2
                # Both terms are needed for correct Newton direction
                risk_XX_sum = risk_XX_sum_sorted[first_idx]
                hess_inner -= risk_XX_sum / denom
                hess_inner += np.outer(S1, S1) / (denom * denom)

            # Add event contribution to gradient
            grad += sum_X_events

        hess = -hess_inner
        return grad, hess
    
    def _compute_gradient_hessian_gpu(
        self, beta, X, time, event, efron_pre=None, return_aux=False
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
        
        # Risk sets
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        X_exp_eta = X * exp_eta[:, cp.newaxis]
        risk_X_sum = cp.cumsum(X_exp_eta[::-1], axis=0)[::-1]
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_pre = _time.perf_counter()
        
        # Efron: when no ties, use Breslow vectorized path.
        if self.ties == "efron":
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
                grad = cp.sum(X[event_mask], axis=0)
                E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
                grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
                use_fused_breslow = (
                    os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "1").strip().lower()
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
        
        # Breslow gradient (vectorized)
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)

        if not cp.any(event_mask):
            out = (grad, cp.zeros((n_features, n_features), dtype=cp.float64))
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

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

        counts_f = counts_uft.astype(cp.float64)
        grad = cp.sum(X[event_mask], axis=0)
        E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
        grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_grad = _time.perf_counter()
        use_fused_breslow = (
            os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "1").strip().lower()
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

    def _compute_gradient_hessian_efron_grouped_gemm_cupy(self, beta, X, efron_pre):
        """Exact Efron grad/hess on CuPy via grouped GEMM updates (no p^2 atomics)."""
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
        try:
            H = -hess
            eps = 1e-11 * (torch.max(torch.abs(torch.diag(H))) + 1.0)
            H = H + eps * torch.eye(p, dtype=torch.float64, device=hess.device)
            return -torch.linalg.solve(H, grad)
        except Exception:
            try:
                return torch.linalg.solve(hess, grad)
            except Exception:
                result = torch.linalg.lstsq(hess, grad)
                return result.solution.flatten()

    @staticmethod
    def _use_torch_fused_efron() -> bool:
        """Opt-in fused Torch-CUDA Efron path (experimental)."""
        v = os.environ.get("STATGPU_TORCH_EFRON_FUSED", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    @staticmethod
    def _torch_to_cupy_view(x_torch):
        import cupy as cp
        import torch
        try:
            return cp.from_dlpack(x_torch)
        except Exception:
            return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x_torch))

    @staticmethod
    def _cupy_to_torch_view(x_cupy, torch_device):
        import torch
        try:
            return torch.utils.dlpack.from_dlpack(x_cupy.toDlpack()).to(torch_device)
        except Exception:
            return torch.as_tensor(x_cupy.get(), dtype=torch.float64, device=torch_device)

    def _compute_gradient_hessian_efron_torch_via_cuda_kernel(
        self, beta, X, efron_pre
    ):
        """Use fused CUDA RawKernel for Torch-CUDA Efron grad/hessian."""
        import cupy as cp
        from ._cox_efron_cuda import compute_efron_grad_hess_raw

        X_cp = self._torch_to_cupy_view(X)
        beta_cp = self._torch_to_cupy_view(beta)
        out = compute_efron_grad_hess_raw(
            X_cp,
            beta_cp,
            efron_pre,
            cupy_module=cp,
            efron_csr=getattr(self, "_efron_pre_csr_gpu", None) or self._efron_pre_csr,
        )
        if out is None:
            raise RuntimeError("CUDA RawKernel grad/hess returned None")
        grad_cp, hess_cp = out
        grad_t = self._cupy_to_torch_view(grad_cp, beta.device)
        hess_t = self._cupy_to_torch_view(hess_cp, beta.device)
        return grad_t, hess_t

    def _compute_gradient_hessian_efron_grouped_gemm_torch(self, beta, X, efron_pre):
        """Exact Efron grad/hess on Torch device via grouped GEMM updates."""
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

    def _compute_gradient_hessian_efron_torch_via_cpu_exact(
        self, beta, X, time, event, efron_pre
    ):
        """Exact Efron grad/hess fallback for Torch via CPU reference core."""
        import torch

        beta_cpu = beta.detach().cpu().numpy()
        X_cpu = X.detach().cpu().numpy()
        time_cpu = time.detach().cpu().numpy()
        event_cpu = event.detach().cpu().numpy()
        grad_cpu, hess_cpu = self._compute_gradient_hessian_efron_backward(
            beta_cpu, X_cpu, time_cpu, event_cpu, efron_pre
        )
        grad = torch.as_tensor(grad_cpu, dtype=torch.float64, device=beta.device)
        hess = torch.as_tensor(hess_cpu, dtype=torch.float64, device=beta.device)
        return grad, hess

    def _compute_log_likelihood_efron_torch_via_cuda_kernel(
        self, eta, exp_eta, risk_sum, efron_pre, torch_device
    ):
        """Use fused CUDA RawKernel for Torch-CUDA Efron log-likelihood."""
        import cupy as cp
        from ._cox_efron_cuda import compute_efron_loglik_raw_csr

        if self._efron_pre_csr is None:
            raise RuntimeError("missing efron_pre_csr")
        csr = getattr(self, "_efron_pre_csr_gpu", None) or self._efron_pre_csr
        _, _, _, _, fail_ptr, fail_ind, first_idx_uft, nuft = csr
        eta_cp = self._torch_to_cupy_view(eta)
        exp_eta_cp = self._torch_to_cupy_view(exp_eta)
        risk_sum_cp = self._torch_to_cupy_view(risk_sum)
        ll_cp = compute_efron_loglik_raw_csr(
            eta_cp,
            exp_eta_cp,
            risk_sum_cp,
            fail_ptr,
            fail_ind,
            first_idx_uft,
            nuft,
            cupy_module=cp,
        )
        return self._cupy_to_torch_view(ll_cp, torch_device)

    def _compute_log_likelihood_torch(self, beta, X, time, event, efron_pre=None):
        """Compute log partial likelihood on Torch."""
        import torch

        eta = X @ beta
        exp_eta = torch.exp(eta)
        risk_sum = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)
        return self._compute_log_likelihood_torch_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre
        )

    def _compute_log_likelihood_torch_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None
    ):
        """Compute log partial likelihood on Torch with precomputed stats."""
        import torch

        ll = torch.tensor(0.0, dtype=torch.float64, device=eta.device)
        event_mask = event == 1

        if not torch.any(event_mask):
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

        # Efron: strict GPU mode forbids CPU fallbacks.
        if efron_pre is not None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)
            if needs_exact_ties:
                if eta.is_cuda and self._efron_pre_csr is not None:
                    return self._compute_log_likelihood_efron_torch_via_cuda_kernel(
                        eta, exp_eta, risk_sum, efron_pre, eta.device
                    )
                raise RuntimeError(
                    "Strict GPU mode: Torch Efron with ties requires CUDA and efron CSR kernel."
                )
            if (
                eta.is_cuda
                and self._efron_pre_csr is not None
                and self._use_torch_fused_efron()
            ):
                try:
                    return self._compute_log_likelihood_efron_torch_via_cuda_kernel(
                        eta, exp_eta, risk_sum, efron_pre, eta.device
                    )
                except Exception:
                    pass
            # No-tie Efron equals Breslow; keep computation on torch device.
            _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
            first_idx_t = torch.as_tensor(first_idx_uft, dtype=torch.int64, device=eta.device)
            counts_t = torch.ones(int(nuft), dtype=torch.float64, device=eta.device)
            risk_at = risk_sum[first_idx_t]
            return torch.sum(eta[event_mask]) - torch.sum(counts_t * torch.log(risk_at))

        # Fallback Efron (loop version)
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
        self, beta, X, time, event, efron_pre=None, return_aux=False
    ):
        """Fully vectorized gradient/Hessian for Torch - Efron and Breslow."""
        import torch

        n_samples, n_features = X.shape
        eta = X @ beta
        exp_eta = torch.exp(eta)
        rev_idx = torch.arange(n_samples - 1, -1, -1, device=beta.device)
        risk_sum = torch.cumsum(exp_eta[rev_idx], dim=0)[rev_idx]

        if self.ties == "efron" and efron_pre is not None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)
            n_samples = int(X.shape[0])
            avg_tie = float(n_samples) / max(1.0, float(_unpack_efron_pre6(efron_pre)[4]))
            use_grouped_gemm = (
                os.environ.get("STATGPU_EFRON_GROUPED_GEMM", "1").strip().lower()
                in ("1", "true", "yes", "on")
            )
            # For real ties, the approximate closed-form torch path is inaccurate.
            # Strict GPU mode: require exact fused CUDA kernel, no CPU fallback.
            if needs_exact_ties:
                if (
                    use_grouped_gemm
                    and beta.is_cuda
                    and n_features <= 192
                    and avg_tie >= 24.0
                ):
                    out = self._compute_gradient_hessian_efron_grouped_gemm_torch(
                        beta, X, efron_pre
                    )
                    if return_aux:
                        return out[0], out[1], (eta, exp_eta, risk_sum)
                    return out
                if beta.is_cuda and self._efron_pre_csr is not None:
                    out = self._compute_gradient_hessian_efron_torch_via_cuda_kernel(
                        beta, X, efron_pre
                    )
                    if return_aux:
                        return out[0], out[1], (eta, exp_eta, risk_sum)
                    return out
                raise RuntimeError(
                    "Strict GPU mode: Torch Efron with ties requires CUDA and efron CSR kernel."
                )
            # No-tie Efron (equivalent to Breslow): optional fused path.
            if (
                beta.is_cuda
                and self._efron_pre_csr is not None
                and self._use_torch_fused_efron()
            ):
                try:
                    out = self._compute_gradient_hessian_efron_torch_via_cuda_kernel(
                        beta, X, efron_pre
                    )
                    if return_aux:
                        return out[0], out[1], (eta, exp_eta, risk_sum)
                    return out
                except Exception:
                    pass

        # Reverse cumsum for risk sets (vectorized)
        risk_X_sum = torch.cumsum((X * exp_eta[:, None])[rev_idx], dim=0)[rev_idx]

        event_mask = event == 1
        if not torch.any(event_mask):
            out = (
                torch.zeros(n_features, dtype=torch.float64, device=beta.device),
                torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device),
            )
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        # Get event data
        event_times = time[event_mask]

        # Unique failure times with inverse mapping
        uft, unique_inv = torch.unique(event_times, sorted=True, return_inverse=True)
        n_uft = len(uft)
        counts = torch.bincount(unique_inv).to(torch.float64)

        # Get first index of each unique time
        sorted_times, sort_idx = torch.sort(time)
        first_in_sorted = torch.searchsorted(sorted_times, uft, side="left")
        first_idx = sort_idx[first_in_sorted]

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

        hess = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
        prev_idx = 0
        for g in range(n_uft):
            idx = int(first_idx[g].item())
            if idx > prev_idx:
                blk = slice(prev_idx, idx)
                risk_X2 = risk_X2 - (X_exp[blk].transpose(0, 1) @ X[blk])
                prev_idx = idx

            rs = risk_at_uft[g]
            if rs <= 0.0:
                continue

            ex = E_X_at_uft[g]
            exx = risk_X2 / rs
            hess = hess - weights[g] * (exx - torch.outer(ex, ex))

        if return_aux:
            return grad, hess, (eta, exp_eta, risk_sum)
        return grad, hess

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

    def _compute_inference_cpu(self, X, time, event, cluster=None):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]

        # Keep inference self-contained (no nested external model fitting),
        # so runtime reflects this implementation directly.
        
        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(
            self.coef_, X, time, event, getattr(self, "_efron_pre", None)
        )
        
        # Bread matrix from observed information.
        try:
            bread = np.linalg.solve(-hess, np.eye(n_features))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(-hess)

        if self.cov_type == "nonrobust":
            self._var_matrix = bread
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
        self._bse = np.sqrt(np.diag(self._var_matrix))
        
        # z-values
        self._zvalues = self.coef_ / self._bse
        
        # p-values (two-sided)
        self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._zvalues)))
        
        # 95% confidence intervals
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha / 2)
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
        self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
        
        # Likelihood ratio test
        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
        
        # Score test (Rao's test) - computed at beta = 0
        ep = getattr(self, "_efron_pre", None)
        grad_0, _ = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep)
        try:
            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep)
            info_0 = -hess_0
            info_0_inv = np.linalg.solve(info_0, np.eye(n_features))
            self._score_test_stat = grad_0 @ info_0_inv @ grad_0
        except:
            self._score_test_stat = np.nan
        self._score_test_pvalue = 1 - stats.chi2.cdf(self._score_test_stat, n_features)

    def _compute_robust_score_residuals(self, X, time, event):
        """
        Per-observation contributions for sandwich (HC0/HC1/cluster).

        When `statsmodels` is available, uses `PHReg.score_residuals`, which
        follows the martingale / leverage construction used by statsmodels for
        cluster-robust covariance (same for both Breslow and Efron partial
        likelihood). This aligns robust SEs with statsmodels much more closely
        than the closed-form Breslow score residual or the fast Efron
        approximation.

        Falls back to `_compute_score_residuals_exact_breslow` (Breslow) or
        `_compute_score_residuals_fast` (Efron) when statsmodels is missing or
        raises.
        """
        sr = self._score_residuals_via_statsmodels_if_available(X, time, event)
        if sr is not None:
            return sr
        if self.ties == "breslow":
            return self._compute_score_residuals_exact_breslow(X, time, event)
        return self._compute_score_residuals_fast(X, time, event)

    def _compute_robust_score_residuals_gpu(self, X, time, event):
        """GPU robust score residuals using event-row approximation."""
        import cupy as cp

        eta = X @ cp.asarray(self.coef_)
        exp_eta = cp.exp(eta)
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = cp.cumsum((X * exp_eta[:, cp.newaxis])[::-1], axis=0)[::-1]
        score_residuals = cp.zeros((X.shape[0], X.shape[1]), dtype=cp.float64)
        event_mask = event == 1
        score_residuals[event_mask] = X[event_mask] - risk_X_sum[event_mask] / risk_sum[event_mask, cp.newaxis]
        return score_residuals

    def _score_residuals_via_statsmodels_if_available(
        self, X: np.ndarray, time: np.ndarray, event: np.ndarray
    ):
        """Return statsmodels-style score residuals, or None if unavailable."""
        try:
            import statsmodels.duration.api as smd
        except Exception:
            return None
        try:
            model = smd.PHReg(time, X, status=event, ties=self.ties)
            sr = model.score_residuals(self.coef_)
            if sr.shape != (X.shape[0], X.shape[1]):
                return None
            # Undefined strata / risk-set rows are NaN in statsmodels; drop from meat.
            sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.asarray(sr, dtype=np.float64)
        except Exception:
            return None

    def _compute_score_residuals_fast(self, X, time, event):
        """
        Fast approximate per-observation score residuals at fitted beta.

        Event-row approximation:
          u_i = x_i - E[X | R(t_i)] for event rows, 0 for censored rows.
        This is substantially faster for larger n.
        """
        n_samples, n_features = X.shape
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = np.cumsum((X * exp_eta[:, np.newaxis])[::-1], axis=0)[::-1]
        u = np.zeros((n_samples, n_features), dtype=np.float64)
        # Vectorized: fill only event rows.
        event_mask = event == 1
        u[event_mask] = X[event_mask] - risk_X_sum[event_mask] / risk_sum[event_mask, np.newaxis]
        return u

    def _compute_score_residuals_exact_breslow(self, X, time, event):
        """
        Exact per-observation score residuals for Breslow ties in O(n p).

        u_j = I(event_j) * s_j - exp_eta_j * sum_{i<=j, event_i=1}(s_i / risk_sum_i),
        where s_i = x_i - E[X|R(t_i)].
        """
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = np.cumsum((X * exp_eta[:, np.newaxis])[::-1], axis=0)[::-1]
        event_mask = (event == 1).astype(np.float64)
        s = X - (risk_X_sum / risk_sum[:, np.newaxis])
        a = (event_mask[:, np.newaxis] * s) / risk_sum[:, np.newaxis]
        csum_a = np.cumsum(a, axis=0)
        u = event_mask[:, np.newaxis] * s - exp_eta[:, np.newaxis] * csum_a
        return u
    
    def _compute_baseline_hazard(self, X, time, event):
        """Compute Breslow estimator of baseline hazard and survival function."""
        # Get unique event times
        event_mask = event == 1
        if not np.any(event_mask):
            self._unique_times = np.array([])
            self._baseline_hazard = np.array([])
            self._baseline_cumulative_hazard = np.array([])
            return
        
        unique_times = np.unique(time[event_mask])
        self._unique_times = unique_times
        
        # Linear predictor
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        
        # Compute baseline cumulative hazard using Breslow estimator
        cumulative_hazard = np.zeros(len(unique_times))
        
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = np.sum((time == t) & (event == 1))
            
            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = np.sum(exp_eta[risk_set])
            
            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum
        
        # Cumulative sum
        self._baseline_cumulative_hazard = np.cumsum(cumulative_hazard)
        
        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

    def _compute_baseline_hazard_gpu(self, X, time, event, beta):
        """Compute Breslow estimator of baseline hazard and survival function on GPU."""
        import cupy as cp

        # Get unique event times
        event_mask = event == 1
        if not cp.any(event_mask):
            self._unique_times = cp.array([])
            self._baseline_hazard = cp.array([])
            self._baseline_cumulative_hazard = cp.array([])
            return

        unique_times = cp.unique(time[event_mask])
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = cp.exp(eta)

        # Compute baseline cumulative hazard using Breslow estimator (vectorized)
        cumulative_hazard = cp.zeros(len(unique_times))

        # Vectorized computation using searchsorted
        # For each unique time, compute d_i / risk_sum
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = int(cp.sum((time == t) & (event == 1)))

            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = cp.sum(exp_eta[risk_set])

            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum

        # Cumulative sum
        self._baseline_cumulative_hazard = cp.cumsum(cumulative_hazard)

        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

        # Transfer to CPU for storage
        self._unique_times = cp.asnumpy(self._unique_times)
        self._baseline_hazard = cp.asnumpy(self._baseline_hazard)
        self._baseline_cumulative_hazard = cp.asnumpy(self._baseline_cumulative_hazard)

    def _compute_baseline_hazard_torch(self, X, time, event, beta):
        """Compute Breslow estimator of baseline hazard and survival function on Torch."""
        import torch

        # Get unique event times
        event_mask = event == 1
        if not torch.any(event_mask):
            self._unique_times = torch.tensor([], dtype=torch.float64, device=beta.device)
            self._baseline_hazard = torch.tensor([], dtype=torch.float64, device=beta.device)
            self._baseline_cumulative_hazard = torch.tensor([], dtype=torch.float64, device=beta.device)
            return

        unique_times = torch.unique(time[event_mask])
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = torch.exp(eta)

        # Compute baseline cumulative hazard using Breslow estimator (vectorized)
        cumulative_hazard = torch.zeros(len(unique_times), dtype=torch.float64, device=beta.device)

        # Vectorized computation
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = int(torch.sum((time == t) & (event == 1)))

            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = torch.sum(exp_eta[risk_set])

            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum

        # Cumulative sum
        self._baseline_cumulative_hazard = torch.cumsum(cumulative_hazard, dim=0)

        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

        # Transfer to CPU for storage
        self._unique_times = self._unique_times.cpu().numpy()
        self._baseline_hazard = self._baseline_hazard.cpu().numpy()
        self._baseline_cumulative_hazard = self._baseline_cumulative_hazard.cpu().numpy()

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
    
    def summary(self):
        """Print summary table similar to R's summary(coxph())."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 80)
        print("                     Cox Proportional Hazards Model")
        print("=" * 80)
        print(f"Call:")
        print(f"  coxph(formula = Surv(time, event) ~ ., ties = '{self.ties}')")
        print()
        print(f"  n= {self._nobs}, number of events= {int(self._nevents)}")
        print(f"  covariance type= {self.cov_type}")
        print()
        if self.compute_inference and self._bse is not None:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12} {'se(coef)':>10} {'z':>10} {'Pr(>|z|)':>10}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f} "
                      f"{self._bse[i]:>10.4f} {self._zvalues[i]:>10.3f} {self._pvalues[i]:>10.4f}")
            
            print("-" * 80)
            print(f"{'':<15} {'exp(coef)':>12} {'exp(-coef)':>12} {'lower .95':>12} {'upper .95':>12}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                hr = self.hazard_ratios_[i]
                print(f"{name:<15} {hr:>12.4f} {1/hr:>12.4f} "
                      f"{np.exp(self._conf_int[i, 0]):>12.4f} {np.exp(self._conf_int[i, 1]):>12.4f}")
        else:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12}")
            print("-" * 80)
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f}")
            print("-" * 80)
            print("Inference statistics disabled (compute_inference=False).")
        
        print("=" * 80)
        if self._cindex is None:
            print("Concordance: skipped (compute_cindex=False)")
        else:
            print(f"Concordance: {self._cindex:.3f} (if 0.5-0.7: moderate, 0.7-0.9: strong)")
        if self.compute_inference and self._lr_test_stat is not None:
            print(f"Likelihood ratio test: {self._lr_test_stat:.2f} on {len(self.coef_)} df, p={self._lr_test_pvalue:.4e}")
            print(f"Wald test:            {self._wald_test_stat:.2f} on {len(self.coef_)} df, p={self._wald_test_pvalue:.4e}")
            print(f"Score (logrank) test: {self._score_test_stat:.2f} on {len(self.coef_)} df, p={self._score_test_pvalue:.4e}")
        else:
            print("Likelihood/Wald/Score tests skipped (compute_inference=False).")
        print(f"Number of Newton-Raphson iterations: {self._iterations}")
        print(f"Converged: {self._converged}")
        print("=" * 80)
    
    def predict_hazard_ratio(self, X):
        """
        Predict hazard ratios (exp(X @ coef)).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        
        Returns
        -------
        hazard_ratios : ndarray of shape (n_samples,)
            Predicted hazard ratios.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.exp(X @ self.coef_)
    
    def predict_risk_score(self, X):
        """
        Predict risk scores (X @ coef).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        
        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores (linear predictor).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_
    
    def predict_survival(self, X, times=None):
        """
        Predict survival function S(t|X) = exp(-H0(t) * exp(X @ coef)).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like, optional
            Times at which to evaluate survival function.
            If None, uses unique event times from training data.
        
        Returns
        -------
        survival : ndarray of shape (n_samples, n_times)
            Predicted survival probabilities.
        times : ndarray
            Times at which survival is evaluated.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if times is None:
            times = self._unique_times
        else:
            times = np.asarray(times)
        
        if len(times) == 0 or self._baseline_cumulative_hazard is None:
            return np.ones((X.shape[0], len(times))), times
        
        # Hazard ratios
        hr = np.exp(X @ self.coef_)
        
        # Survival function: S(t) = exp(-H0(t) * HR)
        survival = np.exp(-self._baseline_cumulative_hazard[np.newaxis, :] * hr[:, np.newaxis])
        
        return survival, times
    
    def predict(self, X):
        """Alias for predict_hazard_ratio."""
        return self.predict_hazard_ratio(X)
    
    def score(self, X, time, event):
        """
        Compute concordance index on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test covariates.
        time : array-like of shape (n_samples,)
            Test event/censoring times.
        event : array-like of shape (n_samples,)
            Test event indicators.

        Returns
        -------
        cindex : float
            Concordance index.
        """
        self._check_is_fitted()

        risk_score = self.predict_risk_score(X)
        time = np.asarray(time)
        event = np.asarray(event)

        n = len(time)
        event_mask = (event == 1)

        if not np.any(event_mask):
            return 0.5

        # Use chunked vectorized approach for memory efficiency
        # Similar to _compute_cindex
        event_idx = np.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return 0.5

        concordant = np.int64(0)
        permissible = np.int64(0)
        tied_risk = np.int64(0)

        # Chunk size: keep each (chunk × n) bool matrix <= 128 MB
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk, np.newaxis]
            risk_i = risk_score[idx_chunk, np.newaxis]
            time_j = time[np.newaxis, :]
            risk_j = risk_score[np.newaxis, :]
            event_j = event[np.newaxis, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))

            # Exclude self-comparisons
            chunk_indices = np.arange(end - start, dtype=np.int64)
            perm[chunk_indices, idx_chunk] = False

            concordant += int(np.sum(perm & (risk_i > risk_j)))
            tied_risk += int(np.sum(perm & (risk_i == risk_j)))
            permissible += int(np.sum(perm))

        if permissible > 0:
            return (concordant + 0.5 * tied_risk) / permissible
        return np.nan
