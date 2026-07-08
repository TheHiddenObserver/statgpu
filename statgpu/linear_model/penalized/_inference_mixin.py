"""Inference mixin for PenalizedGeneralizedLinearModel."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from statgpu.backends import _to_numpy
from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
)

if TYPE_CHECKING:
    from ._base import PenalizedGeneralizedLinearModel as _Self


class _PenalizedInferenceMixin:

    def _weighted_gaussian_fit_inputs(self, X, y, sample_weight=None):
        X_np = np.asarray(_to_numpy(X), dtype=float)
        y_np = np.asarray(_to_numpy(y), dtype=float)
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()
        if sample_weight is None:
            return X_np, y_np
        sw = np.asarray(_to_numpy(sample_weight), dtype=float)
        if sw.ndim != 1 or sw.shape[0] != X_np.shape[0]:
            raise ValueError("sample_weight must be one-dimensional with length n_samples.")
        sqrt_sw = np.sqrt(sw)
        return X_np * sqrt_sw[:, np.newaxis], y_np * sqrt_sw

    def _compute_post_fit_gaussian_inference(self, X, y, sample_weight=None):
        """Populate inference state after fit. Routes to sandwich/debiased/oracle."""
        if not self.compute_inference:
            return

        # Non-squared_error Hessian losses + smooth/L2 penalties: penalized sandwich
        if self.loss != "squared_error":
            loss_has_hessian = getattr(self._loss, 'has_hessian', False)
            penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
            if loss_has_hessian and penalty_name in ("l2", "none", "", "elasticnet", "en"):
                self._compute_penalized_sandwich_inference(X, y, sample_weight)
                return
            # SCAD/MCP + oracle
            if penalty_name in ("scad", "mcp"):
                im = str(getattr(self, "inference_method", "oracle")).lower()
                if im == "oracle":
                    self._compute_oracle_inference(X, y, sample_weight)
                    return
            # Bootstrap for any other combination
            if str(getattr(self, "inference_method", "")).lower() == "bootstrap":
                self._compute_post_fit_bootstrap_inference(X, y)
                return
            return  # no inference available

        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()

        # SCAD/MCP + squared_error: oracle or bootstrap
        if penalty_name in ("scad", "mcp"):
            im = str(getattr(self, "inference_method", "oracle")).lower()
            if im == "oracle":
                self._compute_oracle_inference(X, y, sample_weight)
                return
            elif im == "bootstrap":
                self._compute_post_fit_bootstrap_inference(X, y)
                return
            raise NotImplementedError(
                f"SCAD/MCP inference requires inference_method='oracle' or "
                f"'bootstrap', got '{im}'. "
                f"Set compute_inference=False or choose a supported method."
            )

        if penalty_name in ("l1", "elasticnet", "en"):
            # GPU/Torch backends run their own debiased inference inside
            # _fit_gpu / _fit_torch.  Skip the CPU re-dispatch when inference
            # is already populated so the GPU result is not overwritten.
            if getattr(self, '_inference_result', None) is not None:
                return
            inference_method = str(getattr(self, "inference_method", "debiased")).lower()
            if "debiased" in inference_method:
                self._compute_post_fit_debiased_inference(X, y, sample_weight=sample_weight)
            elif "bootstrap" in inference_method:
                self._compute_post_fit_bootstrap_inference(X, y)
            elif "cpu_ols" in inference_method or "gpu_ols" in inference_method:
                self._compute_post_fit_cpu_ols_inference(X, y)
            else:
                raise NotImplementedError(
                    f"L1/ElasticNet inference requires inference_method='debiased', "
                    f"'cpu_ols', 'gpu_ols', or 'bootstrap', got '{inference_method}'. "
                    f"Set compute_inference=False or choose a supported method."
                )
            return
        if penalty_name != "l2":
            raise NotImplementedError(
                f"Inference not supported for penalty='{penalty_name}' "
                f"with loss='{self.loss}'. "
                f"Set compute_inference=False or use a supported penalty."
            )
        if self._inference_precomputed:
            state = self._precomputed_gaussian_state
            self._resid = np.asarray(state["resid"], dtype=float)
            self._scale = float(state["scale"])
            self._nobs = int(state["nobs"])
            self._df_resid = int(state["df_resid"])
            self._params = np.asarray(state["params"], dtype=float)
            if self._inference_result is not None:
                self._X_design = np.asarray(state["X_design"], dtype=float)
                self._y = np.asarray(state["y"], dtype=float)
                self._inference_result.feature_names = self._inference_feature_names()
                self._inference_result.apply_to(self)
            self._inference_precomputed = False
            self._precomputed_gaussian_state = None
            return
        X_fit, y_fit = self._weighted_gaussian_fit_inputs(X, y, sample_weight=sample_weight)
        state = build_gaussian_fit_state(
            X_fit,
            y_fit,
            self.coef_,
            self.intercept_,
            self._effective_intercept,
        )
        self._X_design = state.X_design
        self._y = state.y
        self._resid = state.resid
        self._scale = state.scale
        self._nobs = state.nobs
        self._df_resid = state.df_resid
        self._params = state.params
        ridge_alpha = float(state.nobs) * self._ridge_alpha_for_exact()
        result = compute_gaussian_inference(
            self._X_design,
            self._params,
            self._resid,
            self._scale,
            self._df_resid,
            self.cov_type,
            hac_maxlags=self.hac_maxlags,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=False if self._effective_intercept else True,
        )
        if result is None:
            self._inference_result = None
            self._bse = None
            self._tvalues = None
            self._pvalues = None
            self._conf_int = None
            return
        result.feature_names = self._inference_feature_names()
        result.apply_to(self)

    def _inference_feature_names(self):
        if self._feature_names is not None:
            names = list(self._feature_names)
            if self._effective_intercept:
                names.insert(0, "(Intercept)")
            return names
        if self.coef_ is None:
            return None
        n_features = int(np.asarray(self.coef_).shape[-1])
        if self._effective_intercept:
            return ["(Intercept)"] + [f"x{i+1}" for i in range(n_features)]
        return [f"x{i+1}" for i in range(n_features)]

    # ----------------------------------------------------------------
    # Debiased Lasso inference (CPU / CuPy / Torch)
    # ----------------------------------------------------------------

    @staticmethod
    def _debiased_stats_from_M(M, Sigma_hat, sigma2, coef, X, y,
                               intercept, fit_intercept, n, xp, arr_norm):
        """Shared post-M computation for debiased Lasso inference.

        Works with any backend (numpy/cupy/torch) via xp module and
        arr_norm function.  Returns (theta_db, se, z_stats, V_diag) for
        coefficient inference, plus intercept SE if fit_intercept.

        Parameters
        ----------
        M : array (p, p) — decorrelation matrix
        Sigma_hat : array (p, p) — X'X / n
        sigma2 : float — noise variance estimate
        coef : array (p,) — Lasso coefficients
        X, y : arrays — design matrix and response
        intercept : float — fitted intercept
        fit_intercept : bool
        n : int — number of observations
        xp : module — numpy/cupy/torch for array ops
        arr_norm : callable — norm function (np.linalg.norm / cp.linalg.norm / torch.linalg.norm)
        """
        resid = y - X @ coef
        if fit_intercept:
            resid = resid - intercept

        theta_db = coef + (M @ X.T @ resid) / n

        V = M @ Sigma_hat @ M.T
        V_diag = xp.diag(V)
        se = xp.sqrt(xp.abs(sigma2 * V_diag / n))

        z_stats = theta_db / (se + 1e-30)

        # Intercept inference
        se_intercept = None
        z_intercept = None
        if fit_intercept:
            if xp.__name__ == "torch":
                _ones = xp.ones((n, 1), dtype=X.dtype, device=X.device)
            else:
                _ones = xp.ones((n, 1), dtype=X.dtype)
            X_full = xp.concatenate([_ones, X], axis=1)
            try:
                XtX_inv = xp.linalg.inv(X_full.T @ X_full)
            except Exception:
                XtX_inv = xp.linalg.pinv(X_full.T @ X_full)
            se_intercept = float(xp.sqrt(sigma2 * XtX_inv[0, 0]))
            z_intercept = float(intercept) / (se_intercept + 1e-30)

        return theta_db, se, z_stats, V_diag, se_intercept, z_intercept

    def _compute_post_fit_debiased_inference(self, X, y, sample_weight=None):
        """Debiased Lasso inference for squared_error + L1/ElasticNet (CPU path).

        Constructs the decorrelation matrix M via node-wise Lasso,
        then computes the debiased estimator, standard errors,
        z-statistics, p-values, and confidence intervals.
        """
        from statgpu.backends import _resolve_backend
        backend = _resolve_backend("auto", X)
        if backend in ("cupy", "torch"):
            raise NotImplementedError(
                f"Debiased Lasso inference is not yet supported on device={backend!r}. "
                f"Use device='cpu' for inference, or set inference_method='cpu_ols' or 'bootstrap'."
            )
        from statgpu.inference._distributions_backend import get_distribution
        _norm_dist = get_distribution("norm", backend="numpy")

        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()

        if sample_weight is not None:
            sw = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
            sqrt_sw = np.sqrt(sw)
            X_np = X_np * sqrt_sw[:, None]
            y_np = y_np * sqrt_sw

        n, p = X_np.shape
        coef = np.asarray(self.coef_, dtype=np.float64).copy()

        Sigma_hat = X_np.T @ X_np / n

        # Compute residuals
        if self._effective_intercept:
            resid = y_np - X_np @ coef - self.intercept_
        else:
            resid = y_np - X_np @ coef

        # Noise variance estimate
        s_hat = int(np.sum(np.abs(coef) > 0))
        sigma2 = np.sum(resid ** 2) / max(n - s_hat, 1)

        # Node-wise Lasso to build M matrix
        from statgpu.linear_model.wrappers._lasso import (
            _debiased_m_cache_get,
            _debiased_m_cache_put,
            _debiased_m_key_from_numpy_design,
        )

        # Scale node-wise lambda by sigma_hat (van de Geer et al. 2014)
        sigma_hat = np.sqrt(sigma2)
        lam_nw = np.sqrt(2.0 * np.log(max(p, 2)) / n) * sigma_hat
        m_cache_key = _debiased_m_key_from_numpy_design(
            X_np, n=n, p=p, lam_nw=lam_nw, tol=float(self.tol),
        )
        M_cached = _debiased_m_cache_get(m_cache_key)
        if M_cached is not None:
            M = np.asarray(M_cached, dtype=np.float64)
        else:
            M = np.zeros((p, p), dtype=np.float64)
            for j in range(p):
                cols = np.concatenate([np.arange(0, j), np.arange(j + 1, p)])
                X_minus_j = X_np[:, cols]
                x_j = X_np[:, j]

                from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
                nw = PenalizedLinearRegression(
                    penalty="l1", alpha=lam_nw,
                    fit_intercept=False, max_iter=500, tol=1e-5,
                    device="cpu", cpu_solver="fista",
                    compute_inference=False, inference_method="none",
                )
                nw.fit(X_minus_j, x_j)
                gamma_j = np.asarray(nw.coef_, dtype=np.float64)

                z_j = x_j - X_minus_j @ gamma_j
                C_j = z_j @ x_j / n

                if abs(C_j) < 1e-30:
                    M[j, j] = 1.0
                    continue
                M[j, j] = 1.0 / C_j
                M[j, cols] = -gamma_j / C_j
            _debiased_m_cache_put(m_cache_key, M)

        # Shared post-M computation: debiased estimates, SE, z-stats, intercept
        theta_db, se, z_stats, _, se_intercept, z_intercept = self._debiased_stats_from_M(
            M, Sigma_hat, sigma2, coef, X_np, y_np,
            self.intercept_, self._effective_intercept, n, np, np.linalg.norm,
        )
        self._debiased_M_cpu = M

        # p-values and CIs (scipy.stats for CPU path)
        pvalues = 2.0 * (1.0 - _norm_dist.cdf(np.abs(z_stats)))
        alpha_ci = 0.05
        z_crit = _norm_dist.ppf(1.0 - alpha_ci / 2.0)
        ci = np.column_stack([theta_db - z_crit * se, theta_db + z_crit * se])

        # Store residuals and design matrix for R² and simultaneous inference
        self._y = y_np
        self._resid = y_np - X_np @ coef - (self.intercept_ if self._effective_intercept else 0)
        self._nobs = n
        self._scale = sigma2
        if self._effective_intercept:
            self._X_design = np.column_stack([np.ones(n), X_np])
        else:
            self._X_design = X_np.copy()

        if self._effective_intercept:
            p_intercept = 2.0 * (1.0 - _norm_dist.cdf(np.abs(z_intercept)))
            ci_intercept = np.array([
                self.intercept_ - z_crit * se_intercept,
                self.intercept_ + z_crit * se_intercept,
            ])
            self._bse = np.concatenate([[se_intercept], se])
            self._tvalues = np.concatenate([[z_intercept], z_stats])
            self._pvalues = np.concatenate([[p_intercept], pvalues])
            self._conf_int = np.vstack([ci_intercept[np.newaxis, :], ci])
            self._params = np.concatenate([[self.intercept_], theta_db])
        else:
            self._bse = se
            self._tvalues = z_stats
            self._pvalues = pvalues
            self._conf_int = ci
            self._params = theta_db

        # Simultaneous inference (max-|Z| bootstrap) if requested
        if getattr(self, 'enable_simultaneous_inference', False):
            self._compute_simultaneous_ci_maxz_bootstrap()

        # Cleanup: free large intermediates that were only needed for bootstrap
        self._resid = None
        self._X_design = None
        self._y = None

        # Populate _inference_result for API consumers
        from statgpu.inference._results import DebiasedInferenceResult
        self._inference_result = DebiasedInferenceResult(
            method="debiased",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._tvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            precision_method="nodewise_lasso",
            metadata={"backend_path": "cpu_debiased", "precision_cache_hit": M_cached is not None},
            simultaneous_conf_int=getattr(self, '_conf_int_simultaneous', None),
            simultaneous_method=getattr(self, 'simultaneous_method', None),
            simultaneous_alpha=getattr(self, 'simultaneous_alpha', None),
            simultaneous_n_bootstrap=getattr(self, 'simultaneous_n_bootstrap', None),
            simultaneous_critical_value=getattr(self, '_simultaneous_critical_value', None),
        )
        self._inference_result.apply_to(self)

    def _compute_post_fit_cpu_ols_inference(self, X, y):
        """Post-selection OLS inference: refit OLS on selected features.

        This is a heuristic approach — it does NOT provide valid selective
        inference coverage.  Use ``inference_method='debiased'`` for
        proper marginal inference.
        """
        from statgpu.backends import _resolve_backend
        backend = _resolve_backend("auto", X)
        if backend in ("cupy", "torch"):
            raise NotImplementedError(
                f"CPU-OLS inference is not yet supported on device={backend!r}. "
                f"Use device='cpu' for inference, or set inference_method='debiased'."
            )
        from statgpu.inference._distributions_backend import get_distribution
        _t_dist = get_distribution("t", backend="numpy")

        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
        n, p_full = X_np.shape

        # Identify selected (non-zero) features
        coef = np.asarray(self.coef_, dtype=np.float64)
        selected = np.abs(coef) > 1e-15
        n_selected = int(np.sum(selected))

        n_params = len(self._params)
        if n_selected == 0:
            self._bse = np.zeros(n_params)
            self._tvalues = np.zeros(n_params)
            self._pvalues = np.ones(n_params)
            self._conf_int = np.zeros((n_params, 2))
            return

        # Build design matrix for selected features only
        if self._effective_intercept:
            X_sel = np.column_stack([np.ones(n), X_np[:, selected]])
            params_sel = np.concatenate([[self.intercept_], coef[selected]])
        else:
            X_sel = X_np[:, selected]
            params_sel = coef[selected]

        try:
            XtX_inv = np.linalg.inv(X_sel.T @ X_sel)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_sel.T @ X_sel)

        resid = y_np - X_sel @ params_sel
        df_resid = max(n - X_sel.shape[1], 1)
        scale = float(np.sum(resid ** 2) / df_resid)

        bse_sel = np.sqrt(scale * np.diag(XtX_inv))
        tvalues_sel = params_sel / (bse_sel + 1e-30)
        pvalues_sel = 2.0 * _t_dist.sf(np.abs(tvalues_sel), df=df_resid)

        t_crit = _t_dist.ppf(0.975, df=df_resid)
        ci_sel = np.column_stack([
            params_sel - t_crit * bse_sel,
            params_sel + t_crit * bse_sel,
        ])

        # Map back to full parameter space (zero for non-selected)
        self._bse = np.zeros(n_params)
        self._tvalues = np.zeros(n_params)
        self._pvalues = np.ones(n_params)
        self._conf_int = np.zeros((n_params, 2))

        if self._effective_intercept:
            self._bse[0] = bse_sel[0]
            self._tvalues[0] = tvalues_sel[0]
            self._pvalues[0] = pvalues_sel[0]
            self._conf_int[0] = ci_sel[0]
            sel_idx = np.where(selected)[0] + 1
            self._bse[sel_idx] = bse_sel[1:]
            self._tvalues[sel_idx] = tvalues_sel[1:]
            self._pvalues[sel_idx] = pvalues_sel[1:]
            self._conf_int[sel_idx] = ci_sel[1:]
        else:
            sel_idx = np.where(selected)[0]
            self._bse[sel_idx] = bse_sel
            self._tvalues[sel_idx] = tvalues_sel
            self._pvalues[sel_idx] = pvalues_sel
            self._conf_int[sel_idx] = ci_sel

        self._df_resid = df_resid
        self._scale = scale
        self._nobs = n

        # Populate _inference_result
        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="post_selection_ols",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._tvalues.copy(),
            statistic_name="t",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="t",
            df=float(df_resid),
            metadata={
                "heuristic_post_selection": True,
                "backend_path": "cpu_ols",
                "n_selected": n_selected,
            },
        )
        self._inference_result.apply_to(self)

    def _compute_post_fit_bootstrap_inference(self, X, y):
        """Residual bootstrap inference for Lasso.

        More robust than naive OLS-based inference, but still not full
        "post-selection inference" for Lasso.
        """
        # Bootstrap currently runs serial refits (CPU-native RNG).
        # GPU-parallel bootstrap with batched solver tracked for follow-up PR.
        if self._X_design is None or self._resid is None or self._y is None:
            # Need to store these first
            X_np = np.asarray(_to_numpy(X), dtype=np.float64)
            y_np = np.asarray(_to_numpy(y), dtype=np.float64).ravel()
            n = X_np.shape[0]
            if self._effective_intercept:
                self._X_design = np.column_stack([np.ones(n), X_np])
            else:
                self._X_design = X_np.copy()
            self._y = y_np
            coef = np.asarray(self.coef_, dtype=np.float64)
            if self._effective_intercept:
                self._resid = y_np - self._X_design @ np.concatenate([[self.intercept_], coef])
            else:
                self._resid = y_np - self._X_design @ coef
            self._nobs = n

        X_design = self._X_design
        y_arr = self._y
        resid = self._resid
        y_pred = y_arr - resid
        n = len(resid)

        B = int(getattr(self, 'n_bootstrap', 200))
        rng = np.random.default_rng(getattr(self, 'bootstrap_random_state', None))

        params_dim = len(self._params)
        boot_params = np.zeros((B, params_dim), dtype=float)

        for b in range(B):
            eps_star = rng.choice(resid, size=n, replace=True)
            y_star = y_pred + eps_star

            # Refit on bootstrap sample using current penalty
            from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
            refit = PenalizedLinearRegression(
                penalty="l1", alpha=float(self.alpha),
                fit_intercept=self._effective_intercept,
                max_iter=self.max_iter, tol=self.tol,
                device="cpu", cpu_solver="fista",
                compute_inference=False, inference_method="none",
            )
            if self._effective_intercept:
                refit.fit(X_design[:, 1:], y_star)
            else:
                refit.fit(X_design, y_star)
            boot_params[b, :] = refit._params

        # Bootstrap SE
        self._bse = np.std(boot_params, axis=0, ddof=1)

        # Two-sided p-values using sign-change probability
        pvalues = np.zeros(params_dim, dtype=float)
        for i in range(params_dim):
            coef_b = boot_params[:, i]
            p_lower = np.mean(coef_b <= 0.0)
            p_upper = np.mean(coef_b >= 0.0)
            p = 2.0 * min(p_lower, p_upper)
            pvalues[i] = min(p, 1.0)
        self._pvalues = pvalues

        # Percentile confidence intervals
        lower_q = 0.025
        upper_q = 0.975
        self._conf_int = np.column_stack([
            np.quantile(boot_params, lower_q, axis=0),
            np.quantile(boot_params, upper_q, axis=0),
        ])

        # t-stats (approx) from bootstrap SE
        self._tvalues = self._params / (self._bse + 1e-30)

        # Populate _inference_result
        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="residual_bootstrap",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._tvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="bootstrap_percentile",
            metadata={
                "n_bootstrap": B,
                "random_state": getattr(self, 'bootstrap_random_state', None),
            },
        )
        self._inference_result.apply_to(self)

    def _compute_inference_debiased_gpu(self, X_gpu, y_gpu, coef_gpu):
        """CuPy GPU path for debiased Lasso inference."""
        import cupy as cp
        from statgpu.inference._distributions_backend import norm as _gpu_norm

        n, p = X_gpu.shape
        Sigma_hat = X_gpu.T @ X_gpu / n

        resid = y_gpu - X_gpu @ coef_gpu
        if self._effective_intercept:
            resid = resid - cp.mean(y_gpu) + cp.mean(X_gpu, axis=0) @ coef_gpu

        s_hat = float(cp.sum(cp.abs(coef_gpu) > 0))
        sigma2 = float(cp.sum(resid ** 2)) / max(n - s_hat, 1)

        from statgpu.linear_model.wrappers._lasso import (
            _debiased_m_cache_get,
            _debiased_m_cache_put,
            _LASSO_DEBIASED_M_GPU_HASH_ROW_CHUNK,
            _solve_lasso_path_gpu_fista_multi_fold_from_gram,
        )

        # Scale node-wise lambda by sigma_hat (van de Geer et al. 2014)
        sigma_hat = np.sqrt(sigma2)
        lam_nw = float(np.sqrt(2.0 * np.log(max(p, 2)) / n) * sigma_hat)
        alpha_nw = np.asarray([lam_nw], dtype=np.float64)

        # GPU-aware cache key
        import hashlib
        x_hasher = hashlib.blake2b(digest_size=32)
        x_hasher.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
        x_hasher.update(str(X_gpu.dtype).encode("utf-8"))
        x_hasher.update(np.asarray([float(lam_nw), float(self.tol)], dtype=np.float64).tobytes())
        row_chunk = max(1, min(int(n), _LASSO_DEBIASED_M_GPU_HASH_ROW_CHUNK))
        for start in range(0, int(n), row_chunk):
            stop = min(int(n), start + row_chunk)
            x_hasher.update(cp.asnumpy(X_gpu[start:stop]).tobytes())
        m_cache_key = x_hasher.hexdigest()

        M_cached = _debiased_m_cache_get(m_cache_key)
        if M_cached is not None:
            M = cp.asarray(M_cached, dtype=X_gpu.dtype)
        else:
            M = cp.zeros((p, p), dtype=X_gpu.dtype)
            # Reuse Sigma_hat * n instead of recomputing X'X
            XtX_full = Sigma_hat * n
            Sigma_diag = cp.diag(Sigma_hat)

            # Precompute global Lipschitz constant once (avoids per-batch eigendecomposition)
            eig_max = float(cp.linalg.eigvalsh(Sigma_hat)[-1])
            L_global = max(eig_max, 1e-12)

            # Adaptive chunk_size: use as much GPU memory as possible
            # Memory per fold: (p-1)^2 * 8 (Gram) + (p-1)^2 * 8 * 3 (FISTA workspace)
            try:
                free_mem, _ = cp.cuda.Device().mem_info
                bytes_per_fold = int((p - 1) * (p - 1) * 8 * 4)  # Gram + FISTA buffers
                chunk_size = int(max(4, min(p, free_mem * 0.7 // max(bytes_per_fold, 1))))
            except Exception:
                chunk_size = 16
            chunk_size = max(4, min(int(p), chunk_size))

            for j0 in range(0, p, chunk_size):
                j1 = min(p, j0 + chunk_size)
                bsz = j1 - j0
                j_batch = cp.arange(j0, j1, dtype=cp.int32)
                if int(j_batch.size) == 0:
                    continue

                base = cp.arange(p - 1, dtype=cp.int32).reshape(1, -1)
                cols_batch = base + (base >= j_batch.reshape(-1, 1))

                XtX_batch = XtX_full[
                    cols_batch[:, :, cp.newaxis],
                    cols_batch[:, cp.newaxis, :],
                ]
                Xty_batch = XtX_full[cols_batch, j_batch.reshape(-1, 1)].reshape(bsz, p - 1)

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
                    XtX_batch, Xty_batch,
                    n_samples_vec=np.full((bsz,), float(n), dtype=np.float64),
                    alphas_desc=alpha_nw,
                    max_iter=500, tol=1e-5, stopping="coef_delta",
                    lipschitz_L=L_global, check_every=8,
                )
                gamma_batch = cp.asarray(coefs_batch_desc[:, 0, :], dtype=X_gpu.dtype)

                sigma_j_cols = Sigma_hat[j_batch[:, cp.newaxis], cols_batch]
                C_batch = Sigma_diag[j_batch] - cp.sum(sigma_j_cols * gamma_batch, axis=1)

                tiny = X_gpu.dtype.type(1e-30)
                zero = X_gpu.dtype.type(0.0)
                one = X_gpu.dtype.type(1.0)
                small_c = cp.abs(C_batch) < tiny
                inv_c = cp.where(small_c, zero, one / C_batch)
                M[j_batch, j_batch] = cp.where(small_c, one, inv_c)
                M[j_batch[:, cp.newaxis], cols_batch] = -gamma_batch * inv_c.reshape(-1, 1)

                del XtX_batch, Xty_batch, coefs_batch_desc, gamma_batch, sigma_j_cols
            _debiased_m_cache_put(m_cache_key, cp.asnumpy(M))

        # Shared post-M computation
        intercept_val = float(self.intercept_) if self._effective_intercept else 0.0
        theta_db, se, z_stats, _, se_intercept, z_intercept = self._debiased_stats_from_M(
            M, Sigma_hat, sigma2, coef_gpu, X_gpu, y_gpu,
            intercept_val, self._effective_intercept, n, cp, cp.linalg.norm,
        )

        # p-values and CIs (CuPy GPU norm distribution)
        pvalues = cp.minimum(1.0, 2.0 * _gpu_norm.sf(cp.abs(z_stats)))
        z_crit = _gpu_norm.ppf(0.975)
        ci = cp.stack([theta_db - z_crit * se, theta_db + z_crit * se], axis=1)

        if self._effective_intercept:
            intercept_gpu = cp.asarray(self.intercept_, dtype=cp.float64)
            p_intercept = cp.minimum(1.0, 2.0 * _gpu_norm.sf(
                cp.abs(cp.asarray(z_intercept)).reshape(1)))
            ci_intercept = cp.stack([
                intercept_gpu - z_crit * cp.asarray(se_intercept),
                intercept_gpu + z_crit * cp.asarray(se_intercept),
            ]).reshape(1, 2)

            self._bse = cp.asnumpy(cp.concatenate([cp.asarray(se_intercept).reshape(1), se]))
            self._tvalues = cp.asnumpy(cp.concatenate([
                cp.asarray(z_intercept).reshape(1), z_stats]))
            self._pvalues = cp.asnumpy(cp.concatenate([p_intercept.reshape(1), pvalues]))
            self._conf_int = cp.asnumpy(cp.concatenate([ci_intercept, ci], axis=0))
            self._params = cp.asnumpy(cp.concatenate([intercept_gpu.reshape(1), theta_db]))
        else:
            self._bse = cp.asnumpy(se)
            self._tvalues = cp.asnumpy(z_stats)
            self._pvalues = cp.asnumpy(pvalues)
            self._conf_int = cp.asnumpy(ci)
            self._params = cp.asnumpy(theta_db)

        # Store state needed for simultaneous CI bootstrap
        self._debiased_M_cpu = cp.asnumpy(M)
        self._y = cp.asnumpy(y_gpu)
        self._resid = cp.asnumpy(resid)
        self._nobs = n
        if self._effective_intercept:
            self._X_design = np.column_stack([np.ones(n), cp.asnumpy(X_gpu)])
        else:
            self._X_design = cp.asnumpy(X_gpu)

        # Simultaneous inference if requested
        if getattr(self, 'enable_simultaneous_inference', False):
            self._compute_simultaneous_ci_maxz_bootstrap()

        # Cleanup: free large intermediates that were only needed for bootstrap
        self._resid = None
        self._X_design = None
        self._y = None

        # Populate _inference_result for API consumers
        from statgpu.inference._results import DebiasedInferenceResult
        self._inference_result = DebiasedInferenceResult(
            method="debiased",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._tvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            precision_method="nodewise_lasso",
            metadata={"backend_path": "cupy_debiased", "precision_cache_hit": M_cached is not None},
            simultaneous_conf_int=getattr(self, '_conf_int_simultaneous', None),
            simultaneous_method=getattr(self, 'simultaneous_method', None),
            simultaneous_alpha=getattr(self, 'simultaneous_alpha', None),
            simultaneous_n_bootstrap=getattr(self, 'simultaneous_n_bootstrap', None),
            simultaneous_critical_value=getattr(self, '_simultaneous_critical_value', None),
        )

    def _compute_inference_debiased_torch(self, X_torch, y_torch, coef_torch):
        """Torch GPU path for debiased Lasso inference."""
        import torch
        from statgpu.inference._distributions_backend import norm as _gpu_norm

        n, p = X_torch.shape
        dtype = torch.float64
        device = X_torch.device

        if X_torch.dtype != dtype:
            X_torch = X_torch.to(dtype)
        if y_torch.dtype != dtype:
            y_torch = y_torch.to(dtype)
        if coef_torch.dtype != dtype:
            coef_torch = coef_torch.to(dtype)

        Sigma_hat = X_torch.T @ X_torch / n
        resid = y_torch - X_torch @ coef_torch
        if self._effective_intercept:
            resid = resid - torch.mean(y_torch) + torch.mean(X_torch, dim=0) @ coef_torch

        s_hat = float(torch.sum(torch.abs(coef_torch) > 0))
        sigma2 = float(torch.sum(resid ** 2)) / max(n - s_hat, 1)

        from statgpu.linear_model.wrappers._lasso import (
            _debiased_m_cache_get,
            _debiased_m_cache_put,
            _debiased_m_key_from_sample,
            _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch,
        )

        # Scale node-wise lambda by sigma_hat (van de Geer et al. 2014)
        sigma_hat = np.sqrt(sigma2)
        lam_nw = float(np.sqrt(2.0 * np.log(max(p, 2)) / n) * sigma_hat)
        alpha_nw = np.asarray([lam_nw], dtype=np.float64)

        X_sample = X_torch[: min(24, n), : min(24, p)].cpu().numpy()
        m_cache_key = _debiased_m_key_from_sample(
            n=n, p=p, dtype_name=str(dtype),
            sample_block=X_sample, lam_nw=lam_nw, tol=float(self.tol),
        )
        M_cached = _debiased_m_cache_get(m_cache_key)

        if M_cached is not None:
            M = torch.from_numpy(M_cached).to(dtype).to(device)
        else:
            M = torch.zeros((p, p), dtype=dtype, device=device)
            # Reuse Sigma_hat * n instead of recomputing X'X
            XtX_full = Sigma_hat * n
            Sigma_diag = torch.diag(Sigma_hat)

            # Precompute global Lipschitz constant once (avoids per-batch eigendecomposition)
            eig_max = float(torch.linalg.eigvalsh(Sigma_hat)[-1])
            L_global = max(eig_max, 1e-12)

            # Adaptive chunk_size: use as much GPU memory as possible
            try:
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info(device)[0]
                    bytes_per_fold = int((p - 1) * (p - 1) * 8 * 4)  # Gram + FISTA buffers
                    chunk_size = int(max(4, min(p, free_mem * 0.7 // max(bytes_per_fold, 1))))
                else:
                    chunk_size = 16
            except Exception:
                chunk_size = 16
            chunk_size = max(4, min(int(p), chunk_size))

            for j0 in range(0, p, chunk_size):
                j1 = min(p, j0 + chunk_size)
                bsz = j1 - j0
                j_batch = torch.arange(j0, j1, dtype=torch.int32, device=device)

                base = torch.arange(p - 1, dtype=torch.int32, device=device).reshape(1, -1)
                cols_batch = base + (base >= j_batch.reshape(-1, 1))

                XtX_batch = XtX_full[
                    cols_batch[:, :, None],
                    cols_batch[:, None, :],
                ]
                Xty_batch = XtX_full[cols_batch, j_batch.reshape(-1, 1)].reshape(bsz, p - 1)

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
                    XtX_batch, Xty_batch,
                    n_samples_vec=torch.full((bsz,), float(n), dtype=torch.float64, device=device),
                    alphas_desc=alpha_nw,
                    max_iter=500, tol=1e-5, stopping="coef_delta",
                    lipschitz_L=L_global, check_every=8,
                )
                if isinstance(coefs_batch_desc, torch.Tensor):
                    gamma_batch = coefs_batch_desc[:, 0, :].to(dtype).to(device)
                else:
                    gamma_batch = torch.from_numpy(
                        np.asarray(coefs_batch_desc[:, 0, :], dtype=np.float64)
                    ).to(dtype).to(device)

                sigma_j_cols = Sigma_hat[j_batch[:, None], cols_batch]
                C_batch = Sigma_diag[j_batch] - torch.sum(sigma_j_cols * gamma_batch, dim=1)

                tiny = 1e-30
                small_c = torch.abs(C_batch) < tiny
                inv_c = torch.where(small_c, torch.tensor(0.0, dtype=dtype, device=device),
                                    torch.tensor(1.0, dtype=dtype, device=device) / C_batch)
                M[j_batch, j_batch] = torch.where(small_c, torch.tensor(1.0, dtype=dtype, device=device), inv_c)
                M[j_batch[:, None], cols_batch] = -gamma_batch * inv_c.reshape(-1, 1)

                del XtX_batch, Xty_batch, coefs_batch_desc, gamma_batch, sigma_j_cols
            _debiased_m_cache_put(m_cache_key, M.cpu().numpy())

        # Shared post-M computation
        intercept_val = float(self.intercept_) if self._effective_intercept else 0.0
        theta_db, se, z_stats, _, se_intercept, z_intercept = self._debiased_stats_from_M(
            M, Sigma_hat, sigma2, coef_torch, X_torch, y_torch,
            intercept_val, self._effective_intercept, n, torch, torch.linalg.norm,
        )

        # p-values and CIs (Torch GPU norm distribution)
        pvalues = torch.minimum(torch.tensor(1.0, dtype=dtype, device=device),
                                 2.0 * _gpu_norm.sf(torch.abs(z_stats)))
        z_crit = _gpu_norm.ppf(0.975)
        ci = torch.stack([theta_db - z_crit * se, theta_db + z_crit * se], dim=1)

        if self._effective_intercept:
            intercept_t = torch.tensor(self.intercept_, dtype=dtype, device=device)
            p_intercept = torch.minimum(torch.tensor(1.0, dtype=dtype, device=device),
                                         2.0 * _gpu_norm.sf(
                                             torch.abs(torch.tensor(z_intercept, dtype=dtype, device=device)).reshape(1)))
            ci_intercept = torch.stack([
                intercept_t - z_crit * torch.tensor(se_intercept, dtype=dtype, device=device),
                intercept_t + z_crit * torch.tensor(se_intercept, dtype=dtype, device=device),
            ]).reshape(1, 2)

            self._bse = torch.cat([torch.tensor(se_intercept, dtype=dtype, device=device).reshape(1), se]).cpu().numpy()
            self._tvalues = torch.cat([torch.tensor(z_intercept, dtype=dtype, device=device).reshape(1), z_stats]).cpu().numpy()
            self._pvalues = torch.cat([p_intercept.reshape(1), pvalues]).cpu().numpy()
            self._conf_int = torch.cat([ci_intercept, ci], dim=0).cpu().numpy()
            self._params = torch.cat([intercept_t.reshape(1), theta_db]).cpu().numpy()
        else:
            self._bse = se.cpu().numpy()
            self._tvalues = z_stats.cpu().numpy()
            self._pvalues = pvalues.cpu().numpy()
            self._conf_int = ci.cpu().numpy()
            self._params = theta_db.cpu().numpy()

        # Store state needed for simultaneous CI bootstrap
        self._debiased_M_cpu = M.cpu().numpy() if hasattr(M, 'cpu') else np.asarray(M)
        self._y = y_torch.cpu().numpy() if hasattr(y_torch, 'cpu') else np.asarray(y_torch)
        self._resid = resid.cpu().numpy() if hasattr(resid, 'cpu') else np.asarray(resid)
        self._nobs = n
        if self._effective_intercept:
            self._X_design = np.column_stack([
                np.ones(n),
                X_torch.cpu().numpy() if hasattr(X_torch, 'cpu') else np.asarray(X_torch),
            ])
        else:
            self._X_design = X_torch.cpu().numpy() if hasattr(X_torch, 'cpu') else np.asarray(X_torch)

        # Simultaneous inference if requested
        if getattr(self, 'enable_simultaneous_inference', False):
            self._compute_simultaneous_ci_maxz_bootstrap()

        # Cleanup: free large intermediates that were only needed for bootstrap
        self._resid = None
        self._X_design = None
        self._y = None

        # Populate _inference_result for API consumers
        from statgpu.inference._results import DebiasedInferenceResult
        self._inference_result = DebiasedInferenceResult(
            method="debiased",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._tvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            precision_method="nodewise_lasso",
            metadata={"backend_path": "torch_debiased", "precision_cache_hit": M_cached is not None},
            simultaneous_conf_int=getattr(self, '_conf_int_simultaneous', None),
            simultaneous_method=getattr(self, 'simultaneous_method', None),
            simultaneous_alpha=getattr(self, 'simultaneous_alpha', None),
            simultaneous_n_bootstrap=getattr(self, 'simultaneous_n_bootstrap', None),
            simultaneous_critical_value=getattr(self, '_simultaneous_critical_value', None),
        )

    # ----------------------------------------------------------------
    # Penalized sandwich for non-squared_error Hessian losses + L2/EN
    # ----------------------------------------------------------------

    def _compute_penalized_sandwich_inference(self, X, y, sample_weight=None):
        """Penalized sandwich inference for Hessian-equipped losses + L2/ElasticNet.

        Uses the penalized Hessian H_pen = H_loss + penalty.curvature_diag()
        as bread.  Backend-aware: works with NumPy, CuPy, and Torch arrays.
        """
        import numpy as np
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_ones, xp_asarray
        from statgpu.inference._sandwich import m_estimation_inference, _infer_covariance_convention
        from statgpu.inference._results import ParameterInferenceResult

        # Resolve backend and keep arrays on native device
        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
        sw_arr = None
        if sample_weight is not None:
            sw_arr = xp_asarray(sample_weight, dtype=xp.float64, xp=xp).ravel()

        # Build aligned design: [1, X] with intercept first
        n, p_feat = X_arr.shape
        if self._effective_intercept:
            ones = xp_ones(n, xp.float64, xp, ref_arr=X_arr)
            if is_torch:
                X_design = xp.cat([ones.reshape(-1, 1), X_arr], dim=1)
            else:
                X_design = xp.column_stack([ones, X_arr])
            params = xp.concatenate([xp.asarray([self.intercept_], dtype=xp.float64),
                                      xp_asarray(self.coef_, dtype=xp.float64, xp=xp)])
            intercept_idx = 0
        else:
            X_design = X_arr
            params = xp_asarray(self.coef_, dtype=xp.float64, xp=xp)
            intercept_idx = None

        # Penalty curvature: features only, intercept gets 0
        curv = xp.zeros(len(params), dtype=xp.float64)
        if self._penalty is not None:
            pen_name = str(getattr(self._penalty, "name", "")).lower()
            if pen_name in ("l2",):
                curv_feat = xp_asarray(
                    self._penalty.curvature_diag(self.coef_), dtype=xp.float64, xp=xp)
            elif pen_name in ("elasticnet", "en"):
                l1r = float(getattr(self._penalty, "l1_ratio", 0.5))
                alpha = float(getattr(self._penalty, "alpha", self.alpha))
                lam2 = alpha * (1.0 - l1r)
                curv_feat = xp.full(p_feat, lam2, dtype=xp.float64)
            else:
                curv_feat = xp.zeros(p_feat, dtype=xp.float64)

            if intercept_idx is not None:
                curv[1:] = curv_feat
            else:
                curv[:] = curv_feat

        has_curv = bool(float(xp.sum(xp.abs(curv))) > 0)

        result = m_estimation_inference(
            self._loss, X_design, y_arr, params,
            cov_type=self.cov_type,
            penalty_curvature_diag=_to_numpy(curv) if has_curv else None,
            sample_weight=sw_arr,
        )

        self._bse = np.asarray(_to_numpy(result["bse"]))
        self._zvalues = np.asarray(_to_numpy(result["statistic"]))
        self._pvalues = np.asarray(_to_numpy(result["pvalues"]))
        self._conf_int = np.asarray(_to_numpy(result["conf_int"]))
        self._params = np.asarray(_to_numpy(params))

        self._inference_result = ParameterInferenceResult(
            method="m_estimation",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._zvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            metadata={
                "dispersion": result["dispersion"],
                "wald_stat": result["wald_stat"],
                "wald_pval": result["wald_pval"],
                "meat_type": self.cov_type,
                "covariance_convention": _infer_covariance_convention(
                    self.cov_type, has_curv
                ),
                "backend": backend,
            },
        )
        self._inference_result.apply_to(self)

    def _compute_oracle_inference(self, X, y, sample_weight=None):
        """Oracle active-set inference for SCAD/MCP.

        Refits unpenalized model on the active set and applies sandwich.
        Backend-aware: works with NumPy, CuPy, and Torch arrays.
        Valid due to the oracle property (Fan & Li 2001).
        """
        import numpy as np
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_asarray, xp_ones
        from statgpu.inference._sandwich import m_estimation_inference, _infer_covariance_convention
        from statgpu.inference._results import ParameterInferenceResult

        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
        coef_arr = xp_asarray(self.coef_, dtype=xp.float64, xp=xp)
        n, p = X_arr.shape

        # Active set
        active = xp.abs(coef_arr) > 1e-10
        n_active = int(xp.sum(active))

        active_cpu = np.asarray(_to_numpy(active))
        n_active = int(np.sum(active_cpu))
        coef_cpu = np.asarray(_to_numpy(coef_arr))

        if n_active == 0:
            full_p = p + (1 if self._effective_intercept else 0)
            self._params = np.concatenate([[self.intercept_], coef_cpu]) if self._effective_intercept else coef_cpu.copy()
            self._bse = np.full(full_p, np.nan)
            self._zvalues = np.full(full_p, np.nan)
            self._pvalues = np.full(full_p, np.nan)
            self._conf_int = np.full((full_p, 2), np.nan)
            self._inference_result = ParameterInferenceResult(
                method="oracle", params=self._params.copy(), bse=self._bse.copy(),
                statistic=self._zvalues.copy(), statistic_name="z",
                pvalues=self._pvalues.copy(), conf_int=self._conf_int.copy(),
                distribution="normal", metadata={"n_active": 0, "active_set": []})
            self._inference_result.apply_to(self)
            return

        # Convert to CPU for refit (model constructors expect numpy)
        X_cpu = np.asarray(_to_numpy(X_arr), dtype=float)
        y_cpu = np.asarray(_to_numpy(y_arr), dtype=float).ravel()

        from statgpu.linear_model.wrappers._poisson import PoissonRegression
        from statgpu.linear_model.wrappers._gamma import GammaRegression
        from statgpu.linear_model.wrappers._inverse_gaussian import InverseGaussianRegression
        from statgpu.linear_model.wrappers._negative_binomial import NegativeBinomialRegression
        from statgpu.linear_model.wrappers._tweedie import TweedieRegression
        from statgpu.linear_model.wrappers._linear import LinearRegression

        _MODEL_MAP = {
            "squared_error": LinearRegression, "poisson": PoissonRegression,
            "logistic": None, "gamma": GammaRegression,
            "inverse_gaussian": InverseGaussianRegression,
            "negative_binomial": NegativeBinomialRegression, "tweedie": TweedieRegression}
        model_cls = _MODEL_MAP.get(self.loss)
        if model_cls is None:
            if self.loss == "logistic":
                from statgpu.linear_model.wrappers._logistic import LogisticRegression as LR
                model_cls = LR
            else:
                raise NotImplementedError(f"Oracle inference not implemented for loss='{self.loss}'")

        X_active = X_cpu[:, active_cpu]
        kwargs = {"fit_intercept": self._effective_intercept}
        loss_kwargs = getattr(self, 'loss_kwargs', None) or {}
        # Pass through loss-specific kwargs with correct parameter names
        if self.loss == "negative_binomial" and "alpha" in model_cls.__init__.__code__.co_varnames:
            kwargs["alpha"] = loss_kwargs.get("alpha", 1.0)
        elif self.loss == "gamma" and "link" in model_cls.__init__.__code__.co_varnames:
            kwargs["link"] = loss_kwargs.get("link", "log")
        elif self.loss == "tweedie" and "power" in model_cls.__init__.__code__.co_varnames:
            kwargs["power"] = loss_kwargs.get("power", 1.5)
        elif loss_kwargs and "loss_kwargs" in model_cls.__init__.__code__.co_varnames:
            kwargs["loss_kwargs"] = loss_kwargs
        if self.loss == "logistic" and "C" in model_cls.__init__.__code__.co_varnames:
            kwargs["C"] = 1e9
        if backend != "numpy" and "device" in model_cls.__init__.__code__.co_varnames:
            kwargs["device"] = backend
        # Ensure sample_weight is CPU numpy (refit runs on CPU)
        sw_cpu = None
        if sample_weight is not None:
            sw_cpu = np.asarray(_to_numpy(sample_weight), dtype=float).ravel()
        refit = model_cls(**kwargs)
        refit.fit(X_active, y_cpu, sample_weight=sw_cpu)

        # Sandwich on refit — use backend-aware m_estimation_inference
        if self._effective_intercept:
            X_design = np.column_stack([np.ones(n), X_active])
            params_active = np.concatenate([[refit.intercept_], refit.coef_])
        else:
            X_design = X_active
            params_active = np.asarray(refit.coef_)

        loss_obj = refit._resolve_loss_for_inference() if hasattr(refit, '_resolve_loss_for_inference') else self._loss
        result = m_estimation_inference(
            loss_obj, X_design, y_cpu, params_active,
            cov_type=self.cov_type, sample_weight=sample_weight)

        # Map back to full parameter space
        full_p = p + (1 if self._effective_intercept else 0)
        bse_full = np.full(full_p, np.nan); z_full = np.full(full_p, np.nan)
        p_full = np.full(full_p, np.nan); ci_full = np.full((full_p, 2), np.nan)
        offset = 1 if self._effective_intercept else 0
        active_idx = np.where(active_cpu)[0] + offset
        bse_full[active_idx] = np.asarray(result["bse"])[offset:]
        z_full[active_idx] = np.asarray(result["statistic"])[offset:]
        p_full[active_idx] = np.asarray(result["pvalues"])[offset:]
        ci_full[active_idx] = np.asarray(result["conf_int"])[offset:]
        if self._effective_intercept:
            bse_full[0] = np.asarray(result["bse"])[0]; z_full[0] = np.asarray(result["statistic"])[0]
            p_full[0] = np.asarray(result["pvalues"])[0]; ci_full[0] = np.asarray(result["conf_int"])[0]
        self._bse = bse_full; self._zvalues = z_full
        self._pvalues = p_full; self._conf_int = ci_full
        params_full = np.full(full_p, np.nan)
        params_full[active_idx] = params_active[offset:]
        if self._effective_intercept:
            params_full[0] = params_active[0]
        self._params = params_full

        self._inference_result = ParameterInferenceResult(
            method="oracle",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._zvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            metadata={
                "n_active": n_active,
                "active_set": active.tolist() if hasattr(active, 'tolist') else list(active),
                "covariance_convention": _infer_covariance_convention(self.cov_type, False),
            },
        )
        self._inference_result.apply_to(self)

    def _compute_simultaneous_ci_maxz_bootstrap(self):
        """Compute simultaneous CIs using max-|Z| multiplier bootstrap.

        Requires debiased inference to have been run first (provides M matrix,
        residuals, SEs). Uses the Zhang & Zhang (2014) max-|Z| procedure.
        """
        if self._debiased_M_cpu is None:
            return
        if self._y is None or self._resid is None or self._bse is None:
            return

        n = self._nobs
        X = self._X_design
        if X is None:
            return
        if self._effective_intercept:
            X_feat = X[:, 1:]
        else:
            X_feat = X
        _, p = X_feat.shape
        M = self._debiased_M_cpu
        resid = np.asarray(self._resid, dtype=float).reshape(-1)

        # Target indices (exclude intercept unless requested)
        include_intercept = getattr(self, 'simultaneous_include_intercept',
                                    getattr(self, '_simultaneous_include_intercept', False))
        if include_intercept and self._effective_intercept:
            param_target_idx = np.arange(len(self._params), dtype=int)
        elif self._effective_intercept:
            param_target_idx = np.arange(1, len(self._params), dtype=int)
        else:
            param_target_idx = np.arange(len(self._params), dtype=int)

        feature_target_idx = param_target_idx - (1 if self._effective_intercept else 0)
        feature_target_idx = feature_target_idx[feature_target_idx >= 0]
        if feature_target_idx.size == 0:
            return

        se_feat = np.asarray(self._bse[(1 if self._effective_intercept else 0):], dtype=float)
        alpha_sim = float(getattr(self, 'simultaneous_alpha',
                                  getattr(self, '_simultaneous_alpha', 0.05)))
        B = int(getattr(self, 'simultaneous_n_bootstrap',
                        getattr(self, '_simultaneous_n_bootstrap', 1000)))
        rng = np.random.default_rng(getattr(self, 'simultaneous_random_state',
                                            getattr(self, '_simultaneous_random_state', None)))

        # Bootstrap max-|Z|
        chunk = min(256, B)
        max_stats = np.empty(B, dtype=float)
        filled = 0
        while filled < B:
            bsz = min(chunk, B - filled)
            xi = rng.standard_normal(size=(bsz, n))
            weighted = xi * resid.reshape(1, -1)
            score = (weighted @ X_feat) @ M.T / float(max(n, 1))
            z_star = score / (se_feat.reshape(1, -1) + 1e-30)
            max_stats[filled:filled + bsz] = np.max(
                np.abs(z_star[:, feature_target_idx]), axis=1
            )
            filled += bsz

        critical = float(np.quantile(max_stats, 1.0 - alpha_sim))
        params = np.asarray(self._params, dtype=float)
        bse = np.asarray(self._bse, dtype=float)
        conf_sim = np.array(self._conf_int, copy=True, dtype=float)
        conf_sim[param_target_idx, 0] = params[param_target_idx] - critical * bse[param_target_idx]
        conf_sim[param_target_idx, 1] = params[param_target_idx] + critical * bse[param_target_idx]

        self._conf_int_simultaneous = conf_sim
        self._simultaneous_critical_value = critical
        self._simultaneous_enabled = True

    def _precompute_exact_l2_inference_cupy(self, X, y, XtX_centered, X_mean, coef_full, n_samples):
        """Compute nonrobust exact L2 inference on CuPy without a CPU Gram rebuild."""
        import cupy as cp
        from statgpu.inference._distributions_backend import t

        p = XtX_centered.shape[0]
        ridge_alpha = float(n_samples) * self._ridge_alpha_for_exact()
        if X_mean is None:
            xtx_full = XtX_centered
            bread = xtx_full + ridge_alpha * cp.eye(p, dtype=XtX_centered.dtype)
        else:
            sum_x = float(n_samples) * X_mean
            xtx_orig = XtX_centered + float(n_samples) * cp.outer(X_mean, X_mean)
            xtx_full = cp.empty((p + 1, p + 1), dtype=XtX_centered.dtype)
            xtx_full[0, 0] = float(n_samples)
            xtx_full[0, 1:] = sum_x
            xtx_full[1:, 0] = sum_x
            xtx_full[1:, 1:] = xtx_orig
            bread = xtx_full.copy()
            bread[1:, 1:] = xtx_orig + ridge_alpha * cp.eye(p, dtype=XtX_centered.dtype)
        try:
            chol = cp.linalg.cholesky(bread)
            bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, cp.eye(bread.shape[0], dtype=bread.dtype)))
        except Exception:
            bread_inv = cp.linalg.pinv(bread)

        if X_mean is None:
            y_pred = X @ coef_full
        else:
            y_pred = coef_full[0] + X @ coef_full[1:]
        resid = y - y_pred
        df_resid = int(n_samples - coef_full.shape[0])
        if df_resid <= 0:
            if X_mean is None:
                X_design = X.get()
            else:
                X_np = X.get()
                X_design = np.column_stack([np.ones(int(n_samples), dtype=X_np.dtype), X_np])
            self._inference_precomputed = True
            self._precomputed_gaussian_state = {
                "params": coef_full.get(),
                "X_design": X_design,
                "y": y.get(),
                "resid": resid.get(),
                "scale": np.nan,
                "nobs": int(n_samples),
                "df_resid": int(df_resid),
            }
            return
        scale = cp.sum(resid ** 2) / df_resid if df_resid > 0 else cp.asarray(cp.nan, dtype=X.dtype)

        # Compute covariance matrix
        if self.cov_type == "nonrobust":
            cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
            distribution = "t"
            method = "classical"
        else:
            # GPU-native robust/HAC covariance
            from statgpu.linear_model._gaussian_inference import robust_covariance_gpu
            if X_mean is None:
                X_design_gpu = X
            else:
                X_design_gpu = cp.column_stack([cp.ones(int(n_samples), dtype=X.dtype), X])
            cov_params = robust_covariance_gpu(
                X_design_gpu, resid, bread_inv, self.cov_type, cp,
                hac_maxlags=self.hac_maxlags,
            )
            distribution = "normal"
            method = "sandwich"

        bse = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
        tvalues = coef_full / (bse + 1e-30)
        if distribution == "t":
            pvalues = t.two_sided_pvalue(tvalues, df=df_resid)
            t_crit = cp.asarray(t.two_sided_critical_value(0.05, df=df_resid), dtype=bse.dtype)
        else:
            from statgpu.inference._distributions_backend import norm
            pvalues = 2.0 * norm.sf(cp.abs(tvalues))
            z_crit = cp.asarray(norm.ppf(0.975), dtype=bse.dtype)
            t_crit = z_crit
        conf_int = cp.stack([coef_full - t_crit * bse, coef_full + t_crit * bse], axis=1)
        from statgpu.inference._results import GaussianInferenceResult
        result = GaussianInferenceResult(
            params=coef_full.get(),
            bse=bse.get(),
            statistic=tvalues.get(),
            pvalues=pvalues.get(),
            conf_int=conf_int.get(),
            cov_type=self.cov_type,
            distribution=distribution,
            df=df_resid,
            method=method,
            metadata={"ridge_alpha": ridge_alpha, "alpha": 0.05},
        )
        result.apply_to(self)
        self._inference_precomputed = True
        if X_mean is None:
            X_design = X.get()
        else:
            X_np = X.get()
            X_design = np.column_stack([np.ones(int(n_samples), dtype=X_np.dtype), X_np])
        self._precomputed_gaussian_state = {
            "params": coef_full.get(),
            "X_design": X_design,
            "y": y.get(),
            "resid": resid.get(),
            "scale": float(scale.get()) if df_resid > 0 else np.nan,
            "nobs": int(n_samples),
            "df_resid": int(df_resid),
        }

    def _precompute_exact_l2_inference_torch(self, X, y, XtX_centered, X_mean, coef_full, n_samples):
        """Compute nonrobust exact L2 inference on Torch without a CPU Gram rebuild."""
        import torch
        from statgpu.inference._distributions_backend import get_distribution

        p = XtX_centered.shape[0]
        ridge_alpha = float(n_samples) * self._ridge_alpha_for_exact()
        eye_p = torch.eye(p, dtype=XtX_centered.dtype, device=XtX_centered.device)
        if X_mean is None:
            xtx_full = XtX_centered
            bread = xtx_full + ridge_alpha * eye_p
        else:
            sum_x = float(n_samples) * X_mean
            xtx_orig = XtX_centered + float(n_samples) * torch.outer(X_mean, X_mean)
            xtx_full = torch.empty((p + 1, p + 1), dtype=XtX_centered.dtype, device=XtX_centered.device)
            xtx_full[0, 0] = float(n_samples)
            xtx_full[0, 1:] = sum_x
            xtx_full[1:, 0] = sum_x
            xtx_full[1:, 1:] = xtx_orig
            bread = xtx_full.clone()
            bread[1:, 1:] = xtx_orig + ridge_alpha * eye_p
        try:
            chol = torch.linalg.cholesky(bread)
            bread_inv = torch.cholesky_inverse(chol)
        except RuntimeError:
            bread_inv = torch.linalg.pinv(bread)

        if X_mean is None:
            y_pred = X @ coef_full
        else:
            y_pred = coef_full[0] + X @ coef_full[1:]
        resid = y - y_pred
        df_resid = int(n_samples - coef_full.shape[0])
        if df_resid <= 0:
            if X_mean is None:
                X_design = X.detach().cpu().numpy()
            else:
                X_np = X.detach().cpu().numpy()
                X_design = np.column_stack([np.ones(int(n_samples), dtype=X_np.dtype), X_np])
            self._inference_precomputed = True
            self._precomputed_gaussian_state = {
                "params": coef_full.detach().cpu().numpy(),
                "X_design": X_design,
                "y": y.detach().cpu().numpy(),
                "resid": resid.detach().cpu().numpy(),
                "scale": np.nan,
                "nobs": int(n_samples),
                "df_resid": int(df_resid),
            }
            return
        scale = torch.sum(resid ** 2) / df_resid if df_resid > 0 else torch.tensor(float("nan"), dtype=X.dtype, device=X.device)

        # Compute covariance matrix
        if self.cov_type == "nonrobust":
            cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
            distribution = "t"
            method = "classical"
        else:
            # GPU-native robust/HAC covariance
            from statgpu.linear_model._gaussian_inference import robust_covariance_gpu
            if X_mean is None:
                X_design_gpu = X
            else:
                X_design_gpu = torch.cat([torch.ones(int(n_samples), 1, dtype=X.dtype, device=X.device), X], dim=1)
            cov_params = robust_covariance_gpu(
                X_design_gpu, resid, bread_inv, self.cov_type, torch,
                hac_maxlags=self.hac_maxlags,
            )
            distribution = "normal"
            method = "sandwich"

        bse = torch.sqrt(torch.clamp(torch.diag(cov_params), min=0.0))
        tvalues = coef_full / (bse + 1e-30)
        if distribution == "t":
            t_dist = get_distribution("t", backend="torch", device=X.device)
            pvalues = t_dist.two_sided_pvalue(tvalues, df=df_resid)
            t_crit = t_dist.two_sided_critical_value(0.05, df=df_resid)
        else:
            norm_dist = get_distribution("norm", backend="torch", device=X.device)
            pvalues = 2.0 * norm_dist.sf(torch.abs(tvalues))
            z_crit = norm_dist.ppf(0.975)
            t_crit = z_crit
        conf_int = torch.stack([coef_full - t_crit * bse, coef_full + t_crit * bse], dim=1)
        from statgpu.inference._results import GaussianInferenceResult
        result = GaussianInferenceResult(
            params=coef_full.detach().cpu().numpy(),
            bse=bse.detach().cpu().numpy(),
            statistic=tvalues.detach().cpu().numpy(),
            pvalues=pvalues.detach().cpu().numpy(),
            conf_int=conf_int.detach().cpu().numpy(),
            cov_type=self.cov_type,
            distribution=distribution,
            df=df_resid,
            method=method,
            metadata={"ridge_alpha": ridge_alpha, "alpha": 0.05},
        )
        result.apply_to(self)
        self._inference_precomputed = True
        if X_mean is None:
            X_design = X.detach().cpu().numpy()
        else:
            X_np = X.detach().cpu().numpy()
            X_design = np.column_stack([np.ones(int(n_samples), dtype=X_np.dtype), X_np])
        self._precomputed_gaussian_state = {
            "params": coef_full.detach().cpu().numpy(),
            "X_design": X_design,
            "y": y.detach().cpu().numpy(),
            "resid": resid.detach().cpu().numpy(),
            "scale": float(scale.detach().cpu().numpy()) if df_resid > 0 else np.nan,
            "nobs": int(n_samples),
            "df_resid": int(df_resid),
        }
