"""Temporary patch script for Ridge weighted-objective consistency review."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_regex(text: str, pattern: str, repl: str, label: str, flags=0) -> str:
    new, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return new


# ---------------------------------------------------------------------------
# Penalized fit paths
# ---------------------------------------------------------------------------
fit_path = ROOT / "statgpu/linear_model/penalized/_fit_mixin.py"
fit = fit_path.read_text()

fit = replace_once(
    fit,
    """        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name)
""",
    """        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name)
            _sw_check = np.asarray(_to_numpy(_sw_arr), dtype=np.float64).reshape(-1)
            if _sw_check.shape[0] != int(X.shape[0]):
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(_sw_check)):
                raise ValueError("sample_weight must be finite")
            if np.any(_sw_check < 0):
                raise ValueError("sample_weight must be non-negative")
            if float(np.sum(_sw_check)) <= 0.0:
                raise ValueError("sample_weight must have a positive sum")
""",
    "fit sample_weight validation",
)

cpu_pattern = r"""        # Original squared-error path \(backward compatible\)\n\n        if sample_weight is not None:.*?        if y_centered\.ndim == 1:\n            y_centered = y_centered\.reshape\(-1, 1\)\n"""
cpu_repl = """        # Original squared-error path (backward compatible)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            n_eff = float(np.sum(sample_weight))
        else:
            n_eff = float(n_samples)

        if self._effective_intercept:
            if sample_weight is None:
                X_mean = np.mean(X, axis=0)
                y_mean = float(np.mean(y))
            else:
                X_mean = np.average(X, axis=0, weights=sample_weight)
                y_mean = float(np.average(y, weights=sample_weight))
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_mean = np.zeros(n_features, dtype=X.dtype)
            y_mean = 0.0
            X_centered = X
            y_centered = y

        if sample_weight is not None:
            sqrt_sw = np.sqrt(sample_weight)
            X_work = X_centered * sqrt_sw[:, np.newaxis]
            y_work = y_centered * sqrt_sw
        else:
            X_work = X_centered
            y_work = y_centered

        if y_work.ndim == 1:
            y_work = y_work.reshape(-1, 1)
"""
fit = replace_regex(fit, cpu_pattern, cpu_repl, "CPU weighted centering", flags=re.S)

start = fit.index("    def _fit_cpu(")
end = fit.index("    def _fit_gpu(", start)
cpu = fit[start:end]
cpu = replace_once(cpu, "XtX = X_centered.T @ X_centered", "XtX = X_work.T @ X_work", "CPU XtX")
cpu = replace_once(cpu, "Xty = X_centered.T @ y_centered.flatten()", "Xty = X_work.T @ y_work.flatten()", "CPU Xty")
cpu = replace_once(cpu, "self._solve_exact_numpy(XtX, Xty, n_samples)", "self._solve_exact_numpy(XtX, Xty, n_eff)", "CPU exact normalization")
cpu = cpu.replace("_max_eigval_power(XtX) / n_samples", "_max_eigval_power(XtX) / n_eff")
cpu = cpu.replace("(XtX @ y_k - Xty) / n_samples", "(XtX @ y_k - Xty) / n_eff")
cpu = cpu.replace("self.alpha * _w * n_samples", "self.alpha * _w * n_eff")
cpu = cpu.replace("self.alpha * n_samples", "self.alpha * n_eff")
cpu = cpu.replace("self.alpha * self.l1_ratio * n_samples", "self.alpha * self.l1_ratio * n_eff")
cpu = cpu.replace("self.alpha * (1 - self.l1_ratio) * n_samples", "self.alpha * (1 - self.l1_ratio) * n_eff")
cpu = cpu.replace("lam = self.alpha * n_samples", "lam = self.alpha * n_eff")
fit = fit[:start] + cpu + fit[end:]

exact_pattern = r"""        # --- Exact solver \(closed-form Ridge\) ---\n        if solver_name == \"exact\":.*?            return\n\n        # Route IRLS/newton/lbfgs through their dedicated backends\."""
exact_repl = """        # --- Exact solver (closed-form Ridge) ---
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            X = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
            y = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=y)
            if is_torch:
                import torch
                if X.dtype != torch.float64:
                    X = X.to(torch.float64)

            sw = None
            n_eff = float(n_samples)
            if sample_weight is not None:
                sw = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X).reshape(-1)
                n_eff = float(np.sum(np.asarray(_to_numpy(sw), dtype=np.float64)))

            if self._effective_intercept:
                if sw is None:
                    X_mean = xp.mean(X, axis=0)
                    y_mean = xp.mean(y)
                else:
                    X_mean = xp.sum(X * sw[:, None], axis=0) / n_eff
                    y_mean = xp.sum(y * sw) / n_eff
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_mean = None
                y_mean = xp_zeros((), X.dtype, xp, ref_arr=X) if is_torch else xp.array(0.0, dtype=X.dtype)
                X_centered = X
                y_centered = y

            if sw is not None:
                sqrt_sw = xp.sqrt(sw)
                X_work = X_centered * sqrt_sw[:, None]
                y_work = y_centered * sqrt_sw
            else:
                X_work = X_centered
                y_work = y_centered

            if y_work.ndim == 1:
                y_work = y_work.reshape(-1)
            _cv = getattr(self, '_cv_cache', None)
            if sw is None and _cv is not None and 'XtX' in _cv:
                XtX = _cv['XtX']
                Xty = _cv['Xty']
            else:
                XtX = X_work.T @ X_work
                Xty = X_work.T @ y_work

            solve_fn = getattr(self, f'_solve_exact_{"torch" if is_torch else "cupy"}')
            coef = solve_fn(XtX, Xty, n_eff)
            self.n_iter_ = 1
            if self._effective_intercept:
                intercept_gpu = (y_mean.reshape(1) - X_mean.reshape(1, -1) @ coef.reshape(-1, 1)).reshape(-1)
                coef_full_gpu = xp.concatenate([intercept_gpu, coef.reshape(-1)])
            else:
                coef_full_gpu = coef.reshape(-1)

            if self.compute_inference:
                infer_fn = getattr(self, f'_precompute_exact_l2_inference_{"torch" if is_torch else "cupy"}')
                infer_fn(
                    X, y, XtX, X_mean, coef_full_gpu, n_samples,
                    sample_weight=sw, normalization=n_eff,
                )

            coef_np = _to_numpy(coef)
            if self._effective_intercept:
                self.intercept_ = float(_to_numpy(y_mean) - _to_numpy(X_mean) @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))
            if is_torch:
                self._cleanup_torch_memory()
            else:
                self._cleanup_cuda_memory()
            return

        # Route IRLS/newton/lbfgs through their dedicated backends."""
fit = replace_regex(fit, exact_pattern, exact_repl, "GPU exact weighted path", flags=re.S)

fit = fit.replace("def _solve_exact_numpy(self, XtX, Xty, n_samples):", "def _solve_exact_numpy(self, XtX, Xty, normalization):")
fit = fit.replace("def _solve_exact_cupy(self, XtX, Xty, n_samples):", "def _solve_exact_cupy(self, XtX, Xty, normalization):")
fit = fit.replace("def _solve_exact_torch(self, XtX, Xty, n_samples):", "def _solve_exact_torch(self, XtX, Xty, normalization):")
fit = fit.replace("(float(n_samples) * alpha)", "(float(normalization) * alpha)")

fit = replace_once(
    fit,
    """        params, n_iter = solver.fit(
            X_work, y_arr,
            sample_weight=sample_weight,
            ridge_alpha=float(n_samples * self.alpha),
""",
    """        ridge_normalization = (
            float(n_samples)
            if sample_weight is None
            else float(np.sum(np.asarray(_to_numpy(sample_weight), dtype=np.float64)))
        )
        params, n_iter = solver.fit(
            X_work, y_arr,
            sample_weight=sample_weight,
            ridge_alpha=float(ridge_normalization * self.alpha),
""",
    "IRLS weighted ridge normalization",
)

fit_path.write_text(fit)

# ---------------------------------------------------------------------------
# Ridge wrapper validation
# ---------------------------------------------------------------------------
ridge_path = ROOT / "statgpu/linear_model/wrappers/_ridge.py"
ridge = ridge_path.read_text()
ridge = replace_once(
    ridge,
    """        X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
        y_np = np.asarray(self._to_array(y, Device.CPU), dtype=np.float64)

        n_samples, n_features = X_np.shape
""",
    """        X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
        y_np = np.asarray(self._to_array(y, Device.CPU), dtype=np.float64)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_np.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        n_samples, n_features = X_np.shape
""",
    "Ridge input validation",
)
ridge = replace_once(
    ridge,
    """        sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None

        if self.fit_intercept:
""",
    """        sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None
        if sw is not None:
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(sw)):
                raise ValueError("sample_weight must be finite")
            if np.any(sw < 0):
                raise ValueError("sample_weight must be non-negative")
            if float(np.sum(sw)) <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        if self.fit_intercept:
""",
    "Ridge weight validation",
)
ridge_path.write_text(ridge)

# ---------------------------------------------------------------------------
# Gaussian Ridge inference
# ---------------------------------------------------------------------------
inf_path = ROOT / "statgpu/linear_model/penalized/_inference_mixin.py"
inf = inf_path.read_text()
inf = replace_once(
    inf,
    """from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
)
""",
    """from statgpu.linear_model._gaussian_inference import (
    GaussianFitState,
    build_gaussian_fit_state,
    compute_gaussian_inference,
)
""",
    "GaussianFitState import",
)

helper_pattern = r"""    def _weighted_gaussian_fit_inputs\(self, X, y, sample_weight=None\):.*?    def _compute_post_fit_gaussian_inference\(self, X, y, sample_weight=None\):"""
helper_repl = """    def _gaussian_fit_state(self, X, y, sample_weight=None):
        \"\"\"Build Gaussian inference state under the fitted average-loss weights.\"\"\"
        X_np = np.asarray(_to_numpy(X), dtype=float)
        y_np = np.asarray(_to_numpy(y), dtype=float)
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()
        if sample_weight is None:
            return build_gaussian_fit_state(
                X_np, y_np, self.coef_, self.intercept_, self._effective_intercept
            )

        sw = np.asarray(_to_numpy(sample_weight), dtype=float).reshape(-1)
        if sw.shape[0] != X_np.shape[0]:
            raise ValueError("sample_weight must be one-dimensional with length n_samples.")
        sqrt_sw = np.sqrt(sw)
        coef = np.asarray(self.coef_, dtype=float)
        if self._effective_intercept:
            params = np.concatenate([[float(self.intercept_)], coef])
            X_design = np.column_stack([sqrt_sw, X_np * sqrt_sw[:, None]])
            y_pred = float(self.intercept_) + X_np @ coef
        else:
            params = coef.copy()
            X_design = X_np * sqrt_sw[:, None]
            y_pred = X_np @ coef
        y_weighted = y_np * sqrt_sw
        resid = (y_np - y_pred) * sqrt_sw
        nobs = int(X_np.shape[0])
        df_resid = nobs - int(X_design.shape[1])
        scale = float(np.sum(resid ** 2) / df_resid) if df_resid > 0 else np.nan
        return GaussianFitState(
            X_design=X_design,
            y=y_weighted,
            resid=resid,
            scale=scale,
            nobs=nobs,
            df_resid=df_resid,
            params=params,
        )

    def _compute_post_fit_gaussian_inference(self, X, y, sample_weight=None):"""
inf = replace_regex(inf, helper_pattern, helper_repl, "weighted inference state", flags=re.S)
inf = replace_once(
    inf,
    """        X_fit, y_fit = self._weighted_gaussian_fit_inputs(X, y, sample_weight=sample_weight)
        state = build_gaussian_fit_state(
            X_fit,
            y_fit,
            self.coef_,
            self.intercept_,
            self._effective_intercept,
        )
""",
    """        state = self._gaussian_fit_state(X, y, sample_weight=sample_weight)
""",
    "post-fit state call",
)
inf = replace_once(
    inf,
    """        ridge_alpha = float(state.nobs) * self._ridge_alpha_for_exact()
""",
    """        ridge_normalization = (
            float(state.nobs)
            if sample_weight is None
            else float(np.sum(np.asarray(_to_numpy(sample_weight), dtype=float)))
        )
        ridge_alpha = ridge_normalization * self._ridge_alpha_for_exact()
""",
    "inference ridge normalization",
)

cupy_start = inf.index("    def _precompute_exact_l2_inference_cupy(")
torch_start = inf.index("    def _precompute_exact_l2_inference_torch(", cupy_start)

cupy_fn = '''    def _precompute_exact_l2_inference_cupy(
        self, X, y, XtX_centered, X_mean, coef_full, n_samples,
        sample_weight=None, normalization=None,
    ):
        \"\"\"Compute exact L2 inference on CuPy using the fitted weighted objective.\"\"\"
        import cupy as cp
        from statgpu.inference._distributions_backend import t

        p = XtX_centered.shape[0]
        normalization = float(n_samples if normalization is None else normalization)
        ridge_alpha = normalization * self._ridge_alpha_for_exact()
        sw = None if sample_weight is None else cp.asarray(sample_weight, dtype=X.dtype).reshape(-1)
        sqrt_sw = None if sw is None else cp.sqrt(sw)

        if X_mean is None:
            xtx_full = XtX_centered
            bread = xtx_full + ridge_alpha * cp.eye(p, dtype=XtX_centered.dtype)
        else:
            sum_x = normalization * X_mean
            xtx_orig = XtX_centered + normalization * cp.outer(X_mean, X_mean)
            xtx_full = cp.empty((p + 1, p + 1), dtype=XtX_centered.dtype)
            xtx_full[0, 0] = normalization
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

        y_pred = X @ coef_full if X_mean is None else coef_full[0] + X @ coef_full[1:]
        resid_raw = y - y_pred
        resid = resid_raw if sqrt_sw is None else resid_raw * sqrt_sw
        df_resid = int(n_samples - coef_full.shape[0])
        scale = cp.sum(resid ** 2) / df_resid if df_resid > 0 else cp.asarray(cp.nan, dtype=X.dtype)

        if X_mean is None:
            X_design_gpu = X if sqrt_sw is None else X * sqrt_sw[:, None]
        else:
            intercept_col = cp.ones(int(n_samples), dtype=X.dtype) if sqrt_sw is None else sqrt_sw
            feature_block = X if sqrt_sw is None else X * sqrt_sw[:, None]
            X_design_gpu = cp.column_stack([intercept_col, feature_block])
        y_state = y if sqrt_sw is None else y * sqrt_sw

        if df_resid <= 0:
            self._inference_precomputed = True
            self._precomputed_gaussian_state = {
                "params": coef_full.get(), "X_design": X_design_gpu.get(),
                "y": y_state.get(), "resid": resid.get(), "scale": np.nan,
                "nobs": int(n_samples), "df_resid": int(df_resid),
            }
            return

        if self.cov_type == "nonrobust":
            cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
            distribution, method = "t", "classical"
        else:
            from statgpu.linear_model._gaussian_inference import robust_covariance_gpu
            cov_params = robust_covariance_gpu(
                X_design_gpu, resid, bread_inv, self.cov_type, cp,
                hac_maxlags=self.hac_maxlags,
            )
            distribution, method = "normal", "sandwich"

        bse = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
        tvalues = coef_full / (bse + 1e-30)
        if distribution == "t":
            pvalues = t.two_sided_pvalue(tvalues, df=df_resid)
            critical = cp.asarray(t.two_sided_critical_value(0.05, df=df_resid), dtype=bse.dtype)
        else:
            from statgpu.inference._distributions_backend import norm
            pvalues = 2.0 * norm.sf(cp.abs(tvalues))
            critical = cp.asarray(norm.ppf(0.975), dtype=bse.dtype)
        conf_int = cp.stack([coef_full - critical * bse, coef_full + critical * bse], axis=1)

        from statgpu.inference._results import GaussianInferenceResult
        result = GaussianInferenceResult(
            params=coef_full.get(), bse=bse.get(), statistic=tvalues.get(),
            pvalues=pvalues.get(), conf_int=conf_int.get(), cov_type=self.cov_type,
            distribution=distribution, df=df_resid, method=method,
            metadata={"ridge_alpha": ridge_alpha, "alpha": 0.05},
        )
        result.apply_to(self)
        self._inference_precomputed = True
        self._precomputed_gaussian_state = {
            "params": coef_full.get(), "X_design": X_design_gpu.get(),
            "y": y_state.get(), "resid": resid.get(), "scale": float(scale.get()),
            "nobs": int(n_samples), "df_resid": int(df_resid),
        }

'''

torch_fn = '''    def _precompute_exact_l2_inference_torch(
        self, X, y, XtX_centered, X_mean, coef_full, n_samples,
        sample_weight=None, normalization=None,
    ):
        \"\"\"Compute exact L2 inference on Torch using the fitted weighted objective.\"\"\"
        import torch
        from statgpu.inference._distributions_backend import get_distribution

        p = XtX_centered.shape[0]
        normalization = float(n_samples if normalization is None else normalization)
        ridge_alpha = normalization * self._ridge_alpha_for_exact()
        eye_p = torch.eye(p, dtype=XtX_centered.dtype, device=XtX_centered.device)
        sw = None if sample_weight is None else torch.as_tensor(
            sample_weight, dtype=X.dtype, device=X.device
        ).reshape(-1)
        sqrt_sw = None if sw is None else torch.sqrt(sw)

        if X_mean is None:
            xtx_full = XtX_centered
            bread = xtx_full + ridge_alpha * eye_p
        else:
            sum_x = normalization * X_mean
            xtx_orig = XtX_centered + normalization * torch.outer(X_mean, X_mean)
            xtx_full = torch.empty((p + 1, p + 1), dtype=XtX_centered.dtype, device=XtX_centered.device)
            xtx_full[0, 0] = normalization
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

        y_pred = X @ coef_full if X_mean is None else coef_full[0] + X @ coef_full[1:]
        resid_raw = y - y_pred
        resid = resid_raw if sqrt_sw is None else resid_raw * sqrt_sw
        df_resid = int(n_samples - coef_full.shape[0])
        scale = torch.sum(resid ** 2) / df_resid if df_resid > 0 else torch.tensor(float("nan"), dtype=X.dtype, device=X.device)

        if X_mean is None:
            X_design_gpu = X if sqrt_sw is None else X * sqrt_sw[:, None]
        else:
            intercept_col = torch.ones(int(n_samples), dtype=X.dtype, device=X.device) if sqrt_sw is None else sqrt_sw
            feature_block = X if sqrt_sw is None else X * sqrt_sw[:, None]
            X_design_gpu = torch.cat([intercept_col.reshape(-1, 1), feature_block], dim=1)
        y_state = y if sqrt_sw is None else y * sqrt_sw

        if df_resid <= 0:
            self._inference_precomputed = True
            self._precomputed_gaussian_state = {
                "params": coef_full.detach().cpu().numpy(),
                "X_design": X_design_gpu.detach().cpu().numpy(),
                "y": y_state.detach().cpu().numpy(),
                "resid": resid.detach().cpu().numpy(), "scale": np.nan,
                "nobs": int(n_samples), "df_resid": int(df_resid),
            }
            return

        if self.cov_type == "nonrobust":
            cov_params = scale * (bread_inv @ xtx_full @ bread_inv)
            distribution, method = "t", "classical"
        else:
            from statgpu.linear_model._gaussian_inference import robust_covariance_gpu
            cov_params = robust_covariance_gpu(
                X_design_gpu, resid, bread_inv, self.cov_type, torch,
                hac_maxlags=self.hac_maxlags,
            )
            distribution, method = "normal", "sandwich"

        bse = torch.sqrt(torch.clamp(torch.diag(cov_params), min=0.0))
        tvalues = coef_full / (bse + 1e-30)
        if distribution == "t":
            dist = get_distribution("t", backend="torch", device=X.device)
            pvalues = dist.two_sided_pvalue(tvalues, df=df_resid)
            critical = dist.two_sided_critical_value(0.05, df=df_resid)
        else:
            dist = get_distribution("norm", backend="torch", device=X.device)
            pvalues = 2.0 * dist.sf(torch.abs(tvalues))
            critical = dist.ppf(0.975)
        conf_int = torch.stack([coef_full - critical * bse, coef_full + critical * bse], dim=1)

        from statgpu.inference._results import GaussianInferenceResult
        result = GaussianInferenceResult(
            params=coef_full.detach().cpu().numpy(),
            bse=bse.detach().cpu().numpy(), statistic=tvalues.detach().cpu().numpy(),
            pvalues=pvalues.detach().cpu().numpy(), conf_int=conf_int.detach().cpu().numpy(),
            cov_type=self.cov_type, distribution=distribution, df=df_resid, method=method,
            metadata={"ridge_alpha": ridge_alpha, "alpha": 0.05},
        )
        result.apply_to(self)
        self._inference_precomputed = True
        self._precomputed_gaussian_state = {
            "params": coef_full.detach().cpu().numpy(),
            "X_design": X_design_gpu.detach().cpu().numpy(),
            "y": y_state.detach().cpu().numpy(),
            "resid": resid.detach().cpu().numpy(),
            "scale": float(scale.detach().cpu().numpy()),
            "nobs": int(n_samples), "df_resid": int(df_resid),
        }
'''

inf = inf[:cupy_start] + cupy_fn + torch_fn + "\n"
inf_path.write_text(inf)

# ---------------------------------------------------------------------------
# RidgeCV alpha-grid weighting
# ---------------------------------------------------------------------------
cv_path = ROOT / "statgpu/linear_model/cv/_ridge_cv.py"
cv = cv_path.read_text()
cv_pattern = r"""def _default_ridge_alpha_grid\(X, y, n_alphas: int = 100, alpha_min_ratio: float = 1e-3\):.*?# =============================================================================\n# Batch MSE computation"""
cv_repl = '''def _default_ridge_alpha_grid(
    X, y, n_alphas: int = 100, alpha_min_ratio: float = 1e-3,
    sample_weight=None,
):
    \"\"\"Generate an alpha grid on the package's average-loss scale.\"\"\"
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if sample_weight is None:
        normalization = float(X_arr.shape[0])
        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        Xty = (X_arr - X_mean).T @ (y_arr - y_mean)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        normalization = float(np.sum(sw))
        X_mean = np.sum(X_arr * sw[:, None], axis=0) / normalization
        y_mean = float(np.sum(y_arr * sw) / normalization)
        Xty = ((X_arr - X_mean) * sw[:, None]).T @ (y_arr - y_mean)
    alpha_max = float(np.max(np.abs(Xty)) * 2.0 / normalization)
    if alpha_max == 0.0:
        alpha_max = 1.0
    if n_alphas <= 1:
        return np.array([alpha_max])
    return np.logspace(
        np.log10(alpha_max * alpha_min_ratio), np.log10(alpha_max),
        num=n_alphas, dtype=np.float64,
    )


def _default_ridge_alpha_grid_backend(
    X, y, backend, n_alphas: int = 100, alpha_min_ratio: float = 1e-3,
    sample_weight=None,
):
    \"\"\"Backend-native alpha grid with the same weighted normalization.\"\"\"
    X_arr = backend.asarray(X)
    y_arr = backend.asarray(y).reshape(-1)
    if sample_weight is None:
        normalization = float(X_arr.shape[0])
        X_mean = backend.mean(X_arr, axis=0)
        y_mean = backend.mean(y_arr)
        Xty = (X_arr - X_mean).T @ (y_arr - y_mean)
    else:
        sw = backend.asarray(sample_weight).reshape(-1)
        normalization = float(backend.sum(sw))
        X_mean = backend.sum(X_arr * sw[:, None], axis=0) / normalization
        y_mean = backend.sum(y_arr * sw) / normalization
        Xty = ((X_arr - X_mean) * sw[:, None]).T @ (y_arr - y_mean)
    alpha_max = float(backend.max(backend.abs(Xty)) * 2.0 / normalization)
    if alpha_max == 0.0:
        alpha_max = 1.0
    if n_alphas <= 1:
        return np.array([alpha_max])
    return np.logspace(
        np.log10(alpha_max * alpha_min_ratio), np.log10(alpha_max),
        num=n_alphas, dtype=np.float64,
    )


# =============================================================================
# Batch MSE computation'''
cv = replace_regex(cv, cv_pattern, cv_repl, "RidgeCV alpha-grid helpers", flags=re.S)

first_grid_pattern = r"""        if gpu_input_cupy or gpu_input_torch or use_gpu:\n            # GPU path for alpha grid generation.*?        else:\n            alpha_grid = _default_ridge_alpha_grid\(X_np, y_np, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio\)"""
first_grid_repl = '''        if gpu_input_cupy or gpu_input_torch or use_gpu:
            backend = get_backend(
                backend='torch' if gpu_input_torch else 'cupy', device='cuda'
            )
            alpha_grid = _default_ridge_alpha_grid_backend(
                X, y, backend, n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight,
            )
        else:
            alpha_grid = _default_ridge_alpha_grid(
                X_np, y_np, n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight_np,
            )'''
cv = replace_regex(cv, first_grid_pattern, first_grid_repl, "first alpha-grid dispatch", flags=re.S)

second_grid_pattern = r"""            if gpu_input_cupy or gpu_input_torch or use_gpu:\n                # GPU path for alpha grid generation.*?            else:\n                alpha_grid = _default_ridge_alpha_grid\(X_np, y_np, n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio\)"""
second_grid_repl = '''            if gpu_input_cupy or gpu_input_torch or use_gpu:
                backend = get_backend(
                    backend='torch' if gpu_input_torch else 'cupy', device='cuda'
                )
                alpha_grid = _default_ridge_alpha_grid_backend(
                    X, y, backend, n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight,
                )
            else:
                alpha_grid = _default_ridge_alpha_grid(
                    X_np, y_np, n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight_np,
                )'''
cv = replace_regex(cv, second_grid_pattern, second_grid_repl, "fallback alpha-grid dispatch", flags=re.S)
cv_path.write_text(cv)

# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------
test_path = ROOT / "dev/tests/test_ridge_weighted_consistency.py"
test_path.write_text('''import numpy as np
import pytest

from statgpu import Ridge
from statgpu.linear_model.cv._ridge_cv import _default_ridge_alpha_grid
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


def _weighted_closed_form(X, y, w, alpha, fit_intercept=True):
    normalizer = float(np.sum(w))
    if fit_intercept:
        x_mean = np.sum(X * w[:, None], axis=0) / normalizer
        y_mean = float(np.sum(y * w) / normalizer)
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = 0.0
    Xc = X - x_mean
    yc = y - y_mean
    XtWX = (Xc * w[:, None]).T @ Xc
    XtWy = (Xc * w[:, None]).T @ yc
    coef = np.linalg.solve(XtWX + normalizer * alpha * np.eye(X.shape[1]), XtWy)
    return coef, float(y_mean - x_mean @ coef) if fit_intercept else 0.0


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_weighted_ridge_exact_matches_average_loss_and_weight_rescaling(fit_intercept):
    rng = np.random.default_rng(1201)
    X = rng.normal(size=(240, 8))
    y = X @ rng.normal(size=8) + 0.6 + rng.normal(scale=0.4, size=240)
    w = rng.uniform(0.1, 3.0, size=240)
    alpha = 0.19

    expected_coef, expected_intercept = _weighted_closed_form(X, y, w, alpha, fit_intercept)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu", compute_inference=False).fit(X, y, sample_weight=w)
    scaled = Ridge(alpha=alpha, fit_intercept=fit_intercept, device="cpu", compute_inference=False).fit(X, y, sample_weight=7.3 * w)

    np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(model.intercept_, expected_intercept, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(scaled.coef_, model.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(scaled.intercept_, model.intercept_, rtol=1e-11, atol=1e-11)


def test_weighted_ridge_formula_and_generic_exact_match_wrapper():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1202)
    X = rng.normal(size=(180, 4))
    y = 1.1 + X @ np.array([0.7, -0.3, 0.9, 0.2]) + rng.normal(scale=0.2, size=180)
    w = rng.uniform(0.2, 2.5, size=180)
    alpha = 0.11
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    frame["y"] = y

    direct = Ridge(alpha=alpha, compute_inference=False, device="cpu").fit(X, y, sample_weight=w)
    formula = Ridge(alpha=alpha, compute_inference=False, device="cpu").fit(
        formula="y ~ x1 + x2 + x3 + x4", data=frame, sample_weight=w
    )
    generic = PenalizedLinearRegression(
        penalty="l2", alpha=alpha, solver="exact", fit_intercept=True,
        compute_inference=False, device="cpu",
    ).fit(X, y, sample_weight=w)

    for other in (formula, generic):
        np.testing.assert_allclose(other.coef_, direct.coef_, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(other.intercept_, direct.intercept_, rtol=1e-10, atol=1e-10)


def test_weighted_ridge_fista_matches_exact_objective():
    rng = np.random.default_rng(1203)
    X = rng.normal(size=(260, 6))
    y = -0.8 + X @ rng.normal(size=6) + rng.normal(scale=0.3, size=260)
    w = rng.uniform(0.05, 4.0, size=260)
    alpha = 0.07

    exact = Ridge(alpha=alpha, solver="exact", compute_inference=False, device="cpu").fit(X, y, sample_weight=w)
    fista = Ridge(
        alpha=alpha, solver="fista", max_iter=20000, tol=1e-12,
        compute_inference=False, device="cpu",
    ).fit(X, y, sample_weight=w)

    np.testing.assert_allclose(fista.coef_, exact.coef_, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(fista.intercept_, exact.intercept_, rtol=2e-7, atol=2e-8)


def test_weighted_ridge_inference_uses_weighted_intercept_column():
    rng = np.random.default_rng(1204)
    n, p = 320, 5
    X = rng.normal(size=(n, p))
    y = 0.9 + X @ rng.normal(size=p) + rng.normal(scale=0.5, size=n)
    w = rng.uniform(0.1, 2.7, size=n)
    alpha = 0.09
    model = Ridge(alpha=alpha, compute_inference=True, device="cpu").fit(X, y, sample_weight=w)

    D = np.column_stack([np.ones(n), X])
    params = np.concatenate([[model.intercept_], model.coef_])
    resid = y - D @ params
    XtWX = D.T @ (D * w[:, None])
    penalty = np.diag(np.r_[0.0, np.repeat(np.sum(w) * alpha, p)])
    bread_inv = np.linalg.inv(XtWX + penalty)
    scale = float(np.sum(w * resid ** 2) / (n - p - 1))
    cov = scale * (bread_inv @ XtWX @ bread_inv)

    np.testing.assert_allclose(model._bse, np.sqrt(np.diag(cov)), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model._X_design[:, 0], np.sqrt(w), rtol=0, atol=0)
    np.testing.assert_allclose(model._resid, np.sqrt(w) * resid, rtol=1e-12, atol=1e-12)


def test_weighted_default_alpha_grid_uses_average_loss_scale():
    rng = np.random.default_rng(1205)
    X = rng.normal(size=(140, 4))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.3, size=140)
    w = np.linspace(0.1, 3.0, 140)
    grid = _default_ridge_alpha_grid(X, y, n_alphas=7, sample_weight=w)
    scaled = _default_ridge_alpha_grid(X, y, n_alphas=7, sample_weight=9.0 * w)
    np.testing.assert_allclose(grid, scaled, rtol=1e-12, atol=1e-12)
''')

print("Ridge weighted-consistency patch applied")
