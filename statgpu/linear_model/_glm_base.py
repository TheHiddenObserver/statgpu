"""
Generalized Linear Model base classes.

Uses sklearn pattern: base class with subclasses overriding _get_family()
and, when needed, the family-to-GLM-loss mapping.
Supports IRLS (smooth penalty) and FISTA (any penalty) solvers.
"""

from typing import Optional, Union, Dict
import numpy as np


def _parse_formula_if_provided(formula, data, X, y):
    """Parse formula+data or fall back to raw arrays. Returns (y, X, info)."""
    if formula is not None:
        from statgpu.core.formula import parse_formula
        return parse_formula(formula, data)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy, _resolve_backend, _is_torch_array
from statgpu.glm_core._irls import IRLSSolver
from statgpu.solvers import fista_solver
from statgpu.glm_core._family import (
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    NegativeBinomial,
    Tweedie,
)


def _np_compat_xp(arr):
    """Return the native array module for the given array: cupy, torch, or numpy."""
    from statgpu.backends._utils import _get_xp
    backend = _resolve_backend("auto", arr)
    if backend == "cupy":
        return _get_xp("cupy")
    if backend == "torch":
        return _get_xp("torch")
    return np


def _ordered_xp(X):
    """Native array module: torch for torch, cupy for cupy, numpy otherwise."""
    from statgpu.backends._utils import _get_xp
    from statgpu.backends import _resolve_backend
    backend = _resolve_backend("auto", X)
    return _get_xp(backend)


def _torch_promoted_float_dtype(X, y):
    """Return a floating dtype that can safely combine Torch X and y."""
    import torch

    x_dtype = X.dtype if X.is_floating_point() else torch.float64
    y_is_float = getattr(y, "is_floating_point", lambda: False)()
    y_dtype = y.dtype if y_is_float else torch.float64
    return torch.promote_types(x_dtype, y_dtype)


def _add_intercept_column(X, backend_name):
    """Prepend an intercept column of ones to X.  Works for numpy/cupy/torch."""
    from statgpu.backends._utils import _get_xp, xp_ones
    xp = _get_xp(backend_name)
    n = X.shape[0]
    ones = xp_ones((n, 1), dtype=X.dtype, xp=xp, ref_arr=X)
    return xp.column_stack([ones, X])


class GeneralizedLinearModel(BaseEstimator):
    """GLM base class with shared IRLS + FISTA paths.

    Subclasses override _get_family() and optionally the GLM loss mapping.

    Parameters
    ----------
    family : str, default='gaussian'
        Distribution family: 'gaussian', 'binomial', 'poisson'.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    C : float, default=1.0
        Inverse regularization strength (for IRLS L2).
    device : str or Device, default='auto'
    solver : str, default='auto'
        'auto', 'irls', 'fista', 'newton', or 'lbfgs'.
    """

    def __init__(
        self,
        family: str = "gaussian",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.family = family
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.solver = solver
        self.gpu_memory_cleanup = gpu_memory_cleanup
        self.compute_inference = compute_inference
        self.cov_type = cov_type

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._use_intercept = None  # formula-derived override; None = use fit_intercept

        # Inference state (populated by _compute_inference)
        self._loss = None
        self._X_design = None
        self._y_inf = None
        self._sample_weight_inf = None
        self._intercept_idx = None
        self._fit_metadata = {}
        self._inference_result = None
        self._bse = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None

    @property
    def _effective_intercept(self):
        """Return the effective intercept flag.

        When formula is used, the formula's intercept semantics take priority
        (stored in ``_use_intercept``).  Otherwise, ``fit_intercept`` is used.
        This avoids mutating ``fit_intercept`` which would break ``sklearn.clone``.
        """
        if self._use_intercept is not None:
            return self._use_intercept
        return self.fit_intercept

    def _get_family(self):
        """Return the GLM Family instance. Override in subclass."""
        family_map = {
            "gaussian": Gaussian,
            "binomial": Binomial,
            "poisson": Poisson,
            "gamma": Gamma,
            "inverse_gaussian": InverseGaussian,
            "negative_binomial": NegativeBinomial,
            "tweedie": Tweedie,
        }
        if self.family not in family_map:
            raise ValueError(
                f"Unknown family '{self.family}'. "
                f"Supported families: {list(family_map.keys())}"
            )
        kwargs = self._get_loss_kwargs()
        return family_map[self.family](**kwargs)

    def _get_penalty_alpha(self):
        """L2 regularization alpha for IRLS: lambda = 1/(2*C)."""
        return 1.0 / (2.0 * self.C) if self.C > 0 else 0.0

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not bool(self.gpu_memory_cleanup):
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA memory cleanup."""
        if not bool(self.gpu_memory_cleanup):
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _cleanup_backend_memory(self, backend_name):
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _resolve_loss_for_inference(self):
        """Create the GLM loss object for inference.

        Returns a loss with ``has_hessian=True`` for sandwich covariance.
        Matches the family used during fitting.
        """
        from statgpu.glm_core import get_glm_loss
        kwargs = self._get_loss_kwargs()
        loss_name = self.family_to_loss()
        return get_glm_loss(loss_name, **kwargs)

    def family_to_loss(self):
        """Map family name to GLM loss name."""
        _map = {
            "gaussian": "squared_error",
            "binomial": "logistic",
            "poisson": "poisson",
            "gamma": "gamma",
            "inverse_gaussian": "inverse_gaussian",
            "negative_binomial": "negative_binomial",
            "tweedie": "tweedie",
        }
        if self.family not in _map:
            raise ValueError(f"Cannot map family '{self.family}' to loss name.")
        return _map[self.family]

    def _get_loss_kwargs(self):
        """Return kwargs for the GLM loss constructor. Override in subclasses."""
        return {}

    def _aligned_inference_design_glm(self, X_orig):
        """Return (X_design, params, intercept_idx) with aligned layout.

        Layout: intercept first → X_design = [1, X], params = [intercept, coef].
        This matches ``statsmodels.add_constant(X, prepend=True)`` order.
        Backend-aware: works with numpy, cupy, and torch arrays.

        ``X_orig`` must be the numeric design matrix after preprocessing
        (dtype/backend conversion), before solver-internal intercept augmentation.
        """
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp

        backend = _resolve_backend("auto", X_orig)
        xp = _get_xp(backend)
        is_gpu = backend != "numpy"

        if self._effective_intercept:
            n = X_orig.shape[0]
            if is_gpu:
                if backend == "torch":
                    import torch
                    dev = X_orig.device; dt = X_orig.dtype
                    ones = torch.ones((n, 1), dtype=dt, device=dev)
                    X_inf = torch.cat([ones, X_orig], dim=1)
                    params_inf = torch.cat([
                        torch.tensor([self.intercept_], dtype=dt, device=dev),
                        torch.as_tensor(self.coef_, dtype=dt, device=dev)
                    ])
                else:
                    ones = xp.ones((n, 1), dtype=X_orig.dtype)
                    X_inf = xp.concatenate([ones, X_orig], axis=1)
                    params_inf = xp.concatenate([
                        xp.asarray([self.intercept_], dtype=X_orig.dtype),
                        xp.asarray(self.coef_, dtype=X_orig.dtype)
                    ])
            else:
                X_np = np.asarray(_to_numpy(X_orig), dtype=float)
                X_inf = np.column_stack([np.ones(n), X_np])
                params_inf = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
            return X_inf, params_inf, 0  # intercept_idx = 0
        else:
            if is_gpu:
                if backend == "torch":
                    import torch
                    return X_orig, torch.as_tensor(self.coef_, dtype=X_orig.dtype, device=X_orig.device), None
                return X_orig, xp.asarray(self.coef_, dtype=X_orig.dtype), None
            else:
                return np.asarray(_to_numpy(X_orig), dtype=float), np.asarray(self.coef_), None

    def _compute_inference(self):
        """Compute M-estimation inference after fit.

        Called automatically at end of ``fit()`` when ``compute_inference=True``.
        Uses fit-time metadata to match the inference to the actual objective.
        Backend-aware: works with NumPy, CuPy, and Torch arrays.
        """
        from statgpu.inference._sandwich import m_estimation_inference, _infer_covariance_convention
        from statgpu.inference._results import ParameterInferenceResult
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp

        curv = self._fit_metadata.get("penalty_curvature_diag")
        backend = _resolve_backend("auto", self._X_design)
        is_gpu = backend != "numpy"

        result = m_estimation_inference(
            self._loss, self._X_design, self._y_inf, self._params,
            cov_type=self.cov_type,
            penalty_curvature_diag=curv,
            sample_weight=self._sample_weight_inf,
        )
        # Convert GPU results to NumPy for storage (API contract: CPU NumPy)
        self._bse = np.asarray(_to_numpy(result["bse"]))
        self._zvalues = np.asarray(_to_numpy(result["statistic"]))
        self._pvalues = np.asarray(_to_numpy(result["pvalues"]))
        self._conf_int = np.asarray(_to_numpy(result["conf_int"]))

        # params may be GPU array
        params_np = np.asarray(_to_numpy(self._params))

        self._inference_result = ParameterInferenceResult(
            method="m_estimation",
            params=params_np.copy(),
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
                    self.cov_type, curv is not None
                ),
                "solver_used": self._fit_metadata.get("solver_used"),
                "inference_backend": backend,
            },
        )
        self._inference_result.apply_to(self)

    # ------------------------------------------------------------------
    # Summary & diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        """Print a summary table of inference results.

        Returns
        -------
        str
            Formatted summary string.
        """
        if not self._fitted:
            return f"{self.__class__.__name__}(not fitted)"

        lines = []
        family_name = getattr(self, 'family', 'unknown')
        lines.append(f"{'='*60}")
        lines.append(f"  {self.__class__.__name__} Results")
        lines.append(f"{'='*60}")
        lines.append(f"  Family: {family_name}")
        lines.append(f"  Solver: {getattr(self, 'solver', 'unknown')}")
        lines.append(f"  No. Observations: {self._nobs}")
        lines.append(f"  Df Residuals: {self._df_resid}")
        lines.append(f"  Covariance Type: {getattr(self, 'cov_type', 'nonrobust')}")
        lines.append("")

        if self._inference_result is not None:
            try:
                df = self._inference_result.to_dataframe()
                lines.append(str(df.to_string(index=False)))
            except Exception:
                lines.append(f"  coef: {self._params}")
                if self._bse is not None:
                    lines.append(f"  std err: {self._bse}")
        else:
            if self._params is not None:
                lines.append(f"  coef: {self._params}")
            lines.append("  (inference not computed)")

        # Model fit statistics
        llf = self.loglikelihood if hasattr(self, 'loglikelihood') else None
        aic = self.aic if hasattr(self, 'aic') else None
        bic = self.bic if hasattr(self, 'bic') else None
        if llf is not None:
            lines.append(f"\n  Log-Likelihood: {llf:.4f}")
        if aic is not None:
            lines.append(f"  AIC: {aic:.4f}")
        if bic is not None:
            lines.append(f"  BIC: {bic:.4f}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    @property
    def llf(self):
        """Log-likelihood of the fitted model (alias for loglikelihood)."""
        return self.loglikelihood

    @property
    def loglikelihood(self):
        """Pseudo-loglikelihood at the fitted coefficients.

        Computed as -sum(loss.per_sample_value(eta, y)).  Additive constants
        that do not depend on the parameters (e.g. -log(y!) for Poisson,
        -n log(2πσ²)/2 for Gaussian) are omitted.  ΔAIC / ΔBIC comparisons
        between nested models on the same data remain valid; absolute values
        should not be compared with statsmodels or R.
        """
        self._check_is_fitted()
        if self._loss is None or self._X_design is None or self._y_inf is None:
            return float("nan")
        from statgpu.backends._utils import _get_xp, xp_asarray
        from statgpu.backends import _resolve_backend
        import numpy as np
        backend = _resolve_backend("auto", self._X_design)
        xp = _get_xp(backend)
        params = xp_asarray(self._params, xp=xp, ref_arr=self._X_design)
        eta = self._X_design @ params
        return -float(xp.sum(self._loss.per_sample_value(eta, self._y_inf)))

    @property
    def aic(self):
        """Akaike Information Criterion: -2*loglik + 2*k."""
        ll = self.loglikelihood
        k = len(self._params) if self._params is not None else 0
        return -2.0 * ll + 2.0 * k

    @property
    def bic(self):
        """Bayesian Information Criterion: -2*loglik + k*log(n)."""
        ll = self.loglikelihood
        k = len(self._params) if self._params is not None else 0
        n = self._nobs if self._nobs else 0
        return -2.0 * ll + k * np.log(max(n, 1))

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
        except Exception:
            pass

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit GLM model.

        Parameters
        ----------
        X : array-like or None
            Predictor matrix. Required if ``formula`` is None.
        y : array-like or None
            Response vector. Required if ``formula`` is None.
        sample_weight : array-like or None
            Sample weights.
        formula : str or None
            R-style formula string (e.g. ``"y ~ x1 + x2"``).
        data : pd.DataFrame or None
            DataFrame used with ``formula`` for column lookup.
        """
        # Resolve backend once for both formula and direct paths
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        # Handle formula interface
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            y_arr, X_arr, design_info = _parse_formula_if_provided(
                formula, data, None, None
            )
            self._design_info = design_info
            formula_column_names = list(design_info.column_names)
            self._formula_has_intercept = "Intercept" in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != "Intercept"]
            if self._formula_has_intercept:
                intercept_idx = formula_column_names.index("Intercept")
                X_arr = np.delete(X_arr, intercept_idx, axis=1)
                self._use_intercept = True
            else:
                self._use_intercept = False
            # Formula produces numpy; convert to backend
            y_arr = self._to_array(y_arr, backend=backend_name)
            X_arr = self._to_array(X_arr, backend=backend_name)
        else:
            if X is None or y is None:
                raise ValueError(
                    "Either formula+data or X+y must be provided."
                )
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None
            self._use_intercept = None
            # _to_array safely handles numpy/cupy/torch inputs
            y_arr = self._to_array(y, backend=backend_name)
            X_arr = self._to_array(X, backend=backend_name)

        # Ensure y is 1D after backend conversion
        if hasattr(y_arr, 'ndim') and y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        self._nobs = X_arr.shape[0]

        family = self._get_family()
        _solver_lower = self.solver.lower() if isinstance(self.solver, str) else self.solver
        if _solver_lower == "auto":
            # Heuristic: IRLS for smooth/no penalties, FISTA for non-smooth
            _pen = getattr(self, "_penalty", None)
            _pname = str(getattr(_pen, "name", "none")).lower() if _pen is not None else "none"
            if _pname in ("l1", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
                          "group_lasso", "group_mcp", "group_scad"):
                solver_name = "fista"
            else:
                solver_name = "irls"
        else:
            solver_name = _solver_lower

        if solver_name == "irls":
            self._fit_irls(X_arr, y_arr, sample_weight, family, backend_name)
        elif solver_name == "fista":
            self._fit_fista(X_arr, y_arr, sample_weight, family, backend_name)
        elif solver_name in ("newton", "lbfgs"):
            self._fit_smooth_solver(
                X_arr, y_arr, sample_weight, solver_name, backend_name
            )
        else:
            raise ValueError(
                "solver must be one of: 'auto', 'irls', 'fista', 'newton', 'lbfgs'"
            )

        # ---- Store design/loss for loglikelihood/aic/bic (always) ----
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp
        inf_backend = _resolve_backend("auto", X_arr)
        inf_xp = _get_xp(inf_backend)
        is_gpu = inf_backend != "numpy"

        # Keep GPU arrays for inference (no CPU transfer)
        if is_gpu:
            self._y_inf = y_arr.ravel() if y_arr.ndim > 1 else y_arr
            self._X_design, self._params, self._intercept_idx = \
                self._aligned_inference_design_glm(X_arr)
        else:
            self._y_inf = np.asarray(_to_numpy(y_arr), dtype=float).ravel()
            self._X_design, self._params, self._intercept_idx = \
                self._aligned_inference_design_glm(X_arr)
        self._loss = self._resolve_loss_for_inference()

        # ---- Compute inference if requested ----
        if self.compute_inference:
            if sample_weight is not None:
                sw = np.asarray(_to_numpy(sample_weight), dtype=float).ravel()
                if is_gpu:
                    self._sample_weight_inf = self._to_array(
                        sw, backend=inf_backend)
                else:
                    self._sample_weight_inf = sw
            else:
                self._sample_weight_inf = None

            self._fit_metadata = {
                "solver_used": solver_name,
                "objective_scale": "mean_loss_plus_penalty",
                "ridge_alpha_avg": None,
                "penalty_curvature_diag": None,
            }
            # IRLS with finite C: add ridge curvature
            if solver_name == "irls" and self.C > 0:
                lam = self._get_penalty_alpha()
                if is_gpu:
                    from statgpu.backends._utils import xp_zeros
                    curv = xp_zeros(self._params.shape[0], self._params.dtype,
                                    inf_xp, ref_arr=self._params)
                else:
                    curv = np.zeros(self._params.shape[0])
                if self._effective_intercept:
                    curv[1:] = lam
                else:
                    curv[:] = lam
                self._fit_metadata["ridge_alpha_avg"] = lam
                self._fit_metadata["penalty_curvature_diag"] = curv

            self._compute_inference()

        self._fitted = True
        self._cleanup_backend_memory(backend_name)
        return self

    def _fit_irls(self, X, y, sample_weight, family, backend_name="numpy"):
        """Fit using IRLS (per-iteration weighted least squares)."""
        # IRLSSolver solves the unnormalized WLS normal equations
        # X'WX + lambda I, while _get_penalty_alpha() is the normalized
        # objective penalty.  Scale by n to keep C semantics consistent.
        ridge_alpha = X.shape[0] * self._get_penalty_alpha()

        if self._effective_intercept:
            X_design = _add_intercept_column(X, backend_name)
        else:
            X_design = X

        solver = IRLSSolver(family, max_iter=self.max_iter, tol=self.tol)
        params, n_iter = solver.fit(
            X_design, y,
            sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=not self._effective_intercept,
            backend=backend_name,
        )

        self.n_iter_ = n_iter
        self._params = params

        # Convert to numpy (params may be cupy/torch array)
        params_np = _to_numpy(params)

        if self._effective_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()

        self._df_resid = self._nobs - (X.shape[1] + (1 if self._effective_intercept else 0))

    def _fit_fista(self, X, y, sample_weight, family, backend_name="numpy"):
        """Fit using FISTA (no penalty; pure loss minimization).

        For GLM losses with intercept, uses iterated intercept estimation
        + coef refinement to converge to the correct joint optimum.
        """
        from statgpu.glm_core import get_glm_loss
        from statgpu.penalties._l2 import L2Penalty

        loss_kwargs = self._get_loss_kwargs()
        loss = get_glm_loss(self.family_to_loss(), **loss_kwargs)

        if not self._effective_intercept:
            X_centered = X
            if backend_name == "torch":
                dtype = _torch_promoted_float_dtype(X_centered, y)
                X_centered = X_centered.to(dtype=dtype)
                y = y.to(X_centered.device).to(dtype)
            init = None
            if self.family == "gamma" and loss_kwargs.get("link") == "inverse_power":
                eta_lo = float(getattr(loss, "_ETA_LO", 1e-4))
                if backend_name == "cupy":
                    import cupy as cp
                    if not cp.issubdtype(X_centered.dtype, cp.floating):
                        X_centered = X_centered.astype(cp.float64)
                    y_cp = cp.asarray(y, dtype=cp.float64)
                    X_cp = cp.asarray(X_centered, dtype=cp.float64)
                    eta_raw = 1.0 / cp.clip(y_cp, 1e-6, None)
                    eta_target = eta_raw - cp.mean(eta_raw)
                    try:
                        init_cp, *_ = cp.linalg.lstsq(X_cp, eta_target, rcond=None)
                    except cp.linalg.LinAlgError:
                        init_cp = cp.zeros(X.shape[1], dtype=cp.float64)
                    eta_init = X_cp @ init_cp
                    eta_abs_max = cp.max(cp.abs(eta_init))
                    min_scale = eta_lo * 10.0
                    if float(eta_abs_max) < min_scale:
                        scale = min_scale / (float(eta_abs_max) + 1e-12)
                        init_cp = init_cp * scale
                        eta_init = X_cp @ init_cp
                    near_zero_frac = cp.mean((cp.abs(eta_init) < (eta_lo * 10.0)).astype(cp.float64))
                    if float(near_zero_frac) > 0.5:
                        g = X_cp.T @ (y_cp - cp.mean(y_cp))
                        g_norm = cp.sqrt(cp.sum(g * g))
                        if float(g_norm) > 0:
                            init_cp = g / g_norm
                            eta_g = X_cp @ init_cp
                            med_abs = float(cp.median(cp.abs(eta_g)))
                            target = eta_lo * 20.0
                            init_cp = init_cp * (target / (med_abs + 1e-12))
                    coef_dtype = (
                        X_centered.dtype
                        if cp.issubdtype(X_centered.dtype, cp.floating)
                        else cp.float64
                    )
                    init = init_cp.astype(coef_dtype, copy=False)
                elif backend_name == "torch":
                    import torch
                    dtype = X_centered.dtype
                    y_t = y.to(X.device).to(torch.float64)
                    X_t = X_centered.to(X.device).to(torch.float64)
                    eta_raw = 1.0 / torch.clamp(y_t, min=1e-6)
                    eta_target = eta_raw - torch.mean(eta_raw)
                    try:
                        init_t = torch.linalg.lstsq(X_t, eta_target).solution
                    except RuntimeError:
                        init_t = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
                    eta_init = X_t @ init_t
                    eta_abs_max = torch.max(torch.abs(eta_init))
                    min_scale = eta_lo * 10.0
                    if float(eta_abs_max.item()) < min_scale:
                        scale = min_scale / (float(eta_abs_max.item()) + 1e-12)
                        init_t = init_t * scale
                        eta_init = X_t @ init_t
                    near_zero_frac = torch.mean((torch.abs(eta_init) < (eta_lo * 10.0)).to(torch.float64))
                    if float(near_zero_frac.item()) > 0.5:
                        g = X_t.T @ (y_t - torch.mean(y_t))
                        g_norm = torch.sqrt(torch.sum(g * g))
                        if float(g_norm.item()) > 0:
                            init_t = g / g_norm
                            eta_g = X_t @ init_t
                            med_abs = float(torch.median(torch.abs(eta_g)).item())
                            target = eta_lo * 20.0
                            init_t = init_t * (target / (med_abs + 1e-12))
                    init = init_t.to(dtype)
                else:
                    if not np.issubdtype(X_centered.dtype, np.floating):
                        X_centered = X_centered.astype(np.float64)
                    y_np = np.asarray(y, dtype=np.float64)
                    X_np = np.asarray(X_centered, dtype=np.float64)
                    eta_raw = 1.0 / np.clip(y_np, 1e-6, None)
                    eta_target = eta_raw - np.mean(eta_raw)
                    try:
                        init = np.linalg.lstsq(X_np, eta_target, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        init = np.zeros(X.shape[1], dtype=np.float64)
                    eta_init = X_np @ init
                    eta_abs_max = float(np.max(np.abs(eta_init))) if eta_init.size else 0.0
                    min_scale = eta_lo * 10.0
                    if eta_abs_max < min_scale:
                        init = init * (min_scale / (eta_abs_max + 1e-12))
                        eta_init = X_np @ init
                    near_zero_frac = float(np.mean(np.abs(eta_init) < (eta_lo * 10.0))) if eta_init.size else 1.0
                    if near_zero_frac > 0.5:
                        g = X_np.T @ (y_np - np.mean(y_np))
                        g_norm = float(np.sqrt(np.sum(g * g)))
                        if g_norm > 0:
                            init = g / g_norm
                            eta_g = X_np @ init
                            med_abs = float(np.median(np.abs(eta_g)))
                            target = eta_lo * 20.0
                            init = init * (target / (med_abs + 1e-12))
            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
            self.coef_ = _to_numpy(coef)
            self.n_iter_ = n_iter
            self.intercept_ = 0.0
            self._params = self.coef_.copy()
            self._df_resid = self._nobs - X.shape[1]
            return

        if loss.name != "squared_error":
            # All non-Gaussian GLM losses must optimize intercept jointly with
            # coefficients. Centering y is only valid for squared-error loss.
            # Augment X with intercept column (no penalty in _fit_fista).
            from statgpu.backends._utils import _get_xp
            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                x_dtype = X.dtype if xp.issubdtype(X.dtype, xp.floating) else xp.float64
                X_float = X.astype(x_dtype, copy=False)
                X_aug = xp.column_stack([X_float, xp.ones(X.shape[0], dtype=x_dtype)])
            elif backend_name == "torch":
                import torch
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_float = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
                X_aug = torch.column_stack([X_float, torch.ones(X.shape[0], dtype=x_dtype, device=X.device)])
            else:
                X_aug = np.column_stack([X, np.ones(X.shape[0])])
            p = X.shape[1]
            # Compute mean on native backend to avoid GPU→CPU transfer
            _xp_mod = _get_xp(backend_name) if backend_name != "numpy" else np
            y_mean = max(float(_xp_mod.mean(y)), 1e-3)
            init = np.zeros(p + 1, dtype=np.float64)
            if self.family == "binomial":
                p_mean = np.clip(y_mean, 1e-3, 1.0 - 1e-3)
                init[-1] = np.log(p_mean / (1.0 - p_mean))
            elif self.family == "gamma" and loss_kwargs.get("link") == "inverse_power":
                init[-1] = 1.0 / y_mean
            elif self.family in (
                "poisson", "gamma", "inverse_gaussian",
                "negative_binomial", "tweedie",
            ):
                init[-1] = np.log(y_mean)
            if backend_name == "cupy":
                init = _xp_mod.asarray(init, dtype=x_dtype)
            elif backend_name == "torch":
                init = torch.from_numpy(init).to(X.device).to(x_dtype)

            full_coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_aug, y,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )

            full_np = _to_numpy(full_coef)
            self.coef_ = full_np[:p]
            self.intercept_ = float(full_np[p])
            self.n_iter_ = n_iter
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            # Squared error: centering X and y preserves the objective.
            from statgpu.backends._utils import _get_xp
            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                X_centered = X - xp.mean(X, axis=0)
                y_centered = y - xp.mean(y)
            elif backend_name == "torch":
                import torch
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_float = X.to(dtype=x_dtype)
                y_float = y.to(X.device).to(x_dtype)
                X_centered = X_float - torch.mean(X_float, dim=0)
                y_centered = y_float - torch.mean(y_float)
            else:
                X_centered = X - X.mean(axis=0)
                y_centered = y - y.mean()

            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y_centered,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            _xp_mod = _get_xp(backend_name) if backend_name != "numpy" else np
            X_mean = _to_numpy(_xp_mod.mean(X, axis=0))
            y_mean = float(_xp_mod.mean(y))
            self.coef_ = _to_numpy(coef)
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self.n_iter_ = n_iter
            self._params = np.concatenate([[self.intercept_], self.coef_])

        self._df_resid = self._nobs - (X.shape[1] + 1)

    def _fit_smooth_solver(self, X, y, sample_weight, solver_name, backend_name):
        """Fit ordinary GLM with backend-native Newton or L-BFGS."""
        from statgpu.glm_core import get_glm_loss
        from statgpu.solvers import lbfgs_solver, newton_solver

        if sample_weight is not None:
            raise ValueError(
                f"solver='{solver_name}' does not support sample_weight yet; "
                "use solver='irls' or solver='fista'."
            )

        loss_kwargs = self._get_loss_kwargs()
        loss = get_glm_loss(self.family_to_loss(), **loss_kwargs)
        if not getattr(loss, "has_hessian", False):
            raise ValueError(f"solver='{solver_name}' requires a Hessian.")

        if self._effective_intercept:
            from statgpu.backends._utils import _get_xp
            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                x_dtype = X.dtype if getattr(X.dtype, "kind", "") == "f" else xp.float64
                X_float = X.astype(x_dtype, copy=False)
                X_work = xp.column_stack([X_float, xp.ones(X.shape[0], dtype=x_dtype)])
            elif backend_name == "torch":
                import torch
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_float = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
                X_work = torch.column_stack([
                    X_float,
                    torch.ones(X.shape[0], dtype=x_dtype, device=X.device),
                ])
            else:
                x_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
                X_float = X.astype(x_dtype, copy=False)
                X_work = np.column_stack([X_float, np.ones(X.shape[0], dtype=x_dtype)])
            p = X.shape[1]
        else:
            if backend_name == "torch":
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_work = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
            else:
                X_work = X
            p = X.shape[1]

        if solver_name == "newton":
            params, n_iter = newton_solver(
                loss, None, X_work, y, max_iter=self.max_iter, tol=self.tol
            )
        else:
            params, n_iter = lbfgs_solver(
                loss, None, X_work, y, max_iter=self.max_iter, tol=self.tol
            )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self._effective_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
        self._params = (
            np.concatenate([[self.intercept_], self.coef_])
            if self._effective_intercept
            else self.coef_.copy()
        )
        self._df_resid = self._nobs - (
            X.shape[1] + (1 if self._effective_intercept else 0)
        )

    def predict(self, X):
        """Predict using fitted model."""
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self._design_info is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None and isinstance(X, pd.DataFrame):
                from statgpu.core.formula import FormulaParser

                parser = FormulaParser.__new__(FormulaParser)
                parser._design_info = self._design_info
                parser.formula = None
                X = parser.transform(X)
                col_names = list(self._design_info.column_names)
                if self._formula_has_intercept and "Intercept" in col_names:
                    X = np.delete(X, col_names.index("Intercept"), axis=1)

        device = self._get_compute_device()
        family = self._get_family()
        from statgpu.backends._utils import _get_xp, xp_asarray
        if device in (Device.CUDA, Device.TORCH):
            backend_name = "cupy" if device == Device.CUDA else "torch"
            xp = _get_xp(backend_name)
            Xb = xp_asarray(self._to_array(X, device), xp=xp)
            coef = xp_asarray(self.coef_, xp=xp, ref_arr=Xb)
            # Ensure float dtype for matmul (CUDA doesn't support Long matmul)
            if hasattr(Xb, 'is_floating_point') and not Xb.is_floating_point():
                Xb = Xb.float()
            elif not hasattr(Xb, 'is_floating_point') and hasattr(Xb, 'dtype') and 'int' in str(Xb.dtype):
                Xb = xp_asarray(Xb, dtype=xp.float64, xp=xp)
            # Align dtypes for torch matmul compatibility
            if hasattr(Xb, 'dtype') and hasattr(coef, 'dtype') and Xb.dtype != coef.dtype:
                coef = coef.to(Xb.dtype) if hasattr(coef, 'to') else xp_asarray(coef, dtype=Xb.dtype, xp=xp)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + xp_asarray(self.intercept_, xp=xp, ref_arr=Xb)
            out = family.link.inverse(raw)
            if device == Device.CUDA:
                self._cleanup_cuda_memory()
            else:
                self._cleanup_torch_memory()
            return out

        X = np.asarray(X)
        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_
        return family.link.inverse(raw)


class OrderedGeneralizedLinearModel(GeneralizedLinearModel):
    """Ordered GLM base class.

    Jointly estimates coefficients + (K-1) thresholds.
    P(y <= j | X) = F(theta_j - X * beta)

    Parameters
    ----------
    n_categories : int, default=3
        Number of ordinal categories.
    family : str, default='binomial'
        Distribution family (should be Binomial for ordered models).
    ... : same as GeneralizedLinearModel
    """

    def __init__(
        self,
        n_categories: int = 3,
        family: str = "binomial",
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
    ):
        # Inference is supported via analytical Hessian in _compute_ordered_inference
        super().__init__(
            family=family,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            compute_inference=compute_inference,
            cov_type=cov_type,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )
        if n_categories < 2:
            raise ValueError(
                f"n_categories must be >= 2, got {n_categories}. "
                "Ordered models require at least 2 ordinal categories."
            )
        self.n_categories = n_categories
        self.thresholds_ = None

    def fit(self, X, y, sample_weight=None):
        """Fit ordered GLM using Newton-Raphson with analytical Hessian.

        Supports numpy (CPU), cupy (GPU), and torch (GPU) via a shared
        trust-region Newton implementation with backend-agnostic operations.
        """
        if sample_weight is not None:
            raise ValueError(
                "OrderedGeneralizedLinearModel does not support sample_weight yet."
            )

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        self._nobs = X.shape[0]

        # Convert to backend format (cupy→cupy zero-copy, numpy→cupy/torch)
        X = self._to_array(X, backend=backend_name)
        y = self._to_array(y, backend=backend_name)

        # Validate labels: must be integers in [0, n_categories)
        from statgpu.backends._utils import _get_xp
        xp = _get_xp(backend_name)
        y_flat = xp.asarray(y).ravel()
        y_min = int(xp.min(y_flat))
        y_max = int(xp.max(y_flat))
        K = self.n_categories
        if y_min < 0 or y_max >= K:
            raise ValueError(
                f"Ordered model labels must be integers in [0, {K - 1}], "
                f"got range [{y_min}, {y_max}]. "
                f"n_categories={K}."
            )
        if xp.any(y_flat != xp.floor(y_flat)):
            raise ValueError(
                "Ordered model labels must be integer-coded categories, "
                "not continuous values. Found non-integer labels."
            )

        family = self._get_family()
        n = X.shape[0]
        p = X.shape[1]

        try:
            if backend_name == "cupy":
                self._fit_cupy_ordered(X, y, family, K, n, p)
            elif backend_name == "torch":
                self._fit_torch_ordered(X, y, family, K, n, p)
            else:
                self._fit_scipy_ordered(X, y, family, K, n, p)

            self._df_resid = self._nobs - (p + K - 1)
            self._params = np.concatenate([self.coef_, self._thresh_est])

            if self.compute_inference:
                self._compute_ordered_inference(X, y)
            self._fitted = True
        finally:
            self._cleanup_backend_memory(backend_name)
        return self

    @property
    def loglikelihood(self):
        """Log-likelihood at the fitted parameters (ordered model)."""
        self._check_is_fitted()
        return -float(self._nobs) * float(self._final_nll)

    # -----------------------------------------------------------------
    # Shared Newton-Raphson trust-region (all 3 backends)
    # -----------------------------------------------------------------

    def _fit_ordered_newton_impl(self, X, y, family, K, n, p, xp, is_torch, is_cupy,
                                  dev=None):
        """Backend-agnostic Newton-Raphson with trust-region for ordered models.

        Parameters
        ----------
        X, y : arrays on the target backend (already converted by caller).
        family : GLMFamily
        K, n, p : int
        xp : module — numpy, cupy, or torch
        is_torch, is_cupy : bool
        dev : torch device or None
        """
        # ---- Standardization ----
        from statgpu.backends._array_ops import _clip
        if is_torch:
            X_mean = X.mean(dim=0)
            X_std = X.std(dim=0)
            X_std = xp.where(X_std < 1e-10, xp.ones_like(X_std), X_std)
        else:
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_std[X_std < 1e-10] = 1.0
        Xs = (X - X_mean) / X_std

        # ---- Initialisation ----
        if is_torch:
            theta = xp.zeros(p + K - 1, dtype=xp.float64, device=dev)
            theta[p:] = xp.arange(0.5, K - 0.5, dtype=xp.float64, device=dev)
            idx = xp.arange(n, device=dev)
        elif is_cupy:
            theta = xp.zeros(p + K - 1, dtype=xp.float64)
            theta[p:] = xp.arange(0.5, K - 0.5, dtype=xp.float64)
            idx = xp.arange(n)
        else:
            theta = xp.zeros(p + K - 1)
            theta[p:] = xp.arange(0.5, K - 0.5, dtype=xp.float64)
            idx = xp.arange(n)

        d = len(theta); nll_old = xp.inf; ridge = 1e-4

        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be > 0, got {self.max_iter}. "
                "Newton-Raphson requires at least 1 iteration."
            )

        # Pre-allocate identity matrix for trust-region (reused across attempts)
        if is_torch:
            eye_d = xp.eye(d, dtype=xp.float64, device=dev)
        elif is_cupy:
            eye_d = xp.eye(d, dtype=xp.float64)
        else:
            eye_d = xp.eye(d)

        # ---- Newton loop ----
        for iteration in range(self.max_iter):
            # Enforce strictly increasing thresholds with minimum gap
            thresh = xp.sort(theta[p:])
            if is_torch:
                thresh = thresh[0]  # torch.sort returns (values, indices)
            if len(thresh) > 1:
                gaps = xp.diff(thresh)
                if is_torch:
                    gaps = xp.clamp(gaps, min=1e-6)
                    thresh = xp.cat([thresh[:1], thresh[:1] + xp.cumsum(gaps, dim=0)])
                else:
                    gaps = xp.maximum(gaps, 1e-6)
                    thresh = xp.concatenate([thresh[:1], thresh[:1] + xp.cumsum(gaps)])
            theta = xp.concatenate([theta[:p], thresh])
            beta = theta[:p]; thresh = theta[p:]
            eta = Xs @ beta  # compute once, pass to all callees

            prob = self._ordered_category_probs(Xs, beta, thresh, family, K, eta=eta)
            prob_c = _clip(prob, 1e-15, None)
            if is_torch:
                nll = -xp.mean(xp.log(prob_c[y, idx]))
            else:
                nll = -xp.sum(xp.log(prob_c[y, idx])) / n

            # Gradient (torch uses its own device-aware path)
            if is_torch:
                grad = self._ordered_gradient_torch(
                    Xs, y, beta, thresh, prob, prob_c, family, K, n, eta=eta)
            else:
                grad = self._ordered_gradient(
                    Xs, y, beta, thresh, prob, prob_c, family, K, n, eta=eta)

            # Convergence: NLL-change + gradient-norm + isfinite guard
            if not xp.isfinite(nll):
                raise RuntimeError(
                    f"NLL became non-finite ({float(nll):.4g}) at iteration "
                    f"{iteration}. Coefficients may have diverged."
                )
            if iteration > 0 and abs(float(nll_old - nll)) < self.tol:
                break
            grad_inf = float(xp.max(xp.abs(grad)))
            if grad_inf < self.tol:
                break
            nll_old = nll

            # Hessian + trust-region
            H = self._ordered_hessian_analytical(
                Xs, y, beta, thresh, family, K, prob, prob_c, eta=eta)
            H_avg = H / n

            for attempt in range(20):
                H_reg = H_avg + ridge * eye_d
                # Catch linalg errors (singular matrix) only; OOM/programming
                # errors re-raise.  CuPy uses generic Exception for linalg.
                try:
                    delta = xp.linalg.solve(H_reg, -grad)
                except (np.linalg.LinAlgError, RuntimeError):
                    ridge *= 10; continue
                except Exception:
                    if is_cupy:
                        ridge *= 10; continue
                    raise

                theta_try = theta + delta
                # Enforce strictly increasing thresholds with minimum gap
                beta_t = theta_try[:p]
                thresh_t = xp.sort(theta_try[p:])
                if is_torch:
                    thresh_t = thresh_t[0]
                if len(thresh_t) > 1:
                    gaps = xp.diff(thresh_t)
                    if is_torch:
                        gaps = xp.clamp(gaps, min=1e-6)
                        thresh_t = xp.cat([thresh_t[:1], thresh_t[:1] + xp.cumsum(gaps, dim=0)])
                    else:
                        gaps = xp.maximum(gaps, 1e-6)
                        thresh_t = xp.concatenate([thresh_t[:1], thresh_t[:1] + xp.cumsum(gaps)])
                theta_try = xp.concatenate([beta_t, thresh_t])
                prob_t = self._ordered_category_probs(Xs, beta_t, thresh_t, family, K)
                pc_t = _clip(prob_t, 1e-15, None)
                if is_torch:
                    nll_try = -xp.mean(xp.log(pc_t[y, idx]))
                else:
                    nll_try = -xp.sum(xp.log(pc_t[y, idx])) / n
                if float(nll_try) < float(nll):
                    ridge *= 0.5; break
                ridge *= 2.0
            else:
                break
            theta = theta_try
            nll = nll_try  # keep NLL in sync with accepted theta

        # ---- Extract results to CPU ----
        self.n_iter_ = iteration + 1
        self._final_nll = float(nll)

        if is_torch:
            beta_scaled = theta[:p]
            self.coef_ = (beta_scaled / X_std).cpu().numpy()
            thresh_est = xp.sort(theta[p:])[0]
            intercept_adj = float(((X_mean / X_std) * beta_scaled).sum().cpu())
            th_est = thresh_est.cpu().numpy()
            self._beta_scaled = beta_scaled.cpu().numpy().copy()
            self._thresh_est = th_est + intercept_adj
            self._X_mean = X_mean.cpu().numpy()
            self._X_std = X_std.cpu().numpy()
        elif is_cupy:
            beta_scaled = theta[:p]
            self.coef_ = (beta_scaled / X_std).get()
            thresh_est = xp.sort(theta[p:])
            self._beta_scaled = beta_scaled.get().copy()
            intercept_adj = float((X_mean / X_std * beta_scaled).sum().get())
            self._thresh_est = thresh_est.get() + intercept_adj
            self._X_mean = X_mean.get()
            self._X_std = X_std.get()
        else:
            beta_scaled = theta[:p]
            self.coef_ = beta_scaled / X_std
            thresh_est = np.sort(theta[p:])
            self._beta_scaled = beta_scaled.copy()
            intercept_adj = float(X_mean @ self.coef_)
            self._thresh_est = thresh_est + intercept_adj
            self._X_mean = X_mean
            self._X_std = X_std

        self.thresholds_ = np.concatenate([[-np.inf], self._thresh_est, [np.inf]])

    def _fit_scipy_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using NumPy Newton-Raphson."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._fit_ordered_newton_impl(X, y, family, K, n, p, np,
                                       is_torch=False, is_cupy=False)

    def _fit_cupy_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using CuPy Newton-Raphson."""
        import cupy as cp
        X = cp.asarray(X, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.int64)
        self._fit_ordered_newton_impl(X, y, family, K, n, p, cp,
                                       is_torch=False, is_cupy=True)

    def _fit_torch_ordered(self, X, y, family, K, n, p):
        """Fit ordered GLM using Torch Newton-Raphson."""
        import torch
        assert isinstance(X, torch.Tensor)
        dev = X.device
        if X.dtype != torch.float64: X = X.to(torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(dev)
        elif y.dtype != torch.int64:
            y = y.to(torch.int64)
        self._fit_ordered_newton_impl(X, y, family, K, n, p, torch,
                                       is_torch=True, is_cupy=False, dev=dev)

    def _ordered_category_probs(self, X, beta, thresh, family, K, eta=None):
        """Compute category probabilities P(y=j|X), shape (K, n)."""
        if eta is None:
            eta = X @ beta  # (n,)
        pi = family.link.inverse(thresh[:, None] - eta[None, :])  # (K-1, n)

        # Use native array module for dtype compatibility (numpy/cupy/torch)
        dt = getattr(X, 'dtype', None)
        is_torch = _is_torch_array(X)
        if is_torch:
            import torch
            prob = torch.zeros((K, X.shape[0]), dtype=dt, device=X.device)
        else:
            xp = _np_compat_xp(X)
            prob = xp.zeros((K, X.shape[0]), dtype=dt)
        prob[0] = pi[0]
        for j in range(1, K - 1):
            prob[j] = pi[j] - pi[j - 1]
        prob[K - 1] = 1.0 - pi[K - 2]
        return prob

    # -----------------------------------------------------------------
    # Ordered model inference
    # -----------------------------------------------------------------

    def _compute_ordered_inference(self, X_orig, y_orig):
        """Backend-aware analytical Hessian inference for ordered models.

        Works with NumPy, CuPy, and Torch arrays.  Uses the vectorized
        ``_ordered_hessian_analytical`` and backend-native linalg + distributions.
        """
        # Only nonrobust covariance is supported for ordered models
        cov_type = self.cov_type.lower()
        if cov_type not in ("nonrobust",):
            raise NotImplementedError(
                f"Ordered model inference only supports cov_type='nonrobust', "
                f"got '{self.cov_type}'. HC0/HC1 sandwich and penalized "
                f"inference are not yet available for ordered models."
            )

        import numpy as np
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_eye
        from statgpu.inference._distributions_backend import get_distribution

        backend = _resolve_backend("auto", X_orig)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")
        is_cupy = (backend == "cupy")

        # Keep arrays on native backend; convert y to int
        X_raw = xp.asarray(X_orig, dtype=xp.float64)
        y = xp.asarray(y_orig, dtype=xp.int64 if not is_torch else None)
        if is_torch:
            y = y.to(xp.int64) if y.dtype != xp.int64 else y
        y = y.ravel()
        n, p = X_raw.shape
        K = self.n_categories; n_thresh = K - 1; d = p + n_thresh
        family = self._get_family()

        # Raw-scale parameters (on same device as X for torch)
        if is_torch:
            beta = xp.asarray(self.coef_, dtype=xp.float64, device=X_raw.device)
            thresh = xp.asarray(self._thresh_est, dtype=xp.float64, device=X_raw.device)
        else:
            beta = xp.asarray(self.coef_, dtype=xp.float64)
            thresh = xp.asarray(self._thresh_est, dtype=xp.float64)

        # Vectorized analytical Hessian
        prob = self._ordered_category_probs(X_raw, beta, thresh, family, K)
        from statgpu.backends._array_ops import _clip
        prob_c = _clip(prob, 1e-15, None)
        H = self._ordered_hessian_analytical(X_raw, y, beta, thresh, family, K, prob, prob_c)

        # Covariance = H^{-1} (strict: raise on singular)
        eye = xp_eye(d, xp.float64, xp, ref_arr=H)
        try:
            H_inv = xp.linalg.solve(H, eye)
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise np.linalg.LinAlgError(
                "Ordered model Hessian is singular — cannot compute standard errors. "
                "This may indicate quasi-complete separation or redundant thresholds. "
                "Consider using inference_method='bootstrap' or reducing n_categories."
            ) from e
        except Exception as e:
            if is_cupy:
                raise np.linalg.LinAlgError(
                    "Ordered model Hessian is singular — cannot compute standard errors. "
                    "This may indicate quasi-complete separation or redundant thresholds. "
                    "Consider using inference_method='bootstrap' or reducing n_categories."
                ) from e
            raise
        cov = H_inv

        # Backend-aware distribution functions
        norm_dist = get_distribution("norm", backend=backend)
        params = xp.concatenate([beta, thresh])

        if is_torch:
            bse = xp.sqrt(xp.clamp(xp.diag(cov), min=0.0))
        else:
            bse = xp.sqrt(xp.maximum(xp.diag(cov), 0.0))
        z_values = params / (bse + 1e-30)
        pvalues = 2.0 * norm_dist.sf(xp.abs(z_values))
        z_crit = norm_dist.ppf(0.975)
        conf_int = xp.column_stack([
            params - z_crit * bse,
            params + z_crit * bse,
        ])

        # Convert to CPU numpy for storage
        bse_cpu = _to_numpy(bse)
        z_cpu = _to_numpy(z_values)
        p_cpu = _to_numpy(pvalues)
        ci_cpu = _to_numpy(conf_int)
        params_cpu = _to_numpy(params)
        beta_cpu = _to_numpy(beta)
        thresh_cpu = _to_numpy(thresh)

        # Store flat arrays (matching parent GLM contract).
        # Users access coef-SEs via _bse[:p], threshold-SEs via _bse[p:].
        self._bse = bse_cpu
        self._zvalues = z_cpu
        self._pvalues = p_cpu
        self._conf_int = ci_cpu
        self._params = np.concatenate([beta_cpu, thresh_cpu])

        from statgpu.inference._results import ParameterInferenceResult
        feat_names = [f"coef_{i}" for i in range(p)] + [f"thresh_{j}" for j in range(n_thresh)]
        self._inference_result = ParameterInferenceResult(
            method="analytical_hessian",
            params=self._params.copy(),
            bse=self._bse.copy(),
            statistic=self._zvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            feature_names=feat_names,
            metadata={"method": "analytical", "n_thresholds": n_thresh,
                       "backend": backend},
        )
        self._inference_result.apply_to(self)

    def _ordered_hessian_analytical(self, X, y, beta, thresh, family, K, prob, prob_c, eta=None):
        """Vectorized analytical observed Hessian. Backend-agnostic (numpy/cupy/torch).

        All operations are fully vectorized — zero per-row or per-category Python
        loops.  Pre-computed (K, n) category mask matrix eliminates repeated
        ``y == k`` mask creation.
        """
        xp = _ordered_xp(X)
        is_torch = _is_torch_array(X)
        dev = X.device if is_torch else None
        p = len(beta); n_thresh = len(thresh); d = p + n_thresh; n = X.shape[0]
        if is_torch:
            _z = lambda sz: xp.zeros(sz, dtype=X.dtype, device=dev)
        else:
            _z = lambda sz: xp.zeros(sz)

        # ---- f and fp (fully vectorized over thresholds) ----
        if eta is None:
            eta = X @ beta
        diff = thresh[:, None] - eta[None, :]  # (n_thresh, n)
        import math as _math
        _sqrt2pi = _math.sqrt(2.0 * _math.pi)
        is_probit = 'probit' in str(type(family.link)).lower()
        if is_probit:
            f_all = xp.exp(-0.5 * diff * diff) / _sqrt2pi
            fp_all = -diff * f_all
        else:
            from statgpu.backends._array_ops import _sigmoid
            F_all = _sigmoid(diff)
            f_all = F_all * (1.0 - F_all)
            fp_all = f_all * (1.0 - 2.0 * F_all)

        # ---- Pre-computed category mask matrix (K, n) — single broadcast ----
        if is_torch:
            y_cat = (y[None, :] == xp.arange(K, device=dev)[:, None])
        else:
            y_cat = (y[None, :] == xp.arange(K)[:, None])

        # ---- a_vec and w_bb (fused single K-loop) ----
        a_vec = _z(n)
        pv_vec = prob_c[y, xp.arange(n, device=dev) if is_torch else xp.arange(n)]
        w_bb = _z(n)
        for k_val in range(K):
            mask = y_cat[k_val]
            if not mask.any():
                continue
            fk = f_all[k_val, mask] if k_val < n_thresh else _z(int(mask.sum()))
            fk1 = f_all[k_val - 1, mask] if k_val > 0 else _z(int(mask.sum()))
            a = fk - fk1
            a_vec[mask] = a
            fpk = fp_all[k_val, mask] if k_val < n_thresh else _z(int(mask.sum()))
            fpk1 = fp_all[k_val - 1, mask] if k_val > 0 else _z(int(mask.sum()))
            pv = pv_vec[mask]
            w_bb[mask] = a * a / (pv * pv) - (fpk - fpk1) / pv

        H = xp.zeros((d, d), dtype=X.dtype) if not is_torch else xp.zeros((d, d), dtype=X.dtype, device=dev)
        H[:p, :p] = (X * w_bb[:, None]).T @ X

        # ---- Beta-theta cross terms ----
        for j in range(n_thresh):
            w_bth = _z(n)
            f_j, fp_j = f_all[j], fp_all[j]
            mk = y_cat[j]
            if mk.any():
                pv = pv_vec[mk]; a = a_vec[mk]
                w_bth[mk] = fp_j[mk] / pv - a * f_j[mk] / (pv * pv)
            if j + 1 < K:
                mk1 = y_cat[j + 1]
                if mk1.any():
                    pv1 = pv_vec[mk1]; a1 = a_vec[mk1]
                    w_bth[mk1] = a1 * f_j[mk1] / (pv1 * pv1) - fp_j[mk1] / pv1
            H[:p, p + j] = X.T @ w_bth
            H[p + j, :p] = H[:p, p + j]

        # ---- Theta-theta block ----
        for k_val in range(n_thresh):
            mk = y_cat[k_val]
            if mk.any():
                pv = pv_vec[mk]; fk = f_all[k_val, mk]; fpk = fp_all[k_val, mk]
                H[p + k_val, p + k_val] += xp.sum(fk * fk / (pv * pv) - fpk / pv)
            mk1 = y_cat[k_val + 1]
            if mk1.any():
                pv1 = pv_vec[mk1]; fk1 = f_all[k_val, mk1]; fpk1 = fp_all[k_val, mk1]
                H[p + k_val, p + k_val] += xp.sum(fk1 * fk1 / (pv1 * pv1) + fpk1 / pv1)
            if k_val + 1 < n_thresh:
                mc = y_cat[k_val + 1]
                if mc.any():
                    pvc = pv_vec[mc]; fk_c = f_all[k_val, mc]; fk1_c = f_all[k_val + 1, mc]
                    cross = -xp.sum(fk_c * fk1_c / (pvc * pvc))
                    H[p + k_val, p + k_val + 1] += cross
                    H[p + k_val + 1, p + k_val] += cross

        return H

    def _ordered_gradient(self, X, y, beta, thresh, prob, prob_clipped, family, K, n, eta=None):
        """Compute analytical gradient of the negative log-likelihood (vectorized)."""
        xp = _np_compat_xp(X)
        p = X.shape[1]
        n_thresh = K - 1
        dim = p + n_thresh
        grad = xp.zeros(dim)

        if eta is None:
            eta = X @ beta  # (n,)

        # Link derivative at all threshold positions: shape (n_thresh, n)
        diff = thresh[:, None] - eta[None, :]  # (n_thresh, n)
        deriv_all = xp.empty_like(diff)
        for j in range(n_thresh):
            deriv_all[j] = self._ordered_link_derivative(diff[j], family)

        # inv_prob[i] = 1 / P(y[i] | X[i]), shape (n,)
        inv_prob = 1.0 / prob_clipped[y, xp.arange(n)]  # (n,)

        # dP_dthresh contribution for each (j, i):
        #   +deriv_all[j, i] if j == y[i]
        #   -deriv_all[j, i] if j == y[i] - 1
        # Vectorized: for each j, count how many samples have y==j (positive)
        # and y==j+1 (negative).
        dP_dthresh_j = xp.zeros(n_thresh)
        for j in range(n_thresh):
            mask_pos = (y == j)
            mask_neg = (y == j + 1)
            dP_dthresh_j[j] = xp.sum(inv_prob * (deriv_all[j] * mask_pos - deriv_all[j] * mask_neg))

        grad[p:] -= dP_dthresh_j / n

        # dP_dbeta for sample i: X[i] * scalar_i
        # scalar_i = -(deriv_all[0, i]) if y[i]==0
        #           (deriv_all[y[i]-1, i] - deriv_all[y[i], i]) if 0 < y[i] < K-1
        #           (deriv_all[n_thresh-1, i]) if y[i]==K-1
        scalar = xp.empty(n)
        mask0 = (y == 0)
        mask_last = (y == K - 1)
        mask_mid = ~mask0 & ~mask_last
        scalar[mask0] = -deriv_all[0, mask0]
        scalar[mask_last] = deriv_all[n_thresh - 1, mask_last]
        # For middle: deriv[y[i]-1] - deriv[y[i]]
        idx_mid = xp.where(mask_mid)[0]
        scalar[idx_mid] = (deriv_all[y[idx_mid] - 1, idx_mid]
                           - deriv_all[y[idx_mid], idx_mid])

        grad[:p] -= X.T @ (inv_prob * scalar) / n

        return grad

    def _ordered_gradient_torch(self, X, y, beta, thresh, prob, prob_clipped, family, K, n, eta=None):
        """Torch-native gradient of NLL for ordered model."""
        import torch
        d = len(beta) + len(thresh); p = len(beta); n_thresh = len(thresh)
        grad = torch.zeros(d, dtype=X.dtype, device=X.device)
        inv_p = 1.0 / prob_clipped[y, torch.arange(n, device=X.device)]
        if eta is None:
            eta = X @ beta
        diff = thresh[:, None] - eta[None, :]
        d_all = torch.empty_like(diff)
        for j in range(n_thresh):
            d_all[j] = self._ordered_link_derivative(diff[j], family)
        for j in range(n_thresh):
            mp = (y == j); mn = (y == j + 1)
            grad[p + j] = -torch.sum(inv_p * (d_all[j] * mp - d_all[j] * mn)) / n
        scalar = torch.zeros(n, dtype=X.dtype, device=X.device)
        mask0 = (y == 0); mask_last = (y == K - 1); mask_mid = ~mask0 & ~mask_last
        scalar[mask0] = -d_all[0, mask0]
        scalar[mask_last] = d_all[n_thresh - 1, mask_last]
        idx_mid = torch.where(mask_mid)[0]
        scalar[idx_mid] = d_all[y[idx_mid] - 1, idx_mid] - d_all[y[idx_mid], idx_mid]
        grad[:p] = -X.T @ (inv_p * scalar) / n
        return grad

    def _ordered_link_derivative(self, x, family):
        """First derivative of link inverse F'(x) = density at x.

        For logit: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
        For probit: normal PDF φ(x).
        Both paths are backend-agnostic (numpy/cupy/torch).
        """
        if family.link.name == "probit":
            from statgpu.backends._array_ops import _xp, _exp, _scalar_tensor
            xp = _xp(x)
            two_pi = _scalar_tensor(2.0 * np.pi, x)
            return _exp(-0.5 * x * x) / xp.sqrt(two_pi)
        # logit: F * (1 - F) — element-wise, works for any backend
        F = family.link.inverse(x)
        return F * (1.0 - F)

    def predict_proba(self, X):
        """Predict class probabilities P(y=j|X).

        Backend-agnostic: uses the same backend that was used during fit().
        Returns a NumPy array by convention (small output, consumed on CPU).
        """
        self._check_is_fitted()
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = self.n_categories

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        X_arr = self._to_array(X, backend=backend_name)

        # Guard: integer X causes torch matmul to fail (matching parent GLM.predict)
        if hasattr(X_arr, 'is_floating_point') and not X_arr.is_floating_point():
            X_arr = X_arr.float()

        from statgpu.backends._utils import _get_xp, xp_asarray
        xp = _get_xp(backend_name)
        is_torch = _is_torch_array(X_arr)
        coef = xp_asarray(self.coef_, xp=xp, ref_arr=X_arr)
        # coef_ is already on raw (unstandardized) scale:
        #   coef_ = beta_fit / X_std
        # Thresholds are also on raw scale:
        #   _thresh_est = theta_fit + X_mean @ coef_
        # So linear predictor is simply X @ coef (no standardization needed).
        thresholds = xp_asarray(self.thresholds_, xp=xp, ref_arr=X_arr)
        eta = X_arr @ coef
        family = self._get_family()
        diff = thresholds[:, None] - eta[None, :]
        pi = family.link.inverse(diff)  # (K+1, n) with -inf/+inf thresholds

        if is_torch:
            proba = xp.diff(pi, dim=0).T  # (n, K)
        else:
            proba = xp.diff(pi, axis=0).T  # (n, K)
        if backend_name != "numpy":
            out = _to_numpy(proba)
            self._cleanup_backend_memory(backend_name)
            return out
        return proba

    def predict(self, X):
        """Predict class labels.

        Backend-agnostic: computes argmax on the native backend, returns NumPy.
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y):
        """Return mean accuracy on the given test data and labels.

        Uses the same backend as fit() for the computation.
        """
        self._check_is_fitted()

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        y_true = self._to_array(y, backend=backend_name)
        y_pred = self.predict(X)
        y_pred_arr = self._to_array(y_pred, backend=backend_name)

        from statgpu.backends._utils import _get_xp, _to_float_scalar
        xp = _get_xp(backend_name)
        matches = xp.asarray(y_pred_arr == y_true, dtype=xp.float64)
        out = _to_float_scalar(xp.mean(matches))
        if backend_name != "numpy":
            self._cleanup_backend_memory(backend_name)
        return out
