"""Core PenalizedGeneralizedLinearModel class and SelectivePenalty.

This module contains the class definition, __init__, and core utility methods.
Fit, inference, and predict methods live in separate mixin modules.
"""

from __future__ import annotations

__all__ = ["PenalizedGeneralizedLinearModel", "SelectivePenalty"]

from contextlib import nullcontext
from typing import Optional, Union, Dict, TYPE_CHECKING
import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy
from statgpu.cross_validation._base import INTERCEPT_CLIP_BOUND as _INTERCEPT_CLIP_BOUND
from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
    validate_cov_type,
    validate_hac_maxlags,
)
from statgpu.penalties._categories import NONSMOOTH as _NONSMOOTH_PENALTIES

from ._fit_mixin import _PenalizedFitMixin
from ._inference_mixin import _PenalizedInferenceMixin
from ._predict_mixin import _PenalizedPredictMixin

if TYPE_CHECKING:
    from statgpu.penalties import Penalty


class SelectivePenalty:
    """Penalty wrapper that leaves the last intercept coefficient free.

    Created once per fit and reused across iterations. The inner penalty,
    feature count p, and backend are set via ``configure()``.
    """

    def __init__(self):
        self._pen = None
        self._p = 0
        self._backend = "numpy"
        self._alpha = 0.0
        self._l1_ratio = 0.0

    def configure(self, pen, p, backend):
        self._pen = pen
        self._p = p
        self._backend = backend
        self._alpha = float(getattr(pen, "alpha", 0.0))
        self._l1_ratio = float(getattr(pen, "l1_ratio", 0.0))
        self.name = pen.name

    def value(self, coef):
        return self._pen.value(coef[:self._p])

    def proximal(self, w, step, backend=None):
        b = backend or self._backend
        w_feat = w[:self._p]
        result_feat = self._pen.proximal(w_feat, step, backend=b)
        if b == "cupy":
            import cupy as cp
            result = cp.empty_like(w)
            result[:self._p] = result_feat
            result[-1] = cp.clip(
                w[-1], -_INTERCEPT_CLIP_BOUND, _INTERCEPT_CLIP_BOUND
            )
        elif b == "torch":
            import torch
            result = torch.empty(w.shape[0], dtype=w.dtype, device=w.device)
            result[:self._p] = result_feat
            result[-1] = torch.clamp(w[-1], -_INTERCEPT_CLIP_BOUND, _INTERCEPT_CLIP_BOUND)
        else:
            result = np.empty(w.shape[0], dtype=w.dtype)
            result[:self._p] = result_feat
            result[-1] = np.clip(w[-1], -_INTERCEPT_CLIP_BOUND, _INTERCEPT_CLIP_BOUND)
        return result

    def _smooth_alpha(self):
        pname = str(self._pen.name).lower()
        if pname == "l2":
            return self._alpha
        if pname == "elasticnet":
            return self._alpha * (1.0 - self._l1_ratio)
        raise ValueError("smooth solvers only support L2/ElasticNet penalties.")

    def smooth_value(self, coef):
        sa = self._smooth_alpha()
        active = coef[:self._p]
        if self._backend == "cupy":
            import cupy as cp
            return 0.5 * sa * cp.sum(active * active)
        if self._backend == "torch":
            import torch
            return 0.5 * sa * torch.sum(active * active)
        return 0.5 * sa * np.sum(active * active)

    def smooth_gradient(self, coef):
        sa = self._smooth_alpha()
        if self._backend == "cupy":
            import cupy as cp
            grad = cp.zeros_like(coef)
        elif self._backend == "torch":
            import torch
            grad = torch.zeros_like(coef)
        else:
            grad = np.zeros_like(coef)
        grad[:self._p] = sa * coef[:self._p]
        return grad

    def smooth_hessian(self, coef):
        """Return smooth penalty Hessian as a dense diagonal matrix.

        WARNING: For p > ~1000, this allocates O(p^2) memory which may cause
        OOM. Consider using the diagonal representation directly when available.
        """
        sa = self._smooth_alpha()
        if self._backend == "cupy":
            import cupy as cp
            diag = cp.zeros_like(coef)
            diag[:self._p] = sa
            return cp.diag(diag)
        if self._backend == "torch":
            import torch
            diag = torch.zeros(coef.shape[0], dtype=coef.dtype, device=coef.device)
            diag[:self._p] = sa
            return torch.diag(diag)
        diag = np.zeros(coef.shape[0], dtype=coef.dtype)
        diag[:self._p] = sa
        return np.diag(diag)



class PenalizedGeneralizedLinearModel(
    _PenalizedFitMixin,
    _PenalizedInferenceMixin,
    _PenalizedPredictMixin,
    BaseEstimator,
):
    """Penalized generalized linear model with a loss and penalty.

    Minimize the average loss plus alpha times the declared slope penalty;
    the fitted intercept is unpenalized. Analytic weights normalize the data-fit
    term by their sum. Solver and inference support depend on the combination.

    Parameters
    ----------
    loss : str, default='squared_error'
        GLM loss name, e.g. squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie; other loss families have separate compatibility limits.
    penalty : str or Penalty, default='l1'
        none, l1, l2, elasticnet, scad, mcp, adaptive_l1, supported group penalties, or a Penalty object.
    alpha : float, default=1.0
        Penalty strength on the average-loss scale. A supplied Penalty object owns its own configuration.
    l1_ratio : float, default=0.5
        L1 fraction for elasticnet.
    penalty_kwargs : dict or None, default=None
        Additional penalty constructor settings, e.g. groups or shape controls.
    fit_intercept : bool, default=True
        Unpenalized intercept; formula syntax takes precedence.
    max_iter : int, default=1000
        Per-solve iteration budget.
    tol : float, default=0.0001
        Numerical tolerance.
    device : str or Device, default='auto'
        cpu, cuda, torch, auto; see the backend guide.
    n_jobs : int or None, default=None
        Shared CPU-worker setting where used.
    cpu_solver : str, default='fista'
        Deprecated compatibility control; use solver. See the solver migration guide.
    solver : str, default='auto'
        Backend-neutral solver request; supported choices depend on loss and penalty.
    lipschitz_L : float or None, default=None
        Optional Lipschitz bound for compatible proximal paths.
    gpu_memory_cleanup : bool, default=False
        Best-effort GPU memory-pool cleanup.
    compute_inference : bool, default=False
        Run supported post-fit inference only when True.
    inference_method : str, default='auto'
        Resolve a supported method from loss/penalty; consult the inference matrix.
    cov_type : str, default='nonrobust'
        Method-specific covariance; non-Gaussian smooth inference supports nonrobust/hc0/hc1.
    hac_maxlags : int or None, default=None
        HAC lag control only on paths supporting HAC.
    stopping : str, default='coef_delta'
        Stored convergence request; direct sparse Gaussian fits currently ignore the kkt choice.
    lla : bool, default=True
        Enable local linear approximation for supported nonconvex penalties.
    max_lla_iters : int, default=50
        Maximum outer LLA iterations.
    lla_tol : float, default=1e-06
        Outer LLA convergence tolerance.
    loss_kwargs : dict or None, default=None
        Loss-specific controls, e.g. link, Negative Binomial dispersion alpha, or Tweedie power.
    nodewise_alpha : float or None, default=None
        Keyword-only tuning for supported debiased precision estimation; not the fit penalty.

    Methods
    -------
    fit(X=None, y=None, sample_weight=None, formula=None, data=None)
        Return self; numeric X has shape (n, p), scalar-response y has shape (n,).
    predict(X, return_cpu=True)
        Return (m,) predictions; default NumPy, native fitted backend when False.
        Logistic returns 0/1 labels (probability strictly above 0.5 selects 1),
        while other GLMs return response means.
    score(X, y, sample_weight=None)
        Response-scale R-squared, including on logistic labels; not accuracy or
        deviance pseudo-R-squared. Pass one-dimensional y and validate finite,
        nonnegative weights with positive sum; shared score checks are incomplete.
    get_params(deep=True), set_params(**params)
        Shared estimator configuration methods.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Shared statistical helpers, not automatic GLM resampling/refitting.

    Notes
    -----
    The generic class has no predict_proba or summary method. Typed wrappers can
    provide additional methods. Read supported coefficient inference through
    _inference_result.to_dict(), or to_dataframe() when pandas is installed.
    Prediction coef_ and intercept_ need not equal corrected/refitted inference
    parameters. The inference guide defines each supported statistical target.
    There are no generic n_bootstrap, bootstrap_random_state or initial_coef
    constructor parameters; specialized wrapper controls are not interchangeable.
    """

    def __init__(
        self,
        loss: str = "squared_error",
        penalty: Union[str, "Penalty"] = "l1",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[Dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
        loss_kwargs: Optional[Dict] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty_kwargs = (
            penalty_kwargs if penalty_kwargs is not None else {}
        )
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        # Preserve original string identity for sklearn clone() compatibility
        _cpu_solver = cpu_solver.lower()
        self.cpu_solver = cpu_solver if cpu_solver == _cpu_solver else _cpu_solver
        _solver = solver.lower()
        self.solver = solver if solver == _solver else _solver
        self.lipschitz_L = lipschitz_L
        self.gpu_memory_cleanup = gpu_memory_cleanup
        self.compute_inference = compute_inference
        _inference_method = inference_method.lower()
        self.inference_method = inference_method if inference_method == _inference_method else _inference_method
        self.cov_type = validate_cov_type(cov_type)
        self.hac_maxlags = validate_hac_maxlags(hac_maxlags)
        # Preserve original object identity for sklearn clone() compatibility
        _stopping = str(stopping).lower()
        self.stopping = stopping if stopping == _stopping else _stopping
        self.lla = lla
        self.max_lla_iters = max_lla_iters
        self.lla_tol = lla_tol
        self.loss_kwargs = loss_kwargs if loss_kwargs is not None else {}

        # Internal state
        self._penalty: Optional["Penalty"] = None
        self._lla_enabled = lla
        self._max_lla_iters = max_lla_iters
        self._lla_tol = lla_tol
        self._lla_n_iters_ = 0
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._bse = None
        self._tvalues = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._selected_solver = None
        self._selected_backend_name = None
        self._selected_backend_device = None
        self._init_coef = None
        self._inference_precomputed = False
        self._precomputed_gaussian_state = None
        self._native_fit_coef = None
        self._native_fit_intercept = None
        # Simultaneous inference state
        self._conf_int_simultaneous = None
        self._simultaneous_enabled = False
        self._debiased_M_cpu = None
        self._use_intercept = None  # formula-derived override; None = use fit_intercept

    @property
    def _effective_intercept(self):
        """Return effective intercept flag. Formula path overrides via _use_intercept."""
        if self._use_intercept is not None:
            return self._use_intercept
        return self._fit_intercept

    @staticmethod
    def _reporting_array(value):
        """Take the allowed post-inference NumPy reporting snapshot."""
        return np.asarray(_to_numpy(value), dtype=float)

    @staticmethod
    def _cupy_linalg_errstate():
        """Enable CuPy solver-status errors while tolerating lightweight test doubles."""
        import cupyx

        errstate = getattr(cupyx, "errstate", None)
        return nullcontext() if errstate is None else errstate(linalg="raise")

    def _apply_gaussian_reporting_state(self, state):
        """Populate established reporting fields after numerical inference."""
        self._X_design = self._reporting_array(state.X_design)
        self._y = self._reporting_array(state.y)
        self._resid = self._reporting_array(state.resid)
        self._params = self._reporting_array(state.params)
        scale = self._reporting_array(state.scale)
        self._scale = float(scale) if scale.ndim == 0 else scale
        self._nobs = int(state.nobs)
        self._df_resid = int(state.df_resid)

    def _solve_exact_cupy(self, XtX, Xty, normalization):
        """Run the existing exact-L2 CuPy solve with cuSOLVER errors enabled."""
        # CuPy>=13 exposes cupyx.errstate. The null context is only for the
        # lightweight maintenance test double that exercises exception routing.
        with self._cupy_linalg_errstate():
            return _PenalizedFitMixin._solve_exact_cupy(
                self, XtX, Xty, normalization
            )

    def _precompute_exact_l2_inference_cupy(self, *args, **kwargs):
        """Run existing exact-L2 inference with cuSOLVER errors enabled."""
        with self._cupy_linalg_errstate():
            return _PenalizedInferenceMixin._precompute_exact_l2_inference_cupy(
                self, *args, **kwargs
            )

    def _compute_post_fit_gaussian_inference(self, X, y, sample_weight=None):
        """Run squared-error L2 inference on the selected fit backend.

        This class-level router covers both the public generic
        ``PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l2')``
        entry point and typed linear/Ridge wrappers. Other penalty/loss branches
        remain owned by ``_PenalizedInferenceMixin``.
        """
        if not self._compute_inference_enabled:
            return

        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        precomputed = bool(getattr(self, "_inference_precomputed", False))
        if self.loss != "squared_error" or penalty_name != "l2" or precomputed:
            _PenalizedInferenceMixin._compute_post_fit_gaussian_inference(
                self, X, y, sample_weight=sample_weight
            )
            if precomputed and self._inference_result is not None:
                metadata = self._inference_result.metadata
                selected_backend = getattr(self, "_selected_backend_name", None)
                if selected_backend not in ("numpy", "cupy", "torch"):
                    raise RuntimeError(
                        "Gaussian inference is missing valid executed-backend provenance; "
                        "refusing to report a silent NumPy fallback."
                    )
                metadata.setdefault("numerical_backend", selected_backend)
                metadata.setdefault("reporting_backend", "numpy")
                metadata.setdefault(
                    "reporting_boundary", "post_numerical_inference"
                )
            return

        selected_backend = getattr(self, "_selected_backend_name", None)
        if selected_backend is None:
            raise RuntimeError(
                "Gaussian inference requires executed-backend provenance from fit; "
                "_selected_backend_name is missing."
            )
        backend_name = str(selected_backend).lower()
        if backend_name not in ("numpy", "cupy", "torch"):
            raise RuntimeError(
                f"Unsupported executed backend provenance {selected_backend!r}; "
                "refusing to fall back to NumPy inference."
            )

        # Generic fit may pass original NumPy inputs here even after a GPU fit.
        # The executed fit backend *and concrete device*, not the input container
        # type or ambient CUDA state, are authoritative.
        selected_device = getattr(self, "_selected_backend_device", None)
        device_label = str(selected_device or "")
        if backend_name == "numpy":
            selected_device = "cpu"
        elif backend_name == "cupy":
            if not device_label.startswith("cuda:"):
                raise RuntimeError(
                    "Gaussian inference is missing concrete executed-device provenance "
                    f"for backend={backend_name!r}: {selected_device!r}"
                )
        elif device_label != "cpu" and not device_label.startswith("cuda:"):
            raise RuntimeError(
                "Gaussian inference is missing concrete executed-device provenance "
                f"for backend={backend_name!r}: {selected_device!r}"
            )
        if backend_name in ("cupy", "torch"):
            coef_for_inference = getattr(self, "_native_fit_coef", None)
            intercept_for_inference = getattr(self, "_native_fit_intercept", None)
            if coef_for_inference is None:
                raise RuntimeError(
                    "Gaussian GPU inference requires native fitted parameters; "
                    "refusing a reporting-array host round trip."
                )
            if self._effective_intercept and intercept_for_inference is None:
                raise RuntimeError(
                    "Gaussian GPU inference is missing the native fitted intercept."
                )
        else:
            coef_for_inference = self.coef_
            intercept_for_inference = self.intercept_

        def _run_gaussian_inference_on_fit_device():
            state = build_gaussian_fit_state(
                X,
                y,
                coef_for_inference,
                intercept_for_inference if self._effective_intercept else 0.0,
                self._effective_intercept,
                sample_weight=sample_weight,
                backend=backend_name,
                device=selected_device,
            )
            ridge_alpha = state.normalization * self._ridge_alpha_for_exact()
            result = compute_gaussian_inference(
                state.X_design,
                state.params,
                state.resid,
                state.scale,
                state.df_resid,
                self._cov_type,
                hac_maxlags=self._hac_maxlags,
                ridge_alpha=ridge_alpha,
                ridge_penalize_intercept=(
                    False if self._effective_intercept else True
                ),
                backend=backend_name,
                device=state.device,
            )
            return state, result

        if backend_name == "cupy":
            import cupy as cp

            cupy_device_id = int(str(selected_device).split(":", 1)[1])
            with cp.cuda.Device(cupy_device_id):
                state, result = _run_gaussian_inference_on_fit_device()
        else:
            state, result = _run_gaussian_inference_on_fit_device()

        # Convert only after covariance, distribution, p-value, and CI work.
        self._apply_gaussian_reporting_state(state)
        if self._effective_intercept:
            self.intercept_ = float(self._params[0])
            self.coef_ = self._params[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self._params.copy()
        self._native_fit_coef = None
        self._native_fit_intercept = None
        if result is None:
            self._inference_result = None
            self._bse = None
            self._tvalues = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            return
        result.feature_names = self._inference_feature_names()
        result.apply_to(self)

    def _resolve_penalty(self) -> "Penalty":
        """Resolve penalty string or instance to a Penalty object."""
        # Lazy import to avoid circular dependency
        from statgpu.penalties import get_penalty, Penalty

        if isinstance(self.penalty, Penalty):
            # Penalties with learned initialization state (currently
            # Adaptive-L1) must be fit-local. Reusing the caller's object would
            # leak learned weights/caches from one dataset or refit into the
            # next. Stateless/fixed penalties retain their historical identity.
            if bool(getattr(self.penalty, "requires_init", False)):
                import copy
                cloned = copy.deepcopy(self.penalty)
                # Adaptive-L1 distinguishes constructor-owned fixed weights
                # (weights=...) from fit-learned _weights. A caller may reuse a
                # template after it has previously learned weights; those
                # learned values are not constructor state and must not seed a
                # new fit on different data.
                if (
                    str(getattr(cloned, "name", "")).lower() == "adaptive_l1"
                    and getattr(cloned, "weights", None) is None
                ):
                    cloned._weights = None
                    cloned._norm_factor = 1.0
                    for key in tuple(vars(cloned)):
                        if key.startswith("_alpha_w_"):
                            delattr(cloned, key)
                return cloned
            return self.penalty

        # Map "none"/"null" to l2 with alpha=0 (no regularization)
        pen_name = str(self.penalty).lower().strip()
        if pen_name in ("none", "null", ""):
            return get_penalty("l2", alpha=0.0)

        kwargs = {**self._penalty_kwargs, "alpha": self.alpha}
        if pen_name in ("elasticnet", "en"):
            kwargs["l1_ratio"] = self.l1_ratio

        return get_penalty(pen_name, **kwargs)

    def _resolve_loss(self):
        """Resolve loss string to a loss object.

        Tries the GLM-specific registry first (squared_error, logistic, etc.),
        then falls back to the generic loss registry (quantile, huber, cox_ph, etc.).
        """
        try:
            from statgpu.glm_core import get_glm_loss
            return get_glm_loss(self.loss, **self._loss_kwargs)
        except (ValueError, KeyError, TypeError):
            from statgpu.losses import get_loss
            return get_loss(self.loss, **self._loss_kwargs)

    def _validate_solver_penalty(self):
        """Validate solver/penalty combinations before backend dispatch."""
        solver_name = self._solver
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        non_smooth = _NONSMOOTH_PENALTIES
        if self._solver == "exact":
            if self.loss != "squared_error" or penalty_name != "l2":
                raise ValueError(
                    "solver='exact' is only supported for squared-error L2/Ridge models."
                )
            return
        if solver_name == "irls" and penalty_name not in ("l2", "none", "null", ""):
            raise ValueError(
                "solver='irls' only supports smooth L2 or no-penalty objectives."
            )
        # Reject irls for losses without IRLS support (not GLM and no custom irls())
        if solver_name == "irls" and not getattr(self._loss, '_supports_irls', False):
            raise ValueError(
                f"solver='irls' requires a loss with IRLS support, "
                f"got loss='{self.loss}'. Use solver='newton' or 'fista'."
            )
        if solver_name in ("newton", "lbfgs") and penalty_name in non_smooth:
            raise ValueError(
                f"solver='{solver_name}' only supports smooth objectives; "
                f"use solver='fista' for penalty='{penalty_name}'."
            )
        # QuantileLoss has no Hessian — cannot use newton/lbfgs/exact.
        # But irls is allowed: quantile has its own IRLS (Frisch-Newton) method.
        if solver_name in ("newton", "lbfgs", "exact") and self.loss == "quantile":
            raise ValueError(
                f"solver='{solver_name}' requires Hessian, but quantile loss has none. "
                f"Use solver='fista', 'irls', or 'auto' for quantile regression."
            )
        if solver_name != "lbfgs":
            return

    def _validate_inference_request(self):
        """Validate and route penalized inference requests.

        Supported paths:
        - squared_error + L2: standard Gaussian inference
        - squared_error + L1/ElasticNet: debiased Lasso / bootstrap
        - Hessian-equipped losses + L2/none/ElasticNet: penalized sandwich
        - SCAD/MCP + oracle/bootstrap: oracle active-set or bootstrap
        - Any loss + bootstrap: universal fallback
        """
        if not self._compute_inference_enabled:
            return
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        inference_method = str(getattr(self, "inference_method", "sandwich")).lower()

        # squared_error + l1/elasticnet: default to debiased (not sandwich)
        if (self.loss == "squared_error"
                and penalty_name in ("l1", "elasticnet", "en")
                and inference_method == "sandwich"):
            inference_method = "debiased"

        # squared_error: existing paths (unchanged)
        if self.loss == "squared_error":
            if penalty_name == "l2":
                return
            if penalty_name in ("l1", "elasticnet", "en"):
                if inference_method in ("debiased", "cpu_ols", "gpu_ols", "bootstrap"):
                    return
            if penalty_name in ("scad", "mcp") and inference_method in ("oracle", "bootstrap"):
                return
            raise NotImplementedError(
                f"squared_error + '{penalty_name}' inference not supported "
                f"with inference_method='{inference_method}'. "
                f"Use inference_method='oracle' or 'bootstrap'."
            )

        # Hessian-equipped losses + smooth penalties: penalized sandwich
        loss_has_hessian = getattr(self._loss, 'has_hessian', False)
        if loss_has_hessian and penalty_name in ("l2", "none", ""):
            return
        if loss_has_hessian and penalty_name in ("elasticnet", "en"):
            return  # L2 curvature component only

        # SCAD/MCP: oracle or bootstrap
        if penalty_name in ("scad", "mcp") and inference_method in ("oracle", "bootstrap"):
            return

        # Bootstrap: universal fallback
        if inference_method == "bootstrap":
            return

        # L1 + non-squared_error: only bootstrap
        if loss_has_hessian and penalty_name in ("l1",) and inference_method == "bootstrap":
            return
        if penalty_name in ("l1",):
            raise NotImplementedError(
                f"loss='{self.loss}' + penalty='l1' does not support "
                f"inference_method='{inference_method}'. "
                f"Use inference_method='bootstrap' or set compute_inference=False."
            )

        raise NotImplementedError(
            f"Inference not supported for loss='{self.loss}' × penalty='{penalty_name}'. "
            f"Use inference_method='bootstrap' or set compute_inference=False."
        )

    def _clear_inference_state(self):
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._bse = None
        self._tvalues = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._family_cache = None  # Clear cached family to avoid stale link after loss change

    def _family_for_loss(self):
        # Cache on first call (avoid re-creating on every predict/score)
        cached = getattr(self, '_family_cache', None)
        if cached is not None:
            return cached

        from statgpu.glm_core._family import (
            Binomial,
            Gaussian,
            Poisson,
            Gamma,
            InverseGaussian,
            NegativeBinomial,
            Tweedie,
        )

        if self.loss == "logistic":
            fam = Binomial()
        elif self.loss == "poisson":
            fam = Poisson()
        elif self.loss == "gamma":
            fam = Gamma()
        elif self.loss == "inverse_gaussian":
            fam = InverseGaussian()
        elif self.loss == "negative_binomial":
            alpha = getattr(
                getattr(self, "_loss", None),
                "alpha",
                getattr(self, "loss_kwargs", {}).get("alpha", 1.0),
            )
            fam = NegativeBinomial(alpha=alpha)
        elif self.loss == "tweedie":
            power = getattr(
                getattr(self, "_loss", None),
                "power",
                getattr(self, "loss_kwargs", {}).get("power", 1.5),
            )
            fam = Tweedie(power=power)
        elif self.loss in ("quantile", "huber", "bisquare", "fair"):
            # Robust/quantile losses use identity link (linear predictor)
            fam = Gaussian()
        else:
            fam = Gaussian()

        self._family_cache = fam
        return fam

    def _column_stack(self, arrays, backend_name):
        if backend_name == "cupy":
            import cupy as cp
            return cp.column_stack(arrays)
        if backend_name == "torch":
            import torch
            return torch.column_stack(arrays)
        return np.column_stack(arrays)

    def _ones(self, n, backend_name, ref):
        if backend_name == "cupy":
            import cupy as cp
            with cp.cuda.Device(int(ref.device.id)):
                return cp.ones(n, dtype=ref.dtype)
        if backend_name == "torch":
            import torch
            return torch.ones(n, dtype=ref.dtype, device=ref.device)
        return np.ones(n, dtype=getattr(ref, "dtype", np.float64))

    def _selective_penalty(self, p, backend_name):
        """Penalty wrapper that leaves the last intercept coefficient free.

        Creates a fresh instance per call to avoid thread-local singleton
        conflicts in nested CV within the same thread.
        """
        sp = SelectivePenalty()
        sp.configure(self._penalty, p, backend_name)
        return sp