"""
Penalized GLM estimators.

This module keeps the GLM-specific optimization path explicit.  The central
implementation accepts a GLM loss name, while public typed estimators expose
gaussian, logistic, and poisson models without the old ``loss=...`` switch on
``PenalizedLinearRegression``.
"""

from __future__ import annotations

__all__ = ["PenalizedGeneralizedLinearModel", "PenalizedLinearRegression", "PenalizedLogisticRegression", "PenalizedPoissonRegression"]

import copy
from typing import Optional, Union, Dict
import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import get_backend, _get_torch_device_str, _to_numpy, _LINALG_ERRORS
from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
    validate_cov_type,
    validate_hac_maxlags,
)
from statgpu.inference._results import GaussianInferenceResult
from statgpu.backends._array_ops import _clip


def _irls_ridge_init(X, y, loss_name, alpha=0.01, max_iter=100, tol=1e-4, loss_kwargs=None):
    """Compute ridge-penalized GLM coefficients for adaptive_l1 init.

    For squared_error uses IRLS-CD (matching R glmnet's ridge solver).
    For GLM losses (logistic, poisson, etc.) uses FISTA with L2 penalty,
    which has proper line search and handles extreme y values robustly.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix (no intercept column).
    y : ndarray of shape (n,)
        Response vector.
    loss_name : str
        GLM loss name: 'logistic', 'poisson', 'squared_error', etc.
    alpha : float
        Ridge penalty strength (lambda in R glmnet).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on coefficient change.

    Returns
    -------
    coef : ndarray of shape (p,)
        Ridge-penalized coefficient estimates (no intercept).
    """
    if loss_name in ("squared_error", ""):
        return _irls_ridge_init_cd(X, y, alpha, max_iter, tol)
    # For GLM losses, use FISTA with L2 penalty (robust line search)
    from statgpu.glm_core._solver import fista_solver
    from statgpu.penalties import get_penalty
    l2_pen = get_penalty("l2", alpha=alpha)
    loss_obj = _resolve_loss_name(loss_name, loss_kwargs=loss_kwargs)
    coef, _ = fista_solver(
        loss_obj, l2_pen, np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        max_iter=max_iter, tol=tol,
    )
    return np.asarray(coef, dtype=np.float64)


def _resolve_loss_name(loss_name, loss_kwargs=None):
    """Resolve loss name string to loss object via the GLM loss registry."""
    from statgpu.glm_core._base import get_glm_loss
    loss_kwargs = loss_kwargs or {}
    return get_glm_loss(loss_name, **loss_kwargs)


# ---------------------------------------------------------------------------
# Solver dispatch table for solver='auto'
# ---------------------------------------------------------------------------
# Each entry is (solver, condition_fn). First match wins.
# condition_fn takes (loss, penalty, backend, l1_ratio, cv_mode, problem_size).

# Import shared penalty categories (single source of truth)
from statgpu.penalties._categories import (
    NONSMOOTH as _NONSMOOTH_PENALTIES,
    NONCONVEX as _NONCONVEX_PENALTIES,
    SPARSE as _SPARSE_PENALTIES,
)
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})

# (solver, condition)
# condition = (loss, penalty, backend, l1_ratio, cv_mode, problem_size) -> bool
_SOLVER_DISPATCH_TABLE = [
    # ── Priority 1: Exact closed-form solutions (highest priority) ──
    # Ridge + squared_error has an exact eigendecomposition solver.
    ("exact", lambda l, p, b, lr, cv, ps: l == "squared_error" and p == "l2"),

    # ── Priority 2: Nonconvex penalties always use FISTA+LLA wrapper ──
    # SCAD/MCP/adaptive_l1 require iteratively reweighted L1 (LLA approximation).
    ("fista", lambda l, p, b, lr, cv, ps: p in _NONCONVEX_PENALTIES),

    # ── Priority 3: Squared error + sparse penalties → FISTA ──
    # Quadratic loss + L1/ElasticNet: FISTA with exact line search.
    ("fista", lambda l, p, b, lr, cv, ps: l == "squared_error" and p in _SPARSE_PENALTIES),

    # ── Priority 4: GLM + GPU + sparse penalties (size-gated) ──
    # Poisson + GPU + L1: fista_bb for small/medium problems (< 2M elements).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and b in ("cupy", "torch") and p == "l1" and (ps is None or ps < 2_000_000)),
    # Poisson + GPU + ElasticNet: fista_bb (BB step adapts well to EN geometry).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and b in ("cupy", "torch") and p in ("elasticnet", "en")),
    # Poisson + CPU + sparse: FISTA (CPU backtracking is cheap).
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and p in _SPARSE_PENALTIES),

    # ── Priority 5: NB + GPU + sparse penalties ──
    # NB + GPU + L1: fista_bb (NB gradient is well-behaved for BB steps).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p == "l1"),
    # NB + GPU + ElasticNet: FISTA for medium problems (200K-1M), fista_bb otherwise.
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p in ("elasticnet", "en") and ps is not None and 200_000 <= ps < 1_000_000),
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p in ("elasticnet", "en")),

    # ── Priority 6: Gamma/IG/Tweedie + sparse → FISTA ──
    # These families have steep loss landscapes; FISTA with backtracking is safer.
    ("fista", lambda l, p, b, lr, cv, ps: l in ("gamma", "inverse_gaussian") and p in _SPARSE_PENALTIES),
    ("fista", lambda l, p, b, lr, cv, ps: l == "tweedie" and b in ("cupy", "torch") and p in _SPARSE_PENALTIES),

    # ── Priority 7: Logistic + sparse → FISTA ──
    # Logistic has iterate-dependent Lipschitz; FISTA with fixed global bound.
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "logistic" and p in _SPARSE_PENALTIES),

    # ── Priority 8: Default sparse → fista_bb ──
    # Catch-all for remaining sparse penalty cases.
    ("fista_bb", lambda l, p, b, lr, cv, ps: p in _SPARSE_PENALTIES),

    # ── Priority 9: CV + L2: loss-specific smooth solvers ──
    # NB needs L-BFGS (non-canonical link issues with IRLS).
    ("lbfgs", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l == "negative_binomial"),
    # Poisson/Tweedie: Newton (canonical link, well-conditioned).
    ("newton", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l in ("poisson", "tweedie")),
    # Gamma/IG: L-BFGS (non-canonical link, better convergence).
    ("lbfgs", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l in ("gamma", "inverse_gaussian")),

    # ── Priority 10: Smooth penalties (L2/none) with loss-specific solvers ──
    ("newton", lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in ("gamma", "tweedie", "inverse_gaussian")),
    ("irls", lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in ("logistic", "poisson", "negative_binomial")),
]


def _preferred_penalized_glm_solver(
    loss_name,
    penalty_name,
    backend_name=None,
    l1_ratio=0.5,
    cv_mode=False,
    problem_size=None,
):
    """Private benchmark-backed solver policy for solver='auto'.

    This helper only chooses an internal solver.  It must never be used to
    override an explicitly requested solver or to change the selected device.

    Dispatch is table-driven: first matching rule wins.
    """
    loss_name = str(loss_name or "").lower()
    penalty_name = str(penalty_name or "").lower()
    backend_name = str(backend_name or "").lower()
    if problem_size is not None:
        problem_size = int(problem_size)

    for solver, cond in _SOLVER_DISPATCH_TABLE:
        if cond(loss_name, penalty_name, backend_name, l1_ratio, cv_mode, problem_size):
            return solver

    return "fista"


def _irls_ridge_init_cd(X, y, alpha, max_iter, tol):
    """CD ridge for squared_error (matching R glmnet's ridge solver)."""
    n, p = X.shape
    feat_norms = np.sqrt(np.sum(X ** 2, axis=0))
    feat_norms = np.maximum(feat_norms, 1e-20)
    scale = np.sqrt(n) / feat_norms
    X_work = X * scale

    beta = np.zeros(p)
    XDX_diag = np.sum(X_work ** 2, axis=0)

    for it in range(max_iter):
        beta_old = beta.copy()
        r = y - X_work @ beta
        for j in range(p):
            rho_j = np.dot(X_work[:, j], r) + XDX_diag[j] * beta[j]
            u_j = rho_j / n
            v_j = XDX_diag[j] / n
            beta[j] = u_j / (v_j + alpha)
            r += X_work[:, j] * (beta_old[j] - beta[j])

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta * scale


# Intercept clipping bound for SelectivePenalty proximal operator
from statgpu.cross_validation._base import INTERCEPT_CLIP_BOUND as _INTERCEPT_CLIP_BOUND

# Eta (linear predictor) clipping bound for numerical stability in GLM link functions.
# Prevents overflow in exp(eta) for log-link families and sigmoid(eta) for logistic.
# Value of 500 is safe because exp(500) ≈ 1.4e217 (within float64 range).
_ETA_CLIP = 500.0


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
            result = cp.empty(w.shape[0], dtype=w.dtype)
            result[:self._p] = result_feat
            result[-1] = cp.clip(w[-1], -_INTERCEPT_CLIP_BOUND, _INTERCEPT_CLIP_BOUND)
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
            diag = cp.zeros(coef.shape[0], dtype=coef.dtype)
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


# Thread-local singleton for _selective_penalty (avoids per-call class creation
# while remaining safe for concurrent CV folds via n_jobs > 1)
import threading
_SELECTIVE_PENALTY_LOCAL = threading.local()


def _get_selective_penalty_singleton():
    """Get or create a thread-local SelectivePenalty instance."""
    obj = getattr(_SELECTIVE_PENALTY_LOCAL, 'instance', None)
    if obj is None:
        obj = SelectivePenalty()
        _SELECTIVE_PENALTY_LOCAL.instance = obj
    return obj


class PenalizedGeneralizedLinearModel(BaseEstimator):
    """
    Penalized generalized linear model with pluggable GLM loss and penalty.

    Minimizes: loss(X, y, w) + penalty(w)

    Parameters
    ----------
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'logistic', 'poisson', 'gamma',
        'negative_binomial', 'tweedie', 'inverse_gaussian'.
    penalty : str or Penalty
        Penalty type: 'l1', 'l2', 'elasticnet', 'scad', 'mcp', 'adaptive_l1',
        'group_lasso', 'group_scad', 'group_mcp', or a Penalty instance.
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'fista_bb', 'irls', 'newton', 'lbfgs', 'exact'.
        'auto' selects the best path for the resolved backend and loss/penalty
        combination (see _SOLVER_DISPATCH_TABLE).
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Only used when penalty='elasticnet'.
    penalty_kwargs : dict, optional
        Additional arguments passed to the penalty constructor.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    cpu_solver : str, default='fista'
        CPU solver: 'fista', 'fista_bb', or 'coordinate_descent'.
    lipschitz_L : float, optional
        Pre-computed Lipschitz constant.
    gpu_memory_cleanup : bool, default=False
        If True, free GPU memory pool after fitting.

    Examples
    --------
    # Lasso
    >>> model = PenalizedLinearRegression(penalty='l1', alpha=0.1)

    # Ridge
    >>> model = PenalizedLinearRegression(penalty='l2', alpha=1.0)

    # Elastic Net
    >>> model = PenalizedLinearRegression(
    ...     penalty='elasticnet', alpha=0.1, l1_ratio=0.5
    ... )
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
        self.penalty_kwargs = penalty_kwargs or {}
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
        self.loss_kwargs = loss_kwargs or {}

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
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._selected_solver = None
        self._selected_backend_name = None
        self._init_coef = None
        self._inference_precomputed = False
        self._precomputed_gaussian_state = None
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
        return self.fit_intercept

    def _resolve_penalty(self) -> "Penalty":
        """Resolve penalty string or instance to a Penalty object."""
        # Lazy import to avoid circular dependency
        from statgpu.penalties import get_penalty, Penalty

        if isinstance(self.penalty, Penalty):
            return self.penalty

        kwargs = {**self.penalty_kwargs, "alpha": self.alpha}
        if self.penalty == "elasticnet":
            kwargs["l1_ratio"] = self.l1_ratio

        return get_penalty(self.penalty, **kwargs)

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """
        Fit penalized GLM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Training data. Required when ``formula`` is None.
        y : array-like of shape (n_samples,), optional
            Target values. Required when ``formula`` is None.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        formula : str, optional
            R-style formula string, e.g. ``"y ~ x1 + C(group)"``.
        data : pandas.DataFrame, optional
            Data used to evaluate ``formula``.

        Returns
        -------
        self : PenalizedLinearRegression
            Fitted estimator.
        """
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            from statgpu.core.formula import FormulaParser

            parser = FormulaParser(formula)
            y, X, design_info = parser.eval(data)
            formula_column_names = list(design_info.column_names)
            self._design_info = design_info
            self._formula_has_intercept = "Intercept" in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != "Intercept"]
            if self._formula_has_intercept:
                X = np.delete(X, formula_column_names.index("Intercept"), axis=1)
                self._use_intercept = True
            else:
                # Formula syntax owns intercept semantics, matching statsmodels/R.
                self._use_intercept = False
        else:
            if X is None or y is None:
                raise ValueError("Either formula+data or X+y must be provided.")
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None
            self._use_intercept = None

        # Record number of features for sklearn compatibility
        if X is not None:
            X_arr = np.asarray(X) if not hasattr(X, 'shape') else X
            self.n_features_in_ = X_arr.shape[1] if X_arr.ndim >= 2 else 1

        self._penalty = self._resolve_penalty()
        self._validate_solver_penalty()
        self._loss = self._resolve_loss()
        self._validate_inference_request()
        self._inference_precomputed = False
        self._precomputed_gaussian_state = None
        self._clear_inference_state()

        # Resolve the actual backend before auto-selecting the solver. This
        # keeps solver="auto" device-aware: CPU can use IRLS for smooth GLMs,
        # while GPU/Torch stays on accelerator-capable FISTA.
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        # Auto-dispatch small problems to CPU only when device="auto".
        # Explicit CUDA/TORCH device selection must never silently fall back.
        if self.device == Device.AUTO and backend_name in ("cupy", "torch") and X is not None:
            _n, _p = X.shape
            if _n * _p < 200_000:
                backend_name = "numpy"

        backend_name = self._auto_backend_override(backend_name, X)
        selected_solver = self._select_solver(
            self._loss, backend_name=backend_name, X=X
        )
        self._selected_solver = selected_solver
        self._selected_backend_name = backend_name

        # Handle penalties requiring initialization (e.g., Adaptive Lasso)
        if self._penalty.requires_init:
            init_coef = self._fit_initial(X, y, backend_name=backend_name)
            self._penalty.set_weights(init_coef)

        # Non-convex penalties (SCAD, MCP) for squared_error: use IRLS-CD
        # directly with a 100-step continuation path from lambda_max.
        # This matches R ncvreg's algorithm for Gaussian regression.
        # GLM+SCAD/MCP must NOT use IRLS-CD — it cycles due to non-convex
        # penalty causing features to flip on/off between IRLS iterations.
        # GLM+SCAD/MCP goes through _fit_lla → FISTA with proximal operator.
        _pen_name = str(getattr(self._penalty, 'name', '')).lower()
        _loss_name = str(getattr(self._loss, 'name', '') if hasattr(self, '_loss') else self.loss).lower()
        _is_glm_loss = _loss_name not in ("squared_error", "")
        if _pen_name in ("scad", "mcp") and self._lla_enabled and not _is_glm_loss:
            # Use fused FISTA+LLA path for all backends (CPU/GPU).
            from statgpu.glm_core._solver import fista_lla_path
            self._nobs = X.shape[0]
            X_arr = self._to_array(X, backend=backend_name)
            y_arr = self._to_array(y, backend=backend_name)
            # Lambda_max computation uses numpy (one-time cost, negligible).
            _X_np = _to_numpy(X_arr)
            _y_np = _to_numpy(y_arr)
            _n = _X_np.shape[0]
            _col_norms = np.sqrt(np.sum(_X_np ** 2, axis=0))
            _col_norms = np.maximum(_col_norms, 1e-20)
            _X_s = _X_np * (np.sqrt(_n) / _col_norms)
            _y_c = _y_np - np.mean(_y_np)
            _lam_max = float(np.max(np.abs(_X_s.T @ _y_c / _n)))
            _target_alpha = float(self._penalty.alpha)
            _n_cont = 20
            _alpha_start = max(_lam_max, _target_alpha * 1.1)
            if (not np.isfinite(_alpha_start)) or _alpha_start <= 0.0 or _target_alpha <= 0.0:
                _alpha_path = np.linspace(max(_lam_max, 0.0), _target_alpha, _n_cont)
            else:
                _alpha_path = np.geomspace(_alpha_start, _target_alpha, _n_cont)
            _max_lla_per_step = max(6, getattr(self, '_max_lla_iters', 50) // _n_cont)
            _saved_mi = self.max_iter
            _mi_path = []
            for _i in range(_n_cont):
                _is_last = (_i == _n_cont - 1)
                _mi_path.append(_saved_mi if _is_last else max(100, _saved_mi // 10))
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_arr, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self._effective_intercept,
                sample_weight=sample_weight,
            )
            self.coef_ = coef_np
            self.intercept_ = intercept
            self.n_iter_ = n_iter
            if self._effective_intercept:
                self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
            else:
                self._params = np.asarray(self.coef_).copy()
            self._df_resid = X.shape[0] - (X.shape[1] + (1 if self._effective_intercept else 0))
            self._compute_post_fit_gaussian_inference(X, y, sample_weight=sample_weight)
            if backend_name == "cupy":
                self._cleanup_cuda_memory()
            elif backend_name == "torch":
                self._cleanup_torch_memory()
            self._fitted = True
            return self

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)

        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif backend_name == "cupy":
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        self._compute_post_fit_gaussian_inference(X, y, sample_weight=sample_weight)
        self._fitted = True
        # Clean up CV cache unless a caller is intentionally reusing one
        # across repeated fits, as PenalizedGLM_CV does within a fold.
        if hasattr(self, '_cv_cache') and not getattr(self, '_preserve_cv_cache', False):
            del self._cv_cache
        return self

    def _resolve_loss(self):
        """Resolve loss string to a GLMLoss object."""
        from statgpu.glm_core import get_glm_loss

        return get_glm_loss(self.loss, **self.loss_kwargs)

    def _validate_solver_penalty(self):
        """Validate solver/penalty combinations before backend dispatch."""
        solver_name = self.solver
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        non_smooth = _NONSMOOTH_PENALTIES
        if self.solver == "exact":
            if self.loss != "squared_error" or penalty_name != "l2":
                raise ValueError(
                    "solver='exact' is only supported for squared-error L2/Ridge models."
                )
            return
        if solver_name == "irls" and penalty_name != "l2":
            raise ValueError(
                "solver='irls' only supports smooth L2 penalized GLM objectives."
            )
        if solver_name in ("newton", "lbfgs") and penalty_name in non_smooth:
            raise ValueError(
                f"solver='{solver_name}' only supports smooth objectives; "
                f"use solver='fista' for penalty='{penalty_name}'."
            )
        if solver_name != "lbfgs":
            return

    def _validate_inference_request(self):
        """Reject unsupported penalized inference paths with a clear error.

        Currently supported:
        - squared_error + L2 (standard OLS inference)
        - squared_error + L1/ElasticNet (debiased Lasso, cpu_ols_inference, bootstrap)
        """
        if not self.compute_inference:
            return
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        if self.loss == "squared_error" and penalty_name == "l2":
            return
        inference_method = str(getattr(self, "inference_method", "debiased")).lower()
        if penalty_name in ("l1", "elasticnet", "en"):
            if "debiased" in inference_method:
                return
            if "cpu_ols" in inference_method or "gpu_ols" in inference_method:
                return
            if "bootstrap" in inference_method:
                return
        raise NotImplementedError(
            f"compute_inference=True with penalty='{penalty_name}' and "
            f"loss='{self.loss}' is not supported. Use inference_method='debiased', "
            f"'cpu_ols_inference', or 'bootstrap' for L1/ElasticNet, "
            f"or compute_inference=False to skip inference."
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
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None

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
        """Populate inference state after fit. Dispatches to debiased for L1/ElasticNet."""
        if not self.compute_inference:
            return
        if self.loss != "squared_error":
            return
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
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
            return
        if penalty_name != "l2":
            return
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
        from scipy.stats import norm as _norm_dist

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
        from statgpu.linear_model._lasso import (
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
        from scipy import stats as _stats

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
        pvalues_sel = 2.0 * (1.0 - _stats.t.cdf(np.abs(tvalues_sel), df_resid))

        t_crit = _stats.t.ppf(0.975, df_resid)
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

        from statgpu.linear_model._lasso import (
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

        from statgpu.linear_model._lasso import (
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

    def _select_solver(self, loss, backend_name=None, X=None):
        """Auto-select solver based on loss, penalty, and backend."""
        if self.solver != "auto":
            return self.solver
        return _preferred_penalized_glm_solver(
            getattr(loss, "name", self.loss),
            getattr(self._penalty, "name", self.penalty),
            backend_name=backend_name,
            l1_ratio=getattr(self._penalty, "l1_ratio", self.l1_ratio),
            cv_mode=False,
            problem_size=None if X is None else int(X.shape[0]) * int(X.shape[1]),
        )

    @staticmethod
    def _torch_cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _cupy_available():
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    # Backend override rules for device='auto' at large scale (problem_size >= 1M).
    # Each entry: (loss, penalties, target_backend, reason_template)
    # First match wins. target_backend="numpy" means always CPU;
    # target_backend="torch" means prefer torch over cupy.
    _AUTO_BACKEND_CPU_OVERRIDES = [
        ("squared_error", ("l2",), "numpy", "large squared-error exact solve is faster on CPU"),
        ("squared_error", ("l1", "elasticnet", "en"), "numpy", "large squared-error l1/elasticnet is faster on CPU"),
        ("negative_binomial", ("l1", "elasticnet", "en"), "numpy", "large negative-binomial l1/elasticnet is faster on CPU"),
        ("logistic", ("l1", "elasticnet", "en"), "numpy", "large logistic {penalty} is faster on CPU"),
        ("gamma", ("l2",), "numpy", "large gamma l2/newton is faster on CPU"),
        ("tweedie", ("l1", "elasticnet", "en"), "numpy", "large tweedie {penalty} is faster on CPU"),
    ]
    _AUTO_BACKEND_CUPY_OVERRIDES = [
        ("negative_binomial", ("l2",), "torch", "large negative-binomial l2 is faster on {target} than cupy"),
        ("logistic", ("l1", "elasticnet", "en"), "torch", "large logistic {penalty} is faster on {target} than cupy"),
        ("poisson", ("l1", "elasticnet", "en"), "torch", "large poisson {penalty} is faster on {target} than cupy"),
    ]

    def _auto_backend_override(self, backend_name, X):
        """Benchmark-backed backend routing for device='auto' only."""
        self._auto_backend_reason = None
        if self.device != Device.AUTO or self.solver != "auto" or X is None:
            return backend_name

        n_samples, n_features = X.shape
        problem_size = int(n_samples) * int(n_features)
        if problem_size < 1_000_000:
            return backend_name

        loss_name = str(getattr(self._loss, "name", self.loss)).lower()
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        torch_ok = self._torch_cuda_available()

        # CPU overrides: always route to numpy
        for loss, penalties, target, reason_tpl in self._AUTO_BACKEND_CPU_OVERRIDES:
            if loss_name == loss and penalty_name in penalties:
                self._auto_backend_reason = reason_tpl.format(penalty=penalty_name)
                return target

        # CuPy→Torch overrides: prefer torch when available, else CPU
        if backend_name == "cupy":
            for loss, penalties, target, reason_tpl in self._AUTO_BACKEND_CUPY_OVERRIDES:
                if loss_name == loss and penalty_name in penalties:
                    if torch_ok:
                        self._auto_backend_reason = reason_tpl.format(
                            penalty=penalty_name, target="torch")
                        return "torch"
                    self._auto_backend_reason = reason_tpl.format(
                        penalty=penalty_name, target="CPU")
                    return "numpy"

        return backend_name

    def _fit_initial(self, X, y, backend_name="numpy"):
        """Fit initial model for penalties requiring initialization.

        Parameters
        ----------
        X : array
            Design matrix.
        y : array
            Target vector.
        backend_name : str
            Backend to use ('numpy', 'torch', 'cupy'). Default 'numpy'.

        Uses OLS when n_samples > n_features (well-determined, unbiased),
        and Ridge otherwise (works for any p, required when p > n).

        The ``init_method`` on the penalty controls which path is taken:
        - 'auto': OLS if n > p, Ridge otherwise
        - 'ols': forced OLS (raises if p > n)
        - 'ridge': forced Ridge (always works)

        OLS is only safe for squared_error (Gaussian) data.  For GLM losses
        (Poisson, logistic, etc.) OLS can produce extreme coefficients whose
        Lipschitz constant is enormous, causing the inner FISTA solver to
        take zero-length steps and exit immediately without moving.

        For GLM losses we use sparse L1 initialization only for non-convex
        penalties (SCAD, MCP) that will enter the LLA outer loop — a sparse
        seed gives LLA differentiated weights and drives genuine sparsity.
        Convex penalties with ``requires_init=True`` (adaptive_l1) need a
        dense seed because their weights are 1/|coef| — zero entries from
        L1 init become permanently frozen."""
        n_samples, n_features = X.shape
        init_method = getattr(self._penalty, "init_method", "auto")
        _is_glm = getattr(self, 'loss', 'squared_error') != "squared_error"
        _is_nonconvex = not getattr(self._penalty, "is_convex", True)

        if not _is_glm and not self._penalty.requires_init and (
            init_method == "ols" or (init_method == "auto" and n_samples > n_features)
        ):
            ols_coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return ols_coef

        if _is_glm and _is_nonconvex:
            # Dense l2-penalized GLM init for non-convex penalties (SCAD, MCP).
            # With the corrected lla_weights (= P'(|coef|), not P'(|coef|)/|coef|),
            # a dense starting point lets the LLA continuation path push small
            # coefficients through the transition region where SCAD and MCP
            # differ, matching the path-based strategy used by R's ncvreg.
            from statgpu.penalties import get_penalty
            from statgpu.glm_core._solver import fista_solver

            l2_pen = get_penalty("l2", alpha=0.001)
            loss_obj = self._resolve_loss()
            # Use matching backend for GPU data
            if backend_name in ("torch", "cupy"):
                backend = get_backend(backend=backend_name, device='cuda')
                X_b = backend.asarray(X, dtype=backend.float64)
                y_b = backend.asarray(y, dtype=backend.float64)
            else:
                X_b = np.asarray(_to_numpy(X), dtype=np.float64)
                y_b = np.asarray(_to_numpy(y), dtype=np.float64)
            init_coef, _ = fista_solver(
                loss_obj, l2_pen, X_b, y_b,
                max_iter=500, tol=1e-4,
            )
            return init_coef

        if self._penalty.requires_init:
            # adaptive_l1: weights = 1/(|init_coef|+eps)^nu, so init must
            # produce well-scaled coefficients.  Use IRLS with coordinate
            # descent (matching R glmnet's ridge solver) instead of FISTA,
            # which converges more tightly and gives larger coefficients
            # → smaller weights → too many features surviving.
            loss_name = getattr(self, 'loss', 'squared_error')
            # Use matching backend for GPU data
            if backend_name in ("torch", "cupy"):
                backend = get_backend(backend=backend_name, device='cuda')
                X_b = backend.asarray(X, dtype=backend.float64)
                y_b = backend.asarray(y, dtype=backend.float64)
            else:
                X_b = np.asarray(_to_numpy(X), dtype=np.float64)
                y_b = np.asarray(_to_numpy(y), dtype=np.float64)
            init_coef = _irls_ridge_init(
                X_b, y_b,
                loss_name=loss_name,
                alpha=0.01,
                max_iter=100,
                tol=1e-4,
                loss_kwargs=getattr(self, "loss_kwargs", None),
            )
            return init_coef

        from statgpu.linear_model._ridge import Ridge

        init_model = Ridge(
            alpha=0.1,
            fit_intercept=self._effective_intercept,
            device=self.device,
        )
        init_model.fit(X, y)
        return init_model.coef_

    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU (FISTA or coordinate descent)."""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Route to loss-aware solver for non-squared_error loss
        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="numpy"
        )
        if self.loss != "squared_error" or solver_name == "admm":
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "numpy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "numpy")
            return
        if solver_name in ("irls", "newton", "lbfgs", "admm"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "numpy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "numpy")
            return

        # Route squared_error + SCAD/MCP/adaptive_l1/group_lasso/elasticnet
        # through _fit_loss_backend so CPU and GPU paths produce identical results.
        _cd_penalties_for_sqerr = ("scad", "mcp", "adaptive_l1", "adaptive_lasso", "group_lasso")
        if getattr(self._penalty, 'name', '') in _cd_penalties_for_sqerr:
            self._fit_loss_backend(X, y, sample_weight, solver_name, "numpy")
            return

        # Original squared-error path (backward compatible)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw

        pen = self._penalty

        if self._effective_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = 0.0
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        # Precompute for gradient (use CV cache if available)
        _cv = getattr(self, '_cv_cache', None)
        if _cv is not None and 'XtX' in _cv:
            XtX = _cv['XtX']
            Xty = _cv['Xty']
        else:
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered.flatten()

        pen = self._penalty
        if solver_name == "exact":
            if pen.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            self.coef_ = self._solve_exact_numpy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self._effective_intercept:
                self.intercept_ = float(y_mean - X_mean @ self.coef_)
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self._params = self.coef_.copy()
            self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))
            return

        # Lipschitz constant: L = λ_max(XtX) / n
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            from statgpu.backends._array_ops import _max_eigval_power
            L = _max_eigval_power(XtX) / n_samples

        if L <= 0:
            self.coef_ = np.zeros(n_features)
            self.n_iter_ = 0
        else:
            step = 1.0 / L

            _cd_penalties = ("adaptive_l1", "adaptive_lasso", "scad", "mcp", "group_lasso")
            if solver_name in ("fista_bb", "fista") and pen.name not in _cd_penalties:
                # FISTA with XtX precomputation.
                # BB step (fista_bb) provides no benefit for quadratic losses
                # (BB1=BB2=1/R_H(dw)), so both use the fixed Lipschitz step.
                if hasattr(self, '_init_coef') and self._init_coef is not None:
                    coef = np.asarray(self._init_coef, dtype=np.float64).copy()
                else:
                    coef = np.zeros(n_features)
                y_k = coef.copy()
                t_k = 1.0

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    grad_at_y = (XtX @ y_k - Xty) / n_samples
                    w_tilde = y_k - step * grad_at_y
                    coef = pen.proximal(w_tilde, step, backend="numpy")

                    # Scheduled momentum restart
                    if iteration > 0 and iteration % 50 == 0:
                        t_k = 1.0

                    # Nesterov momentum
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                    beta = (t_k - 1.0) / t_new
                    y_k = coef + beta * (coef - coef_old)
                    t_k = t_new

                    self.n_iter_ = iteration + 1

                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        break

            else:
                # Coordinate descent (for L1-type penalties)
                X_sq_norms = np.diag(XtX)
                if hasattr(self, '_init_coef') and self._init_coef is not None:
                    coef = np.asarray(self._init_coef, dtype=np.float64).copy()
                else:
                    coef = np.zeros(n_features)

                # Precompute per-coordinate thresholds for adaptive penalties.
                # The penalty object stores mean-normalized weights (w = pf / mean(pf))
                # and _norm_factor = mean(pf).  The CD threshold per coordinate is
                # alpha * w_j * n, matching R glmnet's lambda * pf_j * n / X_j'X_j
                # after dividing by X_sq_norms[j].
                _adaptive_thresh = None
                if pen.name in ("adaptive_l1", "adaptive_lasso"):
                    _w = np.asarray(getattr(pen, '_weights', np.ones(n_features)), dtype=float)
                    _adaptive_thresh = self.alpha * _w * n_samples

                # Precompute SCAD/MCP constants (hoisted out of inner loop)
                _a_scad = float(getattr(pen, 'a', 3.7)) if pen.name == "scad" else 0.0
                _gamma_mcp = float(getattr(pen, 'gamma', 3.0)) if pen.name == "mcp" else 0.0

                # Precompute group info for group_lasso block CD
                _is_group = pen.name == "group_lasso"
                if _is_group:
                    _g_indices = getattr(pen, '_group_indices', None)
                    _sqrt_pg = getattr(pen, '_sqrt_pg', None)
                    if _g_indices is None or _sqrt_pg is None:
                        raise ValueError(
                            "group_lasso penalty must have groups set. "
                            "Pass groups=... in penalty_kwargs."
                        )
                    _n_groups = len(_g_indices)
                    # Precompute XtX blocks per group: XtX[g_idx][:, g_idx]
                    _XtX_blocks = []
                    for g_idx in _g_indices:
                        _XtX_blocks.append(XtX[np.ix_(g_idx, g_idx)])

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    if _is_group:
                        # Block coordinate descent: iterate over groups
                        for g in range(_n_groups):
                            g_idx = _g_indices[g]
                            # Group partial residual:
                            # rho_g = Xty[g] - XtX[g,:] @ coef + XtX[g,g] @ coef[g]
                            rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + _XtX_blocks[g] @ coef[g_idx]
                            # Unpenalized group update: w_g = (X'X)_gg^{-1} @ rho_g
                            try:
                                w_g = np.linalg.solve(_XtX_blocks[g], rho_g)
                            except np.linalg.LinAlgError:
                                w_g = np.zeros(len(g_idx))
                            # Block soft-thresholding
                            norm_w = np.linalg.norm(w_g)
                            thresh_g = self.alpha * _sqrt_pg[g]
                            if norm_w > thresh_g:
                                coef[g_idx] = w_g * (1.0 - thresh_g / norm_w)
                            else:
                                coef[g_idx] = 0.0
                    else:
                        # Per-coordinate CD for L1-type penalties
                        for j in range(n_features):
                            rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]

                            if pen.name in ("adaptive_l1", "adaptive_lasso"):
                                thresh = _adaptive_thresh[j]
                                if X_sq_norms[j] > 1e-10:
                                    coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                                else:
                                    coef[j] = 0.0
                            elif pen.name == "l1":
                                # Soft thresholding
                                thresh = self.alpha * n_samples
                                if X_sq_norms[j] > 1e-10:
                                    coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                                else:
                                    coef[j] = 0.0
                            elif pen.name == "elasticnet":
                                # Elastic net CD matching both sklearn and R glmnet:
                                # beta_j = S(rho_j, alpha*l1_ratio*n) / (X_j'X_j + alpha*(1-l1_ratio)*n)
                                thresh = self.alpha * self.l1_ratio * n_samples
                                if X_sq_norms[j] > 1e-10:
                                    st = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0)
                                    coef[j] = st / (X_sq_norms[j] + self.alpha * (1 - self.l1_ratio) * n_samples)
                                else:
                                    coef[j] = 0.0
                            elif pen.name == "scad":
                                # SCAD CD matching R ncvreg: threshold = alpha * n
                                # Guard: a_scad must be > 1 and != 2 to avoid div/0.
                                a_scad = max(float(_a_scad), 1.0 + 1e-6)
                                if abs(a_scad - 2.0) < 1e-6:
                                    a_scad = 2.0 + 1e-6
                                if X_sq_norms[j] > 1e-10:
                                    w_j = rho_j / X_sq_norms[j]
                                    aw = np.abs(w_j)
                                    lam = self.alpha * n_samples
                                    if aw > a_scad * lam:
                                        coef[j] = w_j
                                    elif aw > lam:
                                        coef[j] = np.sign(w_j) * ((a_scad - 1.0) * aw - a_scad * lam) / (a_scad - 2.0)
                                    else:
                                        coef[j] = 0.0
                                else:
                                    coef[j] = 0.0
                            elif pen.name == "mcp":
                                # MCP CD matching R ncvreg: threshold = alpha * n
                                # Guard: gamma_mcp must be > 1 to avoid div/0.
                                gamma_mcp = max(float(_gamma_mcp), 1.0 + 1e-6)
                                if X_sq_norms[j] > 1e-10:
                                    w_j = rho_j / X_sq_norms[j]
                                    aw = np.abs(w_j)
                                    lam = self.alpha * n_samples
                                    if aw > gamma_mcp * lam:
                                        coef[j] = w_j
                                    elif aw > lam:
                                        coef[j] = np.sign(w_j) * (aw - lam) / (1.0 - 1.0 / gamma_mcp)
                                    else:
                                        coef[j] = 0.0
                                else:
                                    coef[j] = 0.0
                            else:
                                raise NotImplementedError(
                                    f"Coordinate descent not implemented for "
                                    f"penalty '{pen.name}'. Use solver='fista'."
                                )

                    self.n_iter_ = iteration + 1

                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        break

        # Compute intercept and store results
        if L > 0:
            self.coef_ = coef

        if self._effective_intercept:
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self._params = self.coef_.copy()

        self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))

    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU (CuPy) with FISTA."""
        import cupy as cp

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="cupy"
        )
        if solver_name not in ("fista", "fista_bb", "admm", "auto", "exact", "irls", "newton", "lbfgs"):
            raise ValueError(
                "CuPy backend supports solver='fista', 'fista_bb', 'admm', "
                "'exact', 'irls', 'newton', and 'lbfgs'."
            )

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Exact solver (closed-form Ridge) — handle before generic routing
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            X = cp.asarray(X)
            y = cp.asarray(y)
            if sample_weight is not None:
                sw = cp.asarray(sample_weight, dtype=X.dtype)
                sqrt_sw = cp.sqrt(sw)
                X = X * sqrt_sw[:, cp.newaxis]
                y = y * sqrt_sw
            if self._effective_intercept:
                X_mean = cp.mean(X, axis=0)
                y_mean = cp.mean(y)
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_centered = X
                y_mean = cp.array(0.0, dtype=X.dtype)
                y_centered = y
            if y_centered.ndim == 1:
                y_centered = y_centered.reshape(-1)
            _cv = getattr(self, '_cv_cache', None)
            if _cv is not None and 'XtX' in _cv:
                XtX = _cv['XtX']
                Xty = _cv['Xty']
            else:
                XtX = X_centered.T @ X_centered
                Xty = X_centered.T @ y_centered
            coef = self._solve_exact_cupy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self.compute_inference:
                if self._effective_intercept:
                    intercept_gpu = (y_mean.reshape(1) - X_mean.reshape(1, -1) @ coef.reshape(-1, 1)).reshape(-1)
                    coef_full_gpu = cp.concatenate([intercept_gpu, coef.reshape(-1)])
                    self._precompute_exact_l2_inference_cupy(
                        X,
                        y,
                        XtX,
                        X_mean,
                        coef_full_gpu.reshape(-1),
                        n_samples,
                    )
                else:
                    self._precompute_exact_l2_inference_cupy(
                        X,
                        y,
                        XtX,
                        None,
                        coef.reshape(-1),
                        n_samples,
                    )
            coef_np = coef.get()
            if self._effective_intercept:
                self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))
            self._cleanup_cuda_memory()
            return

        # Route IRLS/newton/lbfgs through their dedicated backends.
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "cupy")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "cupy")
            return

        # Route non-L1 and non-squared-error through the generic loss backend.
        # The inline XtX path below is an optimized fast-path for L1+squared_error
        # where the proximal is simple element-wise soft-thresholding.
        if self.loss != "squared_error" or solver_name == "admm" or self._penalty.name not in ("l1", "elasticnet", "en"):
            self._fit_loss_backend(X, y, sample_weight, solver_name, "cupy")
            return

        X = cp.asarray(X)
        y = cp.asarray(y)

        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw

        if self._effective_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1)

        # Precompute (use CV cache if available)
        _cv = getattr(self, '_cv_cache', None)
        if _cv is not None and 'XtX' in _cv:
            XtX = _cv['XtX']
            Xty = _cv['Xty']
        else:
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered

        # Lipschitz constant: L = lambda_max(XtX) / n
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            # eigvalsh is faster for p < ~1000 (single cuBLAS call);
            # power iteration only wins for very large p.
            if n_features < 1000:
                L = float(cp.linalg.eigvalsh(XtX)[-1]) / n_samples
            else:
                v = cp.ones(n_features, dtype=X.dtype)
                v /= cp.linalg.norm(v)
                for _ in range(50):
                    v_new = XtX @ v
                    v_norm = cp.linalg.norm(v_new)
                    if v_norm < 1e-15:
                        break
                    v = v_new / v_norm
                L = float((v @ (XtX @ v)) / n_samples)

        if L <= 0:
            coef = cp.zeros(n_features, dtype=X.dtype)
            self.n_iter_ = 0
        elif solver_name in ("fista_bb", "fista"):
            # Standard FISTA with XtX precomputation.
            # All element-wise ops (gradient finish + proximal + momentum) are
            # fused into a single GPU kernel, reducing ~9 launches to 1.
            step = 1.0 / L
            step_over_n = step / n_samples
            step_over_n_Xty = step_over_n * Xty   # (p,) — precompute once
            if self._penalty.name in ("elasticnet", "en"):
                thresh = self.alpha * self._penalty.l1_ratio * step
                l2_scale = 1.0 + self.alpha * (1.0 - self._penalty.l1_ratio) * step
            else:
                thresh = self.alpha * step
                l2_scale = 1.0
            # When l2_scale ≈ 1.0 (pure L1 or l1_ratio=1), use the simpler
            # kernel without division — CuPy's @cp.fuse treats the constant
            # at compile time, and the division changes the generated code
            # even when the divisor is 1.0, causing different float rounding
            # and more iterations to converge.
            _use_l2 = abs(l2_scale - 1.0) > 1e-12

            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = cp.asarray(self._init_coef, dtype=X.dtype)
            else:
                coef = cp.zeros(n_features, dtype=X.dtype)
            y_k = coef.copy()
            t_k = 1.0
            beta = 0.0  # first iteration: y_k = coef (no momentum)

            # Lazy-compile the fused element-wise step (first call triggers JIT)
            _fused_step = None
            _fused_step_l2 = None

            # Warm-up: compile fused kernel BEFORE the loop to avoid
            # first-iteration JIT compilation overhead.
            if _use_l2:
                try:
                    @cp.fuse()
                    def _fista_elementwise_l2(
                        _y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                        _thresh, _l2_scale, _coef_old, _beta,
                    ):
                        w = (_y_k - _step_over_n * _xtx_y + _step_over_n_Xty)
                        c = (cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0) / _l2_scale)
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step_l2 = _fista_elementwise_l2
                    # Trigger JIT compilation with dummy data
                    _dummy = cp.zeros(1, dtype=X.dtype)
                    _fused_step_l2(_dummy, _dummy, _dummy, 0.0, 0.0, 1.0, _dummy, 0.0)
                except Exception:
                    _fused_step_l2 = None
            else:
                try:
                    @cp.fuse()
                    def _fista_elementwise(
                        _y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                        _thresh, _coef_old, _beta,
                    ):
                        w = (_y_k - _step_over_n * _xtx_y + _step_over_n_Xty)
                        c = (cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0))
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step = _fista_elementwise
                    _dummy = cp.zeros(1, dtype=X.dtype)
                    _fused_step(_dummy, _dummy, _dummy, 0.0, 0.0, _dummy, 0.0)
                except Exception:
                    _fused_step = None

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                # cuBLAS matvec (cannot fuse with element-wise ops)
                xtx_y = XtX @ y_k   # (p,)

                if _use_l2:
                    if _fused_step_l2 is not None:
                        coef, y_k = _fused_step_l2(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, l2_scale, coef_old, beta,
                        )
                    else:
                        w_tilde = (y_k - step_over_n * xtx_y
                                   + step_over_n_Xty)
                        coef = (cp.sign(w_tilde)
                                * cp.maximum(cp.abs(w_tilde) - thresh, 0.0)
                                / l2_scale)
                        y_k = coef + beta * (coef - coef_old)
                else:
                    if _fused_step is not None:
                        coef, y_k = _fused_step(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, coef_old, beta,
                        )
                    else:
                        w_tilde = (y_k - step_over_n * xtx_y
                                   + step_over_n_Xty)
                        coef = (cp.sign(w_tilde)
                                * cp.maximum(cp.abs(w_tilde) - thresh, 0.0))
                        y_k = coef + beta * (coef - coef_old)

                # Scheduled momentum restart (zero sync overhead)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0

                # Nesterov momentum (beta for next iteration)
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                t_k = t_new

                self.n_iter_ = iteration + 1

                if iteration % 5 == 4 and float(cp.sum(cp.abs(coef - coef_old))) < self.tol:
                    break
        else:
            step = 1.0 / L
            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = cp.asarray(self._init_coef, dtype=X.dtype)
            else:
                coef = cp.zeros(n_features, dtype=X.dtype)
            y_k = coef.copy()
            t_k = cp.array(1.0, dtype=X.dtype)

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad

                coef = self._penalty.proximal(w_tilde, step, backend="cupy")

                # Scheduled momentum restart (BEFORE momentum update)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = cp.array(1.0, dtype=X.dtype)

                t_new = (1.0 + cp.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                self.n_iter_ = iteration + 1

                if iteration % 5 == 4 and float(cp.sum(cp.abs(coef - coef_old))) < self.tol:
                    break

        # Transfer to CPU
        coef_np = coef.get()

        if self._effective_intercept:
            self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))

        # Debiased inference on GPU (before cleanup, while arrays are in scope)
        if self.compute_inference and "debiased" in str(getattr(self, "inference_method", "")).lower():
            penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
            if penalty_name in ("l1", "elasticnet", "en"):
                self._compute_inference_debiased_gpu(X, y, coef)

        # Cleanup
        self._cleanup_cuda_memory()

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FISTA."""
        import torch

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="torch"
        )
        if solver_name not in ("fista", "fista_bb", "admm", "auto", "exact", "irls", "newton", "lbfgs"):
            raise ValueError(
                "Torch backend supports solver='fista', 'fista_bb', 'admm', "
                f"'exact', 'irls', 'newton', and 'lbfgs', got '{self.solver}'."
            )

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Exact solver (closed-form Ridge) — handle before generic routing
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            torch_device = _get_torch_device_str()
            if not isinstance(X, torch.Tensor):
                X = torch.from_numpy(np.asarray(X, dtype=np.float64)).to(torch_device)
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(np.asarray(y, dtype=np.float64)).to(torch_device)
            if X.dtype != torch.float64:
                X = X.to(torch.float64)
            if y.dtype != torch.float64:
                y = y.to(torch.float64)
            if sample_weight is not None:
                if not isinstance(sample_weight, torch.Tensor):
                    sample_weight = torch.as_tensor(sample_weight, dtype=X.dtype, device=X.device)
                else:
                    sample_weight = sample_weight.to(dtype=X.dtype, device=X.device)
                sqrt_sw = torch.sqrt(sample_weight)
                X = X * sqrt_sw[:, None]
                y = y * sqrt_sw
            if self._effective_intercept:
                X_mean = torch.mean(X, dim=0)
                y_mean = torch.mean(y)
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_centered = X
                y_mean = torch.tensor(0.0, dtype=X.dtype, device=X.device)
                y_centered = y
            if y_centered.ndim == 1:
                y_centered = y_centered.reshape(-1)
            _cv = getattr(self, '_cv_cache', None)
            if _cv is not None and 'XtX' in _cv:
                XtX = _cv['XtX']
                Xty = _cv['Xty']
            else:
                XtX = X_centered.T @ X_centered
                Xty = X_centered.T @ y_centered
            coef = self._solve_exact_torch(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self.compute_inference:
                if self._effective_intercept:
                    coef_full_torch = torch.cat([
                        (y_mean.reshape(1) - X_mean.reshape(1, -1) @ coef.reshape(-1, 1)).reshape(-1),
                        coef.reshape(-1),
                    ])
                    self._precompute_exact_l2_inference_torch(
                        X,
                        y,
                        XtX,
                        X_mean,
                        coef_full_torch.reshape(-1),
                        n_samples,
                    )
                else:
                    self._precompute_exact_l2_inference_torch(
                        X,
                        y,
                        XtX,
                        None,
                        coef.reshape(-1),
                        n_samples,
                    )
            coef_np = coef.cpu().numpy()
            if self._effective_intercept:
                self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))
            # Debiased inference on Torch GPU (before cleanup)
            if self.compute_inference and "debiased" in str(getattr(self, "inference_method", "")).lower():
                penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
                if penalty_name in ("l1", "elasticnet", "en"):
                    self._compute_inference_debiased_torch(X, y, coef)
            self._cleanup_torch_memory()
            return

        # Route IRLS/newton/lbfgs through their dedicated backends.
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._fit_irls_backend(X, y, sample_weight, "torch")
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, "torch")
            return

        # Route non-L1 and non-squared-error through the generic loss backend.
        if self.loss != "squared_error" or solver_name == "admm" or self._penalty.name not in ("l1", "elasticnet", "en"):
            self._fit_loss_backend(X, y, sample_weight, solver_name, "torch")
            return

        torch_device = _get_torch_device_str()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch_device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(torch_device)
        if X.dtype != torch.float64:
            X = X.to(torch.float64)
        if y.dtype != torch.float64:
            y = y.to(torch.float64)

        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                sample_weight = torch.as_tensor(np.asarray(sample_weight, dtype=np.float64), device=torch_device)
            sqrt_sw = torch.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        if self._effective_intercept:
            X_mean = torch.mean(X, dim=0)
            y_mean = torch.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = torch.tensor(0.0, dtype=torch.float64, device=torch_device)
            y_centered = y

        if y_centered.ndim == 1:
            y_centered = y_centered.reshape(-1)

        # Precompute (use CV cache if available)
        _cv = getattr(self, '_cv_cache', None)
        if _cv is not None and 'XtX' in _cv:
            XtX = _cv['XtX']
            Xty = _cv['Xty']
        else:
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered

        # Lipschitz constant: L = lambda_max(XtX) / n
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            if n_features < 1000:
                L = float(torch.linalg.eigvalsh(XtX)[-1]) / n_samples
            else:
                v = torch.ones(n_features, dtype=X.dtype, device=X.device)
                v /= torch.linalg.norm(v)
                for _ in range(50):
                    v_new = XtX @ v
                    v_norm = torch.linalg.norm(v_new)
                    if v_norm < 1e-15:
                        break
                    v = v_new / v_norm
                L = float((v @ (XtX @ v)) / n_samples)

        if L <= 0:
            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            self.n_iter_ = 0
        elif solver_name in ("fista_bb", "fista"):
            # Standard FISTA with XtX precomputation.
            # BB step provides no benefit for quadratic losses (BB1=BB2).
            # All element-wise ops (gradient finish + proximal + momentum) are
            # fused via torch.compile into fewer CUDA kernels, matching cupy's @cp.fuse().
            step = 1.0 / L
            step_over_n = step / n_samples
            step_over_n_Xty = step_over_n * Xty  # (p,) — precompute once
            if self._penalty.name in ("elasticnet", "en"):
                thresh = self.alpha * self._penalty.l1_ratio * step
                l2_scale = 1.0 + self.alpha * (1.0 - self._penalty.l1_ratio) * step
            else:
                thresh = self.alpha * step
                l2_scale = 1.0
            _use_l2 = abs(l2_scale - 1.0) > 1e-12

            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = torch.tensor(self._init_coef, dtype=X.dtype, device=X.device)
            else:
                coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            y_k = coef.clone()
            t_k = 1.0
            beta = 0.0

            # Warm-up: compile fused kernel BEFORE the loop to avoid
            # first-iteration JIT compilation overhead.
            _fused_step = None
            _fused_step_l2 = None
            from statgpu.penalties import _torch_compile_ok as _tc_ok
            if _tc_ok():
                try:
                    def _fista_elementwise_l2(
                        _y_k, _xtx_y, _step_over_n_Xty,
                        _step_over_n, _thresh, _l2_scale,
                        _coef_old, _beta,
                    ):
                        w = (_y_k - _step_over_n * _xtx_y + _step_over_n_Xty)
                        c = (torch.sign(w) * torch.relu(torch.abs(w) - _thresh) / _l2_scale)
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step_l2 = torch.compile(_fista_elementwise_l2, mode='reduce-overhead')
                    _dummy = torch.zeros(1, dtype=X.dtype, device=X.device)
                    _fused_step_l2(_dummy, _dummy, _dummy, 0.0, 0.0, 1.0, _dummy, 0.0)
                except Exception:
                    _fused_step_l2 = None
                try:
                    def _fista_elementwise(
                        _y_k, _xtx_y, _step_over_n_Xty,
                        _step_over_n, _thresh, _coef_old, _beta,
                    ):
                        w = (_y_k - _step_over_n * _xtx_y + _step_over_n_Xty)
                        c = (torch.sign(w) * torch.relu(torch.abs(w) - _thresh))
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step = torch.compile(_fista_elementwise, mode='reduce-overhead')
                    _dummy = torch.zeros(1, dtype=X.dtype, device=X.device)
                    _fused_step(_dummy, _dummy, _dummy, 0.0, 0.0, _dummy, 0.0)
                except Exception:
                    _fused_step = None

            for iteration in range(self.max_iter):
                coef_old = coef.clone()

                # cuBLAS matvec via ATen (cannot fuse with element-wise ops)
                xtx_y = XtX @ y_k  # (p,)

                if _use_l2:
                    if _fused_step_l2 is not None:
                        coef, y_k = _fused_step_l2(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, l2_scale, coef_old, beta,
                        )
                    else:
                        w_tilde = (y_k - step_over_n * xtx_y
                                   + step_over_n_Xty)
                        coef = (torch.sign(w_tilde)
                                * torch.relu(torch.abs(w_tilde) - thresh)
                                / l2_scale)
                        y_k = coef + beta * (coef - coef_old)
                else:
                    if _fused_step is not None:
                        coef, y_k = _fused_step(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, coef_old, beta,
                        )
                    else:
                        w_tilde = (y_k - step_over_n * xtx_y
                                   + step_over_n_Xty)
                        coef = (torch.sign(w_tilde)
                                * torch.relu(torch.abs(w_tilde) - thresh))
                        y_k = coef + beta * (coef - coef_old)

                # Scheduled momentum restart (zero sync overhead)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0

                # Nesterov momentum (beta for next iteration)
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                t_k = t_new

                self.n_iter_ = iteration + 1

                if iteration % 5 == 4 and float(torch.sum(torch.abs(coef - coef_old)).item()) < self.tol:
                    break
        else:
            step = 1.0 / L
            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = torch.tensor(self._init_coef, dtype=X.dtype, device=X.device)
            else:
                coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            y_k = coef.clone()
            t_k = torch.tensor(1.0, dtype=X.dtype, device=X.device)

            for iteration in range(self.max_iter):
                coef_old = coef.clone()

                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad

                coef = self._penalty.proximal(w_tilde, step, backend="torch")

                # Scheduled momentum restart (BEFORE momentum update)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = torch.tensor(1.0, dtype=X.dtype, device=X.device)

                t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                self.n_iter_ = iteration + 1

                if iteration % 5 == 4 and float(torch.sum(torch.abs(coef - coef_old)).item()) < self.tol:
                    break

        # Transfer to CPU
        coef_np = coef.cpu().numpy()

        if self._effective_intercept:
            self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))

        # Debiased inference on Torch GPU (before cleanup)
        if self.compute_inference and "debiased" in str(getattr(self, "inference_method", "")).lower():
            penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
            if penalty_name in ("l1", "elasticnet", "en"):
                self._compute_inference_debiased_torch(X, y, coef)

        self._cleanup_torch_memory()

    def _ridge_alpha_for_exact(self) -> float:
        """Return L2 alpha for the exact Ridge normal equations."""
        return float(getattr(self._penalty, "alpha", self.alpha))

    def _solve_exact_numpy(self, XtX, Xty, n_samples):
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        # Per-sample convention: XtX is unnormalized (X'X), so we need
        # n*alpha to match loss/n + alpha*||w||^2 used by all other paths.
        A = XtX + (float(n_samples) * alpha) * np.eye(p, dtype=XtX.dtype)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ Xty

    def _solve_exact_cupy(self, XtX, Xty, n_samples):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * cp.eye(p, dtype=XtX.dtype)
        try:
            # Cholesky + triangular solve is faster than general solve
            # for positive-definite matrices (Ridge penalty guarantees PD)
            L = cp.linalg.cholesky(A)
            tmp = cp_solve_triangular(L, Xty, lower=True)
            return cp_solve_triangular(L.T, tmp, lower=False)
        except _LINALG_ERRORS:
            try:
                return cp.linalg.solve(A, Xty)
            except _LINALG_ERRORS:
                return cp.linalg.pinv(A) @ Xty

    def _solve_exact_torch(self, XtX, Xty, n_samples):
        import torch

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * torch.eye(
            p, dtype=XtX.dtype, device=XtX.device
        )
        try:
            # torch.linalg.solve is faster than Cholesky + solve_triangular
            # on PyTorch due to kernel launch overhead for small matrices
            return torch.linalg.solve(A, Xty)
        except RuntimeError:
            return torch.linalg.pinv(A) @ Xty

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

    def _prepare_predict_X(self, X):
        """Apply stored formula design metadata to DataFrame inputs."""
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
        return np.asarray(X)

    def _prediction_backend_name(self):
        backend_name = getattr(self, "_selected_backend_name", None)
        if backend_name == "cupy" and self._cupy_available():
            return "cupy"
        if backend_name == "torch" and self._torch_cuda_available():
            return "torch"
        if backend_name == "numpy":
            return "numpy"
        if self.device == Device.AUTO:
            return "numpy"
        device = self._get_compute_device()
        if device == Device.CUDA:
            if self._cupy_available():
                return "cupy"
            raise RuntimeError(
                "device='cuda' was explicitly requested, but CuPy/CUDA is unavailable at prediction time."
            )
        if device == Device.TORCH:
            if self._torch_cuda_available():
                return "torch"
            raise RuntimeError(
                "device='torch' was explicitly requested, but Torch CUDA is unavailable at prediction time."
            )
        return "numpy"

    def predict(self, X, return_cpu=True):
        """
        Predict using fitted model.

        For squared_error: returns linear prediction.
        For logistic: returns binary class labels.
        For poisson: returns exp(linear prediction) (count values).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        return_cpu : bool, default=True
            If True, always return a numpy ndarray (GPU→CPU transfer happens
            automatically when the model was fitted on GPU).  If False, return
            the result in the same backend as the fitted coefficients (cupy/
            torch when fitted on GPU, numpy when fitted on CPU).  Setting to
            False avoids an unnecessary D→H transfer when chaining GPU
            operations (e.g., ``model.predict(X_gpu) - y_gpu``).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        backend_name = self._prediction_backend_name()
        if backend_name == "cupy":
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self._effective_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            if self.loss == "logistic":
                p = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
                result = (p > 0.5).astype(float)
            elif self.loss != "squared_error":
                result = self._family_for_loss().link.inverse(raw)
            else:
                result = raw
            return _to_numpy(result) if return_cpu else result
        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            if self.loss == "logistic":
                p = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -_ETA_CLIP, _ETA_CLIP)))
                result = (p > 0.5).to(raw.dtype)
            elif self.loss != "squared_error":
                result = self._family_for_loss().link.inverse(raw)
            else:
                result = raw
            return _to_numpy(result) if return_cpu else result

        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_

        # Apply link inverse for GLM losses
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
            return (p > 0.5).astype(float)
        elif self.loss != "squared_error":
            return self._family_for_loss().link.inverse(raw)
        return raw

    def score(self, X, y, sample_weight=None):
        """
        Return goodness-of-fit score.

        For squared_error loss, returns R² (1 - SS_res/SS_tot).
        For GLM losses, returns 1 - deviance/null_deviance (pseudo-R²).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. When provided, returns weighted score.

        Returns
        -------
        score : float
            R² or pseudo-R² score.
        """
        y_pred = self.predict(X, return_cpu=False)
        device = self._get_compute_device()
        sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None

        if device == Device.CUDA:
            import cupy as cp
            yb = cp.asarray(self._to_array(y, Device.CUDA))
            y_pred_dev = cp.asarray(y_pred) if isinstance(y_pred, cp.ndarray) else cp.asarray(_to_numpy(y_pred))
            resid_sq = (yb - y_pred_dev) ** 2
            if sw is not None:
                sw_dev = cp.asarray(sw, dtype=cp.float64)
                w_sum = float(cp.sum(sw_dev).get())
                if w_sum <= 0:
                    return 0.0
                ss_res = float(cp.sum(sw_dev * resid_sq).get())
                ss_tot = float(cp.sum(sw_dev * (yb - cp.average(yb, weights=sw_dev)) ** 2).get())
            else:
                ss_res = float(cp.sum(resid_sq).get())
                ss_tot = float(cp.sum((yb - cp.mean(yb)) ** 2).get())
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if device == Device.TORCH:
            import torch
            yb = self._to_array(y, Device.TORCH, backend="torch").to(y_pred.dtype)
            if isinstance(y_pred, torch.Tensor):
                y_pred_dev = y_pred.to(dtype=yb.dtype, device=yb.device)
            else:
                y_pred_dev = torch.as_tensor(_to_numpy(y_pred), dtype=yb.dtype, device=yb.device)
            resid_sq = (yb - y_pred_dev) ** 2
            if sw is not None:
                sw_dev = torch.as_tensor(sw, dtype=yb.dtype, device=yb.device)
                w_sum = float(sw_dev.sum().item())
                if w_sum <= 0:
                    return 0.0
                ss_res = float((sw_dev * resid_sq).sum().item())
                y_wmean = float((sw_dev * yb).sum().item()) / w_sum
                ss_tot = float((sw_dev * (yb - y_wmean) ** 2).sum().item())
            else:
                ss_res = float(resid_sq.sum().item())
                ss_tot = float(((yb - yb.mean()) ** 2).sum().item())
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        y = np.asarray(y)
        y_pred_np = np.asarray(_to_numpy(y_pred))
        resid_sq = (y - y_pred_np) ** 2
        if sw is not None:
            w_sum = float(np.sum(sw))
            if w_sum <= 0:
                return 0.0
            ss_res = float(np.sum(sw * resid_sq))
            ss_tot = float(np.sum(sw * (y - np.average(y, weights=sw)) ** 2))
        else:
            ss_res = float(np.sum(resid_sq))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _family_for_loss(self):
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
            return Binomial()
        if self.loss == "poisson":
            return Poisson()
        if self.loss == "gamma":
            return Gamma()
        if self.loss == "inverse_gaussian":
            return InverseGaussian()
        if self.loss == "negative_binomial":
            alpha = getattr(
                getattr(self, "_loss", None),
                "alpha",
                getattr(self, "loss_kwargs", {}).get("alpha", 1.0),
            )
            return NegativeBinomial(alpha=alpha)
        if self.loss == "tweedie":
            power = getattr(
                getattr(self, "_loss", None),
                "power",
                getattr(self, "loss_kwargs", {}).get("power", 1.5),
            )
            return Tweedie(power=power)
        return Gaussian()

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
            return cp.ones(n, dtype=ref.dtype)
        if backend_name == "torch":
            import torch
            return torch.ones(n, dtype=ref.dtype, device=ref.device)
        return np.ones(n, dtype=getattr(ref, "dtype", np.float64))

    def _selective_penalty(self, p, backend_name):
        """Penalty wrapper that leaves the last intercept coefficient free.

        Uses a thread-local singleton to avoid per-call class creation
        while remaining safe for concurrent CV folds.
        """
        singleton = _get_selective_penalty_singleton()
        singleton.configure(self._penalty, p, backend_name)
        return singleton

    def _block_cd_group_lasso(self, pen, X_work, y_arr, init):
        """Block coordinate descent for group_lasso penalty.

        Matches R grpreg's block CD algorithm: iterate over groups, compute
        partial residual per group, solve the group subproblem, apply block
        soft-thresholding.
        """
        import numpy as np

        n, pp = X_work.shape
        p = pp - 1 if self._effective_intercept else pp
        alpha = self.alpha

        _inner = getattr(self, '_penalty', pen)
        _g_indices = getattr(_inner, '_group_indices', None)
        _sqrt_pg = getattr(_inner, '_sqrt_pg', None)
        if _g_indices is None or _sqrt_pg is None:
            raise ValueError(
                "group_lasso penalty must have groups set. "
                "Pass groups=... in penalty_kwargs."
            )
        _n_groups = len(_g_indices)

        XtX = X_work.T @ X_work / n
        Xty = (X_work.T @ y_arr.flatten()) / n

        _XtX_blocks = []
        for g_idx in _g_indices:
            _XtX_blocks.append(XtX[np.ix_(g_idx, g_idx)])

        if init is not None:
            coef = np.array(init, dtype=np.float64)
        else:
            coef = np.zeros(pp, dtype=np.float64)

        iteration = -1  # ensure defined when max_iter=0
        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            for g in range(_n_groups):
                g_idx = _g_indices[g]
                rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + _XtX_blocks[g] @ coef[g_idx]
                try:
                    w_g = np.linalg.solve(_XtX_blocks[g], rho_g)
                except np.linalg.LinAlgError:
                    w_g = np.zeros(len(g_idx))
                norm_w = np.linalg.norm(w_g)
                thresh_g = alpha * _sqrt_pg[g]
                if norm_w > thresh_g:
                    coef[g_idx] = w_g * (1.0 - thresh_g / norm_w)
                else:
                    coef[g_idx] = 0.0

            if self._effective_intercept:
                coef[pp - 1] = np.mean(y_arr - X_work[:, :p] @ coef[:p])

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        n_iter = iteration + 1

        if self._effective_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0

        return beta, intercept, n_iter

    def _block_cd_group_lasso_gpu(self, pen, X_work, y_arr, init, backend_name):
        """GPU-native block coordinate descent for group_lasso penalty.

        Same algorithm as _block_cd_group_lasso but keeps all arrays on GPU.
        Enforces float64 precision to avoid NaN from float32 conditioning issues.
        """
        from statgpu.backends._array_ops import _xp_copy, _xp_zeros, _xp_asarray, _xp_eye
        from statgpu.backends._utils import _get_xp, xp_astype
        xp = _get_xp(backend_name)

        # Enforce float64 precision for numerical stability
        X_work = xp_astype(X_work, xp.float64, xp)
        y_arr = xp_astype(y_arr, xp.float64, xp)

        n, pp = X_work.shape
        p = pp - 1 if self._effective_intercept else pp
        alpha = self.alpha

        _inner = getattr(self, '_penalty', pen)
        _g_indices = getattr(_inner, '_group_indices', None)
        _sqrt_pg_np = getattr(_inner, '_sqrt_pg', None)
        if _g_indices is None or _sqrt_pg_np is None:
            raise ValueError(
                "group_lasso penalty must have groups set. "
                "Pass groups=... in penalty_kwargs."
            )
        _n_groups = len(_g_indices)
        _sqrt_pg = [float(s) for s in _sqrt_pg_np]

        XtX = X_work.T @ X_work / n
        Xty = (X_work.T @ y_arr.flatten()) / n

        # Pre-compute XtX blocks with diagonal ridge for conditioning
        from statgpu.backends._array_ops import _scalar_tensor
        _XtX_blocks = []
        _ridge = _scalar_tensor(1e-10, X_work)
        for g_idx in _g_indices:
            block = XtX[g_idx][:, g_idx]
            block = block + _ridge * _xp_eye(block.shape[0], block.dtype, block)
            _XtX_blocks.append(block)

        if init is not None:
            if isinstance(init, np.ndarray):
                coef = _xp_asarray(init, X_work.dtype, X_work)
            else:
                coef = _xp_copy(init)
        else:
            coef = _xp_zeros(pp, X_work.dtype, X_work)

        iteration = -1  # ensure defined when max_iter=0
        for iteration in range(self.max_iter):
            coef_old = _xp_copy(coef)

            for g in range(_n_groups):
                g_idx = _g_indices[g]
                rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + _XtX_blocks[g] @ coef[g_idx]
                try:
                    w_g = xp.linalg.solve(_XtX_blocks[g], rho_g)
                    if xp.any(xp.isnan(w_g)) or xp.any(xp.isinf(w_g)):
                        w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
                except Exception:
                    w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
                norm_w = float(xp.linalg.norm(w_g))
                thresh_g = alpha * _sqrt_pg[g]
                if norm_w > thresh_g:
                    coef[g_idx] = w_g * (1.0 - thresh_g / norm_w)
                else:
                    coef[g_idx] = 0.0

            if self._effective_intercept:
                coef[pp - 1] = float(xp.mean(y_arr - X_work[:, :p] @ coef[:p]))

            _max_change = float(xp.max(xp.abs(coef - coef_old)))
            if _max_change < self.tol:
                break

        n_iter = iteration + 1

        if self._effective_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0

        return beta, intercept, n_iter

    def _fit_loss_backend(self, X, y, sample_weight, solver_name, backend_name):
        """Fit GLMLoss + Penalty without changing the selected backend."""
        from statgpu.glm_core._solver import (
            fista_solver,
            fista_bb_solver,
            admm_solver,
            lbfgs_solver,
            newton_solver,
        )

        # Convert to target backend with float64 precision for numerical stability
        from statgpu.backends._array_ops import _xp_asarray
        from statgpu.backends._utils import _get_xp
        _xp = _get_xp(backend_name)
        _ref = X if not isinstance(X, np.ndarray) else _xp.zeros(1, dtype=_xp.float64)
        X_arr = _xp_asarray(X, _xp.float64, _ref)
        y_arr = _xp_asarray(y, _xp.float64, X_arr)
        if self._effective_intercept:
            p = X_arr.shape[1]
            X_work = self._column_stack(
                [X_arr, self._ones(X_arr.shape[0], backend_name, X_arr)],
                backend_name,
            )
            pen = self._selective_penalty(p, backend_name)
            init = None
            if self._init_coef is not None:
                init_intercept = float(getattr(self, '_init_intercept', 0.0) or 0.0)
                init = np.append(self._init_coef, init_intercept)
                init = _xp_asarray(init, X_arr.dtype, X_arr)
            else:
                # Warm-start intercept for GLM losses (prevents divergence
                # of the unpenalized intercept toward -inf for zero-heavy data).
                _loss_name = getattr(self._loss, 'name', '')
                _y_mean = float(np.mean(_to_numpy(y_arr)))
                if _loss_name == "poisson":
                    _int_init = np.log(max(_y_mean, 1e-3))
                elif _loss_name == "logistic":
                    _y_mean_clipped = np.clip(_y_mean, 1e-3, 1.0 - 1e-3)
                    _int_init = np.log(_y_mean_clipped / (1.0 - _y_mean_clipped))
                elif _loss_name in ("gamma", "inverse_gaussian", "negative_binomial", "tweedie"):
                    # All use log link: intercept init = log(y_mean)
                    _int_init = np.log(max(_y_mean, 1e-3))
                else:
                    _int_init = _y_mean  # identity link (squared_error)
                init = np.zeros(p + 1)
                init[-1] = _int_init
                init = _xp_asarray(init, X_arr.dtype, X_arr)
        else:
            p = X_arr.shape[1]
            X_work = X_arr
            pen = self._penalty
            init = None
            if self._init_coef is not None:
                init = np.asarray(self._init_coef, dtype=np.float64)
                init = _xp_asarray(init, X_arr.dtype, X_arr)

        # SCAD/MCP and adaptive_l1 use IRLS-CD (matching R ncvreg's
        # per-coordinate algorithm).  GLM+SCAD/MCP uses 1 CD sweep per
        # IRLS iteration to avoid cycling.
        _loss_name = getattr(self._loss, 'name', '')
        _pen_name = getattr(pen, 'name', '')
        # SelectivePenalty (intercept wrapper) has no name; fall back to
        # the original penalty's name so SCAD/MCP routing works.
        if not _pen_name:
            _pen_name = getattr(self._penalty, 'name', '')
        _is_glm_loss = _loss_name not in ("squared_error", "")
        # Routing:
        #   adaptive_l1/adaptive_lasso → FISTA (weighted L1 proximal, works
        #     for both GLM and squared_error; avoids slow sequential CD)
        #   squared_error + SCAD/MCP → IRLS-CD (matching R ncvreg)
        #   GLM + SCAD/MCP → IRLS-CD (matching R ncvreg's IRLS+CD algorithm)
        _use_fista = _pen_name in ("adaptive_l1", "adaptive_lasso")
        _use_irls_cd = (
            (_pen_name in ("scad", "mcp") and not _is_glm_loss)
        )
        _use_lla_fista = (
            _pen_name in ("scad", "mcp") and _is_glm_loss
        )
        _use_lla_group = (
            _pen_name in ("group_mcp", "group_scad", "gmcp", "gscad") and _is_glm_loss
        )

        if _use_fista:
            # FISTA for GLM+adaptive_l1 — works on any backend.
            from statgpu.glm_core._solver import fista_solver
            params, n_iter = fista_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        elif _use_irls_cd:
            # squared_error + SCAD/MCP: use fused FISTA+LLA on all backends.
            # Produces identical results across CPU/GPU and avoids slow
            # sequential coordinate descent on GPU.
            from statgpu.glm_core._solver import fista_lla_path
            import numpy as _np

            # Compute continuation path (lambda_max → target alpha)
            _X_feat = _to_numpy(X_work[:, :p] if self._effective_intercept else X_work)
            _y_feat = _to_numpy(y_arr)
            _n = _X_feat.shape[0]
            _col_norms = _np.sqrt(_np.sum(_X_feat ** 2, axis=0))
            _col_norms = _np.maximum(_col_norms, 1e-20)
            _X_s = _X_feat * (_np.sqrt(_n) / _col_norms)
            _y_c = _y_feat - _np.mean(_y_feat)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ _y_c / _n)))
            _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))
            _n_cont = 20
            _alpha_path = _np.geomspace(
                max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
            )
            _max_lla_per_step = max(6, getattr(self, '_max_lla_iters', 50) // _n_cont)
            _saved_mi = self.max_iter
            _mi_path = []
            for _i in range(_n_cont):
                _is_last = (_i == _n_cont - 1)
                _mi_path.append(_saved_mi if _is_last else max(100, _saved_mi // 10))

            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self._effective_intercept,
                sample_weight=sample_weight,
            )
            if self._effective_intercept:
                params_np = np.concatenate([coef_np, [intercept]])
            else:
                params_np = coef_np
            params = params_np
        elif _use_lla_fista:
            # GLM + SCAD/MCP: use LLA outer loop + FISTA inner solve.
            from statgpu.glm_core._solver import fista_lla_path
            import numpy as _np

            xp = get_backend(backend_name).xp

            # lambda_max with backend-native arrays (no CPU-GPU transfer)
            X_feat = X_work[:, :p] if self._effective_intercept else X_work
            _n = X_feat.shape[0]
            _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
            if backend_name == "torch":
                import torch
                _col_norms = torch.clamp(_col_norms, min=1e-20)
            else:
                _col_norms = xp.maximum(_col_norms, 1e-20)
            X_s = X_feat * (float(_n) ** 0.5 / _col_norms)
            y_c = y_arr - xp.mean(y_arr)
            _lam_max = float(xp.max(xp.abs(X_s.T @ y_c / _n)))
            _cv_alpha_path = getattr(self, '_cv_alpha_path', None)
            _cv_return_path = _cv_alpha_path is not None
            if _cv_return_path:
                _targets = _np.asarray(_cv_alpha_path, dtype=float).ravel()
                _targets = _targets[_np.isfinite(_targets) & (_targets > 0.0)]
                if _targets.size == 0:
                    _targets = _np.asarray([float(getattr(self._penalty, 'alpha', self.alpha))])
                _targets = _np.sort(_targets)[::-1]
                _target_alpha = float(_targets[-1])
                _alpha_start = max(_lam_max, float(_targets[0]) * 1.1)
                if _alpha_start > float(_targets[0]) * (1.0 + 1e-10):
                    _alpha_path = _np.concatenate([[_alpha_start], _targets])
                else:
                    _alpha_path = _targets
                _n_cont = int(_alpha_path.size)
            else:
                _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))
                _n_cont = 20
                _alpha_path = _np.geomspace(
                    max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
                )

            _max_lla_per_step = max(6, getattr(self, '_max_lla_iters', 50) // max(_n_cont, 1))
            _saved_mi = self.max_iter
            if _cv_return_path:
                _mi_path = [max(200, _saved_mi // 2)] * max(_n_cont - 1, 0) + [_saved_mi]
            else:
                _mi_path = [_saved_mi if i == _n_cont - 1 else max(100, _saved_mi // 10)
                            for i in range(_n_cont)]

            X_orig = X_work[:, :p] if self._effective_intercept else X_work

            _warm_coef = None
            _warm_intercept = None
            _init = getattr(self, '_init_coef', None)
            if _init is not None:
                _init_np = np.asarray(_to_numpy(_init), dtype=np.float64).ravel()
                if self._effective_intercept and _init_np.size == p + 1:
                    _warm_coef = _init_np[:p]
                    _warm_intercept = float(_init_np[p])
                elif _init_np.size == p:
                    _warm_coef = _init_np
                    if self._effective_intercept:
                        _warm_intercept = float(
                            getattr(self, '_init_intercept', 0.0) or 0.0
                        )

            _lla_result = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self._effective_intercept,
                sample_weight=sample_weight,
                init_coef=_warm_coef,
                init_intercept=_warm_intercept,
                return_path=_cv_return_path,
            )
            if _cv_return_path:
                coef_np, intercept, n_iter, _path_results = _lla_result
                self._cv_path_results = _path_results
            else:
                coef_np, intercept, n_iter = _lla_result
            # fista_lla_path returns numpy, convert back to backend-native
            if self._effective_intercept:
                params = xp.concatenate([xp.asarray(coef_np), xp.asarray([intercept])])
            else:
                params = xp.asarray(coef_np)
        elif _use_lla_group:
            # GLM + group_mcp/group_scad: LLA outer loop + FISTA inner solve
            # with AdaptiveGroupLassoPenalty as inner penalty.
            from statgpu.glm_core._solver import fista_lla_path
            from statgpu.penalties._group_lasso import AdaptiveGroupLassoPenalty
            import numpy as _np

            xp = get_backend(backend_name).xp

            # lambda_max with backend-native arrays
            X_feat = X_work[:, :p] if self._effective_intercept else X_work
            _n = X_feat.shape[0]
            _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
            if backend_name == "torch":
                import torch
                _col_norms = torch.clamp(_col_norms, min=1e-20)
            else:
                _col_norms = xp.maximum(_col_norms, 1e-20)
            X_s = X_feat * (float(_n) ** 0.5 / _col_norms)
            y_c = y_arr - xp.mean(y_arr)
            _lam_max = float(xp.max(xp.abs(X_s.T @ y_c / _n)))
            _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))

            _n_cont = 20
            _alpha_path = _np.geomspace(
                max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
            )
            _max_lla_per_step = max(6, getattr(self, '_max_lla_iters', 50) // _n_cont)
            _saved_mi = self.max_iter
            _mi_path = [_saved_mi if i == _n_cont - 1 else max(100, _saved_mi // 10)
                        for i in range(_n_cont)]

            # Create penalty factory for group LLA
            _orig_pen = self._penalty  # unwrap SelectivePenalty
            _groups = getattr(_orig_pen, '_group_indices', None)
            _pen_alpha = float(_orig_pen.alpha)

            # Create penalty object once; reuse via set_weights() to avoid
            # repeated _init_groups() + object creation overhead.
            _adaptive_pen = AdaptiveGroupLassoPenalty(
                groups=_groups, alpha=_pen_alpha,
            )
            def _group_lla_factory(weights_np):
                # lla_weights returns per-coordinate; compute per-group weights
                # as the norm of the per-coordinate weights within each group
                _gw = np.array([
                    float(np.sqrt(np.sum(weights_np[idx] ** 2))) if len(idx) > 0 else 0.0
                    for idx in _groups
                ])
                _adaptive_pen.set_weights(_gw)
                return _adaptive_pen

            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self._effective_intercept,
                sample_weight=sample_weight,
                lla_penalty_factory=_group_lla_factory,
            )
            # fista_lla_path returns numpy, convert back to backend-native
            if self._effective_intercept:
                params = xp.concatenate([xp.asarray(coef_np), xp.asarray([intercept])])
            else:
                params = xp.asarray(coef_np)
        elif _pen_name == "group_lasso":
            # Block CD for group_lasso: use GPU-native solver on GPU backends.
            if backend_name != "numpy":
                coef_gpu, intercept, n_iter = self._block_cd_group_lasso_gpu(
                    pen, X_work, y_arr, init, backend_name,
                )
                if self._effective_intercept:
                    from statgpu.backends._utils import _get_xp as _get_xp_fn
                    from statgpu.backends._array_ops import _xp_asarray as _xp_asarray_fn
                    _xp = _get_xp_fn(backend_name)
                    _int_arr = _xp_asarray_fn([intercept], coef_gpu.dtype, coef_gpu)
                    params = _xp.concatenate([coef_gpu, _int_arr])
                else:
                    params = coef_gpu
            else:
                coef_np, intercept, n_iter = self._block_cd_group_lasso(
                    pen, X_work, y_arr, init,
                )
                if self._effective_intercept:
                    params = np.concatenate([coef_np, [intercept]])
                else:
                    params = coef_np
        elif solver_name == "auto":
            # For smooth penalties (l2, elasticnet with low l1_ratio),
            # fista_bb with BB step sizes converges much more reliably
            # than standard FISTA with Nesterov momentum + proximal l2.
            _is_smooth = (_pen_name == "l2") or (
                _pen_name == "elasticnet" and
                float(getattr(pen, 'l1_ratio', 1.0)) < 0.5
            )
            if _is_smooth:
                params, n_iter = fista_bb_solver(
                    self._loss, pen, X_work, y_arr,
                    max_iter=self.max_iter, tol=self.tol,
                    init_coef=init, sample_weight=sample_weight,
                )
            else:
                params, n_iter = fista_solver(
                    self._loss, pen, X_work, y_arr,
                    max_iter=self.max_iter, tol=self.tol,
                    init_coef=init, sample_weight=sample_weight,
                )
        elif solver_name == "fista":
            params, n_iter = fista_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        elif solver_name == "fista_bb":
            params, n_iter = fista_bb_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        elif solver_name == "admm":
            params, n_iter = admm_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter,
                tol=self.tol, rho=1.0, adaptive_rho=True,
                init_coef=init, sample_weight=sample_weight,
            )
        elif solver_name == "newton":
            params, n_iter = newton_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        elif solver_name == "lbfgs":
            params, n_iter = lbfgs_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        else:
            raise ValueError(f"Unsupported solver: {solver_name}")

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self._effective_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self._effective_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _fit_irls_backend(self, X, y, sample_weight=None, backend_name="numpy"):
        """Fit smooth L2 GLM via IRLS on the selected backend."""
        from statgpu.glm_core._irls import IRLSSolver

        if str(getattr(self._penalty, "name", self.penalty)).lower() != "l2":
            raise ValueError("solver='irls' only supports L2 penalties.")

        from statgpu.backends._utils import _get_xp, xp_asarray
        _xp = _get_xp(backend_name)
        X_arr = xp_asarray(X, dtype=_xp.float64, xp=_xp, ref_arr=X if not isinstance(X, np.ndarray) else np.zeros(1))
        y_arr = xp_asarray(y, dtype=_xp.float64, xp=_xp, ref_arr=X_arr)
        n_samples = X_arr.shape[0]
        if self._effective_intercept:
            X_work = self._column_stack(
                [self._ones(X_arr.shape[0], backend_name, X_arr), X_arr],
                backend_name,
            )
        else:
            X_work = X_arr

        # Respect CV warm starts first.  IRLS uses [intercept, coef...] while
        # the FISTA design stores the intercept as the final column.
        _loss_name = getattr(self._loss, 'name', '')
        init_coef = None
        init_features = getattr(self, '_init_coef', None)
        if init_features is not None:
            init_features_np = np.asarray(init_features, dtype=np.float64).ravel()
            if self._effective_intercept:
                init_intercept = float(getattr(self, '_init_intercept', 0.0) or 0.0)
                init_coef_np = np.concatenate([[init_intercept], init_features_np])
            else:
                init_coef_np = init_features_np
            if backend_name == "cupy":
                import cupy as cp
                init_coef = cp.asarray(init_coef_np, dtype=cp.float64)
            elif backend_name == "torch":
                import torch
                init_coef = torch.as_tensor(
                    init_coef_np,
                    dtype=torch.float64,
                    device=X_work.device,
                )
            else:
                init_coef = init_coef_np

        # Otherwise warm-start intercept for GLM losses whose default eta=0
        # can be far from the intercept-only optimum.
        _log_link_losses = ("gamma", "poisson", "inverse_gaussian",
                            "negative_binomial", "tweedie")
        if init_coef is None and self._effective_intercept and (
            _loss_name in _log_link_losses or _loss_name == "logistic"
        ):
            _y_mean = float(np.mean(_to_numpy(y_arr)))
            if _loss_name == "logistic":
                _y_mean = float(np.clip(_y_mean, 1e-3, 1.0 - 1e-3))
                _int_init = np.log(_y_mean / (1.0 - _y_mean))
            else:
                _int_init = np.log(max(_y_mean, 1e-3))
            n_feat = X_work.shape[1]
            if backend_name == "numpy":
                init_coef = np.zeros(n_feat, dtype=np.float64)
            elif backend_name == "cupy":
                import cupy as cp
                init_coef = cp.zeros(n_feat, dtype=cp.float64)
            else:
                import torch
                init_coef = torch.zeros(n_feat, dtype=torch.float64,
                                        device=X_work.device)
            init_coef_np = np.zeros(n_feat)
            init_coef_np[0] = _int_init
            if backend_name == "cupy":
                import cupy as cp
                init_coef = cp.asarray(init_coef_np)
            elif backend_name == "torch":
                import torch
                init_coef = torch.from_numpy(init_coef_np).to(X_work.device)
            else:
                init_coef = init_coef_np

        solver = IRLSSolver(
            self._family_for_loss(), max_iter=self.max_iter, tol=self.tol
        )
        params, n_iter = solver.fit(
            X_work, y_arr,
            sample_weight=sample_weight,
            ridge_alpha=float(n_samples * self.alpha),
            ridge_penalize_intercept=False if self._effective_intercept else True,
            backend=backend_name,
            init_coef=init_coef,
        )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self._effective_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self._effective_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _cleanup_cuda_memory(self):
        """Free CuPy memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Free Torch memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


