"""Fit mixin for PenalizedGeneralizedLinearModel."""

from __future__ import annotations

import numpy as np

from statgpu._config import Device
from statgpu.backends import get_backend, _to_numpy, _LINALG_ERRORS
from statgpu.solvers._utils import _nesterov_momentum, _nesterov_update

# ---------------------------------------------------------------------------
# Solver dispatch table for solver='auto'
# ---------------------------------------------------------------------------
# Each entry is (solver, condition_fn). First match wins.
# condition_fn takes (loss, penalty, backend, l1_ratio, cv_mode, problem_size).

# Import shared penalty categories (single source of truth)
from statgpu.penalties._categories import (
    NONCONVEX as _NONCONVEX_PENALTIES,
    SPARSE as _SPARSE_PENALTIES,
)
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})

# Losses with special LLA handling (not routed through generic GLM path).
# squared_error: quadratic, uses fused FISTA-LLA fast path.
# quantile: non-smooth gradient, uses proximal IRLS-CD.
# All others (GLM, robust, CoxPH): use FISTA-LLA with generic gradient().
_SPECIAL_LLA_LOSSES = frozenset({"squared_error", "quantile", ""})

# SCAD/MCP continuation path parameters (shared across all fit paths).
# Reduced from 20/6 to 10/4: benchmark shows 1.4x speedup with <1e-11 error.
_N_CONT_STEPS = 5
_N_CONT_STEPS_NONSMOOTH = 3  # Fewer steps for non-smooth losses (quantile) — FISTA is slow per step
_MAX_LLA_PER_STEP_DEFAULT = 2

# (solver, condition)
# condition = (loss, penalty, backend, l1_ratio, cv_mode, problem_size) -> bool
_SOLVER_DISPATCH_TABLE = [
    # -- Priority 1: Exact closed-form solutions (highest priority) --
    # Ridge + squared_error: exact eigendecomposition on CPU, Newton on GPU
    # (cuSOLVER eigendecomposition has high overhead for small/medium matrices).
    ("exact", lambda l, p, b, lr, cv, ps: l == "squared_error" and p == "l2" and b in ("numpy", "cpu", "")),
    ("newton", lambda l, p, b, lr, cv, ps: l == "squared_error" and p == "l2" and b in ("cupy", "torch")),

    # -- Priority 2: Nonconvex penalties always use FISTA+LLA wrapper --
    # SCAD/MCP/adaptive_l1 require iteratively reweighted L1 (LLA approximation).
    ("fista", lambda l, p, b, lr, cv, ps: p in _NONCONVEX_PENALTIES),

    # -- Priority 2b: Quantile loss has no Hessian -> always FISTA --
    ("fista", lambda l, p, b, lr, cv, ps: l == "quantile"),

    # -- Priority 3: Squared error + sparse penalties -> FISTA --
    # Quadratic loss + L1/ElasticNet: FISTA with exact line search.
    ("fista", lambda l, p, b, lr, cv, ps: l == "squared_error" and p in _SPARSE_PENALTIES),

    # -- Priority 4: GLM + GPU + sparse penalties (size-gated) --
    # Poisson + GPU + L1: fista_bb for small/medium problems (< 2M elements).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and b in ("cupy", "torch") and p == "l1" and (ps is None or ps < 2_000_000)),
    # Poisson + GPU + ElasticNet: fista_bb (BB step adapts well to EN geometry).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and b in ("cupy", "torch") and p in ("elasticnet", "en")),
    # Poisson + CPU + sparse: FISTA (CPU backtracking is cheap).
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "poisson" and p in _SPARSE_PENALTIES),

    # -- Priority 5: NB + GPU + sparse penalties --
    # NB + GPU + L1: fista_bb (NB gradient is well-behaved for BB steps).
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p == "l1"),
    # NB + GPU + ElasticNet: FISTA for medium problems (200K-1M), fista_bb otherwise.
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p in ("elasticnet", "en") and ps is not None and 200_000 <= ps < 1_000_000),
    ("fista_bb", lambda l, p, b, lr, cv, ps: cv and l == "negative_binomial" and b in ("cupy", "torch") and p in ("elasticnet", "en")),

    # -- Priority 6: Gamma/IG/Tweedie + sparse -> FISTA --
    # These families have steep loss landscapes; FISTA with backtracking is safer.
    ("fista", lambda l, p, b, lr, cv, ps: l in ("gamma", "inverse_gaussian") and p in _SPARSE_PENALTIES),
    ("fista", lambda l, p, b, lr, cv, ps: l == "tweedie" and b in ("cupy", "torch") and p in _SPARSE_PENALTIES),

    # -- Priority 7: Logistic + sparse -> FISTA --
    # Logistic has iterate-dependent Lipschitz; FISTA with fixed global bound.
    ("fista", lambda l, p, b, lr, cv, ps: cv and l == "logistic" and p in _SPARSE_PENALTIES),

    # -- Priority 7b: Robust losses + sparse -> FISTA --
    ("fista", lambda l, p, b, lr, cv, ps: l in ("huber", "bisquare", "fair") and p in _SPARSE_PENALTIES),

    # -- Priority 8: Default sparse -> fista_bb --
    # Catch-all for remaining sparse penalty cases.
    ("fista_bb", lambda l, p, b, lr, cv, ps: p in _SPARSE_PENALTIES),

    # -- Priority 9: CV + L2: loss-specific smooth solvers --
    # NB needs L-BFGS (non-canonical link issues with IRLS).
    ("lbfgs", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l == "negative_binomial"),
    # Poisson/Tweedie: Newton (canonical link, well-conditioned).
    ("newton", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l in ("poisson", "tweedie")),
    # Gamma/IG: L-BFGS (non-canonical link, better convergence).
    ("lbfgs", lambda l, p, b, lr, cv, ps: cv and p == "l2" and l in ("gamma", "inverse_gaussian")),

    # -- Priority 10: Smooth penalties (L2/none) with loss-specific solvers --
    # All GLM families: Newton (fastest convergence, 2-11 iterations).
    # Fixed: expected Fisher Hessian (W=mu) for gamma/tweedie/IG ensures
    # positive-definite Hessian and proper convergence.
    ("newton", lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in (
        "gamma", "tweedie", "inverse_gaussian", "logistic", "poisson", "negative_binomial")),
    # Robust losses (Huber/Bisquare/Fair) and CoxPH: Newton for smooth penalties.
    # These losses have Hessian and smooth gradient.
    ("newton", lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in (
        "huber", "bisquare", "fair", "cox_ph")),
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


def _resolve_loss_name(loss_name, loss_kwargs=None):
    """Resolve loss name string to loss object.

    Tries the GLM-specific registry first, then falls back to the generic
    loss registry (quantile, huber, cox_ph, etc.).
    """
    loss_kwargs = loss_kwargs or {}
    try:
        from statgpu.glm_core._base import get_glm_loss
        return get_glm_loss(loss_name, **loss_kwargs)
    except (ValueError, KeyError, TypeError):
        from statgpu.losses import get_loss
        return get_loss(loss_name, **loss_kwargs)


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
        coef = _irls_ridge_init_cd(X, y, alpha, max_iter, tol)
    else:
        # For GLM losses, use FISTA with L2 penalty (robust line search)
        # Pass arrays directly — solver handles backend detection internally
        from statgpu.solvers import fista_solver
        from statgpu.penalties import get_penalty
        l2_pen = get_penalty("l2", alpha=alpha)
        loss_obj = _resolve_loss_name(loss_name, loss_kwargs=loss_kwargs)
        coef, _ = fista_solver(loss_obj, l2_pen, X, y, max_iter=max_iter, tol=tol)
    # Return as numpy array (caller expects numpy for penalty.set_weights)
    from statgpu.backends import _to_numpy
    return np.asarray(_to_numpy(coef), dtype=np.float64)


def _irls_ridge_init_cd(X, y, alpha, max_iter, tol):
    """Ridge regression initialization for adaptive L1 weights.

    Uses closed-form solution: beta = (X'X + alpha*I)^-1 X'y
    which is O(p^3) but fully parallelizable on GPU (single matmul + solve).
    Much faster than sequential coordinate descent on GPU.
    """
    from statgpu.backends import _resolve_backend
    from statgpu.backends._utils import _get_xp

    backend = _resolve_backend("auto", X)
    xp = _get_xp(backend)

    n, p = X.shape
    # Normalize features
    feat_norms = xp.sqrt(xp.sum(X ** 2, axis=0))
    if backend == "torch":
        import torch
        feat_norms = xp.maximum(feat_norms, torch.tensor(1e-20, dtype=feat_norms.dtype, device=feat_norms.device))
        scale = torch.tensor(float(n) ** 0.5, dtype=X.dtype, device=X.device) / feat_norms
    else:
        feat_norms = xp.maximum(feat_norms, 1e-20)
        scale = xp.asarray(float(n) ** 0.5, dtype=X.dtype) / feat_norms
    X_work = X * scale

    # Closed-form Ridge: (X'X + alpha*I)^-1 X'y
    XtX = X_work.T @ X_work / n
    Xty = X_work.T @ y / n

    if backend == "torch":
        import torch
        I_mat = torch.eye(p, dtype=X.dtype, device=X.device)
        beta = torch.linalg.solve(XtX + alpha * I_mat, Xty)
    elif backend == "cupy":
        import cupy as cp
        I_mat = cp.eye(p, dtype=X.dtype)
        beta = cp.linalg.solve(XtX + alpha * I_mat, Xty)
    else:
        I_mat = np.eye(p, dtype=X.dtype)
        beta = np.linalg.solve(XtX + alpha * I_mat, Xty)

    return beta * scale


class _PenalizedFitMixin:

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
        self._loss = self._resolve_loss()
        self._validate_solver_penalty()
        self._validate_inference_request()

        # Pre-compute scale for robust losses (avoids per-iteration overhead)
        if hasattr(self._loss, 'precompute_scale') and X is not None and y is not None:
            self._loss.precompute_scale(X, y)
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

        # Convert sample_weight to target backend once (avoids CPU/CUDA mismatch)
        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name)

        # Handle penalties requiring initialization (e.g., Adaptive Lasso)
        if self._penalty.requires_init:
            init_coef = self._fit_initial(X, y, backend_name=backend_name)
            self._penalty.set_weights(init_coef)

        # Non-convex penalties (SCAD, MCP) for squared_error: use IRLS-CD
        # directly with a 100-step continuation path from lambda_max.
        # This matches R ncvreg's algorithm for Gaussian regression.
        # GLM+SCAD/MCP must NOT use IRLS-CD -- it cycles due to non-convex
        # penalty causing features to flip on/off between IRLS iterations.
        # GLM+SCAD/MCP goes through _fit_lla -> FISTA with proximal operator.
        _pen_name = str(getattr(self._penalty, 'name', '')).lower()
        _loss_name = str(getattr(self._loss, 'name', '') if hasattr(self, '_loss') else self.loss).lower()
        # squared_error/quantile use IRLS-CD for SCAD/MCP (fast quadratic path
        # or quantile-specific IRLS). All other losses (GLM, robust, CoxPH)
        # use FISTA-LLA with the generic loss.gradient() interface.
        _is_glm_loss = _loss_name not in _SPECIAL_LLA_LOSSES
        if _pen_name in ("scad", "mcp") and self._lla_enabled and not _is_glm_loss:
            self._nobs = X.shape[0]
            X_arr = self._to_array(X, backend=backend_name)
            y_arr = self._to_array(y, backend=backend_name)
            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(
                X_arr, y_arr, X_arr.shape[1], _loss_name)

            if _loss_name == "quantile":
                # Quantile + SCAD/MCP: use Proximal IRLS (IRLS quadratic
                # majorization + LLA for nonconvex penalty). Much faster than
                # FISTA-LLA: IRLS provides curvature info, converges in ~20-50
                # iterations vs ~1800 for FISTA.
                from statgpu.solvers import proximal_irls_quantile_solver
                coef_np, intercept, n_iter = proximal_irls_quantile_solver(
                    self._loss, self._penalty,
                    X_arr, y_arr,
                    alpha_path=_alpha_path,
                    max_lla_per_step=_max_lla_per_step,
                    lla_tol=getattr(self, '_lla_tol', 1e-6),
                    max_iter=_mi_path,
                    tol=self.tol,
                    fit_intercept=self._effective_intercept,
                    sample_weight=_sw_arr,
                )
            else:
                # squared_error + SCAD/MCP: use fused FISTA+LLA path.
                from statgpu.solvers import fista_lla_path
                coef_np, intercept, n_iter = fista_lla_path(
                    self._loss, self._penalty,
                    X_arr, y_arr,
                    alpha_path=_alpha_path,
                    max_lla_per_step=_max_lla_per_step,
                    lla_tol=getattr(self, '_lla_tol', 1e-6),
                    max_iter=_mi_path,
                    tol=self.tol,
                    fit_intercept=self._effective_intercept,
                    sample_weight=_sw_arr,
                )
            self.coef_ = coef_np
            self.intercept_ = intercept
            self.n_iter_ = n_iter
            if self._effective_intercept:
                self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
            else:
                self._params = np.asarray(self.coef_).copy()
            self._df_resid = X.shape[0] - (X.shape[1] + (1 if self._effective_intercept else 0))
            self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)
            if backend_name == "cupy":
                self._cleanup_cuda_memory()
            elif backend_name == "torch":
                self._cleanup_torch_memory()
            self._fitted = True
            return self

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)

        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, _sw_arr)
        elif backend_name == "cupy":
            self._fit_gpu(X_arr, y_arr, _sw_arr)
        else:
            self._fit_cpu(X_arr, y_arr, _sw_arr)

        self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)
        self._fitted = True
        # Clean up CV cache unless a caller is intentionally reusing one
        # across repeated fits, as PenalizedGLM_CV does within a fold.
        if hasattr(self, '_cv_cache') and not getattr(self, '_preserve_cv_cache', False):
            del self._cv_cache
        return self

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

        # CuPy->Torch overrides: prefer torch when available, else CPU
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
        penalties (SCAD, MCP) that will enter the LLA outer loop -- a sparse
        seed gives LLA differentiated weights and drives genuine sparsity.
        Convex penalties with ``requires_init=True`` (adaptive_l1) need a
        dense seed because their weights are 1/|coef| -- zero entries from
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
            from statgpu.solvers import fista_solver

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
            # -> smaller weights -> too many features surviving.
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

        from statgpu.linear_model.wrappers._ridge import Ridge

        init_model = Ridge(
            alpha=0.1,
            fit_intercept=self._effective_intercept,
            device=self.device,
        )
        init_model.fit(X, y)
        return init_model.coef_

    def _compute_lla_path(self, X_work, y_arr, p, loss_name, n_cont=None):
        """Compute LLA continuation path (lambda_max -> target alpha).

        Shared helper for quantile+SCAD/MCP and squared_error+SCAD/MCP paths.
        Returns (alpha_path, max_lla_per_step, mi_path).

        For quantile: lambda_max uses pinball subgradient X'@psi_tau(y-intercept)/n
        For others: lambda_max uses X'@centered(y)/n (squared-error style)
        """
        import numpy as _np

        _X_feat = _to_numpy(X_work[:, :p] if self._effective_intercept else X_work)
        _y_feat = _to_numpy(y_arr)
        _n = _X_feat.shape[0]

        if loss_name == "quantile":
            # Quantile-specific lambda_max: max_j |X_j' @ psi_tau(y - intercept) / n|
            _tau = getattr(self._loss, '_tau', 0.5)
            _intercept = float(_np.quantile(_y_feat, _tau))
            _r = _y_feat - _intercept
            _psi = _np.where(_r >= 0, _tau, -(1.0 - _tau))
            _lam_max = float(_np.max(_np.abs(_X_feat.T @ _psi / _n)))
        else:
            # Squared-error style: max_j |X_j' @ centered(y) / n|
            _col_norms = _np.sqrt(_np.sum(_X_feat ** 2, axis=0))
            _col_norms = _np.maximum(_col_norms, 1e-20)
            _X_s = _X_feat * (_np.sqrt(_n) / _col_norms)
            _y_c = _y_feat - _np.mean(_y_feat)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ _y_c / _n)))
        _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))

        if n_cont is None:
            n_cont = _N_CONT_STEPS_NONSMOOTH if loss_name == "quantile" else _N_CONT_STEPS

        _alpha_start = max(_lam_max, _target_alpha * 1.1)
        if (not _np.isfinite(_alpha_start)) or _alpha_start <= 0.0 or _target_alpha <= 0.0:
            _alpha_path = _np.linspace(max(_lam_max, 0.0), _target_alpha, n_cont)
        else:
            _alpha_path = _np.geomspace(_alpha_start, _target_alpha, n_cont)

        _max_lla = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // n_cont)
        _saved_mi = self.max_iter
        _mi_path = [_saved_mi if i == n_cont - 1 else max(100, _saved_mi // 10)
                    for i in range(n_cont)]

        return _alpha_path, _max_lla, _mi_path

    def _dispatch_irls(self, X, y, sample_weight, solver_name, backend_name):
        """Route IRLS to the correct backend.

        GLM losses use family-based IRLS (_fit_irls_backend).
        Non-GLM losses with _supports_irls (quantile, bisquare, fair)
        use loss-specific irls() via _fit_loss_backend.
        """
        from statgpu.glm_core._base import GLMLoss
        if isinstance(self._loss, GLMLoss):
            self._fit_irls_backend(X, y, sample_weight, backend_name)
        else:
            self._fit_loss_backend(X, y, sample_weight, solver_name, backend_name)

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
        if self.loss != "squared_error" or solver_name in ("irls", "newton", "lbfgs", "admm"):
            if solver_name == "irls":
                self._dispatch_irls(X, y, sample_weight, solver_name, "numpy")
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

        # Lipschitz constant: L = lambda_max(XtX) / n
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
                    y_k, t_k = _nesterov_update(coef, coef_old, t_k)

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
        self._fit_gpu_backend(X, y, sample_weight, backend_name="cupy")

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FISTA. Delegates to unified backend."""
        self._fit_gpu_backend(X, y, sample_weight, backend_name="torch")

    # ------------------------------------------------------------------
    # Unified GPU backend (replaces _fit_gpu + _fit_torch)
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_threshold_gpu(w, thresh, xp):
        """Backend-agnostic soft-thresholding on GPU."""
        if xp.__name__ == "torch":
            import torch
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh)
        return xp.sign(w) * xp.maximum(xp.abs(w) - thresh, 0.0)

    def _fit_gpu_backend(self, X, y, sample_weight=None, backend_name="cupy"):
        """Unified GPU fit method for both CuPy and Torch backends.

        Handles exact (L2), FISTA, and FISTA-BE solvers with inline
        XtX precomputation and fused element-wise kernels.
        """
        from statgpu.backends._utils import _get_xp, xp_asarray, xp_zeros, xp_copy, xp_ones
        from statgpu.backends import _to_numpy
        from statgpu.backends._array_ops import _abs_sum_dev

        xp = _get_xp(backend_name)
        is_torch = (backend_name == "torch")

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name=backend_name
        )
        _backend_label = "Torch" if is_torch else "CuPy"
        if solver_name not in ("fista", "fista_bb", "admm", "auto", "exact", "irls", "newton", "lbfgs"):
            raise ValueError(
                f"{_backend_label} backend supports solver='fista', 'fista_bb', 'admm', "
                f"'exact', 'irls', 'newton', and 'lbfgs', got '{solver_name}'."
            )

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # --- Exact solver (closed-form Ridge) ---
        if solver_name == "exact":
            if self._penalty.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            X = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
            y = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=y)
            if is_torch:
                import torch
                if X.dtype != torch.float64:
                    X = X.to(torch.float64)
            if sample_weight is not None:
                sw = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X)
                sqrt_sw = xp.sqrt(sw)
                X = X * sqrt_sw[:, None]
                y = y * sqrt_sw
            if self._effective_intercept:
                X_mean = xp.mean(X, axis=0)
                y_mean = xp.mean(y)
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_centered = X
                y_mean = xp_zeros((), X.dtype, xp, ref_arr=X) if is_torch else xp.array(0.0, dtype=X.dtype)
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

            # Dispatch to backend-specific exact solver
            solve_fn = getattr(self, f'_solve_exact_{"torch" if is_torch else "cupy"}')
            coef = solve_fn(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self.compute_inference:
                infer_fn = getattr(self, f'_precompute_exact_l2_inference_{"torch" if is_torch else "cupy"}')
                if self._effective_intercept:
                    intercept_gpu = (y_mean.reshape(1) - X_mean.reshape(1, -1) @ coef.reshape(-1, 1)).reshape(-1)
                    coef_full_gpu = xp.concatenate([intercept_gpu, coef.reshape(-1)])
                    infer_fn(X, y, XtX, X_mean, coef_full_gpu.reshape(-1), n_samples)
                else:
                    infer_fn(X, y, XtX, None, coef.reshape(-1), n_samples)
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

        # Route IRLS/newton/lbfgs through their dedicated backends.
        if solver_name in ("irls", "newton", "lbfgs"):
            if solver_name == "irls":
                self._dispatch_irls(X, y, sample_weight, solver_name, backend_name)
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, backend_name)
            return

        # Route non-L1 and non-squared-error through the generic loss backend.
        if self.loss != "squared_error" or solver_name == "admm" or self._penalty.name not in ("l1", "elasticnet", "en"):
            self._fit_loss_backend(X, y, sample_weight, solver_name, backend_name)
            return

        # --- Inline FISTA fast-path for L1 + squared_error ---
        X = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
        y = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=y)
        if is_torch:
            import torch
            if X.dtype != torch.float64:
                X = X.to(torch.float64)

        if sample_weight is not None:
            sample_weight = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X)
            sqrt_sw = xp.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        if self._effective_intercept:
            X_mean = xp.mean(X, axis=0)
            y_mean = xp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = xp_zeros((), X.dtype, xp, ref_arr=X) if is_torch else xp.array(0.0, dtype=X.dtype)
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

        # Lipschitz constant: L = lambda_max(XtX) / n
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            if n_features < 1000:
                L = float(xp.linalg.eigvalsh(XtX)[-1]) / n_samples
            else:
                v = xp_ones(n_features, X.dtype, xp, ref_arr=X)
                v = v / xp.linalg.norm(v)
                for _ in range(50):
                    v_new = XtX @ v
                    v_norm = xp.linalg.norm(v_new)
                    if v_norm < 1e-15:
                        break
                    v = v_new / v_norm
                L = float(_to_numpy(v @ (XtX @ v))) / n_samples

        if L <= 0:
            coef = xp_zeros(n_features, X.dtype, xp, ref_arr=X)
            self.n_iter_ = 0
        elif solver_name in ("fista_bb", "fista"):
            step = 1.0 / L
            step_over_n = step / n_samples
            step_over_n_Xty = step_over_n * Xty
            if self._penalty.name in ("elasticnet", "en"):
                thresh = self.alpha * self._penalty.l1_ratio * step
                l2_scale = 1.0 + self.alpha * (1.0 - self._penalty.l1_ratio) * step
            else:
                thresh = self.alpha * step
                l2_scale = 1.0
            _use_l2 = abs(l2_scale - 1.0) > 1e-12

            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = xp_asarray(self._init_coef, dtype=X.dtype, xp=xp, ref_arr=X)
            else:
                coef = xp_zeros(n_features, X.dtype, xp, ref_arr=X)
            y_k = xp_copy(coef)
            t_k = 1.0
            beta = 0.0

            # Build fused element-wise kernel (backend-specific JIT)
            _fused_step = None
            _fused_step_l2 = None
            _st_fn = self._soft_threshold_gpu

            if is_torch:
                import torch
                if _use_l2:
                    # torch.compile requires Triton (CUDA capability >= 7.0).
                    # On older GPUs (e.g. Tesla P100, CUDA 6.0), skip compilation
                    # and use the plain eager-mode function directly.
                    _can_compile = (torch.cuda.is_available()
                                    and torch.cuda.get_device_capability()[0] >= 7)
                    if _can_compile:
                        try:
                            def _fista_elementwise_l2(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                                                      _thresh, _l2_scale, _coef_old, _beta):
                                w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                                c = _st_fn(w, _thresh, xp) / _l2_scale
                                y = c + _beta * (c - _coef_old)
                                return c, y
                            _fused_step_l2 = torch.compile(_fista_elementwise_l2, mode='reduce-overhead')
                        except Exception:
                            _fused_step_l2 = None
                else:
                    _can_compile = (torch.cuda.is_available()
                                    and torch.cuda.get_device_capability()[0] >= 7)
                    if _can_compile:
                        try:
                            def _fista_elementwise(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                                                   _thresh, _coef_old, _beta):
                                w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                                c = _st_fn(w, _thresh, xp)
                                y = c + _beta * (c - _coef_old)
                                return c, y
                            _fused_step = torch.compile(_fista_elementwise, mode='reduce-overhead')
                        except Exception:
                            _fused_step = None
            else:
                import cupy as cp
                if _use_l2:
                    try:
                        @cp.fuse()
                        def _fista_elementwise_l2(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                                                  _thresh, _l2_scale, _coef_old, _beta):
                            w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                            c = (cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0) / _l2_scale)
                            y = c + _beta * (c - _coef_old)
                            return c, y
                        _fused_step_l2 = _fista_elementwise_l2
                        _dummy = cp.zeros(1, dtype=X.dtype)
                        _fused_step_l2(_dummy, _dummy, _dummy, 0.0, 0.0, 1.0, _dummy, 0.0)
                    except Exception:
                        _fused_step_l2 = None
                else:
                    try:
                        @cp.fuse()
                        def _fista_elementwise(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                                               _thresh, _coef_old, _beta):
                            w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                            c = (cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0))
                            y = c + _beta * (c - _coef_old)
                            return c, y
                        _fused_step = _fista_elementwise
                        _dummy = cp.zeros(1, dtype=X.dtype)
                        _fused_step(_dummy, _dummy, _dummy, 0.0, 0.0, _dummy, 0.0)
                    except Exception:
                        _fused_step = None

            for iteration in range(self.max_iter):
                coef_old = xp_copy(coef)
                xtx_y = XtX @ y_k

                if _use_l2:
                    if _fused_step_l2 is not None:
                        coef, y_k = _fused_step_l2(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, l2_scale, coef_old, beta,
                        )
                    else:
                        w_tilde = y_k - step_over_n * xtx_y + step_over_n_Xty
                        coef = _st_fn(w_tilde, thresh, xp) / l2_scale
                        y_k = coef + beta * (coef - coef_old)
                else:
                    if _fused_step is not None:
                        coef, y_k = _fused_step(
                            y_k, xtx_y, step_over_n_Xty, step_over_n,
                            thresh, coef_old, beta,
                        )
                    else:
                        w_tilde = y_k - step_over_n * xtx_y + step_over_n_Xty
                        coef = _st_fn(w_tilde, thresh, xp)
                        y_k = coef + beta * (coef - coef_old)

                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0

                beta, t_k = _nesterov_momentum(t_k)

                self.n_iter_ = iteration + 1
                if iteration % 5 == 4 and float(_to_numpy(_abs_sum_dev(coef - coef_old))) < self.tol:
                    break
        else:
            step = 1.0 / L
            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = xp_asarray(self._init_coef, dtype=X.dtype, xp=xp, ref_arr=X)
            else:
                coef = xp_zeros(n_features, X.dtype, xp, ref_arr=X)
            y_k = xp_copy(coef)
            t_k = 1.0

            for iteration in range(self.max_iter):
                coef_old = xp_copy(coef)
                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad
                coef = self._penalty.proximal(w_tilde, step, backend=backend_name)

                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0

                y_k, t_k = _nesterov_update(coef, coef_old, t_k)

                self.n_iter_ = iteration + 1
                if iteration % 5 == 4 and float(_to_numpy(_abs_sum_dev(coef - coef_old))) < self.tol:
                    break

        # Transfer to CPU
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

        # Debiased inference on GPU (before cleanup)
        if self.compute_inference and "debiased" in str(getattr(self, "inference_method", "")).lower():
            penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
            if penalty_name in ("l1", "elasticnet", "en"):
                infer_fn = getattr(self, f'_compute_inference_debiased_{"torch" if is_torch else "gpu"}')
                infer_fn(X, y, coef)

        if is_torch:
            self._cleanup_torch_memory()
        else:
            self._cleanup_cuda_memory()

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

        # Pre-compute batched XtX blocks for vectorized solve (equal-size groups)
        _equal_size = len(set(len(g) for g in _g_indices)) == 1
        _gs = len(_g_indices[0]) if _equal_size else 0
        _contiguous = _equal_size and all(
            _g_indices[g][0] == g * _gs for g in range(_n_groups)
        )
        if _equal_size and _n_groups > 1:
            _XtX_batched = xp.stack(_XtX_blocks)  # (G, gs, gs)
            _sqrt_pg_arr = xp.asarray(_sqrt_pg, dtype=X_work.dtype)

        iteration = -1  # ensure defined when max_iter=0
        for iteration in range(self.max_iter):
            coef_old = _xp_copy(coef)

            if _equal_size and _n_groups > 1:
                # ── Vectorized path: all groups at once ──
                # Compute XtX @ coef once (shared across groups)
                XtX_coef = XtX @ coef  # (pp,)

                if _contiguous:
                    coef_mat = coef[:p].reshape(_n_groups, _gs)
                    XtX_coef_mat = XtX_coef[:p].reshape(_n_groups, _gs)
                    Xty_mat = Xty[:p].reshape(_n_groups, _gs)
                else:
                    flat_idx = xp.asarray([i for g in _g_indices for i in g], dtype=xp.int64)
                    coef_mat = coef[flat_idx].reshape(_n_groups, _gs)
                    XtX_coef_mat = XtX_coef[flat_idx].reshape(_n_groups, _gs)
                    Xty_mat = Xty[flat_idx].reshape(_n_groups, _gs)

                # rho_g = Xty[g] - XtX[g,:] @ coef + XtX_blocks[g] @ coef[g]
                #       = Xty[g] - XtX_coef[g] + diag_blocks @ coef_g
                diag_contrib = xp.einsum('gsj,gj->gs', _XtX_batched, coef_mat)
                rho_mat = Xty_mat - XtX_coef_mat + diag_contrib  # (G, gs)

                # Batched solve: w_g = XtX_blocks[g]^{-1} @ rho_g
                try:
                    w_mat = xp.linalg.solve(_XtX_batched, rho_mat)  # (G, gs)
                except Exception:
                    w_mat = xp.zeros_like(rho_mat)
                bad = xp.isnan(w_mat) | xp.isinf(w_mat)
                if xp.any(bad):
                    w_mat = xp.where(bad, 0.0, w_mat)

                # Vectorized group thresholding
                norms = xp.sqrt(xp.sum(w_mat ** 2, axis=1))  # (G,)
                thresh = alpha * _sqrt_pg_arr  # (G,)
                scale = xp.where(norms > thresh, 1.0 - thresh / (norms + 1e-300), 0.0)
                scaled_mat = w_mat * scale[:, None]  # (G, gs)

                # Scatter back
                if _contiguous:
                    coef[:p] = scaled_mat.reshape(-1)
                else:
                    coef[flat_idx] = scaled_mat.reshape(-1)
            else:
                # ── Serial path: unequal groups ──
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
        from statgpu.solvers import (
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
                elif _loss_name in ("gamma", "inverse_gaussian", "negative_binomial", "tweedie", "cox_ph"):
                    # All use log link: intercept init = log(y_mean)
                    _int_init = np.log(max(_y_mean, 1e-3))
                elif _loss_name == "quantile":
                    # Use empirical quantile as intercept warm start
                    _tau = getattr(self._loss, '_tau', 0.5)
                    _int_init = float(np.quantile(_to_numpy(y_arr), _tau))
                else:
                    _int_init = _y_mean  # identity link (squared_error)
                # For robust/quantile losses: use OLS as warm start
                # (zeros is a poor starting point for non-quadratic losses)
                _robust_losses = ("quantile", "huber", "bisquare", "fair")
                if _loss_name in _robust_losses:
                    _X_np = _to_numpy(X_arr)
                    _y_np = _to_numpy(y_arr)
                    _ols_coef = np.linalg.lstsq(_X_np, _y_np, rcond=None)[0]
                    init = np.append(_ols_coef, _int_init)
                else:
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
        # Routing:
        #   adaptive_l1/adaptive_lasso -> FISTA (weighted L1 proximal)
        #   quantile + SCAD/MCP -> CD solver (coordinate descent, much faster)
        #   squared_error + SCAD/MCP -> IRLS-CD (matching R ncvreg)
        #   GLM + SCAD/MCP -> FISTA-LLA (Proximal Newton for losses with Hessian)
        _is_glm_loss = _loss_name not in _SPECIAL_LLA_LOSSES
        _use_fista = _pen_name in ("adaptive_l1", "adaptive_lasso")
        _use_quantile_cd = (_loss_name == "quantile" and _pen_name in ("scad", "mcp"))
        _use_irls_cd = (
            (_pen_name in ("scad", "mcp") and _loss_name == "squared_error")
        )
        _use_lla_fista = (
            _pen_name in ("scad", "mcp") and _is_glm_loss and _loss_name != "squared_error"
        )
        _use_lla_group = (
            _pen_name in ("group_mcp", "group_scad", "gmcp", "gscad") and _is_glm_loss
        )

        if _use_fista:
            # FISTA for GLM+adaptive_l1 -- works on any backend.
            params, n_iter = fista_solver(
                self._loss, pen, X_work, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=init, sample_weight=sample_weight,
            )
        elif _use_quantile_cd:
            # Quantile + SCAD/MCP: use Proximal IRLS (IRLS quadratic majorization
            # + LLA for nonconvex penalty). Much faster than FISTA-LLA or subgradient CD.
            from statgpu.solvers import proximal_irls_quantile_solver
            import numpy as _np

            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(
                X_work, y_arr, p, _loss_name)
            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = proximal_irls_quantile_solver(
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
                params_np = _np.concatenate([coef_np, [intercept]])
            else:
                params_np = coef_np
            params = _xp_asarray(params_np, X_arr.dtype, X_arr)
        elif _use_irls_cd:
            # squared_error + SCAD/MCP: use fused FISTA+LLA on all backends.
            # Produces identical results across CPU/GPU and avoids slow
            # sequential coordinate descent on GPU.
            from statgpu.solvers import fista_lla_path
            import numpy as _np

            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(
                X_work, y_arr, p, _loss_name)

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
            from statgpu.solvers import fista_lla_path
            import numpy as _np

            xp = get_backend(backend_name).xp

            # lambda_max with backend-native arrays (no CPU-GPU transfer).
            # Cox has a two-column (time, event) response, so the GLM-style
            # X.T @ centered(y) expression is both dimensionally wrong for a
            # coefficient path and unrelated to the Cox score.  At beta=0 the
            # maximum absolute partial-likelihood gradient is the correct
            # zero-solution threshold for the weighted-L1 LLA subproblem.
            X_feat = X_work[:, :p] if self._effective_intercept else X_work
            _n = X_feat.shape[0]
            if _loss_name == "cox_ph":
                if backend_name == "torch":
                    import torch
                    _zero_coef = torch.zeros(
                        p, dtype=X_feat.dtype, device=X_feat.device
                    )
                else:
                    _zero_coef = xp.zeros(p, dtype=X_feat.dtype)
                _score_at_zero = self._loss.gradient(
                    X_feat, y_arr, _zero_coef, sample_weight=sample_weight
                )
                _lam_max = float(xp.max(xp.abs(_score_at_zero)))
            else:
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
                _n_cont = _N_CONT_STEPS_NONSMOOTH if _loss_name == "quantile" else _N_CONT_STEPS
                _alpha_path = _np.geomspace(
                    max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
                )

            _max_lla_per_step = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // max(_n_cont, 1))
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

            # For one-dimensional losses with Hessian (Bisquare, Huber,
            # etc.): use OLS as
            # warm-start if no explicit init_coef is provided. This prevents
            # the continuation path from shrinking everything to zero at the
            # first (large-alpha) step.  Cox's response is (time, event): OLS
            # would return a (p, 2) matrix which cannot warm-start a p-vector.
            # Cox therefore follows the continuation path from zero unless an
            # explicit p-vector warm start is supplied by the caller/CV layer.
            _y_ndim = getattr(y_arr, "ndim", None)
            if _y_ndim is None:
                _y_ndim = np.asarray(y_arr).ndim
            _y_ndim = int(_y_ndim)
            if (_warm_coef is None
                    and getattr(self._loss, 'has_hessian', False)
                    and _y_ndim == 1):
                _X_np = np.asarray(_to_numpy(X_orig), dtype=np.float64)
                _y_np = np.asarray(_to_numpy(y_arr), dtype=np.float64)
                _warm_coef = np.linalg.lstsq(_X_np, _y_np, rcond=None)[0]

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
            from statgpu.solvers import fista_lla_path
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

            # Fewer continuation steps for non-smooth losses (FISTA is slow per step)
            _n_cont = _N_CONT_STEPS_NONSMOOTH if _loss_name == "quantile" else _N_CONT_STEPS
            _alpha_path = _np.geomspace(
                max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
            )
            _max_lla_per_step = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // _n_cont)
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
            # Block CD for group_lasso (converges in 2-5 iterations).
            # CoxPH has 2D y (time, event) — BCD doesn't handle this,
            # so route through FISTA which calls loss.preprocess() internally.
            _use_bcd = _loss_name != "cox_ph"
            if not _use_bcd:
                # CoxPH: BCD doesn't handle 2D y, use FISTA
                from statgpu.solvers import fista_solver
                params, n_iter = fista_solver(
                    self._loss, pen, X_work, y_arr,
                    max_iter=self.max_iter, tol=self.tol,
                    init_coef=init, sample_weight=sample_weight,
                )
            elif backend_name != "numpy":
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
        elif solver_name == "fista":
            # For quantile loss with smooth penalty: use IRLS (FISTA diverges
            # on non-smooth losses). IRLS converges to the same solution as
            # sklearn's HiGHS LP solver.
            # IRLS is backend-aware — no _to_numpy() needed.
            _loss_name = getattr(self._loss, 'name', '')
            _has_irls = hasattr(self._loss, 'irls')
            _is_smooth_pen = _pen_name in ("l2", "none", "null", "")
            if _loss_name == "quantile" and _has_irls and _is_smooth_pen:
                _inner_pen = getattr(self._penalty, '_pen', self._penalty)
                _irls_tol = min(self.tol, 1e-8)
                params_irls, n_iter = self._loss.irls(
                    X_work, y_arr,
                    penalty=_inner_pen,
                    max_iter=self.max_iter, tol=_irls_tol,
                    init_coef=None,
                    sample_weight=sample_weight,
                    fit_intercept=self._effective_intercept,
                )
                params = _xp_asarray(params_irls, X_arr.dtype, X_arr)
            else:
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
        elif solver_name == "irls":
            # Non-GLM losses with _supports_irls (quantile, bisquare, fair).
            # Call loss.irls() directly — these losses have their own IRLS
            # implementation that doesn't need a GLM family.
            # (Validation in _validate_solver_penalty already rejected
            # losses without _supports_irls.)
            _inner_pen = getattr(self._penalty, '_pen', self._penalty)
            # Use tighter tolerance for quantile (1e-4 is too loose)
            _loss_name = getattr(self._loss, 'name', '')
            _irls_tol = min(self.tol, 1e-8) if _loss_name == "quantile" else self.tol
            # Pass arrays directly — IRLS uses _get_xp(X) for backend dispatch
            params_irls, n_iter = self._loss.irls(
                X_work, y_arr,
                penalty=_inner_pen,
                max_iter=self.max_iter, tol=_irls_tol,
                init_coef=None,
                sample_weight=sample_weight,
                fit_intercept=self._effective_intercept,
            )
            params = _xp_asarray(params_irls, X_arr.dtype, X_arr)
        elif solver_name == "auto":
            # Fallback: resolve auto to fista (default for non-smooth losses)
            params, n_iter = fista_solver(
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
