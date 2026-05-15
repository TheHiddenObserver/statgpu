"""
Penalized GLM estimators.

This module keeps the GLM-specific optimization path explicit.  The central
implementation accepts a GLM loss name, while public typed estimators expose
gaussian, logistic, and poisson models without the old ``loss=...`` switch on
``PenalizedLinearRegression``.
"""

from typing import Optional, Union, Any, Dict, List
import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import get_backend, _get_torch_device_str, _to_numpy


def _irls_ridge_init(X, y, loss_name, alpha=0.01, max_iter=100, tol=1e-4):
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
    loss_obj = _resolve_loss_name(loss_name)
    coef, _ = fista_solver(
        loss_obj, l2_pen, np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        max_iter=max_iter, tol=tol,
    )
    return np.asarray(coef, dtype=np.float64)


def _resolve_loss_name(loss_name):
    """Resolve loss name string to loss object."""
    if loss_name == "logistic":
        from statgpu.glm_core._logistic import LogisticLoss
        return LogisticLoss()
    elif loss_name == "poisson":
        from statgpu.glm_core._poisson import PoissonLoss
        return PoissonLoss()
    elif loss_name == "gamma":
        from statgpu.glm_core._gamma import GammaLoss
        return GammaLoss()
    elif loss_name == "inverse_gaussian":
        from statgpu.glm_core._inverse_gaussian import InverseGaussianLoss
        return InverseGaussianLoss()
    elif loss_name == "negative_binomial":
        from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
        return NegativeBinomialLoss()
    elif loss_name == "tweedie":
        from statgpu.glm_core._tweedie import TweedieLoss
        return TweedieLoss()
    else:
        from statgpu.glm_core._squared import SquaredErrorLoss
        return SquaredErrorLoss()


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


class PenalizedGeneralizedLinearModel(BaseEstimator):
    """
    Penalized generalized linear model with pluggable GLM loss and penalty.

    Minimizes: loss(X, y, w) + penalty(w)

    Parameters
    ----------
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'logistic', 'poisson'.
    penalty : str or Penalty
        Penalty type: 'l1', 'l2', 'elasticnet', or a Penalty instance.
    loss : str, default='squared_error'
        Loss function: 'squared_error', 'logistic', 'poisson'.
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'irls', 'newton'.
        'auto' selects the current best path for the resolved backend:
        exact for Gaussian L2, CPU IRLS for smooth logistic/poisson L2, and
        GPU/Torch FISTA for GPU-backed penalized GLMs.
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
        CPU solver: 'fista' or 'coordinate_descent'.
    solver : str, default='fista'
        GPU solver: 'fista'.
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
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
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
        self.cpu_solver = cpu_solver.lower()
        self.solver = solver.lower()
        self.lipschitz_L = lipschitz_L
        self.gpu_memory_cleanup = gpu_memory_cleanup
        self.compute_inference = compute_inference
        self.cov_type = str(cov_type).lower()
        self.hac_maxlags = hac_maxlags
        self.stopping = str(stopping).lower()
        self.lla = lla
        self.max_lla_iters = max_lla_iters
        self.lla_tol = lla_tol

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
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._selected_solver = None
        self._selected_backend_name = None
        self._init_coef = None

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
                self.fit_intercept = True
            else:
                # Formula syntax owns intercept semantics, matching statsmodels/R.
                self.fit_intercept = False
        else:
            if X is None or y is None:
                raise ValueError("Either formula+data or X+y must be provided.")
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None

        self._penalty = self._resolve_penalty()
        self._validate_solver_penalty()
        self._loss = self._resolve_loss()

        # Resolve the actual backend before auto-selecting the solver. This
        # keeps solver="auto" device-aware: CPU can use IRLS for smooth GLMs,
        # while GPU/Torch stays on accelerator-capable FISTA.
        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        selected_solver = self._select_solver(self._loss, backend_name=backend_name)
        selected_solver = self._validate_solver_for_penalty(selected_solver, backend_name)
        self._selected_solver = selected_solver
        self._selected_backend_name = backend_name

        # Handle penalties requiring initialization (e.g., Adaptive Lasso)
        if self._penalty.requires_init:
            init_coef = self._fit_initial(X, y)
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
            _alpha_path = np.geomspace(
                max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont,
            )
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
                fit_intercept=self.fit_intercept,
                sample_weight=sample_weight,
            )
            self.coef_ = coef_np
            self.intercept_ = intercept
            self.n_iter_ = n_iter
            if self.fit_intercept:
                self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
            else:
                self._params = np.asarray(self.coef_).copy()
            self._df_resid = X.shape[0] - (X.shape[1] + (1 if self.fit_intercept else 0))
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

        self._fitted = True
        return self

    def _resolve_loss(self):
        """Resolve loss string to a GLMLoss object."""
        from statgpu.glm_core import get_glm_loss

        return get_glm_loss(self.loss)

    def _validate_solver_penalty(self):
        """Validate solver/penalty combinations before backend dispatch."""
        solver_name = self.solver
        penalty_name = str(getattr(self._penalty, "name", self.penalty)).lower()
        non_smooth = {
            "l1",
            "elasticnet",
            "adaptive_l1",
            "adaptive_lasso",
            "group_lasso",
            "gl",
            "scad",
            "mcp",
        }
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

    def _select_solver(self, loss, backend_name=None):
        """Auto-select solver based on loss and penalty (same across all backends)."""
        if self.solver != "auto":
            return self.solver
        if self.loss == "squared_error" and self._penalty.name == "l2":
            return "exact"
        _non_smooth = {
            "l1", "elasticnet", "en",
            "scad", "mcp", "adaptive_l1", "adaptive_lasso",
            "group_lasso", "gl", "group_mcp", "gmcp", "group_scad", "gscad",
        }
        if self._penalty.name in _non_smooth:
            # Gamma and InverseGaussian with L1/elasticnet: Lipschitz varies
            # with y and iterate, fista_bb burn-in (no backtracking) can't
            # handle the curvature changes.  fista's line search compensates.
            _loss_name = getattr(loss, 'name', '')
            if _loss_name in ("gamma", "inverse_gaussian"):
                return "fista"
            return "fista_bb"  # fista_bb auto-disables BB for non-smooth penalties
        if getattr(loss, "has_hessian", False):
            # For log-link families (gamma, tweedie, inverse_gaussian),
            # IRLS weights are constant or near-constant, making IRLS a
            # slow fixed-point iteration.  Newton with line search converges
            # much faster for these families.
            _loss_name = getattr(loss, 'name', '')
            if _loss_name in ("gamma", "tweedie", "inverse_gaussian"):
                return "newton"
            return "irls"
        return "fista"

    def _validate_solver_for_penalty(self, solver_name, backend_name):
        if solver_name != "fista_bb":
            return solver_name
        return solver_name

    def _fit_initial(self, X, y):
        """Fit initial model for penalties requiring initialization.

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
            init_coef, _ = fista_solver(
                loss_obj, l2_pen, np.asarray(X, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
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
            init_coef = _irls_ridge_init(
                np.asarray(X, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                loss_name=loss_name,
                alpha=0.01,
                max_iter=100,
                tol=1e-4,
            )
            return init_coef

        from statgpu.linear_model._ridge import Ridge

        init_model = Ridge(
            alpha=0.1,
            fit_intercept=self.fit_intercept,
            device=self.device,
        )
        init_model.fit(X, y)
        return init_model.coef_

    def _fit_lla(self, X, y, sample_weight, backend_name, init_coef=None):
        """Fit non-convex penalty via Local Linear Approximation.

        Outer loop reweights the non-convex penalty as per-coordinate
        weighted L1.  Each inner iteration solves a convex problem
        (ADMM for squared-error, FISTA for GLM) with the current weights.

        A **continuation path** is used for all losses: alpha is stepped
        down geometrically from 15× the target to the target (8 steps).
        Without this, small coefficients from the init receive weak L1
        weights (= P'(|coef|) ≈ alpha) and survive the inner solve,
        producing too many non-zeros.  Starting from a larger alpha and
        stepping down forces coefficients to cross the SCAD/MCP transition
        region (alpha .. a·alpha) where the two penalties differ — the
        same strategy used internally by R's ncvreg.

        For the inner loop the penalty is temporarily swapped for an
        ``AdaptiveL1Penalty`` whose per-coordinate weights are set from
        ``penalty.lla_weights(coef)``.
        """
        n_features = X.shape[1]

        if init_coef is not None:
            coef_lla = np.asarray(init_coef, dtype=float).copy()
        elif self._penalty.requires_init:
            coef_lla = np.zeros(n_features)
        else:
            coef_lla = self._fit_initial(X, y)

        # For GLM + SCAD/MCP direct IRLS-CD path, override init to zeros.
        # R's ncvreg starts from lambda_max with all-zero coefficients and
        # warm-starts down the continuation path.  The L2-penalized GLM
        # init gives large coefficients that cause numerical overflow in
        # the IRLS working response when eta is extreme.
        _pen_name_init = str(getattr(self._penalty, 'name', '')).lower()
        _is_glm_scad_mcp = (self.loss != "squared_error") and _pen_name_init in ("scad", "mcp")
        _is_scad_mcp = _pen_name_init in ("scad", "mcp")
        if _is_scad_mcp:
            coef_lla = np.zeros(n_features)

        from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty

        # ADMM inner solver was used for squared_error CPU path for cross-backend
        # consistency, but on CPU it is 4000× slower than FISTA (admm_solver
        # recomputes X@w and X.T@g per CG iteration instead of precomputing XtX
        # once).  On GPU the cuBLAS matmuls are fast enough that ADMM is
        # competitive.  Use fista_bb for CPU (O(p²) gradient with XtX precompute)
        # GLM losses: use fista_bb for early continuation steps (large alpha,
        # small coef — exp(X@coef) ≈ 1, BB steps are safe and 3-10× faster),
        # then switch to fista (backtracking) only for the final step where
        # coefficients may grow large enough to cause exp-link explosion.
        # Gamma is excluded — its gradient scale (1/mu) makes BB step estimates
        # unreliable even at small coefficients.
        saved_cpu_solver = self.cpu_solver
        saved_selected_solver = self._selected_solver
        _is_glm = (self.loss != "squared_error")
        _glm_bb_safe = _is_glm and self.loss in ("poisson", "logistic")
        if _is_glm and not _glm_bb_safe:
            self.cpu_solver = "fista"
            self._selected_solver = "fista"
        elif not _is_glm:
            if _is_scad_mcp:
                # SCAD/MCP uses direct FISTA+proximal (not ADMM)
                self.cpu_solver = "fista_bb"
                self._selected_solver = "fista_bb"
            else:
                # CPU: use fista_bb (precomputes XtX, O(p²) per iter, ~9ms total)
                # GPU: use admm (cuBLAS matmuls, ~40ms total with perfect x-backend consistency)
                if backend_name == "numpy":
                    self.cpu_solver = "fista_bb"
                    self._selected_solver = "fista_bb"
                else:
                    self.cpu_solver = "admm"
                    self._selected_solver = "admm"

        # Continuation path for all losses: start from a larger alpha and
        # step down geometrically to the target.  This forces coefficients
        # to cross the SCAD/MCP transition region (alpha .. a·alpha).
        # Squared-error + ADMM uses a wider path (20× / 8 steps) because
        # the OLS init produces many small but non-zero coefficients that
        # need stronger initial shrinkage to match R's ncvreg.  GLM losses
        # use a moderate path (10× / 5 steps) to balance sparsity and
        # convergence — larger paths cause FISTA to overshoot.
        import numpy as _np

        # Compute lambda_max — the smallest penalty where all coefficients are zero.
        # Matches R ncvreg: lambda_max = max_j |sum(x_s_j * resid)| / n
        # on standardized X (||X_j|| = sqrt(n)).  The IRLS-CD gradient
        # u_j = rho_j/n equals this at the null model, and the SCAD/MCP
        # threshold is l1 = alpha on u_j.
        _X_np = _np.asarray(X, dtype=float)
        _y_np = _np.asarray(y, dtype=float)
        _n = _X_np.shape[0]
        # Standardize X to match ncvreg: ||X_j|| = sqrt(n), i.e. mean(x^2) = 1
        _col_norms = _np.sqrt(_np.sum(_X_np ** 2, axis=0))
        _col_norms = _np.maximum(_col_norms, 1e-20)
        _X_s = _X_np * (_np.sqrt(_n) / _col_norms)
        if self.loss == "logistic":
            _p0 = _np.clip(_np.mean(_y_np), 1e-3, 1-1e-3)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ (_y_np - _p0) / _n)))
        elif self.loss == "poisson":
            _mu0 = max(float(_np.mean(_y_np)), 1e-3)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ (_y_np - _mu0) / _n)))
        elif self.loss == "gamma":
            _mu0 = max(float(_np.mean(_y_np)), 1e-3)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ ((_y_np - _mu0) / _mu0) / _n)))
        elif self.loss == "squared_error":
            _y_centered = _y_np - _np.mean(_y_np)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ _y_centered / _n)))
        else:
            _lam_max = self.alpha * 15.0  # fallback

        _n_cont = 20 if not _is_glm and _is_scad_mcp else (100 if _is_scad_mcp else 10)
        # Start from lambda_max to match R ncvreg's pathwise approach.
        # lambda_max is the smallest penalty where all coefficients are zero.
        _alpha_start = _lam_max
        _alpha_path = _np.geomspace(_alpha_start, self.alpha, _n_cont)
        _max_lla_per_step = max(6, self._max_lla_iters // _n_cont)

        saved_penalty_alpha = self._penalty.alpha
        saved_max_iter = self.max_iter

        try:
            # squared_error+SCAD/MCP: fused LLA+FISTA path.
            # Runs entire continuation+LLA+FISTA loop in one tight function
            # to eliminate per-call overhead (300+ fista_solver calls).
            if _is_scad_mcp and not _is_glm:
                from statgpu.glm_core._solver import fista_lla_path
                X_cached = self._to_array(X, backend=backend_name)
                y_cached = self._to_array(y, backend=backend_name)

                # Build max_iter schedule: early steps need fewer iterations
                _mi_path = []
                for _i in range(_n_cont):
                    _is_last = (_i == _n_cont - 1)
                    _mi_path.append(saved_max_iter if _is_last else max(100, saved_max_iter // 10))

                coef_np, intercept, n_iter = fista_lla_path(
                    self._loss, self._penalty,
                    X_cached, y_cached,
                    alpha_path=_alpha_path,
                    max_lla_per_step=_max_lla_per_step,
                    lla_tol=self._lla_tol,
                    max_iter=_mi_path,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    sample_weight=sample_weight,
                )
                coef_lla = coef_np
                self.coef_ = coef_np
                self.intercept_ = intercept
                self.n_iter_ = n_iter
                self._lla_n_iters_ = _n_cont * _max_lla_per_step
            else:
             for _cont_step, _cont_alpha in enumerate(_alpha_path):
                    self._penalty.alpha = float(_cont_alpha)

                    _is_last_cont = (_cont_step == _n_cont - 1)
                    if _is_glm_scad_mcp:
                        self.max_iter = 500 if _is_last_cont else 100
                    elif _is_last_cont:
                        self.max_iter = saved_max_iter
                    else:
                        self.max_iter = max(200, saved_max_iter // 3)
                    _is_gamma = (self.loss == "gamma")
                    if _is_gamma:
                        self.max_iter = max(300, self.max_iter // 2)
                    if _glm_bb_safe:
                        self.cpu_solver = "fista_bb"
                        self._selected_solver = "fista_bb"

                    if _is_scad_mcp and not _is_glm:
                        # This branch is now handled above by fista_lla_path
                        pass
                    else:
                        # Cache GPU arrays outside the LLA inner loop to avoid
                        # repeated CPU->GPU transfers (major overhead for small datasets)
                        X_cached = self._to_array(X, backend=backend_name)
                        y_cached = self._to_array(y, backend=backend_name)

                        for _lla_local in range(_max_lla_per_step):
                            # Compute LLA weights from current estimate
                            lla_w = self._penalty.lla_weights(coef_lla)

                            # SelectivePenalty wrapper handles intercept separately
                            # (clips to [-15,15] then sets penalty gradient to 0).
                            # Weights stay at p entries — no intercept padding needed.
                            # lla_weights() already returns alpha-scaled derivative
                            # weights (e.g. SCAD: alpha for |coef| <= alpha).
                            # AdaptiveL1Penalty applies: alpha_inner * weight_j * |coef_j|,
                            # so with alpha_inner=1 and weight=lla_w we get exactly
                            # the LLA penalty: sum_j lla_w_j * |coef_j|.
                            #
                            inner_pen = AdaptiveL1Penalty(alpha=1.0)
                            inner_pen._weights = lla_w

                            # Swap penalty
                            orig_penalty = self._penalty
                            self._penalty = inner_pen

                            # Run inner FISTA with warm-start from previous LLA estimate
                            # Use cached arrays to avoid repeated GPU transfers
                            self._init_coef = coef_lla.copy()

                            if backend_name == "torch":
                                self._fit_torch(X_cached, y_cached, sample_weight)
                            elif backend_name == "cupy":
                                self._fit_gpu(X_cached, y_cached, sample_weight)
                            else:
                                self._fit_cpu(X_cached, y_cached, sample_weight)

                            self._init_coef = None

                            # Restore original penalty
                            self._penalty = orig_penalty

                            # LLA convergence
                            coef_new = self.coef_.copy()
                            delta = float(np.sum(np.abs(coef_new - coef_lla)))
                            self._lla_n_iters_ = getattr(self, '_lla_n_iters_', 0) + 1

                            if delta < self._lla_tol:
                                coef_lla = coef_new
                                break

                            coef_lla = coef_new

        # Store final results.  For GLM+SCAD/MCP, _fit_cpu/_fit_gpu/_fit_torch
        # already set self.coef_ and self.intercept_.  For squared_error+SCAD/MCP,
        # _irls_cd returned params but didn't set them on self.
            if self.coef_ is None and coef_lla is not None:
                self.coef_ = np.asarray(coef_lla[:X.shape[1]], dtype=float)
                if self.fit_intercept:
                    X_np = np.asarray(X, dtype=float)
                    y_np = np.asarray(y, dtype=float)
                    self.intercept_ = float(np.mean(y_np) - np.mean(X_np, axis=0) @ self.coef_)
                else:
                    self.intercept_ = 0.0
                self._params = np.concatenate([[self.intercept_], self.coef_])
                self._df_resid = X.shape[0] - (X.shape[1] + (1 if self.fit_intercept else 0))
        finally:
            self._penalty.alpha = saved_penalty_alpha
            self.cpu_solver = saved_cpu_solver
            self._selected_solver = saved_selected_solver

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

        if self.fit_intercept:
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

        # Precompute for gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered.flatten()

        pen = self._penalty
        if solver_name == "exact":
            if pen.name != "l2":
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            self.coef_ = self._solve_exact_numpy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            if self.fit_intercept:
                self.intercept_ = float(y_mean - X_mean @ self.coef_)
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self._params = self.coef_.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
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
                                a_scad = float(getattr(pen, 'a', 3.7))
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
                                gamma_mcp = float(getattr(pen, 'gamma', 3.0))
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

        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self._params = self.coef_.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

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
            if self.fit_intercept:
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
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            coef = self._solve_exact_cupy(XtX, Xty, n_samples)
            self.n_iter_ = 1
            coef_np = coef.get()
            if self.fit_intercept:
                self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
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

        if self.fit_intercept:
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

        # Precompute
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Lipschitz constant via power iteration (O(p^2) vs O(p^3) for eigvalsh)
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            # Deterministic init to avoid different L for same X across calls
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

            coef = cp.zeros(n_features, dtype=X.dtype)
            y_k = coef.copy()
            t_k = 1.0
            beta = 0.0  # first iteration: y_k = coef (no momentum)

            # Lazy-compile the fused element-wise step (first call triggers JIT)
            _fused_step = None
            _fused_step_l2 = None

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
                        try:
                            @cp.fuse()
                            def _fista_elementwise_l2(
                                _y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                                _thresh, _l2_scale, _coef_old, _beta,
                            ):
                                w = (_y_k - _step_over_n * _xtx_y
                                     + _step_over_n_Xty)
                                c = (cp.sign(w)
                                     * cp.maximum(cp.abs(w) - _thresh, 0.0)
                                     / _l2_scale)
                                y = c + _beta * (c - _coef_old)
                                return c, y
                            _fused_step_l2 = _fista_elementwise_l2
                        except Exception:
                            _fused_step_l2 = None
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
                        try:
                            @cp.fuse()
                            def _fista_elementwise(
                                _y_k, _xtx_y, _step_over_n_Xty,
                                _step_over_n, _thresh, _coef_old, _beta,
                            ):
                                w = (_y_k - _step_over_n * _xtx_y
                                     + _step_over_n_Xty)
                                c = (cp.sign(w)
                                     * cp.maximum(cp.abs(w) - _thresh, 0.0))
                                y = c + _beta * (c - _coef_old)
                                return c, y
                            _fused_step = _fista_elementwise
                        except Exception:
                            _fused_step = None
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

        if self.fit_intercept:
            self.intercept_ = float(y_mean.get() - X_mean.get() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

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
            if self.fit_intercept:
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
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            coef = self._solve_exact_torch(XtX, Xty, n_samples)
            self.n_iter_ = 1
            coef_np = coef.cpu().numpy()
            if self.fit_intercept:
                self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
                self.coef_ = coef_np
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np
                self._params = coef_np.copy()
            self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
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
                sample_weight = torch.from_numpy(sample_weight).to(torch_device)
            sqrt_sw = torch.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        if self.fit_intercept:
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

        # Precompute
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Lipschitz constant via power iteration (O(p^2) vs O(p^3) for eigvalsh)
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            # Deterministic init to avoid different L for same X across calls
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

            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            y_k = coef.clone()
            t_k = 1.0
            beta = 0.0

            # Lazy-compile the fused element-wise step.
            # torch.compile JIT is lazy — triggers on first CALL.
            # Both phases are wrapped so old GPUs (P100 SM 6.0) fall back.
            _fused_step = None
            _fused_step_l2 = None
            _compile_tried = False
            _compile_tried_l2 = False

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
                        if not _compile_tried_l2:
                            _compile_tried_l2 = True
                            from statgpu.penalties import _torch_compile_ok as _tc_ok
                            if _tc_ok():
                                try:
                                    def _fista_elementwise_l2(
                                        _y_k, _xtx_y, _step_over_n_Xty,
                                        _step_over_n, _thresh, _l2_scale,
                                        _coef_old, _beta,
                                    ):
                                        w = (_y_k - _step_over_n * _xtx_y
                                             + _step_over_n_Xty)
                                        c = (torch.sign(w)
                                             * torch.relu(torch.abs(w) - _thresh)
                                             / _l2_scale)
                                        y = c + _beta * (c - _coef_old)
                                        return c, y
                                    _fused_step_l2 = torch.compile(
                                        _fista_elementwise_l2, mode='reduce-overhead',
                                    )
                                    coef, y_k = _fused_step_l2(
                                        y_k, xtx_y, step_over_n_Xty, step_over_n,
                                        thresh, l2_scale, coef_old, beta,
                                    )
                                    continue
                                except Exception:
                                    _fused_step_l2 = None
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
                        if not _compile_tried:
                            _compile_tried = True
                            from statgpu.penalties import _torch_compile_ok as _tc_ok
                            if _tc_ok():
                                try:
                                    def _fista_elementwise(
                                        _y_k, _xtx_y, _step_over_n_Xty,
                                        _step_over_n, _thresh,
                                        _coef_old, _beta,
                                    ):
                                        w = (_y_k - _step_over_n * _xtx_y
                                             + _step_over_n_Xty)
                                        c = (torch.sign(w)
                                             * torch.relu(torch.abs(w) - _thresh))
                                        y = c + _beta * (c - _coef_old)
                                        return c, y
                                    _fused_step = torch.compile(
                                        _fista_elementwise, mode='reduce-overhead',
                                    )
                                    coef, y_k = _fused_step(
                                        y_k, xtx_y, step_over_n_Xty, step_over_n,
                                        thresh, coef_old, beta,
                                    )
                                    continue
                                except Exception:
                                    _fused_step = None
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

        if self.fit_intercept:
            self.intercept_ = float(y_mean.cpu().numpy() - X_mean.cpu().numpy() @ coef_np)
            self.coef_ = coef_np
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        self._cleanup_torch_memory()

    def _fit_gpu_loss(self, X, y, sample_weight=None):
        """CuPy FISTA for non-squared-error losses (logistic, poisson).

        Mirrors _fit_cpu_loss but keeps arrays on GPU during computation.
        """
        import cupy as cp
        from statgpu.glm_core._solver import fista_solver, fista_bb_solver

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="cupy"
        )
        _solver = fista_bb_solver if solver_name == "fista_bb" else fista_solver

        X_arr = cp.asarray(X)
        y_arr = cp.asarray(y)

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            X_aug = cp.column_stack([X_arr, cp.ones(X_arr.shape[0])])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="cupy"):
                    import cupy as cp
                    w_feat = w[:-1]
                    result_feat = pen.proximal(w_feat, step, backend=backend)
                    result = cp.empty(w.shape[0], dtype=w.dtype)
                    result[:-1] = result_feat
                    result[-1] = cp.clip(w[-1], -15.0, 15.0)
                    return result
                def value(self, coef):
                    return pen.value(coef[:-1])
                name = pen.name

            full_coef, n_iter = _solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = full_coef.get()[:p]
            self.intercept_ = float(full_coef.get()[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            X_arr = X_arr - cp.mean(X_arr, axis=0)
            y_arr = y_arr - cp.mean(y_arr)

            coef, n_iter = _solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.get()
            self.n_iter_ = n_iter
            self.intercept_ = float(cp.mean(y_arr) - cp.mean(X_arr, axis=0) @ self.coef_)
        else:
            coef, n_iter = _solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.get()
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))
        self._cleanup_cuda_memory()

    def _fit_torch_loss(self, X, y, sample_weight=None):
        """Torch FISTA for non-squared-error losses (logistic, poisson).

        Mirrors _fit_cpu_loss but keeps arrays on GPU during computation.
        """
        import torch
        from statgpu.glm_core._solver import fista_solver, fista_bb_solver

        solver_name = self._selected_solver or self._select_solver(
            self._loss, backend_name="torch"
        )
        _solver = fista_bb_solver if solver_name == "fista_bb" else fista_solver

        torch_device = _get_torch_device_str()

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch_device).to(torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(torch_device).to(torch.float64)

        X_arr = X
        y_arr = y

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            ones_col = torch.ones(X_arr.shape[0], dtype=torch.float64, device=torch_device)
            X_aug = torch.column_stack([X_arr, ones_col])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="torch"):
                    w_feat = w[:-1]
                    result_feat = pen.proximal(w_feat, step, backend=backend)
                    result = torch.empty(w.shape[0], dtype=w.dtype, device=w.device)
                    result[:-1] = result_feat
                    result[-1] = torch.clamp(w[-1], -15.0, 15.0)
                    return result
                def value(self, coef):
                    return pen.value(coef[:-1])
                name = pen.name

            full_coef, n_iter = _solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            full_np = full_coef.cpu().numpy()
            self.coef_ = full_np[:p]
            self.intercept_ = float(full_np[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            X_arr = X_arr - torch.mean(X_arr, dim=0)
            y_arr = y_arr - torch.mean(y_arr)

            coef, n_iter = _solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.cpu().numpy()
            self.n_iter_ = n_iter
            self.intercept_ = float(torch.mean(y_arr) - torch.mean(X_arr, dim=0) @ self.coef_)
        else:
            coef, n_iter = _solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef.cpu().numpy()
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))
        self._cleanup_torch_memory()

    def _ridge_alpha_for_exact(self) -> float:
        """Return L2 alpha for the exact Ridge normal equations."""
        return float(getattr(self._penalty, "alpha", self.alpha))

    def _solve_exact_numpy(self, XtX, Xty, n_samples):
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * np.eye(p, dtype=XtX.dtype)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ Xty

    def _solve_exact_cupy(self, XtX, Xty, n_samples):
        import cupy as cp

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * cp.eye(p, dtype=XtX.dtype)
        try:
            return cp.linalg.solve(A, Xty)
        except Exception:
            return cp.linalg.pinv(A) @ Xty

    def _solve_exact_torch(self, XtX, Xty, n_samples):
        import torch

        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + (float(n_samples) * alpha) * torch.eye(
            p, dtype=XtX.dtype, device=XtX.device
        )
        try:
            return torch.linalg.solve(A, Xty)
        except RuntimeError:
            return torch.linalg.pinv(A) @ Xty

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

    def predict(self, X):
        """
        Predict using fitted model.

        For squared_error: returns linear prediction.
        For logistic: returns binary class labels.
        For poisson: returns exp(linear prediction) (count values).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self.fit_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            if self.loss == "logistic":
                p = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -500, 500)))
                return (p > 0.5).astype(float)
            if self.loss == "poisson":
                return cp.exp(raw)
            return raw
        if device == Device.TORCH:
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self.fit_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            if self.loss == "logistic":
                p = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -500, 500)))
                return (p > 0.5).to(raw.dtype)
            if self.loss == "poisson":
                return torch.exp(raw)
            return raw

        raw = X @ self.coef_
        if self.fit_intercept:
            raw += self.intercept_

        # Apply link inverse for GLM losses
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
            return (p > 0.5).astype(float)
        elif self.loss == "poisson":
            return np.exp(raw)
        return raw

    def score(self, X, y):
        """
        Return R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        r2 : float
            R² score.
        """
        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            yb = cp.asarray(self._to_array(y, Device.CUDA))
            ss_res = cp.sum((yb - y_pred) ** 2)
            ss_tot = cp.sum((yb - cp.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).get()) if float(ss_tot.get()) > 0 else 0.0
        if device == Device.TORCH:
            import torch
            yb = self._to_array(y, Device.TORCH, backend="torch").to(y_pred.dtype)
            ss_res = torch.sum((yb - y_pred) ** 2)
            ss_tot = torch.sum((yb - torch.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
        y = np.asarray(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
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
            alpha = getattr(self._loss, "alpha", 1.0)
            return NegativeBinomial(alpha=alpha)
        if self.loss == "tweedie":
            power = getattr(self._loss, "power", 1.5)
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
        """Penalty wrapper that leaves the last intercept coefficient free."""
        pen = self._penalty
        alpha = float(getattr(pen, "alpha", self.alpha))

        class SelectivePenalty:
            name = pen.name

            def value(self, coef):
                return pen.value(coef[:-1])

            def proximal(self, w, step, backend=backend_name):
                # Always apply penalty only to features (skip intercept at position -1)
                w_feat = w[:-1]
                result_feat = pen.proximal(w_feat, step, backend=backend)
                if backend_name == "cupy":
                    import cupy as cp
                    result = cp.empty(w.shape[0], dtype=w.dtype)
                    result[:-1] = result_feat
                    result[-1] = cp.clip(w[-1], -15.0, 15.0)
                elif backend_name == "torch":
                    import torch
                    result = torch.empty(w.shape[0], dtype=w.dtype, device=w.device)
                    result[:-1] = result_feat
                    result[-1] = torch.clamp(w[-1], -15.0, 15.0)
                else:
                    result = np.empty(w.shape[0], dtype=w.dtype)
                    result[:-1] = result_feat
                    result[-1] = np.clip(w[-1], -15.0, 15.0)
                return result

            def smooth_value(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                active = coef[:p]
                if backend_name == "cupy":
                    import cupy as cp
                    return 0.5 * alpha * cp.sum(active * active)
                if backend_name == "torch":
                    import torch
                    return 0.5 * alpha * torch.sum(active * active)
                return 0.5 * alpha * np.sum(active * active)

            def smooth_gradient(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                if backend_name == "cupy":
                    import cupy as cp
                    grad = cp.zeros_like(coef)
                elif backend_name == "torch":
                    import torch
                    grad = torch.zeros_like(coef)
                else:
                    grad = np.zeros_like(coef)
                grad[:p] = alpha * coef[:p]
                return grad

            def smooth_hessian(self, coef):
                if str(pen.name).lower() != "l2":
                    raise ValueError("smooth solvers only support L2 penalties.")
                if backend_name == "cupy":
                    import cupy as cp
                    diag = cp.zeros(coef.shape[0], dtype=coef.dtype)
                    diag[:p] = alpha
                    return cp.diag(diag)
                if backend_name == "torch":
                    import torch
                    diag = torch.zeros(
                        coef.shape[0], dtype=coef.dtype, device=coef.device
                    )
                    diag[:p] = alpha
                    return torch.diag(diag)
                diag = np.zeros(coef.shape[0], dtype=coef.dtype)
                diag[:p] = alpha
                return np.diag(diag)

        return SelectivePenalty()

    def _irls_cd(self, pen, X_work, y_arr, init, _lla_continuation=False):
        """IRLS with coordinate descent for GLM + non-smooth penalties.

        Matches R glmnet/ncvreg algorithm: outer IRLS loop computes working
        response and weights, inner CD loop solves the weighted penalized
        least squares subproblem with per-coordinate thresholds.
        Supports: adaptive_l1, scad, mcp.
        """
        import numpy as np

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp

        # Access weights from the original penalty (not the SelectivePenalty wrapper)
        _inner = getattr(self, '_penalty', pen)
        _w = np.asarray(getattr(_inner, '_weights', np.ones(p)), dtype=float)
        # Read alpha from the penalty object.  The threshold per coordinate
        # is alpha * _w[j] where _w has mean=1 (matching R glmnet convention).
        alpha = float(getattr(_inner, 'alpha', self.alpha))
        _nf = float(getattr(_inner, '_norm_factor', 1.0))
        pen_name = getattr(pen, 'name', '') or getattr(_inner, 'name', '')

        # SCAD/MCP parameters
        a_scad = float(getattr(_inner, 'a', 3.7)) if pen_name == "scad" else 0.0
        gamma_mcp = float(getattr(_inner, 'gamma', 3.0)) if pen_name == "mcp" else 0.0

        if init is not None:
            beta = np.asarray(init, dtype=float).copy()
        else:
            beta = np.zeros(pp)

        loss_name = self._loss.name
        _is_glm = (loss_name != "squared_error")

        # Continuation path for SCAD/MCP: trace the solution from lambda_max
        # down to the target alpha, matching R ncvreg's pathwise approach.
        # Without this, solving directly at the target alpha can converge to
        # a different local minimum than ncvreg (non-convex penalties have
        # multiple local minima that depend on the starting point).
        # Skip when _lla_continuation=True (outer _fit_lla handles the path).
        _cont_path = [alpha]
        if pen_name in ("scad", "mcp") and not _lla_continuation:
            # lambda_max = max(|X_j^T resid| / ||X_j||^2) at the null model.
            # For squared_error: resid = y - mean(y)
            # For GLM: resid = (y - mu0) / mu0 (working residual at null)
            if loss_name == "logistic":
                _p0 = np.clip(np.mean(y_arr), 1e-3, 1 - 1e-3)
                _resid = y_arr - _p0
            elif loss_name == "poisson":
                _mu0 = max(float(np.mean(y_arr)), 1e-3)
                _resid = y_arr - _mu0
            elif loss_name == "gamma":
                _mu0 = max(float(np.mean(y_arr)), 1e-3)
                _resid = (y_arr - _mu0) / _mu0
            else:
                _resid = y_arr - np.mean(y_arr)
            _xty = np.abs(X_work[:, :p].T @ _resid)
            _xnorm_sq = np.sum(X_work[:, :p] ** 2, axis=0)
            _xnorm_sq = np.maximum(_xnorm_sq, 1e-20)
            _lam_max = float(np.max(_xty / _xnorm_sq))
            if _lam_max > alpha * 1.1:
                _n_cont = 100  # match ncvreg's default nlambda
                _cont_path = np.geomspace(_lam_max, alpha, _n_cont)

        # For GLM losses, do ONE CD sweep per IRLS iteration (matching
        # R ncvreg/glmnet).  The IRLS outer loop handles convergence.
        # For squared_error, use the convergence-based CD loop since
        # there is no outer IRLS loop.
        _n_cd_sweeps_base = 1 if _is_glm else min(self.max_iter, 200)
        # For squared_error, the outer IRLS loop is redundant (d=1, z=y
        # are constant).  Run the outer loop only once.
        _n_outer_base = self.max_iter if _is_glm else 1

        # For squared_error, d/z/XDX_diag are constant across continuation
        # steps — compute once before the loop.
        if not _is_glm:
            d = np.ones(n)
            z = y_arr
            XDX_diag = np.sum(d[:, None] * X_work ** 2, axis=0)

        for _cont_idx, _cont_alpha in enumerate(_cont_path):
            # Update alpha for this continuation step
            if len(_cont_path) > 1:
                alpha = float(_cont_alpha)
                _is_last = (_cont_idx == len(_cont_path) - 1)
                _n_cd_sweeps = _n_cd_sweeps_base if _is_last else 20
                # For GLM with continuation: limit IRLS iterations on
                # non-final steps.  ncvreg does ~10 IRLS per lambda value.
                if _is_glm:
                    _n_outer = _n_outer_base if _is_last else min(20, _n_outer_base)
                else:
                    _n_outer = _n_outer_base
            else:
                _n_cd_sweeps = _n_cd_sweeps_base
                _n_outer = _n_outer_base

            for it in range(_n_outer):
                beta_old = beta.copy()

                if _is_glm:
                    eta = X_work @ beta
                    if loss_name == "logistic":
                        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
                        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
                        d = mu * (1.0 - mu)
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "poisson":
                        mu = np.exp(np.clip(eta, -500, 500))
                        mu = np.maximum(mu, 1e-15)
                        d = mu
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "gamma":
                        mu = np.exp(np.clip(eta, -500, 500))
                        mu = np.maximum(mu, 1e-15)
                        d = np.ones(n)
                        z = eta + (y_arr - mu) / mu
                    elif loss_name == "inverse_gaussian":
                        mu = np.exp(np.clip(eta, -500, 500))
                        mu = np.maximum(mu, 1e-15)
                        d = 1.0 / (mu ** 3)
                        z = eta + (y_arr - mu) / (d * mu ** 2)
                    elif loss_name == "negative_binomial":
                        mu = np.exp(np.clip(eta, -500, 500))
                        mu = np.maximum(mu, 1e-15)
                        theta_nb = float(getattr(self._loss, 'alpha', 1.0))
                        d = mu / (1.0 + mu / theta_nb)
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "tweedie":
                        mu = np.exp(np.clip(eta, -500, 500))
                        mu = np.maximum(mu, 1e-15)
                        tweedie_p = float(getattr(self._loss, 'power', 1.5))
                        d = mu ** tweedie_p
                        d = np.maximum(d, 1e-15)
                        z = eta + (y_arr - mu) / (d * mu)
                    else:
                        grad = self._loss.gradient(X_work, y_arr, beta)
                        d = np.ones(n)
                        z = eta - grad * n
                    XDX_diag = np.sum(d[:, None] * X_work ** 2, axis=0)

                r = z - X_work @ beta

                # Compute penalized objective before CD (for step-halving)
                if _is_glm:
                    _obj_before = float(self._loss.value(X_work[:, :p], y_arr, beta[:p]))
                    _abs_b = np.abs(beta[:p])
                    if pen_name == "scad":
                        _pen_val = np.where(_abs_b <= alpha, alpha * _abs_b,
                                   np.where(_abs_b <= a_scad * alpha,
                                       (a_scad * alpha * _abs_b - 0.5 * (beta[:p]**2 + alpha**2)) / (a_scad - 1.0),
                                       0.5 * (a_scad + 1.0) * alpha**2))
                        _obj_before += float(np.sum(_pen_val))
                    elif pen_name == "mcp":
                        _pen_val = np.where(_abs_b <= gamma_mcp * alpha,
                                    alpha * _abs_b - 0.5 * beta[:p]**2 / gamma_mcp,
                                    0.5 * gamma_mcp * alpha**2)
                        _obj_before += float(np.sum(_pen_val))

                for _cd in range(_n_cd_sweeps):
                    _max_cd_change = 0.0
                    for j in range(pp):
                        if XDX_diag[j] < 1e-20:
                            beta[j] = 0.0
                            continue

                        rho_j = np.dot(d * X_work[:, j], r) + XDX_diag[j] * beta[j]
                        old_bj = beta[j]

                        u_j = rho_j / n
                        v_j = XDX_diag[j] / n

                        if j >= p:
                            beta[j] = u_j / v_j
                        elif pen_name in ("adaptive_l1", "adaptive_lasso"):
                            l1 = alpha * _w[j]
                            w_j = u_j / v_j
                            if w_j > l1:
                                beta[j] = (w_j - l1)
                            elif w_j < -l1:
                                beta[j] = (w_j + l1)
                            else:
                                beta[j] = 0.0
                        elif pen_name == "scad":
                            l1 = alpha
                            w_j = u_j / v_j
                            aw = np.abs(w_j)
                            if aw > a_scad * l1:
                                beta[j] = w_j
                            elif aw > l1:
                                beta[j] = np.sign(w_j) * ((a_scad - 1.0) * aw - a_scad * l1) / (a_scad - 2.0)
                            else:
                                beta[j] = 0.0
                        elif pen_name == "mcp":
                            l1 = alpha
                            w_j = u_j / v_j
                            aw = np.abs(w_j)
                            if aw > gamma_mcp * l1:
                                beta[j] = w_j
                            elif aw > l1:
                                beta[j] = np.sign(w_j) * (aw - l1) / (1.0 - 1.0 / gamma_mcp)
                            else:
                                beta[j] = 0.0
                        else:
                            l1 = alpha
                            w_j = u_j / v_j
                            if w_j > l1:
                                beta[j] = (w_j - l1)
                            elif w_j < -l1:
                                beta[j] = (w_j + l1)
                            else:
                                beta[j] = 0.0

                        if beta[j] != old_bj:
                            r += X_work[:, j] * (old_bj - beta[j])
                            _cd_change = abs(beta[j] - old_bj)
                            if _cd_change > _max_cd_change:
                                _max_cd_change = _cd_change

                    # Inner CD convergence check (only for squared_error)
                    if not _is_glm and _max_cd_change < self.tol:
                        break

                # Step-halving for GLM: ensure penalized objective decreases.
                # ncvreg uses step-halving to prevent IRLS overshooting.
                if _is_glm:
                    _obj_after = float(self._loss.value(X_work[:, :p], y_arr, beta[:p]))
                    _abs_b2 = np.abs(beta[:p])
                    if pen_name == "scad":
                        _pen_val2 = np.where(_abs_b2 <= alpha, alpha * _abs_b2,
                                    np.where(_abs_b2 <= a_scad * alpha,
                                        (a_scad * alpha * _abs_b2 - 0.5 * (beta[:p]**2 + alpha**2)) / (a_scad - 1.0),
                                        0.5 * (a_scad + 1.0) * alpha**2))
                        _obj_after += float(np.sum(_pen_val2))
                    elif pen_name == "mcp":
                        _pen_val2 = np.where(_abs_b2 <= gamma_mcp * alpha,
                                     alpha * _abs_b2 - 0.5 * beta[:p]**2 / gamma_mcp,
                                     0.5 * gamma_mcp * alpha**2)
                        _obj_after += float(np.sum(_pen_val2))
                    if _obj_after > _obj_before + 1e-10:
                        # Step-halving: revert halfway toward beta_old
                        beta[:] = 0.5 * (beta + beta_old)

                # IRLS-level convergence check.
                _delta = np.max(np.abs(beta[:p] - beta_old[:p]))
                if not _is_glm and _delta < self.tol:
                    break
                # For GLM with continuation: early exit on convergence
                # for non-final steps (avoids wasting iterations).
                if _is_glm and len(_cont_path) > 1 and not _is_last:
                    if _delta < self.tol * 10:
                        break

        n_iter = it + 1
        return beta, n_iter

    def _irls_cd_gpu(self, pen, X_work, y_arr, init, backend_name, _lla_continuation=False):
        """GPU-native IRLS with coordinate descent for GLM + non-smooth penalties.

        Same algorithm as _irls_cd but keeps all arrays on GPU to avoid
        CPU-GPU transfer overhead.  Supports cupy and torch backends.
        """
        if backend_name == "cupy":
            import cupy as xp
        elif backend_name == "torch":
            import torch
            xp = torch
        else:
            raise ValueError(f"GPU backend required, got {backend_name}")

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp

        # Access weights from the original penalty
        _inner = getattr(self, '_penalty', pen)
        _w_np = np.asarray(getattr(_inner, '_weights', np.ones(p)), dtype=float)
        _w = xp.asarray(_w_np) if backend_name == "cupy" else torch.from_numpy(_w_np).to(X_work.device)
        alpha = float(getattr(_inner, 'alpha', self.alpha))
        pen_name = getattr(pen, 'name', '') or getattr(_inner, 'name', '')

        # SCAD/MCP parameters
        a_scad = float(getattr(_inner, 'a', 3.7)) if pen_name == "scad" else 0.0
        gamma_mcp = float(getattr(_inner, 'gamma', 3.0)) if pen_name == "mcp" else 0.0

        if init is not None:
            if isinstance(init, np.ndarray):
                beta = xp.asarray(init) if backend_name == "cupy" else torch.from_numpy(init).to(X_work.device)
            else:
                beta = init.clone() if backend_name == "torch" else init.copy()
        else:
            beta = xp.zeros(pp, dtype=X_work.dtype) if backend_name == "cupy" else torch.zeros(pp, dtype=X_work.dtype, device=X_work.device)

        loss_name = self._loss.name
        _is_glm = (loss_name != "squared_error")

        # Continuation path for SCAD/MCP
        _cont_path = [alpha]
        if pen_name in ("scad", "mcp") and not _lla_continuation:
            _y_np = _to_numpy(y_arr)
            if loss_name == "logistic":
                _p0 = np.clip(np.mean(_y_np), 1e-3, 1 - 1e-3)
                _resid = _y_np - _p0
            elif loss_name == "poisson":
                _mu0 = max(float(np.mean(_y_np)), 1e-3)
                _resid = _y_np - _mu0
            elif loss_name == "gamma":
                _mu0 = max(float(np.mean(_y_np)), 1e-3)
                _resid = (_y_np - _mu0) / _mu0
            else:
                _resid = _y_np - np.mean(_y_np)
            _X_np = _to_numpy(X_work)
            _xty = np.abs(_X_np[:, :p].T @ _resid)
            _xnorm_sq = np.sum(_X_np[:, :p] ** 2, axis=0)
            _xnorm_sq = np.maximum(_xnorm_sq, 1e-20)
            _lam_max = float(np.max(_xty / _xnorm_sq))
            if _lam_max > alpha * 1.1:
                _n_cont = 100
                _cont_path = np.geomspace(_lam_max, alpha, _n_cont)

        _n_cd_sweeps_base = 1 if _is_glm else min(self.max_iter, 200)
        _n_outer_base = self.max_iter if _is_glm else 1

        # Precompute X^T X diagonal for squared_error
        if not _is_glm:
            d = xp.ones(n, dtype=X_work.dtype) if backend_name == "cupy" else torch.ones(n, dtype=X_work.dtype, device=X_work.device)
            z = y_arr
            XDX_diag = xp.sum(d[:, None] * X_work ** 2, axis=0) if backend_name == "cupy" else torch.sum(d[:, None] * X_work ** 2, dim=0)

        for _cont_idx, _cont_alpha in enumerate(_cont_path):
            if len(_cont_path) > 1:
                alpha = float(_cont_alpha)
                _is_last = (_cont_idx == len(_cont_path) - 1)
                _n_cd_sweeps = _n_cd_sweeps_base if _is_last else 20
                if _is_glm:
                    _n_outer = _n_outer_base if _is_last else min(20, _n_outer_base)
                else:
                    _n_outer = _n_outer_base
            else:
                _n_cd_sweeps = _n_cd_sweeps_base
                _n_outer = _n_outer_base

            for it in range(_n_outer):
                beta_old = beta.clone() if backend_name == "torch" else beta.copy()

                if _is_glm:
                    eta = X_work @ beta
                    if loss_name == "logistic":
                        mu = 1.0 / (1.0 + xp.exp(-xp.clip(eta, -500, 500)))
                        mu = xp.clip(mu, 1e-15, 1.0 - 1e-15) if backend_name == "cupy" else torch.clamp(mu, 1e-15, 1.0 - 1e-15)
                        d = mu * (1.0 - mu)
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "poisson":
                        mu = xp.exp(xp.clip(eta, -500, 500)) if backend_name == "cupy" else torch.exp(torch.clamp(eta, -500, 500))
                        mu = xp.maximum(mu, 1e-15) if backend_name == "cupy" else torch.clamp(mu, min=1e-15)
                        d = mu
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "gamma":
                        mu = xp.exp(xp.clip(eta, -500, 500)) if backend_name == "cupy" else torch.exp(torch.clamp(eta, -500, 500))
                        mu = xp.maximum(mu, 1e-15) if backend_name == "cupy" else torch.clamp(mu, min=1e-15)
                        d = xp.ones(n, dtype=X_work.dtype) if backend_name == "cupy" else torch.ones(n, dtype=X_work.dtype, device=X_work.device)
                        z = eta + (y_arr - mu) / mu
                    elif loss_name == "inverse_gaussian":
                        mu = xp.exp(xp.clip(eta, -500, 500)) if backend_name == "cupy" else torch.exp(torch.clamp(eta, -500, 500))
                        mu = xp.maximum(mu, 1e-15) if backend_name == "cupy" else torch.clamp(mu, min=1e-15)
                        d = 1.0 / (mu ** 3)
                        z = eta + (y_arr - mu) / (d * mu ** 2)
                    elif loss_name == "negative_binomial":
                        mu = xp.exp(xp.clip(eta, -500, 500)) if backend_name == "cupy" else torch.exp(torch.clamp(eta, -500, 500))
                        mu = xp.maximum(mu, 1e-15) if backend_name == "cupy" else torch.clamp(mu, min=1e-15)
                        theta_nb = float(getattr(self._loss, 'alpha', 1.0))
                        d = mu / (1.0 + mu / theta_nb)
                        z = eta + (y_arr - mu) / d
                    elif loss_name == "tweedie":
                        mu = xp.exp(xp.clip(eta, -500, 500)) if backend_name == "cupy" else torch.exp(torch.clamp(eta, -500, 500))
                        mu = xp.maximum(mu, 1e-15) if backend_name == "cupy" else torch.clamp(mu, min=1e-15)
                        tweedie_p = float(getattr(self._loss, 'power', 1.5))
                        d = mu ** tweedie_p
                        d = xp.maximum(d, 1e-15) if backend_name == "cupy" else torch.clamp(d, min=1e-15)
                        z = eta + (y_arr - mu) / (d * mu)
                    else:
                        grad = self._loss.gradient(X_work, y_arr, beta)
                        d = xp.ones(n, dtype=X_work.dtype) if backend_name == "cupy" else torch.ones(n, dtype=X_work.dtype, device=X_work.device)
                        z = eta - grad * n
                    XDX_diag = xp.sum(d[:, None] * X_work ** 2, axis=0) if backend_name == "cupy" else torch.sum(d[:, None] * X_work ** 2, dim=0)

                r = z - X_work @ beta

                for _cd in range(_n_cd_sweeps):
                    _max_cd_change = 0.0
                    for j in range(pp):
                        if float(XDX_diag[j]) < 1e-20:
                            beta[j] = 0.0
                            continue

                        rho_j = float(xp.dot(d * X_work[:, j], r)) + float(XDX_diag[j]) * float(beta[j])
                        old_bj = float(beta[j])

                        u_j = rho_j / n
                        v_j = float(XDX_diag[j]) / n

                        if j >= p:
                            beta[j] = u_j / v_j
                        elif pen_name in ("adaptive_l1", "adaptive_lasso"):
                            l1 = alpha * float(_w[j])
                            w_j = u_j / v_j
                            if w_j > l1:
                                beta[j] = (w_j - l1)
                            elif w_j < -l1:
                                beta[j] = (w_j + l1)
                            else:
                                beta[j] = 0.0
                        elif pen_name == "scad":
                            l1 = alpha
                            w_j = u_j / v_j
                            aw = abs(w_j)
                            if aw > a_scad * l1:
                                beta[j] = w_j
                            elif aw > l1:
                                beta[j] = np.sign(w_j) * ((a_scad - 1.0) * aw - a_scad * l1) / (a_scad - 2.0)
                            else:
                                beta[j] = 0.0
                        elif pen_name == "mcp":
                            l1 = alpha
                            w_j = u_j / v_j
                            aw = abs(w_j)
                            if aw > gamma_mcp * l1:
                                beta[j] = w_j
                            elif aw > l1:
                                beta[j] = np.sign(w_j) * (aw - l1) / (1.0 - 1.0 / gamma_mcp)
                            else:
                                beta[j] = 0.0
                        else:
                            l1 = alpha
                            w_j = u_j / v_j
                            if w_j > l1:
                                beta[j] = (w_j - l1)
                            elif w_j < -l1:
                                beta[j] = (w_j + l1)
                            else:
                                beta[j] = 0.0

                        if float(beta[j]) != old_bj:
                            r = r + X_work[:, j] * (old_bj - float(beta[j]))
                            _cd_change = abs(float(beta[j]) - old_bj)
                            if _cd_change > _max_cd_change:
                                _max_cd_change = _cd_change

                    if not _is_glm and _max_cd_change < self.tol:
                        break

                # IRLS-level convergence check
                _delta = float(xp.max(xp.abs(beta[:p] - beta_old[:p]))) if backend_name == "cupy" else float(torch.max(torch.abs(beta[:p] - beta_old[:p])))
                if not _is_glm and _delta < self.tol:
                    break
                if _is_glm and len(_cont_path) > 1 and not _is_last:
                    if _delta < self.tol * 10:
                        break

        n_iter = it + 1
        return beta, n_iter

    def _block_cd_group_lasso(self, pen, X_work, y_arr, init):
        """Block coordinate descent for group_lasso penalty.

        Matches R grpreg's block CD algorithm: iterate over groups, compute
        partial residual per group, solve the group subproblem, apply block
        soft-thresholding.
        """
        import numpy as np

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp
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

            if self.fit_intercept:
                coef[pp - 1] = np.mean(y_arr - X_work[:, :p] @ coef[:p])

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        n_iter = iteration + 1

        if self.fit_intercept:
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
        if backend_name == "cupy":
            import cupy as xp
        elif backend_name == "torch":
            import torch
            xp = torch
        else:
            raise ValueError(f"GPU backend required, got {backend_name}")

        # Enforce float64 precision for numerical stability
        if backend_name == "cupy":
            X_work = xp.asarray(X_work, dtype=xp.float64)
            y_arr = xp.asarray(y_arr, dtype=xp.float64)
        else:
            X_work = X_work.to(dtype=torch.float64)
            y_arr = y_arr.to(dtype=torch.float64)

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp
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
        _XtX_blocks = []
        _ridge = 1e-10 if backend_name == "cupy" else torch.tensor(1e-10, dtype=torch.float64, device=X_work.device)
        for g_idx in _g_indices:
            block = XtX[g_idx][:, g_idx]
            # Add diagonal ridge to ensure positive definiteness
            if backend_name == "cupy":
                block = block + _ridge * xp.eye(block.shape[0], dtype=block.dtype)
            else:
                block = block + _ridge * torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
            _XtX_blocks.append(block)

        if init is not None:
            if isinstance(init, np.ndarray):
                coef = xp.asarray(init, dtype=X_work.dtype) if backend_name == "cupy" else torch.from_numpy(init).to(dtype=torch.float64, device=X_work.device)
            else:
                coef = init.clone() if backend_name == "torch" else init.copy()
        else:
            coef = xp.zeros(pp, dtype=X_work.dtype) if backend_name == "cupy" else torch.zeros(pp, dtype=torch.float64, device=X_work.device)

        for iteration in range(self.max_iter):
            coef_old = coef.clone() if backend_name == "torch" else coef.copy()

            for g in range(_n_groups):
                g_idx = _g_indices[g]
                rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + _XtX_blocks[g] @ coef[g_idx]
                try:
                    if backend_name == "cupy":
                        w_g = xp.linalg.solve(_XtX_blocks[g], rho_g)
                    else:
                        w_g = torch.linalg.solve(_XtX_blocks[g], rho_g)
                    # Check for NaN/Inf in solution
                    if backend_name == "cupy":
                        if xp.any(xp.isnan(w_g)) or xp.any(xp.isinf(w_g)):
                            w_g = xp.zeros(len(g_idx), dtype=X_work.dtype)
                    else:
                        if torch.any(torch.isnan(w_g)) or torch.any(torch.isinf(w_g)):
                            w_g = torch.zeros(len(g_idx), dtype=torch.float64, device=X_work.device)
                except Exception:
                    w_g = xp.zeros(len(g_idx), dtype=X_work.dtype) if backend_name == "cupy" else torch.zeros(len(g_idx), dtype=torch.float64, device=X_work.device)
                norm_w = float(xp.linalg.norm(w_g)) if backend_name == "cupy" else float(torch.linalg.norm(w_g))
                thresh_g = alpha * _sqrt_pg[g]
                if norm_w > thresh_g:
                    coef[g_idx] = w_g * (1.0 - thresh_g / norm_w)
                else:
                    coef[g_idx] = 0.0

            if self.fit_intercept:
                coef[pp - 1] = float(xp.mean(y_arr - X_work[:, :p] @ coef[:p])) if backend_name == "cupy" else float(torch.mean(y_arr - X_work[:, :p] @ coef[:p]))

            _max_change = float(xp.max(xp.abs(coef - coef_old))) if backend_name == "cupy" else float(torch.max(torch.abs(coef - coef_old)))
            if _max_change < self.tol:
                break

        n_iter = iteration + 1

        if self.fit_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0

        return beta, intercept, n_iter

    def _block_cd_group_lasso_gpu_batched(self, pen, X_work, y_arr, init, backend_name):
        """Batched GPU block coordinate descent for group_lasso penalty.

        Processes all groups in parallel within each iteration to minimize
        kernel launch overhead. Groups of the same size are batched together
        for efficient linear solves.
        """
        if backend_name == "cupy":
            import cupy as xp
        elif backend_name == "torch":
            import torch
            xp = torch
        else:
            raise ValueError(f"GPU backend required, got {backend_name}")

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp
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

        # Pre-compute XtX and Xty once
        XtX = X_work.T @ X_work / n
        Xty = (X_work.T @ y_arr.flatten()) / n

        # Pre-compute XtX blocks for each group
        _XtX_blocks = []
        for g_idx in _g_indices:
            _XtX_blocks.append(XtX[g_idx][:, g_idx])

        # Group indices by size for batched solving
        _size_groups = {}  # size -> list of (group_idx, indices)
        for g, g_idx in enumerate(_g_indices):
            sz = len(g_idx)
            if sz not in _size_groups:
                _size_groups[sz] = []
            _size_groups[sz].append((g, g_idx))

        if init is not None:
            if isinstance(init, np.ndarray):
                coef = xp.asarray(init) if backend_name == "cupy" else torch.from_numpy(init).to(X_work.device)
            else:
                coef = init.clone() if backend_name == "torch" else init.copy()
        else:
            coef = xp.zeros(pp, dtype=X_work.dtype) if backend_name == "cupy" else torch.zeros(pp, dtype=X_work.dtype, device=X_work.device)

        for iteration in range(self.max_iter):
            coef_old = coef.clone() if backend_name == "torch" else coef.copy()

            # Process groups by size for batched solving
            for sz, size_groups in _size_groups.items():
                n_batch = len(size_groups)
                if n_batch == 0:
                    continue

                # Collect indices for all groups of this size
                all_indices = []
                batch_g_indices = []
                for g, g_idx in size_groups:
                    all_indices.extend(g_idx)
                    batch_g_indices.append(g)

                # Compute rho_g for all groups of this size in one shot
                # rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + XtX_block[g] @ coef[g_idx]
                if backend_name == "cupy":
                    import cupy as cp
                    # Stack all indices for batched indexing
                    idx_arr = cp.array(all_indices, dtype=cp.int32)
                    # Compute XtX[g_idx, :] @ coef for all groups at once
                    XtX_coef = XtX[idx_arr, :] @ coef  # shape: (n_batch * sz,)
                    # Compute Xty for all groups
                    Xty_all = Xty[idx_arr]
                    # Compute block diagonal contributions
                    block_contrib = xp.zeros_like(Xty_all)
                    for i, (g, g_idx) in enumerate(size_groups):
                        block_contrib[i*sz:(i+1)*sz] = _XtX_blocks[g] @ coef[g_idx]
                    # rho_g = Xty - XtX_coef + block_contrib
                    rho_all = Xty_all - XtX_coef + block_contrib

                    # Solve all group systems in one batched call
                    # Reshape to (n_batch, sz, 1) for batched solve
                    rho_mat = rho_all.reshape(n_batch, sz, 1)
                    # Stack all XtX blocks into a single (n_batch, sz, sz) tensor
                    XtX_batch = xp.stack([_XtX_blocks[g] for g in batch_g_indices])
                    try:
                        w_all = xp.linalg.solve(XtX_batch, rho_mat)  # (n_batch, sz, 1)
                        w_all = w_all.reshape(n_batch, sz)
                    except Exception:
                        w_all = xp.zeros((n_batch, sz), dtype=X_work.dtype)

                    # Apply soft-thresholding to all groups at once
                    norms = xp.linalg.norm(w_all, axis=1)  # (n_batch,)
                    thresh = xp.array([alpha * _sqrt_pg[g] for g in batch_g_indices])
                    scale = xp.where(norms > thresh, 1.0 - thresh / (norms + 1e-12), 0.0)

                    # Write back coefficients
                    for i, (g, g_idx) in enumerate(size_groups):
                        coef[g_idx] = w_all[i] * scale[i]

                else:  # torch
                    import torch
                    # Stack all indices for batched indexing
                    idx_arr = torch.tensor(all_indices, dtype=torch.long, device=X_work.device)
                    # Compute XtX[g_idx, :] @ coef for all groups at once
                    XtX_coef = XtX[idx_arr, :] @ coef  # shape: (n_batch * sz,)
                    # Compute Xty for all groups
                    Xty_all = Xty[idx_arr]
                    # Compute block diagonal contributions
                    block_contrib = torch.zeros_like(Xty_all)
                    for i, (g, g_idx) in enumerate(size_groups):
                        block_contrib[i*sz:(i+1)*sz] = _XtX_blocks[g] @ coef[g_idx]
                    # rho_g = Xty - XtX_coef + block_contrib
                    rho_all = Xty_all - XtX_coef + block_contrib

                    # Solve all group systems in one batched call
                    rho_mat = rho_all.reshape(n_batch, sz, 1)
                    XtX_batch = torch.stack([_XtX_blocks[g] for g in batch_g_indices])
                    try:
                        w_all = torch.linalg.solve(XtX_batch, rho_mat)  # (n_batch, sz, 1)
                        w_all = w_all.reshape(n_batch, sz)
                    except Exception:
                        w_all = torch.zeros((n_batch, sz), dtype=X_work.dtype, device=X_work.device)

                    # Apply soft-thresholding to all groups at once
                    norms = torch.linalg.norm(w_all, dim=1)  # (n_batch,)
                    thresh = torch.tensor([alpha * _sqrt_pg[g] for g in batch_g_indices],
                                         dtype=X_work.dtype, device=X_work.device)
                    scale = torch.where(norms > thresh, 1.0 - thresh / (norms + 1e-12), torch.tensor(0.0, dtype=X_work.dtype, device=X_work.device))

                    # Write back coefficients
                    for i, (g, g_idx) in enumerate(size_groups):
                        coef[g_idx] = w_all[i] * scale[i]

            if self.fit_intercept:
                coef[pp - 1] = float(xp.mean(y_arr - X_work[:, :p] @ coef[:p])) if backend_name == "cupy" else float(torch.mean(y_arr - X_work[:, :p] @ coef[:p]))

            _max_change = float(xp.max(xp.abs(coef - coef_old))) if backend_name == "cupy" else float(torch.max(torch.abs(coef - coef_old)))
            if _max_change < self.tol:
                break

        n_iter = iteration + 1

        if self.fit_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0

        return beta, intercept, n_iter

    def _cd_elasticnet(self, pen, X_work, y_arr, init):
        """Coordinate descent for elasticnet penalty (squared_error loss).

        Matches R glmnet's CD algorithm for elasticnet:
        beta_j = S(rho_j, alpha*l1_ratio*n) / (X_j'X_j + alpha*(1-l1_ratio)*n)
        """
        import numpy as np

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp
        alpha = self.alpha
        l1_ratio = getattr(pen, 'l1_ratio', getattr(self, 'l1_ratio', 0.5))

        XtX = X_work.T @ X_work
        Xty = X_work.T @ y_arr.flatten()
        X_sq_norms = np.diag(XtX)

        if init is not None:
            coef = np.array(init, dtype=np.float64)
        else:
            coef = np.zeros(pp, dtype=np.float64)

        thresh = alpha * l1_ratio * n

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            for j in range(p):
                rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]
                if X_sq_norms[j] > 1e-10:
                    st = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0)
                    coef[j] = st / (X_sq_norms[j] + alpha * (1 - l1_ratio) * n)
                else:
                    coef[j] = 0.0

            if self.fit_intercept:
                coef[pp - 1] = np.mean(y_arr - X_work[:, :p] @ coef[:p])

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        n_iter = iteration + 1

        if self.fit_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0

        return beta, intercept, n_iter

    def _cd_l1(self, pen, X_work, y_arr, init):
        """Coordinate descent for L1 (lasso) penalty (squared_error loss).

        Matches R glmnet's CD algorithm:
        beta_j = S(rho_j, alpha*n) / X_j'X_j
        """
        import numpy as np

        n, pp = X_work.shape
        p = pp - 1 if self.fit_intercept else pp
        alpha = self.alpha

        XtX = X_work.T @ X_work
        Xty = X_work.T @ y_arr.flatten()
        X_sq_norms = np.diag(XtX)

        if init is not None:
            coef = np.array(init, dtype=np.float64)
        else:
            coef = np.zeros(pp, dtype=np.float64)

        thresh = alpha * n

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            for j in range(p):
                rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]
                if X_sq_norms[j] > 1e-10:
                    coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                else:
                    coef[j] = 0.0

            if self.fit_intercept:
                coef[pp - 1] = np.mean(y_arr - X_work[:, :p] @ coef[:p])

            if np.max(np.abs(coef - coef_old)) < self.tol:
                break

        n_iter = iteration + 1

        if self.fit_intercept:
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
        if backend_name == "cupy":
            import cupy as cp
            X_arr = cp.asarray(X, dtype=cp.float64) if not isinstance(X, cp.ndarray) else cp.asarray(X, dtype=cp.float64)
            y_arr = cp.asarray(y, dtype=cp.float64) if not isinstance(y, cp.ndarray) else cp.asarray(y, dtype=cp.float64)
        elif backend_name == "torch":
            import torch
            if not isinstance(X, torch.Tensor):
                X_arr = torch.from_numpy(np.asarray(X, dtype=np.float64)).to(_get_torch_device_str())
            else:
                X_arr = X.to(dtype=torch.float64)
            if not isinstance(y, torch.Tensor):
                y_arr = torch.from_numpy(np.asarray(y, dtype=np.float64)).to(_get_torch_device_str())
            else:
                y_arr = y.to(dtype=torch.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            p = X_arr.shape[1]
            X_work = self._column_stack(
                [X_arr, self._ones(X_arr.shape[0], backend_name, X_arr)],
                backend_name,
            )
            pen = self._selective_penalty(p, backend_name)
            init = None
            if self._init_coef is not None:
                init = np.append(self._init_coef, 0.0)
                if backend_name == "cupy":
                    init = cp.asarray(init)
                elif backend_name == "torch":
                    init = torch.from_numpy(init).to(X_arr.device)
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
                if backend_name == "cupy":
                    init = cp.asarray(init)
                elif backend_name == "torch":
                    init = torch.from_numpy(init).to(X_arr.device)
        else:
            p = X_arr.shape[1]
            X_work = X_arr
            pen = self._penalty
            init = None
            if self._init_coef is not None:
                init = np.asarray(self._init_coef, dtype=np.float64)
                if backend_name == "cupy":
                    init = cp.asarray(init)
                elif backend_name == "torch":
                    init = torch.from_numpy(init).to(X_arr.device)

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
        _use_fista = (
            (_pen_name in ("adaptive_l1", "adaptive_lasso") and _is_glm_loss)
            or (_pen_name in ("adaptive_l1", "adaptive_lasso") and not _is_glm_loss)
        )
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
            _X_feat = _to_numpy(X_work[:, :p] if self.fit_intercept else X_work)
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

            X_orig = X_work[:, :p] if self.fit_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                sample_weight=sample_weight,
            )
            if self.fit_intercept:
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
            X_feat = X_work[:, :p] if self.fit_intercept else X_work
            _n = X_feat.shape[0]
            _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
            _col_norms = xp.maximum(_col_norms, 1e-20)
            X_s = X_feat * (xp.sqrt(float(_n)) / _col_norms)
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

            X_orig = X_work[:, :p] if self.fit_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                sample_weight=sample_weight,
            )
            # fista_lla_path returns numpy, convert back to backend-native
            if self.fit_intercept:
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
            X_feat = X_work[:, :p] if self.fit_intercept else X_work
            _n = X_feat.shape[0]
            _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
            _col_norms = xp.maximum(_col_norms, 1e-20)
            X_s = X_feat * (xp.sqrt(float(_n)) / _col_norms)
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

            def _group_lla_factory(weights_np):
                # lla_weights returns per-coordinate; extract per-group weights
                _gw = np.array([float(weights_np[idx[0]]) if len(idx) > 0 else 0.0
                                for idx in _groups])
                return AdaptiveGroupLassoPenalty(
                    groups=_groups, alpha=_pen_alpha, weights=_gw,
                )

            X_orig = X_work[:, :p] if self.fit_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(
                self._loss, self._penalty,
                X_orig, y_arr,
                alpha_path=_alpha_path,
                max_lla_per_step=_max_lla_per_step,
                lla_tol=getattr(self, '_lla_tol', 1e-6),
                max_iter=_mi_path,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                sample_weight=sample_weight,
                lla_penalty_factory=_group_lla_factory,
            )
            # fista_lla_path returns numpy, convert back to backend-native
            if self.fit_intercept:
                params = xp.concatenate([xp.asarray(coef_np), xp.asarray([intercept])])
            else:
                params = xp.asarray(coef_np)
        elif _pen_name == "group_lasso":
            # Block CD for group_lasso: use GPU-native solver on GPU backends.
            if backend_name != "numpy":
                coef_gpu, intercept, n_iter = self._block_cd_group_lasso_gpu(
                    pen, X_work, y_arr, init, backend_name,
                )
                if self.fit_intercept:
                    if backend_name == "cupy":
                        import cupy as cp
                        params = cp.concatenate([coef_gpu, cp.array([intercept])])
                    elif backend_name == "torch":
                        import torch
                        params = torch.cat([coef_gpu, torch.tensor([intercept], device=coef_gpu.device)])
                else:
                    params = coef_gpu
            else:
                coef_np, intercept, n_iter = self._block_cd_group_lasso(
                    pen, X_work, y_arr, init,
                )
                if self.fit_intercept:
                    params = np.concatenate([coef_np, [intercept]])
                else:
                    params = coef_np
        elif solver_name in ("auto", "fista"):
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
                init_coef=init,
            )
        else:
            raise ValueError(f"Unsupported solver: {solver_name}")

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self.fit_intercept else 0)
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

        X_arr = X
        y_arr = y
        n_samples = X_arr.shape[0]
        if self.fit_intercept:
            X_work = self._column_stack(
                [self._ones(X_arr.shape[0], backend_name, X_arr), X_arr],
                backend_name,
            )
        else:
            X_work = X_arr

        # Warm-start intercept for log-link models to prevent divergence.
        # Starting from mu=1 (eta=0) is disastrous when y is far from 1.
        _loss_name = getattr(self._loss, 'name', '')
        _log_link_losses = ("gamma", "poisson", "inverse_gaussian",
                            "negative_binomial", "tweedie")
        if self.fit_intercept and _loss_name in _log_link_losses:
            _y_mean = float(np.mean(_to_numpy(y_arr)))
            _int_init = np.log(max(_y_mean, 1e-3))
            n_feat = X_work.shape[1]
            if backend_name == "numpy":
                init_coef = np.zeros(n_feat)
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
        else:
            init_coef = None

        solver = IRLSSolver(
            self._family_for_loss(), max_iter=self.max_iter, tol=self.tol
        )
        params, n_iter = solver.fit(
            X_work, y_arr,
            sample_weight=sample_weight,
            ridge_alpha=float(n_samples * self.alpha),
            ridge_penalize_intercept=False if self.fit_intercept else True,
            backend=backend_name,
            init_coef=init_coef,
        )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            X_arr.shape[1] + (1 if self.fit_intercept else 0)
        )
        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def _fit_cpu_loss(self, X, y, sample_weight=None, solver="fista"):
        """Fit using loss-aware solver (FISTA with arbitrary loss).

        For GLM losses (logistic, poisson) with intercept, augments X with
        a column of ones and uses a selective penalty (no penalty on intercept)
        to converge to the correct joint optimum.
        """
        from statgpu.glm_core._solver import fista_solver

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.loss in ("logistic", "poisson") and self.fit_intercept:
            # Augment X with intercept column
            X_aug = np.column_stack([X_arr, np.ones(X_arr.shape[0])])
            p = X_arr.shape[1]
            pen = self._penalty

            class SelectivePenalty:
                """Penalty wrapper: apply to first p entries, skip last (intercept)."""
                def proximal(self, w, step, backend="numpy"):
                    w_feat = w[:-1]
                    result_feat = pen.proximal(w_feat, step, backend=backend)
                    result = np.empty(w.shape[0], dtype=w.dtype)
                    result[:-1] = result_feat
                    result[-1] = np.clip(w[-1], -15.0, 15.0)
                    return result
                def value(self, coef):
                    return pen.value(coef[:-1])
                name = pen.name

            full_coef, n_iter = fista_solver(
                self._loss, SelectivePenalty(), X_aug, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = full_coef[:p]
            self.intercept_ = float(full_coef[p])
            self.n_iter_ = n_iter
        elif self.fit_intercept:
            # Squared error: center X and y, fit once
            X_arr = X_arr - X_arr.mean(axis=0)
            y_arr = y_arr - y_arr.mean()

            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef
            self.n_iter_ = n_iter
            self.intercept_ = float(np.mean(y) - np.mean(X, axis=0) @ coef)
        else:
            coef, n_iter = fista_solver(
                self._loss, self._penalty, X_arr, y_arr,
                max_iter=self.max_iter, tol=self.tol,
                init_coef=None, sample_weight=sample_weight,
            )

            self.coef_ = coef
            self.n_iter_ = n_iter
            self.intercept_ = 0.0

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))

    def _fit_cpu_irls(self, X, y, sample_weight=None):
        """Fit using IRLS for smooth penalty + smooth loss (e.g., Logistic/Poisson + L2).

        Each IRLS iteration:
            1. Compute working response z and weights W
            2. Solve: (X'WX + n*alpha*I) params = X'Wz
        """
        from statgpu.glm_core._irls import IRLSSolver
        from statgpu.glm_core._family import (
            Binomial, Poisson, Gaussian, Gamma,
            InverseGaussian, NegativeBinomial, Tweedie,
        )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = X_arr.shape[0]

        # Add intercept column if needed
        if self.fit_intercept:
            X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        # L2 penalty: for objective min loss/n + alpha*0.5*||w||^2,
        # IRLS uses unnormalized X'WX, so ridge = n * alpha.
        # Don't penalize the intercept column (matches sklearn/FISTA behavior).
        ridge_alpha = float(n_samples * self.alpha)
        ridge_penalize_intercept = False if self.fit_intercept else True

        # Select family
        if self.loss == "logistic":
            family = Binomial()
        elif self.loss == "poisson":
            family = Poisson()
        elif self.loss == "gamma":
            family = Gamma()
        elif self.loss == "inverse_gaussian":
            family = InverseGaussian()
        elif self.loss == "negative_binomial":
            family = NegativeBinomial()
        elif self.loss == "tweedie":
            family = Tweedie()
        else:
            family = Gaussian()

        solver = IRLSSolver(family, max_iter=self.max_iter, tol=self.tol)
        params, n_iter = solver.fit(
            X_arr, y_arr, sample_weight=sample_weight,
            ridge_alpha=ridge_alpha,
            ridge_penalize_intercept=ridge_penalize_intercept,
            backend="numpy",
        )

        self.n_iter_ = n_iter

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
            self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
        else:
            self.intercept_ = 0.0
            self.coef_ = params.copy()
            self._params = np.asarray(self.coef_).copy()

        self._df_resid = self._nobs - (X.shape[1] + (1 if self.fit_intercept else 0))

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


class PenalizedLinearRegression(PenalizedGeneralizedLinearModel):
    """Gaussian penalized regression.

    This typed estimator replaces the old ``PenalizedLinearRegression(loss=...)``
    entry point.  Use ``PenalizedLogisticRegression`` or
    ``PenalizedPoissonRegression`` for non-gaussian GLMs.
    """

    def __init__(
        self,
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
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
    ):
        super().__init__(
            loss="squared_error",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )


class PenalizedLogisticRegression(PenalizedGeneralizedLinearModel):
    """Binomial/logistic penalized GLM."""

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
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
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
    ):
        super().__init__(
            loss="logistic",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )

    def predict_proba(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_predict_X(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self.fit_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            p1 = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -500, 500)))
            return cp.column_stack([1.0 - p1, p1])
        if device == Device.TORCH:
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self.fit_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            p1 = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -500, 500)))
            return torch.column_stack([1.0 - p1, p1])
        raw = X @ self.coef_
        if self.fit_intercept:
            raw += self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
        return np.column_stack([1.0 - p1, p1])


class PenalizedPoissonRegression(PenalizedGeneralizedLinearModel):
    """Poisson penalized GLM."""

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
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
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
    ):
        super().__init__(
            loss="poisson",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )
