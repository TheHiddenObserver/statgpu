"""Fit mixin for PenalizedGeneralizedLinearModel."""
from __future__ import annotations
from statgpu.backends._torch_compile import compile_torch
import numpy as np
from statgpu._config import Device
from statgpu.backends import get_backend, _to_numpy, _LINALG_ERRORS
from statgpu.solvers._utils import _nesterov_momentum, _nesterov_update
from statgpu.penalties._categories import NONCONVEX as _NONCONVEX_PENALTIES, SPARSE as _SPARSE_PENALTIES
_SMOOTH_PENALTIES = frozenset({'l2', 'none', 'null', ''})

def _validate_sample_weight_backend(sample_weight, n_samples, backend_name):
    """Validate sample weights in place and synchronize only scalar reductions."""
    if getattr(sample_weight, 'ndim', None) != 1:
        raise ValueError('sample_weight must be one-dimensional')
    if int(sample_weight.shape[0]) != int(n_samples):
        raise ValueError('sample_weight must have length n_samples')
    if backend_name == 'torch':
        import torch
        if not bool(torch.all(torch.isfinite(sample_weight)).item()):
            raise ValueError('sample_weight must be finite')
        if bool(torch.any(sample_weight < 0).item()):
            raise ValueError('sample_weight must be non-negative')
        total = float(torch.sum(sample_weight).item())
    elif backend_name == 'cupy':
        import cupy as cp
        if not bool(cp.all(cp.isfinite(sample_weight)).item()):
            raise ValueError('sample_weight must be finite')
        if bool(cp.any(sample_weight < 0).item()):
            raise ValueError('sample_weight must be non-negative')
        total = float(cp.sum(sample_weight).item())
    else:
        weights = np.asarray(sample_weight)
        if not np.all(np.isfinite(weights)):
            raise ValueError('sample_weight must be finite')
        if np.any(weights < 0):
            raise ValueError('sample_weight must be non-negative')
        total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError('sample_weight must have a positive sum')
    return total
_SPECIAL_LLA_LOSSES = frozenset({'squared_error', 'quantile', ''})
_N_CONT_STEPS = 5
_N_CONT_STEPS_NONSMOOTH = 3
_MAX_LLA_PER_STEP_DEFAULT = 2
_SOLVER_DISPATCH_TABLE = [('exact', lambda l, p, b, lr, cv, ps: l == 'squared_error' and p == 'l2' and (b in ('numpy', 'cpu', ''))), ('newton', lambda l, p, b, lr, cv, ps: l == 'squared_error' and p == 'l2' and (b in ('cupy', 'torch'))), ('fista', lambda l, p, b, lr, cv, ps: p in _NONCONVEX_PENALTIES), ('fista', lambda l, p, b, lr, cv, ps: l == 'quantile'), ('fista', lambda l, p, b, lr, cv, ps: l == 'squared_error' and p in _SPARSE_PENALTIES), ('fista_bb', lambda l, p, b, lr, cv, ps: cv and l == 'poisson' and (b in ('cupy', 'torch')) and (p == 'l1') and (ps is None or ps < 2000000)), ('fista_bb', lambda l, p, b, lr, cv, ps: cv and l == 'poisson' and (b in ('cupy', 'torch')) and (p in ('elasticnet', 'en'))), ('fista', lambda l, p, b, lr, cv, ps: cv and l == 'poisson' and (p in _SPARSE_PENALTIES)), ('fista_bb', lambda l, p, b, lr, cv, ps: cv and l == 'negative_binomial' and (b in ('cupy', 'torch')) and (p == 'l1')), ('fista', lambda l, p, b, lr, cv, ps: cv and l == 'negative_binomial' and (b in ('cupy', 'torch')) and (p in ('elasticnet', 'en')) and (ps is not None) and (200000 <= ps < 1000000)), ('fista_bb', lambda l, p, b, lr, cv, ps: cv and l == 'negative_binomial' and (b in ('cupy', 'torch')) and (p in ('elasticnet', 'en'))), ('fista', lambda l, p, b, lr, cv, ps: l in ('gamma', 'inverse_gaussian') and p in _SPARSE_PENALTIES), ('fista', lambda l, p, b, lr, cv, ps: l == 'tweedie' and b in ('cupy', 'torch') and (p in _SPARSE_PENALTIES)), ('fista', lambda l, p, b, lr, cv, ps: cv and l == 'logistic' and (p in _SPARSE_PENALTIES)), ('fista', lambda l, p, b, lr, cv, ps: l in ('huber', 'bisquare', 'fair') and p in _SPARSE_PENALTIES), ('fista_bb', lambda l, p, b, lr, cv, ps: p in _SPARSE_PENALTIES), ('lbfgs', lambda l, p, b, lr, cv, ps: cv and p == 'l2' and (l == 'negative_binomial')), ('newton', lambda l, p, b, lr, cv, ps: cv and p == 'l2' and (l in ('poisson', 'tweedie'))), ('lbfgs', lambda l, p, b, lr, cv, ps: cv and p == 'l2' and (l in ('gamma', 'inverse_gaussian'))), ('newton', lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in ('gamma', 'tweedie', 'inverse_gaussian', 'logistic', 'poisson', 'negative_binomial')), ('newton', lambda l, p, b, lr, cv, ps: p in _SMOOTH_PENALTIES and l in ('huber', 'bisquare', 'fair', 'cox_ph'))]

def _preferred_penalized_glm_solver(loss_name, penalty_name, backend_name=None, l1_ratio=0.5, cv_mode=False, problem_size=None):
    """Private benchmark-backed solver policy for solver='auto'.

    This helper only chooses an internal solver.  It must never be used to
    override an explicitly requested solver or to change the selected device.

    Dispatch is table-driven: first matching rule wins.
    """
    loss_name = str(loss_name or '').lower()
    penalty_name = str(penalty_name or '').lower()
    backend_name = str(backend_name or '').lower()
    if problem_size is not None:
        problem_size = int(problem_size)
    for solver, cond in _SOLVER_DISPATCH_TABLE:
        if cond(loss_name, penalty_name, backend_name, l1_ratio, cv_mode, problem_size):
            return solver
    return 'fista'

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

def _irls_ridge_init(X, y, loss_name, alpha=0.01, max_iter=100, tol=0.0001, loss_kwargs=None):
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
    if loss_name in ('squared_error', ''):
        coef = _irls_ridge_init_cd(X, y, alpha, max_iter, tol)
    else:
        from statgpu.solvers import fista_solver
        from statgpu.penalties import get_penalty
        l2_pen = get_penalty('l2', alpha=alpha)
        loss_obj = _resolve_loss_name(loss_name, loss_kwargs=loss_kwargs)
        coef, _ = fista_solver(loss_obj, l2_pen, X, y, max_iter=max_iter, tol=tol)
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
    backend = _resolve_backend('auto', X)
    xp = _get_xp(backend)
    n, p = X.shape
    feat_norms = xp.sqrt(xp.sum(X ** 2, axis=0))
    if backend == 'torch':
        import torch
        feat_norms = xp.maximum(feat_norms, torch.tensor(1e-20, dtype=feat_norms.dtype, device=feat_norms.device))
        scale = torch.tensor(float(n) ** 0.5, dtype=X.dtype, device=X.device) / feat_norms
    else:
        feat_norms = xp.maximum(feat_norms, 1e-20)
        scale = xp.asarray(float(n) ** 0.5, dtype=X.dtype) / feat_norms
    X_work = X * scale
    XtX = X_work.T @ X_work / n
    Xty = X_work.T @ y / n
    if backend == 'torch':
        import torch
        I_mat = torch.eye(p, dtype=X.dtype, device=X.device)
        beta = torch.linalg.solve(XtX + alpha * I_mat, Xty)
    elif backend == 'cupy':
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
        # Direct public parameter replacement is part of the established
        # refit contract. Keep runtime aliases synchronized before any group
        # validation, loss construction, or penalty resolution.
        self._penalty_kwargs = (
            self.penalty_kwargs if self.penalty_kwargs is not None else {}
        )
        self._loss_kwargs = self.loss_kwargs if self.loss_kwargs is not None else {}

        if formula is not None:
            if data is None:
                raise ValueError('formula was provided but data is None. Pass data=your_dataframe when using formula.')
            from statgpu.core.formula import FormulaParser
            parser = FormulaParser(formula)
            y, X, design_info = parser.eval(data)
            if sample_weight is not None:
                sw_formula = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
                row_positions = parser.row_positions
                if sw_formula.shape[0] == len(data):
                    sample_weight = sw_formula[row_positions]
                elif sw_formula.shape[0] == X.shape[0]:
                    sample_weight = sw_formula
                else:
                    raise ValueError('For formula fitting, sample_weight must have length len(data) or the number of rows retained by the formula.')
            formula_column_names = list(design_info.column_names)
            self._design_info = design_info
            self._formula_has_intercept = 'Intercept' in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != 'Intercept']
            if self._formula_has_intercept:
                X = np.delete(X, formula_column_names.index('Intercept'), axis=1)
                self._use_intercept = True
            else:
                self._use_intercept = False
        else:
            if X is None or y is None:
                raise ValueError('Either formula+data or X+y must be provided.')
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None
            self._use_intercept = None
        if X is not None:
            X_arr = np.asarray(X) if not hasattr(X, 'shape') else X
            self.n_features_in_ = X_arr.shape[1] if X_arr.ndim >= 2 else 1
        self._penalty = self._resolve_penalty()
        self._loss = self._resolve_loss()
        self._validate_solver_penalty()
        self._validate_inference_request()
        if hasattr(self._loss, 'precompute_scale') and X is not None and (y is not None):
            self._loss.precompute_scale(X, y)
        self._inference_precomputed = False
        self._precomputed_gaussian_state = None
        self._clear_inference_state()
        backend = self._get_backend(backend='auto')
        backend_name = backend.name
        if self._device == Device.AUTO and backend_name in ('cupy', 'torch') and (X is not None):
            _n, _p = X.shape
            if _n * _p < 200000:
                backend_name = 'numpy'
        backend_name = self._auto_backend_override(backend_name, X)
        selected_solver = self._select_solver(self._loss, backend_name=backend_name, X=X)
        self._selected_solver = selected_solver
        self._selected_backend_name = backend_name
        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name).reshape(-1)
            _validate_sample_weight_backend(_sw_arr, X.shape[0], backend_name)
        if self._penalty.requires_init:
            init_coef = self._fit_initial(X, y, backend_name=backend_name)
            self._penalty.set_weights(init_coef)
        _pen_name = str(getattr(self._penalty, 'name', '')).lower()
        _loss_name = str(getattr(self._loss, 'name', '') if hasattr(self, '_loss') else self.loss).lower()
        _is_glm_loss = _loss_name not in _SPECIAL_LLA_LOSSES
        if _pen_name in ('scad', 'mcp') and self._lla_enabled and (not _is_glm_loss):
            self._nobs = X.shape[0]
            X_arr = self._to_array(X, backend=backend_name)
            y_arr = self._to_array(y, backend=backend_name)
            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(X_arr, y_arr, X_arr.shape[1], _loss_name)
            if _loss_name == 'quantile':
                from statgpu.solvers import proximal_irls_quantile_solver
                coef_np, intercept, n_iter = proximal_irls_quantile_solver(self._loss, self._penalty, X_arr, y_arr, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=_sw_arr)
            else:
                from statgpu.solvers import fista_lla_path
                coef_np, intercept, n_iter = fista_lla_path(self._loss, self._penalty, X_arr, y_arr, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=_sw_arr)
            self.coef_ = coef_np
            self.intercept_ = intercept
            self.n_iter_ = n_iter
            if self._effective_intercept:
                self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
            else:
                self._params = np.asarray(self.coef_).copy()
            self._df_resid = X.shape[0] - (X.shape[1] + (1 if self._effective_intercept else 0))
            self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)
            if backend_name == 'cupy':
                self._cleanup_cuda_memory()
            elif backend_name == 'torch':
                self._cleanup_torch_memory()
            self._fitted = True
            return self
        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)
        if backend_name == 'torch':
            self._fit_torch(X_arr, y_arr, _sw_arr)
        elif backend_name == 'cupy':
            self._fit_gpu(X_arr, y_arr, _sw_arr)
        else:
            self._fit_cpu(X_arr, y_arr, _sw_arr)
        self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)
        self._fitted = True
        if hasattr(self, '_cv_cache') and (not getattr(self, '_preserve_cv_cache', False)):
            del self._cv_cache
        return self

    def _select_solver(self, loss, backend_name=None, X=None):
        """Auto-select solver based on loss, penalty, and backend."""
        if self._solver != 'auto':
            return self._solver
        return _preferred_penalized_glm_solver(getattr(loss, 'name', self.loss), getattr(self._penalty, 'name', self.penalty), backend_name=backend_name, l1_ratio=getattr(self._penalty, 'l1_ratio', self.l1_ratio), cv_mode=False, problem_size=None if X is None else int(X.shape[0]) * int(X.shape[1]))

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
    _AUTO_BACKEND_CPU_OVERRIDES = [('squared_error', ('l2',), 'numpy', 'large squared-error exact solve is faster on CPU'), ('squared_error', ('l1', 'elasticnet', 'en'), 'numpy', 'large squared-error l1/elasticnet is faster on CPU'), ('negative_binomial', ('l1', 'elasticnet', 'en'), 'numpy', 'large negative-binomial l1/elasticnet is faster on CPU'), ('logistic', ('l1', 'elasticnet', 'en'), 'numpy', 'large logistic {penalty} is faster on CPU'), ('gamma', ('l2',), 'numpy', 'large gamma l2/newton is faster on CPU'), ('tweedie', ('l1', 'elasticnet', 'en'), 'numpy', 'large tweedie {penalty} is faster on CPU')]
    _AUTO_BACKEND_CUPY_OVERRIDES = [('negative_binomial', ('l2',), 'torch', 'large negative-binomial l2 is faster on {target} than cupy'), ('logistic', ('l1', 'elasticnet', 'en'), 'torch', 'large logistic {penalty} is faster on {target} than cupy'), ('poisson', ('l1', 'elasticnet', 'en'), 'torch', 'large poisson {penalty} is faster on {target} than cupy')]

    def _auto_backend_override(self, backend_name, X):
        """Benchmark-backed backend routing for device='auto' only."""
        self._auto_backend_reason = None
        if self._device != Device.AUTO or self._solver != 'auto' or X is None:
            return backend_name
        n_samples, n_features = X.shape
        problem_size = int(n_samples) * int(n_features)
        if problem_size < 1000000:
            return backend_name
        loss_name = str(getattr(self._loss, 'name', self.loss)).lower()
        penalty_name = str(getattr(self._penalty, 'name', self.penalty)).lower()
        torch_ok = self._torch_cuda_available()
        for loss, penalties, target, reason_tpl in self._AUTO_BACKEND_CPU_OVERRIDES:
            if loss_name == loss and penalty_name in penalties:
                self._auto_backend_reason = reason_tpl.format(penalty=penalty_name)
                return target
        if backend_name == 'cupy':
            for loss, penalties, target, reason_tpl in self._AUTO_BACKEND_CUPY_OVERRIDES:
                if loss_name == loss and penalty_name in penalties:
                    if torch_ok:
                        self._auto_backend_reason = reason_tpl.format(penalty=penalty_name, target='torch')
                        return 'torch'
                    self._auto_backend_reason = reason_tpl.format(penalty=penalty_name, target='CPU')
                    return 'numpy'
        return backend_name

    def _fit_initial(self, X, y, backend_name='numpy'):
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
        init_method = getattr(self._penalty, 'init_method', 'auto')
        _is_glm = getattr(self, 'loss', 'squared_error') != 'squared_error'
        _is_nonconvex = not getattr(self._penalty, 'is_convex', True)
        if not _is_glm and (not self._penalty.requires_init) and (init_method == 'ols' or (init_method == 'auto' and n_samples > n_features)):
            ols_coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return ols_coef
        if _is_glm and _is_nonconvex:
            from statgpu.penalties import get_penalty
            from statgpu.solvers import fista_solver
            l2_pen = get_penalty('l2', alpha=0.001)
            loss_obj = self._resolve_loss()
            if backend_name in ('torch', 'cupy'):
                backend = get_backend(backend=backend_name, device='cuda')
                X_b = backend.asarray(X, dtype=backend.float64)
                y_b = backend.asarray(y, dtype=backend.float64)
            else:
                X_b = np.asarray(_to_numpy(X), dtype=np.float64)
                y_b = np.asarray(_to_numpy(y), dtype=np.float64)
            init_coef, _ = fista_solver(loss_obj, l2_pen, X_b, y_b, max_iter=500, tol=0.0001)
            return init_coef
        if self._penalty.requires_init:
            loss_name = getattr(self, 'loss', 'squared_error')
            if backend_name in ('torch', 'cupy'):
                backend = get_backend(backend=backend_name, device='cuda')
                X_b = backend.asarray(X, dtype=backend.float64)
                y_b = backend.asarray(y, dtype=backend.float64)
            else:
                X_b = np.asarray(_to_numpy(X), dtype=np.float64)
                y_b = np.asarray(_to_numpy(y), dtype=np.float64)
            init_coef = _irls_ridge_init(X_b, y_b, loss_name=loss_name, alpha=0.01, max_iter=100, tol=0.0001, loss_kwargs=getattr(self, 'loss_kwargs', None))
            return init_coef
        from statgpu.linear_model.wrappers._ridge import Ridge
        init_model = Ridge(alpha=0.1, fit_intercept=self._effective_intercept, device=self._device)
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
        if loss_name == 'quantile':
            _tau = getattr(self._loss, '_tau', 0.5)
            _intercept = float(_np.quantile(_y_feat, _tau))
            _r = _y_feat - _intercept
            _psi = _np.where(_r >= 0, _tau, -(1.0 - _tau))
            _lam_max = float(_np.max(_np.abs(_X_feat.T @ _psi / _n)))
        else:
            _col_norms = _np.sqrt(_np.sum(_X_feat ** 2, axis=0))
            _col_norms = _np.maximum(_col_norms, 1e-20)
            _X_s = _X_feat * (_np.sqrt(_n) / _col_norms)
            _y_c = _y_feat - _np.mean(_y_feat)
            _lam_max = float(_np.max(_np.abs(_X_s.T @ _y_c / _n)))
        _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))
        if n_cont is None:
            n_cont = _N_CONT_STEPS_NONSMOOTH if loss_name == 'quantile' else _N_CONT_STEPS
        _alpha_start = max(_lam_max, _target_alpha * 1.1)
        if not _np.isfinite(_alpha_start) or _alpha_start <= 0.0 or _target_alpha <= 0.0:
            _alpha_path = _np.linspace(max(_lam_max, 0.0), _target_alpha, n_cont)
        else:
            _alpha_path = _np.geomspace(_alpha_start, _target_alpha, n_cont)
        _max_lla = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // n_cont)
        _saved_mi = self._max_iter
        _mi_path = [_saved_mi if i == n_cont - 1 else max(100, _saved_mi // 10) for i in range(n_cont)]
        return (_alpha_path, _max_lla, _mi_path)

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
        solver_name = self._selected_solver or self._select_solver(self._loss, backend_name='numpy')
        if self.loss != 'squared_error' or solver_name in ('irls', 'newton', 'lbfgs', 'admm'):
            if solver_name == 'irls':
                self._dispatch_irls(X, y, sample_weight, solver_name, 'numpy')
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, 'numpy')
            return
        _cd_penalties_for_sqerr = ('scad', 'mcp', 'adaptive_l1', 'adaptive_lasso', 'group_lasso')
        if getattr(self._penalty, 'name', '') in _cd_penalties_for_sqerr:
            self._fit_loss_backend(X, y, sample_weight, solver_name, 'numpy')
            return
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
        _cv = getattr(self, '_cv_cache', None)
        if _cv is not None and 'XtX' in _cv:
            XtX = _cv['XtX']
            Xty = _cv['Xty']
        else:
            XtX = X_work.T @ X_work
            Xty = X_work.T @ y_work.flatten()
        pen = self._penalty
        if solver_name == 'exact':
            if pen.name != 'l2':
                raise ValueError("solver='exact' is only supported for L2/Ridge penalty.")
            self.coef_ = self._solve_exact_numpy(XtX, Xty, n_eff)
            self.n_iter_ = 1
            if self._effective_intercept:
                self.intercept_ = float(y_mean - X_mean @ self.coef_)
                self._params = np.concatenate([[self.intercept_], self.coef_])
            else:
                self.intercept_ = 0.0
                self._params = self.coef_.copy()
            self._df_resid = n_samples - (n_features + (1 if self._effective_intercept else 0))
            return
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            from statgpu.backends._array_ops import _max_eigval_power
            L = _max_eigval_power(XtX) / n_eff
        if L <= 0:
            self.coef_ = np.zeros(n_features)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            _cd_penalties = ('adaptive_l1', 'adaptive_lasso', 'scad', 'mcp', 'group_lasso')
            if solver_name in ('fista_bb', 'fista') and pen.name not in _cd_penalties:
                if hasattr(self, '_init_coef') and self._init_coef is not None:
                    coef = np.asarray(self._init_coef, dtype=np.float64).copy()
                else:
                    coef = np.zeros(n_features)
                y_k = coef.copy()
                t_k = 1.0
                for iteration in range(self._max_iter):
                    coef_old = coef.copy()
                    grad_at_y = (XtX @ y_k - Xty) / n_eff
                    w_tilde = y_k - step * grad_at_y
                    coef = pen.proximal(w_tilde, step, backend='numpy')
                    if iteration > 0 and iteration % 50 == 0:
                        t_k = 1.0
                    y_k, t_k = _nesterov_update(coef, coef_old, t_k)
                    self.n_iter_ = iteration + 1
                    if np.sum(np.abs(coef - coef_old)) < self._tol:
                        break
            else:
                X_sq_norms = np.diag(XtX)
                if hasattr(self, '_init_coef') and self._init_coef is not None:
                    coef = np.asarray(self._init_coef, dtype=np.float64).copy()
                else:
                    coef = np.zeros(n_features)
                _adaptive_thresh = None
                if pen.name in ('adaptive_l1', 'adaptive_lasso'):
                    _w = np.asarray(getattr(pen, '_weights', np.ones(n_features)), dtype=float)
                    _adaptive_thresh = self.alpha * _w * n_eff
                _a_scad = float(getattr(pen, 'a', 3.7)) if pen.name == 'scad' else 0.0
                _gamma_mcp = float(getattr(pen, 'gamma', 3.0)) if pen.name == 'mcp' else 0.0
                _is_group = pen.name == 'group_lasso'
                if _is_group:
                    _g_indices = getattr(pen, '_group_indices', None)
                    _sqrt_pg = getattr(pen, '_sqrt_pg', None)
                    if _g_indices is None or _sqrt_pg is None:
                        raise ValueError('group_lasso penalty must have groups set. Pass groups=... in penalty_kwargs.')
                    _n_groups = len(_g_indices)
                    _XtX_blocks = []
                    for g_idx in _g_indices:
                        _XtX_blocks.append(XtX[np.ix_(g_idx, g_idx)])
                for iteration in range(self._max_iter):
                    coef_old = coef.copy()
                    if _is_group:
                        for g in range(_n_groups):
                            g_idx = _g_indices[g]
                            rho_g = Xty[g_idx] - XtX[g_idx, :] @ coef + _XtX_blocks[g] @ coef[g_idx]
                            try:
                                w_g = np.linalg.solve(_XtX_blocks[g], rho_g)
                            except np.linalg.LinAlgError:
                                w_g = np.zeros(len(g_idx))
                            norm_w = np.linalg.norm(w_g)
                            thresh_g = self.alpha * _sqrt_pg[g]
                            if norm_w > thresh_g:
                                coef[g_idx] = w_g * (1.0 - thresh_g / norm_w)
                            else:
                                coef[g_idx] = 0.0
                    else:
                        for j in range(n_features):
                            rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]
                            if pen.name in ('adaptive_l1', 'adaptive_lasso'):
                                thresh = _adaptive_thresh[j]
                                if X_sq_norms[j] > 1e-10:
                                    coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                                else:
                                    coef[j] = 0.0
                            elif pen.name == 'l1':
                                thresh = self.alpha * n_eff
                                if X_sq_norms[j] > 1e-10:
                                    coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
                                else:
                                    coef[j] = 0.0
                            elif pen.name == 'elasticnet':
                                thresh = self.alpha * self.l1_ratio * n_eff
                                if X_sq_norms[j] > 1e-10:
                                    st = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0)
                                    coef[j] = st / (X_sq_norms[j] + self.alpha * (1 - self.l1_ratio) * n_eff)
                                else:
                                    coef[j] = 0.0
                            elif pen.name == 'scad':
                                a_scad = max(float(_a_scad), 1.0 + 1e-06)
                                if abs(a_scad - 2.0) < 1e-06:
                                    a_scad = 2.0 + 1e-06
                                if X_sq_norms[j] > 1e-10:
                                    w_j = rho_j / X_sq_norms[j]
                                    aw = np.abs(w_j)
                                    lam = self.alpha * n_eff
                                    if aw > a_scad * lam:
                                        coef[j] = w_j
                                    elif aw > lam:
                                        coef[j] = np.sign(w_j) * ((a_scad - 1.0) * aw - a_scad * lam) / (a_scad - 2.0)
                                    else:
                                        coef[j] = 0.0
                                else:
                                    coef[j] = 0.0
                            elif pen.name == 'mcp':
                                gamma_mcp = max(float(_gamma_mcp), 1.0 + 1e-06)
                                if X_sq_norms[j] > 1e-10:
                                    w_j = rho_j / X_sq_norms[j]
                                    aw = np.abs(w_j)
                                    lam = self.alpha * n_eff
                                    if aw > gamma_mcp * lam:
                                        coef[j] = w_j
                                    elif aw > lam:
                                        coef[j] = np.sign(w_j) * (aw - lam) / (1.0 - 1.0 / gamma_mcp)
                                    else:
                                        coef[j] = 0.0
                                else:
                                    coef[j] = 0.0
                            else:
                                raise NotImplementedError(f"Coordinate descent not implemented for penalty '{pen.name}'. Use solver='fista'.")
                    self.n_iter_ = iteration + 1
                    if np.sum(np.abs(coef - coef_old)) < self._tol:
                        break
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
        self._fit_gpu_backend(X, y, sample_weight, backend_name='cupy')

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FISTA. Delegates to unified backend."""
        self._fit_gpu_backend(X, y, sample_weight, backend_name='torch')

    @staticmethod
    def _soft_threshold_gpu(w, thresh, xp):
        """Backend-agnostic soft-thresholding on GPU."""
        if xp.__name__ == 'torch':
            import torch
            return torch.sign(w) * torch.relu(torch.abs(w) - thresh)
        return xp.sign(w) * xp.maximum(xp.abs(w) - thresh, 0.0)

    def _fit_gpu_backend(self, X, y, sample_weight=None, backend_name='cupy'):
        """Unified GPU fit method for both CuPy and Torch backends.

        Handles exact (L2), FISTA, and FISTA-BE solvers with inline
        XtX precomputation and fused element-wise kernels.
        """
        from statgpu.backends._utils import _get_xp, xp_asarray, xp_zeros, xp_copy, xp_ones
        from statgpu.backends import _to_numpy
        from statgpu.backends._array_ops import _abs_sum_dev
        xp = _get_xp(backend_name)
        is_torch = backend_name == 'torch'
        solver_name = self._selected_solver or self._select_solver(self._loss, backend_name=backend_name)
        _backend_label = 'Torch' if is_torch else 'CuPy'
        if solver_name not in ('fista', 'fista_bb', 'admm', 'auto', 'exact', 'irls', 'newton', 'lbfgs'):
            raise ValueError(f"{_backend_label} backend supports solver='fista', 'fista_bb', 'admm', 'exact', 'irls', 'newton', and 'lbfgs', got '{solver_name}'.")
        n_samples, n_features = X.shape
        self._nobs = n_samples
        if solver_name == 'exact':
            if self._penalty.name != 'l2':
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
                n_eff = _validate_sample_weight_backend(sw, n_samples, backend_name)
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
            if sw is None and _cv is not None and ('XtX' in _cv):
                XtX = _cv['XtX']
                Xty = _cv['Xty']
            else:
                XtX = X_work.T @ X_work
                Xty = X_work.T @ y_work
            solve_fn = getattr(self, f"_solve_exact_{('torch' if is_torch else 'cupy')}")
            coef = solve_fn(XtX, Xty, n_eff)
            self.n_iter_ = 1
            if self._effective_intercept:
                intercept_gpu = (y_mean.reshape(1) - X_mean.reshape(1, -1) @ coef.reshape(-1, 1)).reshape(-1)
                coef_full_gpu = xp.concatenate([intercept_gpu, coef.reshape(-1)])
            else:
                coef_full_gpu = coef.reshape(-1)
            if self._compute_inference_enabled:
                infer_fn = getattr(self, f"_precompute_exact_l2_inference_{('torch' if is_torch else 'cupy')}")
                infer_fn(X, y, XtX, X_mean, coef_full_gpu, n_samples, sample_weight=sw, normalization=n_eff)
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
        if solver_name in ('irls', 'newton', 'lbfgs'):
            if solver_name == 'irls':
                self._dispatch_irls(X, y, sample_weight, solver_name, backend_name)
            else:
                self._fit_loss_backend(X, y, sample_weight, solver_name, backend_name)
            return
        if self.loss != 'squared_error' or solver_name == 'admm' or self._penalty.name not in ('l1', 'elasticnet', 'en'):
            self._fit_loss_backend(X, y, sample_weight, solver_name, backend_name)
            return
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
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        elif n_features < 1000:
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
        elif solver_name in ('fista_bb', 'fista'):
            step = 1.0 / L
            step_over_n = step / n_samples
            step_over_n_Xty = step_over_n * Xty
            if self._penalty.name in ('elasticnet', 'en'):
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
            _fused_step = None
            _fused_step_l2 = None
            _st_fn = self._soft_threshold_gpu
            if is_torch:
                import torch
                if _use_l2:

                    def _fista_elementwise_l2(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n, _thresh, _l2_scale, _coef_old, _beta):
                        w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                        c = _st_fn(w, _thresh, xp) / _l2_scale
                        y = c + _beta * (c - _coef_old)
                        return (c, y)
                    _fused_step_l2 = compile_torch(_fista_elementwise_l2, workload='iterative')
                else:

                    def _fista_elementwise(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n, _thresh, _coef_old, _beta):
                        w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                        c = _st_fn(w, _thresh, xp)
                        y = c + _beta * (c - _coef_old)
                        return (c, y)
                    _fused_step = compile_torch(_fista_elementwise, workload='iterative')
            else:
                import cupy as cp
                if _use_l2:
                    try:

                        @cp.fuse()
                        def _fista_elementwise_l2(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n, _thresh, _l2_scale, _coef_old, _beta):
                            w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                            c = cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0) / _l2_scale
                            y = c + _beta * (c - _coef_old)
                            return (c, y)
                        _fused_step_l2 = _fista_elementwise_l2
                        _dummy = cp.zeros(1, dtype=X.dtype)
                        _fused_step_l2(_dummy, _dummy, _dummy, 0.0, 0.0, 1.0, _dummy, 0.0)
                    except Exception:
                        _fused_step_l2 = None
                else:
                    try:

                        @cp.fuse()
                        def _fista_elementwise(_y_k, _xtx_y, _step_over_n_Xty, _step_over_n, _thresh, _coef_old, _beta):
                            w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                            c = cp.sign(w) * cp.maximum(cp.abs(w) - _thresh, 0.0)
                            y = c + _beta * (c - _coef_old)
                            return (c, y)
                        _fused_step = _fista_elementwise
                        _dummy = cp.zeros(1, dtype=X.dtype)
                        _fused_step(_dummy, _dummy, _dummy, 0.0, 0.0, _dummy, 0.0)
                    except Exception:
                        _fused_step = None
            for iteration in range(self._max_iter):
                coef_old = xp_copy(coef)
                xtx_y = XtX @ y_k
                if _use_l2:
                    if _fused_step_l2 is not None:
                        coef, y_k = _fused_step_l2(y_k, xtx_y, step_over_n_Xty, step_over_n, thresh, l2_scale, coef_old, beta)
                    else:
                        w_tilde = y_k - step_over_n * xtx_y + step_over_n_Xty
                        coef = _st_fn(w_tilde, thresh, xp) / l2_scale
                        y_k = coef + beta * (coef - coef_old)
                elif _fused_step is not None:
                    coef, y_k = _fused_step(y_k, xtx_y, step_over_n_Xty, step_over_n, thresh, coef_old, beta)
                else:
                    w_tilde = y_k - step_over_n * xtx_y + step_over_n_Xty
                    coef = _st_fn(w_tilde, thresh, xp)
                    y_k = coef + beta * (coef - coef_old)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0
                beta, t_k = _nesterov_momentum(t_k)
                self.n_iter_ = iteration + 1
                if iteration % 5 == 4 and float(_to_numpy(_abs_sum_dev(coef - coef_old))) < self._tol:
                    break
        else:
            step = 1.0 / L
            if hasattr(self, '_init_coef') and self._init_coef is not None:
                coef = xp_asarray(self._init_coef, dtype=X.dtype, xp=xp, ref_arr=X)
            else:
                coef = xp_zeros(n_features, X.dtype, xp, ref_arr=X)
            y_k = xp_copy(coef)
            t_k = 1.0
            for iteration in range(self._max_iter):
                coef_old = xp_copy(coef)
                grad = (XtX @ y_k - Xty) / n_samples
                w_tilde = y_k - step * grad
                coef = self._penalty.proximal(w_tilde, step, backend=backend_name)
                if iteration > 0 and iteration % 50 == 0:
                    t_k = 1.0
                y_k, t_k = _nesterov_update(coef, coef_old, t_k)
                self.n_iter_ = iteration + 1
                if iteration % 5 == 4 and float(_to_numpy(_abs_sum_dev(coef - coef_old))) < self._tol:
                    break
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
        if self._compute_inference_enabled and 'debiased' in str(getattr(self, 'inference_method', '')).lower():
            penalty_name = str(getattr(self._penalty, 'name', self.penalty)).lower()
            if penalty_name in ('l1', 'elasticnet', 'en'):
                infer_fn = getattr(self, f"_compute_inference_debiased_{('torch' if is_torch else 'gpu')}")
                infer_fn(X, y, coef)
        if is_torch:
            self._cleanup_torch_memory()
        else:
            self._cleanup_cuda_memory()

    def _ridge_alpha_for_exact(self) -> float:
        """Return L2 alpha for the exact Ridge normal equations."""
        return float(getattr(self._penalty, 'alpha', self.alpha))

    def _solve_exact_numpy(self, XtX, Xty, normalization):
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + float(normalization) * alpha * np.eye(p, dtype=XtX.dtype)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ Xty

    def _solve_exact_cupy(self, XtX, Xty, normalization):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + float(normalization) * alpha * cp.eye(p, dtype=XtX.dtype)
        try:
            L = cp.linalg.cholesky(A)
            tmp = cp_solve_triangular(L, Xty, lower=True)
            return cp_solve_triangular(L.T, tmp, lower=False)
        except _LINALG_ERRORS:
            try:
                return cp.linalg.solve(A, Xty)
            except _LINALG_ERRORS:
                return cp.linalg.pinv(A) @ Xty

    def _solve_exact_torch(self, XtX, Xty, normalization):
        import torch
        alpha = self._ridge_alpha_for_exact()
        p = XtX.shape[0]
        A = XtX + float(normalization) * alpha * torch.eye(p, dtype=XtX.dtype, device=XtX.device)
        try:
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
            raise ValueError('group_lasso penalty must have groups set. Pass groups=... in penalty_kwargs.')
        _n_groups = len(_g_indices)
        XtX = X_work.T @ X_work / n
        Xty = X_work.T @ y_arr.flatten() / n
        _XtX_blocks = []
        for g_idx in _g_indices:
            _XtX_blocks.append(XtX[np.ix_(g_idx, g_idx)])
        if init is not None:
            coef = np.array(init, dtype=np.float64)
        else:
            coef = np.zeros(pp, dtype=np.float64)
        iteration = -1
        for iteration in range(self._max_iter):
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
            if np.max(np.abs(coef - coef_old)) < self._tol:
                break
        n_iter = iteration + 1
        if self._effective_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0
        return (beta, intercept, n_iter)

    def _block_cd_group_lasso_gpu(self, pen, X_work, y_arr, init, backend_name):
        """GPU-native block coordinate descent for group_lasso penalty.

        Same algorithm as _block_cd_group_lasso but keeps all arrays on GPU.
        Enforces float64 precision to avoid NaN from float32 conditioning issues.
        """
        from statgpu.backends._array_ops import _xp_copy, _xp_zeros, _xp_asarray, _xp_eye
        from statgpu.backends._utils import _get_xp, xp_astype
        xp = _get_xp(backend_name)
        X_work = xp_astype(X_work, xp.float64, xp)
        y_arr = xp_astype(y_arr, xp.float64, xp)
        n, pp = X_work.shape
        p = pp - 1 if self._effective_intercept else pp
        alpha = self.alpha
        _inner = getattr(self, '_penalty', pen)
        _g_indices = getattr(_inner, '_group_indices', None)
        _sqrt_pg_np = getattr(_inner, '_sqrt_pg', None)
        if _g_indices is None or _sqrt_pg_np is None:
            raise ValueError('group_lasso penalty must have groups set. Pass groups=... in penalty_kwargs.')
        _n_groups = len(_g_indices)
        _sqrt_pg = [float(s) for s in _sqrt_pg_np]
        _g_indices_backend = [_xp_asarray(np.asarray(g_idx, dtype=np.int64), xp.int64, X_work) for g_idx in _g_indices]
        _sqrt_pg_arr = _xp_asarray(np.asarray(_sqrt_pg, dtype=np.float64), X_work.dtype, X_work)
        XtX = X_work.T @ X_work / n
        Xty = X_work.T @ y_arr.flatten() / n
        from statgpu.backends._array_ops import _scalar_tensor
        _XtX_blocks = []
        _ridge = _scalar_tensor(1e-10, X_work)
        for g_idx in _g_indices_backend:
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
        _equal_size = len(set((len(g) for g in _g_indices))) == 1
        _gs = len(_g_indices[0]) if _equal_size else 0
        _contiguous = _equal_size and all((_g_indices[g][0] == g * _gs for g in range(_n_groups)))
        _flat_idx_backend = None
        if _equal_size and (not _contiguous):
            _flat_idx_backend = _xp_asarray(np.asarray([i for group in _g_indices for i in group], dtype=np.int64), xp.int64, X_work)
        if _equal_size and _n_groups > 1:
            _XtX_batched = xp.stack(_XtX_blocks)
        iteration = -1
        for iteration in range(self._max_iter):
            coef_old = _xp_copy(coef)
            if _equal_size and _n_groups > 1:
                XtX_coef = XtX @ coef
                if _contiguous:
                    coef_mat = coef[:p].reshape(_n_groups, _gs)
                    XtX_coef_mat = XtX_coef[:p].reshape(_n_groups, _gs)
                    Xty_mat = Xty[:p].reshape(_n_groups, _gs)
                else:
                    coef_mat = coef[_flat_idx_backend].reshape(_n_groups, _gs)
                    XtX_coef_mat = XtX_coef[_flat_idx_backend].reshape(_n_groups, _gs)
                    Xty_mat = Xty[_flat_idx_backend].reshape(_n_groups, _gs)
                diag_contrib = xp.einsum('gsj,gj->gs', _XtX_batched, coef_mat)
                rho_mat = Xty_mat - XtX_coef_mat + diag_contrib
                try:
                    w_mat = xp.linalg.solve(_XtX_batched, rho_mat)
                except Exception:
                    w_mat = xp.zeros_like(rho_mat)
                bad = xp.isnan(w_mat) | xp.isinf(w_mat)
                if xp.any(bad):
                    w_mat = xp.where(bad, 0.0, w_mat)
                norms = xp.sqrt(xp.sum(w_mat ** 2, axis=1))
                thresh = alpha * _sqrt_pg_arr
                scale = xp.where(norms > thresh, 1.0 - thresh / (norms + 1e-300), 0.0)
                scaled_mat = w_mat * scale[:, None]
                if _contiguous:
                    coef[:p] = scaled_mat.reshape(-1)
                else:
                    coef[_flat_idx_backend] = scaled_mat.reshape(-1)
            else:
                for g in range(_n_groups):
                    g_idx = _g_indices_backend[g]
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
            if _max_change < self._tol:
                break
        n_iter = iteration + 1
        if self._effective_intercept:
            beta = coef[:p]
            intercept = float(coef[p])
        else:
            beta = coef
            intercept = 0.0
        return (beta, intercept, n_iter)

    def _fit_loss_backend(self, X, y, sample_weight, solver_name, backend_name):
        """Fit GLMLoss + Penalty without changing the selected backend."""
        from statgpu.solvers import fista_solver, fista_bb_solver, admm_solver, lbfgs_solver, newton_solver
        from statgpu.backends._array_ops import _xp_asarray
        from statgpu.backends._utils import _get_xp
        _xp = _get_xp(backend_name)
        _ref = X if not isinstance(X, np.ndarray) else _xp.zeros(1, dtype=_xp.float64)
        X_arr = _xp_asarray(X, _xp.float64, _ref)
        y_arr = _xp_asarray(y, _xp.float64, X_arr)
        if self._effective_intercept:
            p = X_arr.shape[1]
            X_work = self._column_stack([X_arr, self._ones(X_arr.shape[0], backend_name, X_arr)], backend_name)
            pen = self._selective_penalty(p, backend_name)
            init = None
            if self._init_coef is not None:
                init_intercept = float(getattr(self, '_init_intercept', 0.0) or 0.0)
                init = np.append(self._init_coef, init_intercept)
                init = _xp_asarray(init, X_arr.dtype, X_arr)
            else:
                _loss_name = getattr(self._loss, 'name', '')
                _y_mean = float(np.mean(_to_numpy(y_arr)))
                if _loss_name == 'poisson':
                    _int_init = np.log(max(_y_mean, 0.001))
                elif _loss_name == 'logistic':
                    _y_mean_clipped = np.clip(_y_mean, 0.001, 1.0 - 0.001)
                    _int_init = np.log(_y_mean_clipped / (1.0 - _y_mean_clipped))
                elif _loss_name in ('gamma', 'inverse_gaussian', 'negative_binomial', 'tweedie', 'cox_ph'):
                    _int_init = np.log(max(_y_mean, 0.001))
                elif _loss_name == 'quantile':
                    _tau = getattr(self._loss, '_tau', 0.5)
                    _int_init = float(np.quantile(_to_numpy(y_arr), _tau))
                else:
                    _int_init = _y_mean
                _robust_losses = ('quantile', 'huber', 'bisquare', 'fair')
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
        _loss_name = getattr(self._loss, 'name', '')
        _pen_name = getattr(pen, 'name', '')
        if not _pen_name:
            _pen_name = getattr(self._penalty, 'name', '')
        _is_glm_loss = _loss_name not in _SPECIAL_LLA_LOSSES
        _use_fista = _pen_name in ('adaptive_l1', 'adaptive_lasso')
        _use_quantile_cd = _loss_name == 'quantile' and _pen_name in ('scad', 'mcp')
        _use_irls_cd = _pen_name in ('scad', 'mcp') and _loss_name == 'squared_error'
        _use_lla_fista = _pen_name in ('scad', 'mcp') and _is_glm_loss and (_loss_name != 'squared_error')
        _use_lla_group = _pen_name in ('group_mcp', 'group_scad', 'gmcp', 'gscad') and _is_glm_loss
        if _use_fista:
            params, n_iter = fista_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        elif _use_quantile_cd:
            from statgpu.solvers import proximal_irls_quantile_solver
            import numpy as _np
            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(X_work, y_arr, p, _loss_name)
            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = proximal_irls_quantile_solver(self._loss, self._penalty, X_orig, y_arr, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=sample_weight)
            if self._effective_intercept:
                params_np = _np.concatenate([coef_np, [intercept]])
            else:
                params_np = coef_np
            params = _xp_asarray(params_np, X_arr.dtype, X_arr)
        elif _use_irls_cd:
            from statgpu.solvers import fista_lla_path
            import numpy as _np
            _alpha_path, _max_lla_per_step, _mi_path = self._compute_lla_path(X_work, y_arr, p, _loss_name)
            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(self._loss, self._penalty, X_orig, y_arr, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=sample_weight)
            if self._effective_intercept:
                params_np = np.concatenate([coef_np, [intercept]])
            else:
                params_np = coef_np
            params = params_np
        elif _use_lla_fista:
            from statgpu.solvers import fista_lla_path
            import numpy as _np
            xp = get_backend(backend_name).xp
            X_feat = X_work[:, :p] if self._effective_intercept else X_work
            _n = X_feat.shape[0]
            if _loss_name == 'cox_ph':
                X_feat, y_lla = self._loss.preprocess(X_feat, y_arr)
                if backend_name == 'torch':
                    import torch
                    _zero_coef = torch.zeros(p, dtype=X_feat.dtype, device=X_feat.device)
                else:
                    _zero_coef = xp.zeros(p, dtype=X_feat.dtype)
                _score_at_zero = self._loss.gradient(X_feat, y_lla, _zero_coef, sample_weight=sample_weight)
                _lam_max = float(xp.max(xp.abs(_score_at_zero)))
            else:
                _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
                if backend_name == 'torch':
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
                _n_cont = _N_CONT_STEPS_NONSMOOTH if _loss_name == 'quantile' else _N_CONT_STEPS
                _alpha_path = _np.geomspace(max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont)
            _max_lla_per_step = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // max(_n_cont, 1))
            _saved_mi = self._max_iter
            if _cv_return_path:
                _mi_path = [max(200, _saved_mi // 2)] * max(_n_cont - 1, 0) + [_saved_mi]
            else:
                _mi_path = [_saved_mi if i == _n_cont - 1 else max(100, _saved_mi // 10) for i in range(_n_cont)]
            X_orig = X_feat if _loss_name == 'cox_ph' else X_work[:, :p] if self._effective_intercept else X_work
            y_lla = y_lla if _loss_name == 'cox_ph' else y_arr
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
                        _warm_intercept = float(getattr(self, '_init_intercept', 0.0) or 0.0)
            _y_ndim = getattr(y_arr, 'ndim', None)
            if _y_ndim is None:
                _y_ndim = np.asarray(y_arr).ndim
            _y_ndim = int(_y_ndim)
            if _warm_coef is None and getattr(self._loss, 'has_hessian', False) and (_y_ndim == 1):
                _X_np = np.asarray(_to_numpy(X_orig), dtype=np.float64)
                _y_np = np.asarray(_to_numpy(y_arr), dtype=np.float64)
                _warm_coef = np.linalg.lstsq(_X_np, _y_np, rcond=None)[0]
            _lla_result = fista_lla_path(self._loss, self._penalty, X_orig, y_lla, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=sample_weight, init_coef=_warm_coef, init_intercept=_warm_intercept, return_path=_cv_return_path)
            if _cv_return_path:
                coef_np, intercept, n_iter, _path_results = _lla_result
                self._cv_path_results = _path_results
            else:
                coef_np, intercept, n_iter = _lla_result
            if self._effective_intercept:
                params = xp.concatenate([xp.asarray(coef_np), xp.asarray([intercept])])
            else:
                params = xp.asarray(coef_np)
        elif _use_lla_group:
            from statgpu.solvers import fista_lla_path
            from statgpu.penalties._group_lasso import AdaptiveGroupLassoPenalty
            import numpy as _np
            xp = get_backend(backend_name).xp
            X_feat = X_work[:, :p] if self._effective_intercept else X_work
            _n = X_feat.shape[0]
            _col_norms = xp.sqrt(xp.sum(X_feat ** 2, axis=0))
            if backend_name == 'torch':
                import torch
                _col_norms = torch.clamp(_col_norms, min=1e-20)
            else:
                _col_norms = xp.maximum(_col_norms, 1e-20)
            X_s = X_feat * (float(_n) ** 0.5 / _col_norms)
            y_c = y_arr - xp.mean(y_arr)
            _lam_max = float(xp.max(xp.abs(X_s.T @ y_c / _n)))
            _target_alpha = float(getattr(self._penalty, 'alpha', self.alpha))
            _n_cont = _N_CONT_STEPS_NONSMOOTH if _loss_name == 'quantile' else _N_CONT_STEPS
            _alpha_path = _np.geomspace(max(_lam_max, _target_alpha * 1.1), _target_alpha, _n_cont)
            _max_lla_per_step = max(_MAX_LLA_PER_STEP_DEFAULT, getattr(self, '_max_lla_iters', 50) // _n_cont)
            _saved_mi = self._max_iter
            _mi_path = [_saved_mi if i == _n_cont - 1 else max(100, _saved_mi // 10) for i in range(_n_cont)]
            _orig_pen = self._penalty
            _groups = getattr(_orig_pen, '_group_indices', None)
            _pen_alpha = float(_orig_pen.alpha)
            _adaptive_pen = AdaptiveGroupLassoPenalty(groups=_groups, alpha=_pen_alpha)

            def _group_lla_factory(weights_np):
                _gw = np.array([float(np.sqrt(np.sum(weights_np[idx] ** 2))) if len(idx) > 0 else 0.0 for idx in _groups])
                _adaptive_pen.set_weights(_gw)
                return _adaptive_pen
            X_orig = X_work[:, :p] if self._effective_intercept else X_work
            coef_np, intercept, n_iter = fista_lla_path(self._loss, self._penalty, X_orig, y_arr, alpha_path=_alpha_path, max_lla_per_step=_max_lla_per_step, lla_tol=getattr(self, '_lla_tol', 1e-06), max_iter=_mi_path, tol=self._tol, fit_intercept=self._effective_intercept, sample_weight=sample_weight, lla_penalty_factory=_group_lla_factory)
            if self._effective_intercept:
                params = xp.concatenate([xp.asarray(coef_np), xp.asarray([intercept])])
            else:
                params = xp.asarray(coef_np)
        elif _pen_name == 'group_lasso':
            _use_bcd = _loss_name != 'cox_ph'
            if not _use_bcd:
                from statgpu.solvers import fista_solver
                params, n_iter = fista_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
            elif backend_name != 'numpy':
                coef_gpu, intercept, n_iter = self._block_cd_group_lasso_gpu(pen, X_work, y_arr, init, backend_name)
                if self._effective_intercept:
                    from statgpu.backends._utils import _get_xp as _get_xp_fn
                    from statgpu.backends._array_ops import _xp_asarray as _xp_asarray_fn
                    _xp = _get_xp_fn(backend_name)
                    _int_arr = _xp_asarray_fn([intercept], coef_gpu.dtype, coef_gpu)
                    params = _xp.concatenate([coef_gpu, _int_arr])
                else:
                    params = coef_gpu
            else:
                coef_np, intercept, n_iter = self._block_cd_group_lasso(pen, X_work, y_arr, init)
                if self._effective_intercept:
                    params = np.concatenate([coef_np, [intercept]])
                else:
                    params = coef_np
        elif solver_name == 'fista':
            _loss_name = getattr(self._loss, 'name', '')
            _has_irls = hasattr(self._loss, 'irls')
            _is_smooth_pen = _pen_name in ('l2', 'none', 'null', '')
            if _loss_name == 'quantile' and _has_irls and _is_smooth_pen:
                _inner_pen = getattr(self._penalty, '_pen', self._penalty)
                _irls_tol = min(self._tol, 1e-08)
                params_irls, n_iter = self._loss.irls(X_work, y_arr, penalty=_inner_pen, max_iter=self._max_iter, tol=_irls_tol, init_coef=None, sample_weight=sample_weight, fit_intercept=self._effective_intercept)
                params = _xp_asarray(params_irls, X_arr.dtype, X_arr)
            else:
                params, n_iter = fista_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        elif solver_name == 'fista_bb':
            params, n_iter = fista_bb_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        elif solver_name == 'admm':
            params, n_iter = admm_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, rho=1.0, adaptive_rho=True, init_coef=init, sample_weight=sample_weight)
        elif solver_name == 'newton':
            params, n_iter = newton_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        elif solver_name == 'lbfgs':
            params, n_iter = lbfgs_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        elif solver_name == 'irls':
            _inner_pen = getattr(self._penalty, '_pen', self._penalty)
            _loss_name = getattr(self._loss, 'name', '')
            _irls_tol = min(self._tol, 1e-08) if _loss_name == 'quantile' else self._tol
            params_irls, n_iter = self._loss.irls(X_work, y_arr, penalty=_inner_pen, max_iter=self._max_iter, tol=_irls_tol, init_coef=None, sample_weight=sample_weight, fit_intercept=self._effective_intercept)
            params = _xp_asarray(params_irls, X_arr.dtype, X_arr)
        elif solver_name == 'auto':
            params, n_iter = fista_solver(self._loss, pen, X_work, y_arr, max_iter=self._max_iter, tol=self._tol, init_coef=init, sample_weight=sample_weight)
        else:
            raise ValueError(f'Unsupported solver: {solver_name}')
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
        self._df_resid = self._nobs - (X_arr.shape[1] + (1 if self._effective_intercept else 0))
        if backend_name == 'cupy':
            self._cleanup_cuda_memory()
        elif backend_name == 'torch':
            self._cleanup_torch_memory()

    def _fit_irls_backend(self, X, y, sample_weight=None, backend_name='numpy'):
        """Fit smooth L2 GLM via IRLS on the selected backend."""
        from statgpu.glm_core._irls import IRLSSolver
        if str(getattr(self._penalty, 'name', self.penalty)).lower() != 'l2':
            raise ValueError("solver='irls' only supports L2 penalties.")
        from statgpu.backends._utils import _get_xp, xp_asarray
        _xp = _get_xp(backend_name)
        X_arr = xp_asarray(X, dtype=_xp.float64, xp=_xp, ref_arr=X if not isinstance(X, np.ndarray) else np.zeros(1))
        y_arr = xp_asarray(y, dtype=_xp.float64, xp=_xp, ref_arr=X_arr)
        n_samples = X_arr.shape[0]
        if self._effective_intercept:
            X_work = self._column_stack([self._ones(X_arr.shape[0], backend_name, X_arr), X_arr], backend_name)
        else:
            X_work = X_arr
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
            if backend_name == 'cupy':
                import cupy as cp
                init_coef = cp.asarray(init_coef_np, dtype=cp.float64)
            elif backend_name == 'torch':
                import torch
                init_coef = torch.as_tensor(init_coef_np, dtype=torch.float64, device=X_work.device)
            else:
                init_coef = init_coef_np
        _log_link_losses = ('gamma', 'poisson', 'inverse_gaussian', 'negative_binomial', 'tweedie')
        if init_coef is None and self._effective_intercept and (_loss_name in _log_link_losses or _loss_name == 'logistic'):
            _y_mean = float(np.mean(_to_numpy(y_arr)))
            if _loss_name == 'logistic':
                _y_mean = float(np.clip(_y_mean, 0.001, 1.0 - 0.001))
                _int_init = np.log(_y_mean / (1.0 - _y_mean))
            else:
                _int_init = np.log(max(_y_mean, 0.001))
            n_feat = X_work.shape[1]
            init_coef_np = np.zeros(n_feat)
            init_coef_np[0] = _int_init
            if backend_name == 'cupy':
                import cupy as cp
                init_coef = cp.asarray(init_coef_np)
            elif backend_name == 'torch':
                import torch
                init_coef = torch.from_numpy(init_coef_np).to(X_work.device)
            else:
                init_coef = init_coef_np
        solver = IRLSSolver(self._family_for_loss(), max_iter=self._max_iter, tol=self._tol)
        ridge_normalization = float(n_samples) if sample_weight is None else _validate_sample_weight_backend(sample_weight, n_samples, backend_name)
        params, n_iter = solver.fit(X_work, y_arr, sample_weight=sample_weight, ridge_alpha=float(ridge_normalization * self.alpha), ridge_penalize_intercept=False if self._effective_intercept else True, backend=backend_name, init_coef=init_coef)
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
        self._df_resid = self._nobs - (X_arr.shape[1] + (1 if self._effective_intercept else 0))
        if backend_name == 'cupy':
            self._cleanup_cuda_memory()
        elif backend_name == 'torch':
            self._cleanup_torch_memory()

    def _cleanup_cuda_memory(self):
        """Free CuPy memory pool."""
        if not self._gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Free Torch memory pool."""
        if not self._gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
