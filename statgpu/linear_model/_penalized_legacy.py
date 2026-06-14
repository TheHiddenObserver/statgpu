"""Legacy solver/CD methods from _penalized.py.

These methods were replaced by newer implementations (FISTA, _fit_loss_backend)
but are retained for reference and backward compatibility.

DO NOT import or use in production code.
"""

from __future__ import annotations

import numpy as np

# Methods extracted from PenalizedGeneralizedLinearModel:
def _irls_cd_gpu(self, pen, X_work, y_arr, init, backend_name, _lla_continuation=False):
    """GPU-native IRLS with coordinate descent for GLM + non-smooth penalties.

    Same algorithm as _irls_cd but keeps all arrays on GPU to avoid
    CPU-GPU transfer overhead.  Supports cupy and torch backends.
    """
    from statgpu.backends._array_ops import _xp_copy, _xp_zeros, _xp_asarray
    from statgpu.backends._utils import _get_xp
    xp = _get_xp(backend_name)

    n, pp = X_work.shape
    p = pp - 1 if self._effective_intercept else pp

    # Access weights from the original penalty
    _inner = getattr(self, '_penalty', pen)
    _w_np = np.asarray(getattr(_inner, '_weights', np.ones(p)), dtype=float)
    _w = _xp_asarray(_w_np, X_work.dtype, X_work)
    alpha = float(getattr(_inner, 'alpha', self.alpha))
    pen_name = getattr(pen, 'name', '') or getattr(_inner, 'name', '')

    # SCAD/MCP parameters (guard against division-by-zero)
    a_scad = float(getattr(_inner, 'a', 3.7)) if pen_name == "scad" else 0.0
    if pen_name == "scad":
        a_scad = max(a_scad, 1.0 + 1e-6)
        if abs(a_scad - 2.0) < 1e-6:
            a_scad = 2.0 + 1e-6
    gamma_mcp = float(getattr(_inner, 'gamma', 3.0)) if pen_name == "mcp" else 0.0
    if pen_name == "mcp":
        gamma_mcp = max(gamma_mcp, 1.0 + 1e-6)

    # Penalty value helper (uses numpy for portability; coef_slice is numpy)
    def _nonconvex_penalty_value(coef_slice, _pen_name, _alpha, _a_scad, _gamma_mcp):
        _abs_b = np.abs(coef_slice)
        if _pen_name == "scad":
            return float(np.sum(np.where(
                _abs_b <= _alpha, _alpha * _abs_b,
                np.where(_abs_b <= _a_scad * _alpha,
                    (_a_scad * _alpha * _abs_b - 0.5 * (coef_slice**2 + _alpha**2)) / (_a_scad - 1.0),
                    0.5 * (_a_scad + 1.0) * _alpha**2))))
        if _pen_name == "mcp":
            return float(np.sum(np.where(
                _abs_b <= _gamma_mcp * _alpha,
                _alpha * _abs_b - 0.5 * coef_slice**2 / _gamma_mcp,
                0.5 * _gamma_mcp * _alpha**2)))
        return 0.0

    if init is not None:
        if isinstance(init, np.ndarray):
            beta = _xp_asarray(init, X_work.dtype, X_work)
        else:
            beta = _xp_copy(init)
    else:
        beta = _xp_zeros(pp, X_work.dtype, X_work)

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
        d = _xp_zeros((n,), X_work.dtype, X_work) + 1.0  # ones on correct device
        z = y_arr
        XDX_diag = xp.sum(d[:, None] * X_work ** 2, axis=0)

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

        it = -1
        for it in range(_n_outer):
            beta_old = beta.clone() if backend_name == "torch" else beta.copy()

            # Compute objective before CD for step-halving (GLM only)
            _obj_before = None
            if _is_glm:
                try:
                    _obj_before = float(xp.sum(self._loss.per_sample_value(X_work, y_arr, beta_old)))
                    _obj_before += _nonconvex_penalty_value(
                        _to_numpy(beta_old[:p]) if backend_name != "numpy" else beta_old[:p],
                        pen_name, alpha, a_scad, gamma_mcp)
                except Exception:
                    _obj_before = None

            if _is_glm:
                eta = X_work @ beta
                if loss_name == "logistic":
                    mu = 1.0 / (1.0 + _exp(-_clip(eta, -500, 500)))
                    mu = _clip(mu, 1e-15, 1.0 - 1e-15)
                    d = mu * (1.0 - mu)
                    z = eta + (y_arr - mu) / d
                elif loss_name == "poisson":
                    mu = _clip(_exp(_clip(eta, -500, 500)), 1e-15, None)
                    d = mu
                    z = eta + (y_arr - mu) / d
                elif loss_name == "gamma":
                    mu = _clip(_exp(_clip(eta, -500, 500)), 1e-15, None)
                    d = _xp_zeros((n,), X_work.dtype, X_work) + 1.0
                    z = eta + (y_arr - mu) / mu
                elif loss_name == "inverse_gaussian":
                    mu = _clip(_exp(_clip(eta, -500, 500)), 1e-15, None)
                    d = 1.0 / mu
                    z = eta + (y_arr - mu) / mu
                elif loss_name == "negative_binomial":
                    mu = _clip(_exp(_clip(eta, -500, 500)), 1e-15, None)
                    theta_nb = float(getattr(self._loss, 'alpha', 1.0))
                    d = mu / (1.0 + mu / theta_nb)
                    z = eta + (y_arr - mu) / d
                elif loss_name == "tweedie":
                    mu = _clip(_exp(_clip(eta, -500, 500)), 1e-15, None)
                    tweedie_p = float(getattr(self._loss, 'power', 1.5))
                    d = mu ** tweedie_p
                    d = _clip(d, 1e-15, None)
                    z = eta + (y_arr - mu) / (d * mu)
                else:
                    grad = self._loss.gradient(X_work, y_arr, beta)
                    d = _xp_zeros((n,), X_work.dtype, X_work) + 1.0
                    z = eta - grad * n
                XDX_diag = xp.sum(d[:, None] * X_work ** 2, axis=0)

            # Effective sample size for correct normalization with sample weights
            n_eff = float(xp.sum(d))

            r = z - X_work @ beta

            # Precompute active mask and vectorized penalty weights
            _active = XDX_diag >= 1e-20
            _v_all = XDX_diag / n_eff
            _v_safe = xp.where(_active, _v_all, 1.0)  # avoid division by zero
            if pen_name in ("adaptive_l1", "adaptive_lasso"):
                _l1_all = alpha * _w  # shape (p,)

            for _cd in range(_n_cd_sweeps):
                # --- Vectorized block coordinate descent ---
                # 1. Batch gradient: rho_all = X' (d * r) + XDX_diag * beta
                rho_all = X_work.T @ (d * r) + XDX_diag * beta
                w_all = rho_all / (n_eff * _v_safe)  # un-penalized solution

                # 2. Save old beta for residual update
                old_beta = beta

                # 3. Vectorized thresholding (penalty-specific)
                if self._effective_intercept:
                    new_beta = xp.zeros_like(beta)
                    w_feat = w_all[:p]
                else:
                    w_feat = w_all
                    new_beta = xp.zeros_like(beta)

                if pen_name in ("adaptive_l1", "adaptive_lasso"):
                    aw = xp.abs(w_feat)
                    new_beta_feat = xp.sign(w_feat) * xp.maximum(aw - _l1_all, 0.0)
                elif pen_name == "scad":
                    aw = xp.abs(w_feat)
                    l1 = alpha
                    new_beta_feat = xp.where(
                        aw > a_scad * l1, w_feat,
                        xp.where(
                            aw > l1,
                            xp.sign(w_feat) * ((a_scad - 1.0) * aw - a_scad * l1) / (a_scad - 2.0),
                            0.0,
                        ),
                    )
                elif pen_name == "mcp":
                    aw = xp.abs(w_feat)
                    l1 = alpha
                    new_beta_feat = xp.where(
                        aw > gamma_mcp * l1, w_feat,
                        xp.where(
                            aw > l1,
                            xp.sign(w_feat) * (aw - l1) / (1.0 - 1.0 / gamma_mcp),
                            0.0,
                        ),
                    )
                else:
                    # lasso / elasticnet (pure L1)
                    aw = xp.abs(w_feat)
                    new_beta_feat = xp.sign(w_feat) * xp.maximum(aw - alpha, 0.0)

                # Zero out degenerate columns
                if self._effective_intercept:
                    new_beta[:p] = new_beta_feat * _active[:p]
                    new_beta[p:] = w_all[p:]  # intercept: no penalty
                else:
                    new_beta = new_beta_feat * _active

                # 4. Residual update (single matvec instead of p dot products)
                delta = new_beta - old_beta
                r = r - X_work @ delta
                beta = new_beta

                # 5. Convergence check (single GPU reduction + one sync)
                _max_cd_change = float(xp.max(xp.abs(delta)))

                if not _is_glm and _max_cd_change < self.tol:
                    break

            # Step-halving for GLM: ensure penalized objective decreases.
            # Mirrors the CPU path (_irls_cd) to prevent IRLS overshooting.
            if _is_glm:
                _obj_after = float(xp.sum(self._loss.per_sample_value(X_work, y_arr, beta)))
                _obj_after += _nonconvex_penalty_value(
                    _to_numpy(beta[:p]) if backend_name != "numpy" else beta[:p],
                    pen_name, alpha, a_scad, gamma_mcp)
                if _obj_before is not None and _obj_after > _obj_before + 1e-10:
                    beta_new_gpu = beta.clone() if backend_name == "torch" else beta.copy()
                    for _sh in range(1, 11):
                        _frac = 0.5 ** _sh
                        beta_sh = beta_old + _frac * (beta_new_gpu - beta_old)
                        _obj_sh = float(xp.sum(self._loss.per_sample_value(X_work, y_arr, beta_sh)))
                        _obj_sh += _nonconvex_penalty_value(
                            _to_numpy(beta_sh[:p]) if backend_name != "numpy" else beta_sh[:p],
                            pen_name, alpha, a_scad, gamma_mcp)
                        if _obj_sh <= _obj_before + 1e-10:
                            beta = beta_sh
                            break
                    else:
                        # All step-halving attempts failed — revert to previous iterate
                        beta = beta_old

            # IRLS-level convergence check
            _delta = float(xp.max(xp.abs(beta[:p] - beta_old[:p])))
            if not _is_glm and _delta < self.tol:
                break
            if _is_glm and len(_cont_path) > 1 and not _is_last:
                if _delta < self.tol * 10:
                    break

    n_iter = it + 1 if _n_outer > 0 else 0
    return beta, n_iter


def _block_cd_group_lasso_gpu_batched(self, pen, X_work, y_arr, init, backend_name):
    """Batched GPU block coordinate descent for group_lasso penalty.

    Processes all groups in parallel within each iteration to minimize
    kernel launch overhead. Groups of the same size are batched together
    for efficient linear solves.
    """
    from statgpu.backends._array_ops import _xp_copy, _xp_zeros, _xp_asarray, _scalar_tensor
    from statgpu.backends._utils import _get_xp
    xp = _get_xp(backend_name)

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
            coef = _xp_asarray(init, X_work.dtype, X_work)
        else:
            coef = _xp_copy(init)
    else:
        coef = _xp_zeros(pp, X_work.dtype, X_work)

    iteration = -1  # ensure defined when max_iter=0
    for iteration in range(self.max_iter):
        coef_old = _xp_copy(coef)

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
            # Stack all indices for batched indexing
            idx_arr = _xp_asarray(all_indices, xp.int32 if backend_name == "cupy" else None, X_work)
            # Compute XtX[g_idx, :] @ coef for all groups at once
            XtX_coef = XtX[idx_arr, :] @ coef  # shape: (n_batch * sz,)
            # Compute Xty for all groups
            Xty_all = Xty[idx_arr]
            # Compute block diagonal contributions
            block_contrib = _xp_zeros(Xty_all.shape, Xty_all.dtype, Xty_all)
            for i, (g, g_idx) in enumerate(size_groups):
                block_contrib[i*sz:(i+1)*sz] = _XtX_blocks[g] @ coef[g_idx]
            # rho_g = Xty - XtX_coef + block_contrib
            rho_all = Xty_all - XtX_coef + block_contrib

            # Solve all group systems in one batched call
            rho_mat = rho_all.reshape(n_batch, sz, 1)
            XtX_batch = xp.stack([_XtX_blocks[g] for g in batch_g_indices])
            try:
                w_all = xp.linalg.solve(XtX_batch, rho_mat)  # (n_batch, sz, 1)
                w_all = w_all.reshape(n_batch, sz)
            except Exception:
                w_all = _xp_zeros((n_batch, sz), X_work.dtype, X_work)

            # Apply soft-thresholding to all groups at once
            _norm_dim = 1  # axis for numpy/cupy, dim for torch (both use 1)
            norms = xp.linalg.norm(w_all, axis=_norm_dim)  # (n_batch,)
            thresh = _xp_asarray(
                [alpha * _sqrt_pg[g] for g in batch_g_indices],
                X_work.dtype, X_work,
            )
            scale = xp.where(norms > thresh, 1.0 - thresh / (norms + 1e-12), 0.0)

            # Write back coefficients
            for i, (g, g_idx) in enumerate(size_groups):
                coef[g_idx] = w_all[i] * scale[i]

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


def _cd_elasticnet(self, pen, X_work, y_arr, init):
    """Coordinate descent for elasticnet penalty (squared_error loss).

    Matches R glmnet's CD algorithm for elasticnet:
    beta_j = S(rho_j, alpha*l1_ratio*n) / (X_j'X_j + alpha*(1-l1_ratio)*n)
    """
    import numpy as np

    n, pp = X_work.shape
    p = pp - 1 if self._effective_intercept else pp
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

    iteration = -1  # ensure defined when max_iter=0
    for iteration in range(self.max_iter):
        coef_old = coef.copy()

        for j in range(p):
            rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]
            if X_sq_norms[j] > 1e-10:
                st = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0)
                coef[j] = st / (X_sq_norms[j] + alpha * (1 - l1_ratio) * n)
            else:
                coef[j] = 0.0

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


def _cd_l1(self, pen, X_work, y_arr, init):
    """Coordinate descent for L1 (lasso) penalty (squared_error loss).

    Matches R glmnet's CD algorithm:
    beta_j = S(rho_j, alpha*n) / X_j'X_j
    """
    import numpy as np

    n, pp = X_work.shape
    p = pp - 1 if self._effective_intercept else pp
    alpha = self.alpha

    XtX = X_work.T @ X_work
    Xty = X_work.T @ y_arr.flatten()
    X_sq_norms = np.diag(XtX)

    if init is not None:
        coef = np.array(init, dtype=np.float64)
    else:
        coef = np.zeros(pp, dtype=np.float64)

    thresh = alpha * n

    iteration = -1  # ensure defined when max_iter=0
    for iteration in range(self.max_iter):
        coef_old = coef.copy()

        for j in range(p):
            rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]
            if X_sq_norms[j] > 1e-10:
                coef[j] = np.sign(rho_j) * np.maximum(np.abs(rho_j) - thresh, 0) / X_sq_norms[j]
            else:
                coef[j] = 0.0

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


def _fit_cpu_loss(self, X, y, sample_weight=None, solver="fista"):
    """Fit using loss-aware solver (FISTA with arbitrary loss).

    For GLM losses (logistic, poisson) with intercept, augments X with
    a column of ones and uses a selective penalty (no penalty on intercept)
    to converge to the correct joint optimum.
    """
    from statgpu.glm_core._solver import fista_solver

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if self.loss in ("logistic", "poisson") and self._effective_intercept:
        # Augment X with intercept column
        X_aug = np.column_stack([X_arr, np.ones(X_arr.shape[0])])
        p = X_arr.shape[1]
        pen = self._penalty

        # Reuse thread-local SelectivePenalty singleton
        singleton = _get_selective_penalty_singleton()
        singleton.configure(self._penalty, p, "numpy")

        full_coef, n_iter = fista_solver(
            self._loss, singleton, X_aug, y_arr,
            max_iter=self.max_iter, tol=self.tol,
            init_coef=None, sample_weight=sample_weight,
        )

        self.coef_ = full_coef[:p]
        self.intercept_ = float(full_coef[p])
        self.n_iter_ = n_iter
    elif self._effective_intercept:
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

    self._df_resid = self._nobs - (X.shape[1] + (1 if self._effective_intercept else 0))


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
    if self._effective_intercept:
        X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

    # L2 penalty: for objective min loss/n + alpha*0.5*||w||^2,
    # IRLS uses unnormalized X'WX, so ridge = n * alpha.
    # Don't penalize the intercept column (matches sklearn/FISTA behavior).
    ridge_alpha = float(n_samples * self.alpha)
    ridge_penalize_intercept = False if self._effective_intercept else True

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

    if self._effective_intercept:
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]
        self._params = np.concatenate([[self.intercept_], np.asarray(self.coef_)])
    else:
        self.intercept_ = 0.0
        self.coef_ = params.copy()
        self._params = np.asarray(self.coef_).copy()

    self._df_resid = self._nobs - (X.shape[1] + (1 if self._effective_intercept else 0))



# --- _irls_cd (dead code, moved from _penalized.py) ---
def _irls_cd(self, pen, X_work, y_arr, init, _lla_continuation=False):
    """IRLS with coordinate descent for GLM + non-smooth penalties.

    Matches R glmnet/ncvreg algorithm: outer IRLS loop computes working
    response and weights, inner CD loop solves the weighted penalized
    least squares subproblem with per-coordinate thresholds.
    Supports: adaptive_l1, scad, mcp.
    """
    import numpy as np

    n, pp = X_work.shape
    p = pp - 1 if self._effective_intercept else pp

    # Access weights from the original penalty (not the SelectivePenalty wrapper)
    _inner = getattr(self, '_penalty', pen)
    _w = np.asarray(getattr(_inner, '_weights', np.ones(p)), dtype=float)
    # Read alpha from the penalty object.  The threshold per coordinate
    # is alpha * _w[j] where _w has mean=1 (matching R glmnet convention).
    alpha = float(getattr(_inner, 'alpha', self.alpha))
    _nf = float(getattr(_inner, '_norm_factor', 1.0))
    pen_name = getattr(pen, 'name', '') or getattr(_inner, 'name', '')

    # SCAD/MCP parameters (guard against division-by-zero)
    a_scad = float(getattr(_inner, 'a', 3.7)) if pen_name == "scad" else 0.0
    if pen_name == "scad":
        a_scad = max(a_scad, 1.0 + 1e-6)
        if abs(a_scad - 2.0) < 1e-6:
            a_scad = 2.0 + 1e-6
    gamma_mcp = float(getattr(_inner, 'gamma', 3.0)) if pen_name == "mcp" else 0.0
    if pen_name == "mcp":
        gamma_mcp = max(gamma_mcp, 1.0 + 1e-6)

    if init is not None:
        beta = np.asarray(init, dtype=float).copy()
    else:
        beta = np.zeros(pp)

    loss_name = self._loss.name
    _is_glm = (loss_name != "squared_error")

    def _nonconvex_penalty_value(coef_slice, _pen_name, _alpha, _a_scad, _gamma_mcp):
        """Compute SCAD/MCP penalty value for a coefficient vector."""
        _abs_b = np.abs(coef_slice)
        if _pen_name == "scad":
            return float(np.sum(np.where(
                _abs_b <= _alpha, _alpha * _abs_b,
                np.where(_abs_b <= _a_scad * _alpha,
                    (_a_scad * _alpha * _abs_b - 0.5 * (coef_slice**2 + _alpha**2)) / (_a_scad - 1.0),
                    0.5 * (_a_scad + 1.0) * _alpha**2))))
        if _pen_name == "mcp":
            return float(np.sum(np.where(
                _abs_b <= _gamma_mcp * _alpha,
                _alpha * _abs_b - 0.5 * coef_slice**2 / _gamma_mcp,
                0.5 * _gamma_mcp * _alpha**2)))
        return 0.0

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

        it = -1
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
                    # V(mu) = mu^3, log link g'(mu) = 1/mu
                    # IRLS weight: w = 1/(V(mu) * [g'(mu)]^2) = 1/(mu^3 * 1/mu^2) = 1/mu
                    # Working response: z = eta + (y - mu) * g'(mu) = eta + (y - mu)/mu
                    mu = np.exp(np.clip(eta, -500, 500))
                    mu = np.maximum(mu, 1e-15)
                    d = 1.0 / mu
                    z = eta + (y_arr - mu) / mu
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

            # Effective sample size: use sum(d) for correct normalization
            # when sample weights are present (d already includes sw scaling).
            n_eff = float(np.sum(d))

            r = z - X_work @ beta

            # Compute penalized objective before CD (for step-halving)
            if _is_glm:
                # Use full design matrix (including intercept) for correct objective
                _obj_before = float(self._loss.value(X_work, y_arr, beta))
                _obj_before += _nonconvex_penalty_value(beta[:p], pen_name, alpha, a_scad, gamma_mcp)

            for _cd in range(_n_cd_sweeps):
                _max_cd_change = 0.0
                for j in range(pp):
                    if XDX_diag[j] < 1e-20:
                        beta[j] = 0.0
                        continue

                    rho_j = np.dot(d * X_work[:, j], r) + XDX_diag[j] * beta[j]
                    old_bj = beta[j]

                    u_j = rho_j / n_eff
                    v_j = XDX_diag[j] / n_eff

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
                _obj_after = float(self._loss.value(X_work, y_arr, beta))
                _obj_after += _nonconvex_penalty_value(beta[:p], pen_name, alpha, a_scad, gamma_mcp)
                if _obj_after > _obj_before + 1e-10:
                    # Step-halving: interpolate between old and new beta
                    # beta_sh = beta_old + 0.5^k * (beta_new - beta_old)
                    beta_new = beta.copy()
                    for _sh in range(1, 11):
                        _frac = 0.5 ** _sh
                        beta[:] = beta_old + _frac * (beta_new - beta_old)
                        _obj_after = float(self._loss.value(X_work, y_arr, beta))
                        _obj_after += _nonconvex_penalty_value(beta[:p], pen_name, alpha, a_scad, gamma_mcp)
                        if _obj_after <= _obj_before + 1e-10:
                            break

            # IRLS-level convergence check.
            _delta = np.max(np.abs(beta[:p] - beta_old[:p]))
            if not _is_glm and _delta < self.tol:
                break
            # For GLM with continuation: early exit on convergence
            # for non-final steps (avoids wasting iterations).
            if _is_glm and len(_cont_path) > 1 and not _is_last:
                if _delta < self.tol * 10:
                    break

    n_iter = (it + 1) if _n_outer > 0 else 0
    return beta, n_iter


# --- _fit_lla (dead code, moved from _penalized.py) ---
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
        coef_lla = self._fit_initial(X, y, backend_name=backend_name)

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

    _n_cont = 20 if _is_scad_mcp else 10
    # Start from lambda_max to match R ncvreg's pathwise approach.
    # lambda_max is the smallest penalty where all coefficients are zero.
    _alpha_start = float(_lam_max)
    _alpha_end = float(self.alpha)
    if _alpha_start <= 0.0 or _alpha_end <= 0.0:
        _lo = max(min(_alpha_start, _alpha_end), 1e-12)
        _hi = max(_alpha_start, _alpha_end, 1e-12)
        if _hi <= _lo:
            _alpha_path = _np.full(_n_cont, _hi, dtype=float)
        else:
            _alpha_path = _np.linspace(_hi, _lo, _n_cont, dtype=float)
            _alpha_path[-1] = max(_alpha_end, 1e-12)
    else:
        _alpha_path = _np.geomspace(_alpha_start, _alpha_end, _n_cont)
    _max_lla_per_step = max(6, self._max_lla_iters // _n_cont)

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
                fit_intercept=self._effective_intercept,
                sample_weight=sample_weight,
            )
            coef_lla = coef_np
            self.coef_ = coef_np
            self.intercept_ = intercept
            self.n_iter_ = n_iter
            self._lla_n_iters_ = _n_cont * _max_lla_per_step
        else:
            # Cache GPU arrays once outside the continuation loop
            X_cached = self._to_array(X, backend=backend_name)
            y_cached = self._to_array(y, backend=backend_name)

            for _cont_step, _cont_alpha in enumerate(_alpha_path):
                # Create a copy with the continuation alpha to avoid
                # mutating the shared penalty object (thread-safety).
                _pen_step = copy.copy(self._penalty)
                _pen_step.alpha = float(_cont_alpha)

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
                    for _lla_local in range(_max_lla_per_step):
                        # Compute LLA weights from current estimate
                        lla_w = _pen_step.lla_weights(coef_lla)

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

                        # Swap penalty (protected by try/finally)
                        # Use copy to avoid thread-safety issues with shared instances
                        import copy
                        orig_penalty = copy.copy(self._penalty)
                        self._penalty = inner_pen
                        try:
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
                        finally:
                            # Restore original penalty even if inner fit raises
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
            if self._effective_intercept:
                X_np = np.asarray(X, dtype=float)
                y_np = np.asarray(y, dtype=float)
                if sample_weight is not None:
                    sw_np = np.asarray(sample_weight, dtype=float).ravel()
                    sw_sum = max(float(np.sum(sw_np)), 1e-15)
                    X_wmean = np.sum(X_np * sw_np[:, None], axis=0) / sw_sum
                    y_wmean = float(np.sum(y_np * sw_np)) / sw_sum
                    self.intercept_ = float(y_wmean - X_wmean @ self.coef_)
                else:
                    self.intercept_ = float(np.mean(y_np) - np.mean(X_np, axis=0) @ self.coef_)
            else:
                self.intercept_ = 0.0
            self._params = np.concatenate([[self.intercept_], self.coef_])
            self._df_resid = X.shape[0] - (X.shape[1] + (1 if self._effective_intercept else 0))
    finally:
        self.cpu_solver = saved_cpu_solver
        self._selected_solver = saved_selected_solver
        self.max_iter = saved_max_iter

