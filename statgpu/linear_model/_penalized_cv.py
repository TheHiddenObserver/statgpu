"""
Unified cross-validated penalized GLM estimator.

Supports all GLM loss functions (squared_error, logistic, poisson, gamma,
inverse_gaussian, negative_binomial, tweedie) with all penalty types
(l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso).

Optimizations:
- Warm-start across alpha values (descending order)
- Batch eigendecomposition for squared_error + l2
- Precomputed XtX/Lipschitz per fold
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_numpy
from statgpu.linear_model._cv_base import CVEstimatorBase, kfold_indices


class PenalizedGLM_CV(CVEstimatorBase):
    """Cross-validated penalized GLM supporting all loss + penalty combinations."""

    def __init__(
        self,
        loss: str = 'squared_error',
        penalty: str = 'l2',
        alpha_grid=None,
        n_alphas: int = 100,
        l1_ratio: float = 0.5,
        cv: int = 5,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'auto',
    ):
        super().__init__(cv=cv, random_state=random_state, device=device)
        self.loss = loss
        self.penalty = penalty
        self._alpha_grid_input = alpha_grid
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

        self.alpha_ = None
        self.alpha_grid_ = None

    def _generate_alpha_grid(self, X, y):
        """Auto-generate alpha grid based on loss and penalty type."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        X_np = _to_numpy(X).astype(np.float64)
        y_np = _to_numpy(y).astype(np.float64).ravel()
        n = X_np.shape[0]

        if self.loss == 'squared_error':
            alpha_max = float(np.max(np.abs(X_np.T @ y_np))) / n
        elif self.loss == 'logistic':
            y_centered = y_np - 0.5
            alpha_max = float(np.max(np.abs(X_np.T @ y_centered))) / n
        else:
            try:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty='l2', alpha=0.0,
                    device='cpu', compute_inference=False, max_iter=5,
                )
                model.fit(X_np, y_np)
                grad = X_np.T @ (y_np - _to_numpy(model.predict(X_np))) / n
                alpha_max = float(np.max(np.abs(grad)))
            except Exception:
                alpha_max = 1.0

        if alpha_max <= 0:
            alpha_max = 1.0

        if self.penalty in ('l1', 'elasticnet', 'scad', 'mcp', 'adaptive_l1'):
            grid = np.geomspace(alpha_max, alpha_max * 1e-4, self.n_alphas)
        else:
            grid = np.logspace(
                np.log10(max(alpha_max * 1e-4, 1e-12)),
                np.log10(alpha_max),
                self.n_alphas,
            )

        return grid

    def _solve_ridge_fold_batch(self, X_train, y_train, X_val, y_val, alphas):
        """Batch solve Ridge CV for all alphas using eigendecomposition.

        This is O(eigendecomposition + n_alphas * O(diagonal_scaling))
        instead of O(n_alphas * O(solve)).
        """
        X_train_np = _to_numpy(X_train).astype(np.float64)
        y_train_np = _to_numpy(y_train).astype(np.float64).ravel()
        X_val_np = _to_numpy(X_val).astype(np.float64)
        y_val_np = _to_numpy(y_val).astype(np.float64).ravel()

        n, p = X_train_np.shape
        n_alphas = len(alphas)

        # Center data
        X_mean = np.mean(X_train_np, axis=0)
        y_mean = np.mean(y_train_np)
        Xc = X_train_np - X_mean
        yc = y_train_np - y_mean

        # Eigendecomposition of X'X
        XtX = Xc.T @ Xc
        eigvals, Q = np.linalg.eigh(XtX)
        eigvals = np.maximum(eigvals, 1e-15)

        # Q'X'y
        QtXty = Q.T @ (Xc.T @ yc)

        # Solve for all alphas at once
        # coef = Q @ diag(1/(eigvals + n*alpha)) @ Q' @ X'y
        n_alpha = n * alphas  # per-sample convention
        inv_diag = 1.0 / (eigvals[:, None] + n_alpha[None, :])  # (p, n_alphas)
        coefs = Q @ (inv_diag * QtXty[:, None])  # (p, n_alphas)

        # Compute intercepts
        intercepts = y_mean - X_mean @ coefs  # (n_alphas,)

        # Compute validation MSE for all alphas
        X_val_centered = X_val_np - X_mean
        y_pred = X_val_centered @ coefs + intercepts[None, :]  # (n_val, n_alphas)
        mse = np.mean((y_val_np[:, None] - y_pred) ** 2, axis=0)  # (n_alphas,)

        return mse, coefs, intercepts

    def _evaluate_single(self, X_train, y_train, X_val, y_val, alpha,
                         sample_weight_train=None, prev_coef=None):
        """Train on training fold, return (validation_loss, model)."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel, _resolve_loss_name

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=alpha,
            l1_ratio=self.l1_ratio,
            device=self.device,
            compute_inference=False,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        if prev_coef is not None:
            model._init_coef = np.asarray(prev_coef, dtype=np.float64)
        model.fit(X_train, y_train, sample_weight=sample_weight_train)

        loss_fn = _resolve_loss_name(self.loss)
        X_val_np = _to_numpy(X_val).astype(np.float64)
        y_val_np = _to_numpy(y_val).astype(np.float64).ravel()
        n_val = X_val_np.shape[0]

        if model.fit_intercept:
            X_design = np.column_stack([np.ones(n_val), X_val_np])
            coef_with_intercept = np.concatenate([
                [float(model.intercept_)],
                _to_numpy(model.coef_).ravel(),
            ])
        else:
            X_design = X_val_np
            coef_with_intercept = _to_numpy(model.coef_).ravel()

        try:
            val_loss = float(loss_fn.value(X_design, y_val_np, coef_with_intercept))
        except Exception:
            y_pred_np = _to_numpy(model.predict(X_val_np)).ravel()
            val_loss = float(np.mean((y_val_np - y_pred_np) ** 2))

        return val_loss, model

    def _refit_best(self, X, y, best_alpha):
        """Refit on full data with best alpha."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        can_infer = (self.loss == 'squared_error' and self.penalty == 'l2')

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=best_alpha,
            l1_ratio=self.l1_ratio,
            device=self.device,
            compute_inference=can_infer,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        model.fit(X, y)
        return model

    def fit(self, X, y, sample_weight=None):
        """Fit the CV model with optimized CV loop.

        For squared_error + l2: uses batch eigendecomposition.
        For other combos: uses warm-start across descending alphas.
        """
        # Generate alpha grid
        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(X, y)

        self.alpha_grid_ = alpha_grid
        n_samples = X.shape[0]
        n_alphas = len(alpha_grid)

        # Generate folds
        folds = kfold_indices(n_samples, self.cv, self.random_state)

        # Fast path: squared_error + l2 uses eigendecomposition
        if self.loss == 'squared_error' and self.penalty == 'l2':
            all_scores = np.full((self.cv, n_alphas), np.nan)
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                try:
                    mse, _, _ = self._solve_ridge_fold_batch(
                        X_train, y_train, X_val, y_val, alpha_grid,
                    )
                    all_scores[fold_idx, :] = mse
                except Exception:
                    pass

            mean_scores = np.nanmean(all_scores, axis=0)
            best_idx = int(np.nanargmin(mean_scores))
            best_alpha = float(alpha_grid[best_idx])

        else:
            # General path: warm-start across descending alphas
            sort_idx = np.argsort(-alpha_grid)
            alpha_sorted = alpha_grid[sort_idx]
            all_scores = np.full((self.cv, n_alphas), np.nan)

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                sw_train = sample_weight[train_idx] if sample_weight is not None else None

                prev_coef = None
                for alpha_idx_sorted, alpha in enumerate(alpha_sorted):
                    try:
                        val_loss, model = self._evaluate_single(
                            X_train, y_train, X_val, y_val, alpha,
                            sample_weight_train=sw_train,
                            prev_coef=prev_coef,
                        )
                        orig_idx = sort_idx[alpha_idx_sorted]
                        all_scores[fold_idx, orig_idx] = val_loss
                        prev_coef = _to_numpy(model.coef_).copy()
                    except Exception:
                        orig_idx = sort_idx[alpha_idx_sorted]
                        all_scores[fold_idx, orig_idx] = np.nan

            mean_scores = np.nanmean(all_scores, axis=0)
            best_idx = int(np.nanargmin(mean_scores))
            best_alpha = float(alpha_grid[best_idx])

        self.alpha_ = best_alpha
        self.best_score_ = float(np.nanmin(mean_scores))
        self.cv_results_ = {
            'alpha': alpha_grid,
            'mean_score': mean_scores,
            'all_scores': all_scores,
        }

        # Refit on full data
        self.estimator_ = self._refit_best(X, y, best_alpha)
        self.coef_ = self.estimator_.coef_
        self.intercept_ = self.estimator_.intercept_

        self._fitted = True
        return self
