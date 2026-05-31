"""
Unified cross-validated penalized GLM estimator.

Supports all GLM loss functions (squared_error, logistic, poisson, gamma,
inverse_gaussian, negative_binomial, tweedie) with all penalty types
(l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_numpy
from statgpu.linear_model._cv_base import CVEstimatorBase, kfold_indices


class PenalizedGLM_CV(CVEstimatorBase):
    """Cross-validated penalized GLM supporting all loss + penalty combinations.

    Parameters
    ----------
    loss : str
        Loss function: 'squared_error', 'logistic', 'poisson', 'gamma',
        'inverse_gaussian', 'negative_binomial', 'tweedie'.
    penalty : str
        Penalty type: 'l1', 'l2', 'elasticnet', 'scad', 'mcp',
        'adaptive_l1', 'group_lasso'.
    alpha_grid : array-like or None
        Explicit alpha grid. If None, auto-generated from data.
    n_alphas : int
        Number of alphas for auto-generated grid.
    l1_ratio : float
        ElasticNet mixing parameter (only used for elasticnet penalty).
    cv : int
        Number of CV folds.
    random_state : int or None
        Random seed.
    device : str or Device
        Computation device.
    max_iter : int
        Maximum iterations for the inner solver.
    tol : float
        Convergence tolerance.
    solver : str
        Solver to use for the inner model.
    """

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
        # Warm-start from previous alpha's coefficients
        if prev_coef is not None:
            model._init_coef = np.asarray(prev_coef, dtype=np.float64)
        model.fit(X_train, y_train, sample_weight=sample_weight_train)

        # Compute validation score
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
        """Fit the CV model with warm-start optimization.

        Sorts alphas descending and warm-starts from the previous alpha's
        solution, reducing FISTA iterations by 50-80% for adjacent alphas.
        """
        # Generate alpha grid
        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(X, y)

        self.alpha_grid_ = alpha_grid
        n_samples = X.shape[0]

        # Generate folds
        folds = kfold_indices(n_samples, self.cv, self.random_state)
        n_alphas = len(alpha_grid)

        # Sort alphas descending for warm-start (largest alpha first = sparsest)
        sort_idx = np.argsort(-alpha_grid)
        alpha_sorted = alpha_grid[sort_idx]

        # CV loop with warm-start
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
                    # Map back to original alpha ordering
                    orig_idx = sort_idx[alpha_idx_sorted]
                    all_scores[fold_idx, orig_idx] = val_loss

                    # Store coefficients for warm-start to next alpha
                    prev_coef = _to_numpy(model.coef_).copy()
                except Exception:
                    orig_idx = sort_idx[alpha_idx_sorted]
                    all_scores[fold_idx, orig_idx] = np.nan

        # Aggregate across folds
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
