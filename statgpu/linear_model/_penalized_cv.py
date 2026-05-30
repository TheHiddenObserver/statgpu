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
from statgpu.linear_model._cv_base import CVCache, CVEstimatorBase
from statgpu.linear_model._cv_engine import run_cv


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

    # Map loss names to score function descriptions
    _LOSS_METRICS = {
        'squared_error': ('mse', True),       # minimize MSE
        'logistic': ('logloss', True),         # minimize log-loss
        'poisson': ('deviance', True),         # minimize deviance
        'gamma': ('deviance', True),
        'inverse_gaussian': ('deviance', True),
        'negative_binomial': ('deviance', True),
        'tweedie': ('deviance', True),
    }

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

        n = X.shape[0]

        # Estimate alpha_max from the gradient at zero
        if self.loss == 'squared_error':
            # For MSE: alpha_max ~ max(|X'y|) / n
            alpha_max = float(np.max(np.abs(X.T @ y))) / n
        elif self.loss == 'logistic':
            # For logistic: alpha_max ~ max(|X'(y-0.5)|) / n
            y_centered = y - 0.5
            alpha_max = float(np.max(np.abs(X.T @ y_centered))) / n
        else:
            # For other GLMs: fit an unpenalized model and get gradient
            try:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty='l2', alpha=0.0,
                    device='cpu', compute_inference=False, max_iter=5,
                )
                model.fit(X, y)
                grad = X.T @ (y - _to_numpy(model.predict(X))) / n
                alpha_max = float(np.max(np.abs(grad)))
            except Exception:
                alpha_max = 1.0

        if alpha_max <= 0:
            alpha_max = 1.0

        # Generate grid based on penalty type
        if self.penalty in ('l1', 'elasticnet', 'scad', 'mcp', 'adaptive_l1'):
            # L1-type: search from alpha_max down
            grid = np.geomspace(alpha_max, alpha_max * 1e-4, self.n_alphas)
        else:
            # L2-type: search from small to large
            grid = np.logspace(
                np.log10(max(alpha_max * 1e-4, 1e-12)),
                np.log10(alpha_max),
                self.n_alphas,
            )

        return grid

    def _evaluate_fold(self, X_train, y_train, X_val, y_val, alpha,
                       sample_weight_train=None, sample_weight_val=None):
        """Train on training fold, return validation score."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=alpha,
            l1_ratio=self.l1_ratio,
            device='cpu',  # CV evaluation on CPU for reliability
            compute_inference=False,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
        y_pred = _to_numpy(model.predict(X_val))
        y_val_np = _to_numpy(y_val).ravel()

        # Compute score based on loss type
        if self.loss == 'squared_error':
            if sample_weight_val is not None:
                sw = _to_numpy(sample_weight_val).ravel()
                return float(np.mean((y_val_np - y_pred) ** 2 * sw))
            return float(np.mean((y_val_np - y_pred) ** 2))

        elif self.loss == 'logistic':
            p = np.clip(y_pred, 1e-10, 1 - 1e-10)
            return float(-np.mean(y_val_np * np.log(p) + (1 - y_val_np) * np.log(1 - p)))

        else:
            # GLM deviance: 2 * (log-likelihood(saturated) - log-likelihood(model))
            # Simplified: use residual deviance
            p = np.clip(y_pred, 1e-10, None)
            return float(np.sum((y_val_np - p) ** 2 / p))

    def _refit_best(self, X, y, best_alpha):
        """Refit on full data with best alpha."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=best_alpha,
            l1_ratio=self.l1_ratio,
            device=self.device,
            compute_inference=True,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        model.fit(X, y)
        return model

    def fit(self, X, y, sample_weight=None):
        """Fit the CV model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()

        # Generate or use provided alpha grid
        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(X, y)

        self.alpha_grid_ = alpha_grid

        # Run CV
        best_alpha, mean_scores, all_scores = run_cv(
            X, y,
            alpha_grid=alpha_grid,
            evaluate_fold_fn=self._evaluate_fold,
            n_folds=self.cv,
            random_state=self.random_state,
            minimize=True,
            sample_weight=sample_weight,
        )

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
