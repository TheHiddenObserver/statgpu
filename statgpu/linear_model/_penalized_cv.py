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

        # Convert to numpy for alpha grid computation
        X_np = _to_numpy(X)
        y_np = _to_numpy(y).ravel()
        n = X_np.shape[0]

        # Estimate alpha_max from the gradient at zero
        if self.loss == 'squared_error':
            # For MSE: alpha_max ~ max(|X'y|) / n
            alpha_max = float(np.max(np.abs(X_np.T @ y_np))) / n
        elif self.loss == 'logistic':
            # For logistic: alpha_max ~ max(|X'(y-0.5)|) / n
            y_centered = y_np - 0.5
            alpha_max = float(np.max(np.abs(X_np.T @ y_centered))) / n
        else:
            # For other GLMs: fit an unpenalized model and get gradient
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

        # Convert to numpy for CV evaluation (avoids device-specific issues)
        X_train_np = _to_numpy(X_train).astype(np.float64)
        y_train_np = _to_numpy(y_train).astype(np.float64).ravel()
        X_val_np = _to_numpy(X_val).astype(np.float64)
        y_val_np = _to_numpy(y_val).astype(np.float64).ravel()

        model = PenalizedGeneralizedLinearModel(
            loss=self.loss,
            penalty=self.penalty,
            alpha=alpha,
            l1_ratio=self.l1_ratio,
            device='cpu',
            compute_inference=False,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )
        model.fit(X_train_np, y_train_np)

        # Compute score using the GLM loss function
        from statgpu.linear_model._penalized import _resolve_loss_name

        loss_fn = _resolve_loss_name(self.loss)

        # Build design matrix for loss evaluation
        n_val = X_val_np.shape[0]
        if model.fit_intercept:
            X_design = np.column_stack([np.ones(n_val), X_val_np])
            coef_with_intercept = np.concatenate([[float(model.intercept_)], _to_numpy(model.coef_)])
        else:
            X_design = X_val_np
            coef_with_intercept = _to_numpy(model.coef_)

        # Compute validation loss using the GLM loss function
        try:
            val_loss = loss_fn.value(X_design, y_val_np, coef_with_intercept)
            return float(val_loss)
        except Exception:
            # Fallback: MSE
            y_pred_np = _to_numpy(model.predict(X_val_np)).ravel()
            return float(np.mean((y_val_np - y_pred_np) ** 2))

    def _refit_best(self, X, y, best_alpha):
        """Refit on full data with best alpha."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

        # Only request inference for squared_error + l2 (the only supported combo)
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
        # Keep arrays on their original device; _to_numpy() used where needed
        X = _to_numpy(X).astype(np.float64)
        y = _to_numpy(y).astype(np.float64).ravel()
        if sample_weight is not None:
            sample_weight = _to_numpy(sample_weight).astype(np.float64).ravel()

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
