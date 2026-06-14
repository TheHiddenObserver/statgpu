"""
Elastic Net regression with GPU support.

The V9 ElasticNet class is a thin wrapper over PenalizedLinearRegression
with penalty="elasticnet" and solver="exact".

The legacy standalone implementation has been moved to _elasticnet_legacy.py.
"""

from __future__ import annotations

__all__ = ["ElasticNet"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu._base import BaseEstimator
from statgpu.linear_model._penalized_linear import PenalizedLinearRegression as _PenalizedLinearRegression

# Backward-compat import for legacy implementation
from statgpu.linear_model._elasticnet_legacy import ElasticNet as _ElasticNetLegacy  # noqa: F401

class ElasticNet(_PenalizedLinearRegression):
    """Thin sklearn-style wrapper over ``PenalizedLinearRegression`` with Elastic Net penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "fista",
        cpu_solver: str = "fista",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.stopping = str(stopping).lower()
        super().__init__(
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            stopping=stopping,
        )

    def fit(self, X=None, y=None, sample_weight=None, initial_coef=None, **kwargs):
        """Fit Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        initial_coef : array-like of shape (n_features,), optional
            Warm-start coefficients. Passed to the underlying solver.
        """
        if initial_coef is not None:
            self._init_coef = np.asarray(initial_coef, dtype=np.float64)
        return super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
