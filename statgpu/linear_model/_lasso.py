"""
Lasso regression (L1 penalty) via PenalizedLinearRegression.

The V9 ``Lasso`` class is a thin wrapper over ``PenalizedLinearRegression``
with ``penalty="l1"``.

The legacy standalone implementation has been moved to ``_lasso_legacy.py``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device

from ._penalized import PenalizedLinearRegression as _PenalizedLinearRegression

# Backward-compat imports for legacy implementation
from ._lasso_legacy import Lasso as _LassoLegacy  # noqa: F401
from ._lasso_legacy import _InferenceCapableLasso  # noqa: F401
from ._lasso_legacy import (  # noqa: F401 — used by LassoCV and knockoff
    _fit_lasso_single_alpha_fast,
    _select_lasso_alpha_cv,
    _solve_lasso_path_gpu_fista_multi_fold_from_gram,
    _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch,
)


class Lasso(_PenalizedLinearRegression):
    """Thin sklearn-style wrapper over ``PenalizedLinearRegression`` with L1 penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = True,
        inference_method: str = "cpu_ols_inference",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        enable_simultaneous_inference: bool = False,
        simultaneous_method: str = "maxz_bootstrap",
        simultaneous_alpha: float = 0.05,
        simultaneous_n_bootstrap: int = 1000,
        simultaneous_random_state: Optional[int] = None,
        simultaneous_include_intercept: bool = False,
        admm_rho: float = 1.0,
        **kwargs,
    ):
        # Store Lasso-specific inference parameters
        self.inference_method = str(inference_method).lower()
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.enable_simultaneous_inference = bool(enable_simultaneous_inference)
        self.simultaneous_method = str(simultaneous_method).lower()
        self.simultaneous_alpha = float(simultaneous_alpha)
        self.simultaneous_n_bootstrap = int(simultaneous_n_bootstrap)
        self.simultaneous_random_state = simultaneous_random_state
        self.simultaneous_include_intercept = bool(simultaneous_include_intercept)
        self.admm_rho = float(admm_rho)

        super().__init__(
            penalty="l1",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            stopping=stopping,
        )

    def fit(self, X=None, y=None, sample_weight=None, initial_coef=None, **kwargs):
        """Fit Lasso model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        initial_coef : array-like of shape (n_features,), optional
            Warm-start coefficients.
        """
        if initial_coef is not None:
            self._init_coef = np.asarray(initial_coef, dtype=np.float64)
        return super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
