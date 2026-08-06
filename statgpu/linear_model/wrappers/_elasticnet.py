"""Elastic Net regression with GPU support.

``ElasticNet`` is a thin public wrapper over
:class:`~statgpu.linear_model.penalized.PenalizedLinearRegression` with
``penalty="elasticnet"``. It preserves the shared NumPy/CuPy/Torch solver and
post-fit inference contracts; the public default solver is FISTA.

The legacy standalone implementation has been moved to
``_elasticnet_legacy.py``.
"""

from __future__ import annotations

__all__ = ["ElasticNet"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression as _PenalizedLinearRegression


class ElasticNet(_PenalizedLinearRegression):
    """Elastic Net regression through the shared penalized-linear engine.

    Parameters
    ----------
    alpha : float, default=1.0
        Overall regularization strength.
    l1_ratio : float, default=0.5
        Mixing proportion between L1 and L2 penalties.
    solver : str, default="fista"
        Backend-aware optimization method.
    compute_inference : bool, default=False
        Whether to compute post-fit coefficient inference.
    inference_method : str, default="debiased"
        Post-fit inference method. Supported values are inherited from
        ``PenalizedLinearRegression``.
    cov_type : str, default="nonrobust"
        Covariance convention where the selected inference method uses it.
    hac_maxlags : int, optional
        HAC lag count where supported by the selected inference method.

    Notes
    -----
    ``compute_inference=True`` does not alter the penalized fit. Inference is
    computed after estimation and is conditional on the chosen regularization
    parameters.
    """

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
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
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
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
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
