"""
Lasso regression with statistical inference and GPU support.

The V9 Lasso class is a thin wrapper over PenalizedLinearRegression
with penalty="l1" and solver="exact".

The legacy standalone implementation has been moved to _lasso_legacy.py.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu._base import BaseEstimator

# Backward-compat import for legacy implementation
from ._lasso_legacy import Lasso as _LassoLegacy  # noqa: F401
from ._lasso_legacy import _InferenceCapableLasso  # noqa: F401

class Lasso(_InferenceCapableLasso):
    """Inference-capable Lasso estimator.

    This public wrapper keeps the existing Lasso inference algorithms available
    while the shared inference result containers are introduced.  The pure
    GLM+penalty L1 path remains available through ``PenalizedLinearRegression``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        inference_method: str = "cpu_ols_inference",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        enable_simultaneous_inference: bool = False,
        simultaneous_method: str = "maxz_bootstrap",
        simultaneous_alpha: float = 0.05,
        simultaneous_n_bootstrap: int = 1000,
        simultaneous_random_state: Optional[int] = None,
        simultaneous_include_intercept: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
        **kwargs,
    ):
        self.stopping = str(stopping).lower()
        self.inference_method = str(inference_method).lower()
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.enable_simultaneous_inference = bool(enable_simultaneous_inference)
        self.simultaneous_method = str(simultaneous_method).lower()
        self.simultaneous_alpha = float(simultaneous_alpha)
        self.simultaneous_n_bootstrap = int(simultaneous_n_bootstrap)
        self.simultaneous_random_state = simultaneous_random_state
        self.simultaneous_include_intercept = bool(simultaneous_include_intercept)
        self.compute_inference = bool(compute_inference)
        self.admm_rho = float(admm_rho)
        self._ignored_kwargs = dict(kwargs)
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            stopping=stopping,
            inference_method=inference_method,
            n_bootstrap=n_bootstrap,
            bootstrap_random_state=bootstrap_random_state,
            enable_simultaneous_inference=enable_simultaneous_inference,
            simultaneous_method=simultaneous_method,
            simultaneous_alpha=simultaneous_alpha,
            simultaneous_n_bootstrap=simultaneous_n_bootstrap,
            simultaneous_random_state=simultaneous_random_state,
            simultaneous_include_intercept=simultaneous_include_intercept,
            device=device,
            n_jobs=n_jobs,
            compute_inference=compute_inference,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            admm_rho=admm_rho,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )
