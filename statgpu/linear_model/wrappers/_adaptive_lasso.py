"""Adaptive Lasso regression (Zou, JASA 2006)."""

from __future__ import annotations

from typing import Union

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


class AdaptiveLasso(PenalizedLinearRegression):
    """Adaptive Lasso regression.

    Uses data-driven per-coordinate weights: w_j = 1/(|init_coef_j| + eps)^nu.
    Provides oracle property under regularity conditions (Zou 2006).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    nu : float, default=1.0
        Exponent for weight computation.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device.
    compute_inference : bool, default=False
        Whether to compute post-fit inference.
    inference_method : str, default='debiased'
        Inference method.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        nu: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        self.nu = nu
        super().__init__(
            penalty="adaptive_l1",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            compute_inference=compute_inference,
            inference_method=inference_method,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
            penalty_kwargs={"nu": nu},
        )
