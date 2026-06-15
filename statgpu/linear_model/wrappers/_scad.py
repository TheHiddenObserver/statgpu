"""SCAD-penalized regression (Fan & Li, JASA 2001)."""

from __future__ import annotations

from typing import Union

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


class SCADRegression(PenalizedLinearRegression):
    """SCAD-penalized regression.

    Non-convex penalty with oracle property. Uses LLA+FISTA for optimization.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    a : float, default=3.7
        Concavity parameter (Fan & Li recommend 3.7).
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device.
    compute_inference : bool, default=False
        Whether to compute post-fit inference (SCAD does not support debiased).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        a: float = 3.7,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        compute_inference: bool = False,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        self.a = a
        super().__init__(
            penalty="scad",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            compute_inference=compute_inference,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
            penalty_kwargs={"a": a},
        )
