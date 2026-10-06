"""MCP-penalized regression (Zhang, Annals of Statistics 2010)."""

from __future__ import annotations

from typing import Union

from statgpu._config import Device
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


class MCPRegression(PenalizedLinearRegression):
    """MCP-penalized regression.

    Non-convex penalty with an oracle property under asymptotic regularity
    and tuning conditions. This is not a finite-sample unbiasedness guarantee.
    Uses LLA+FISTA for optimization.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    gamma : float, default=3.0
        Concavity parameter (Zhang recommends gamma > 1).
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device.
    compute_inference : bool, default=False
        Keep False on this specialized wrapper: it does not expose
        inference_method, and fitting with inference enabled raises. For an explicit
        oracle or bootstrap request, use PenalizedLinearRegression with
        penalty="mcp" and the desired inference_method. Ordinary active-set
        intervals do not correct selection uncertainty.
    solver : str, default='auto'
        Optimization algorithm for this non-convex penalty; consult the solver
        compatibility guide for supported choices.
    gpu_memory_cleanup : bool, default=False
        Request best-effort release of reclaimable GPU cache memory.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 3.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        compute_inference: bool = False,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        self.gamma = gamma
        super().__init__(
            penalty="mcp",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            compute_inference=compute_inference,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
            penalty_kwargs={"gamma": gamma},
        )
