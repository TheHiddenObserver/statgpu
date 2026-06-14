"""Penalized Gamma regression wrapper."""

from __future__ import annotations

from typing import Optional, Union
from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


class PenalizedGammaRegression(PenalizedGeneralizedLinearModel):
    """Penalized Gamma regression with log or inverse-power link.

    Thin wrapper over ``PenalizedGeneralizedLinearModel(loss="gamma", ...)``.

    Parameters
    ----------
    penalty : str, default='l2'
        Penalty type: 'l1', 'l2', 'elasticnet', 'scad', 'mcp', etc.
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only used with penalty='elasticnet').
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4
    solver : str, default='auto'
    device : str or Device, default='auto'
    link : str, default='log'
        Link function: 'log' or 'inverse_power'.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss constructor.
    """

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
        link: str = "log",
    ):
        super().__init__(
        super().__init__(
            loss="gamma",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )
            )
