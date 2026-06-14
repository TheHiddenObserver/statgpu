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
        penalty: str = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = "auto",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        link: str = "log",
        loss_kwargs: Optional[dict] = None,
    ):
        _lk = loss_kwargs or {}
        _lk["link"] = link
        super().__init__(
            loss="gamma",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            device=device,
            n_jobs=n_jobs,
            loss_kwargs=_lk,
        )
