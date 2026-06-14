"""Penalized Tweedie regression wrapper."""

from __future__ import annotations

from typing import Optional, Union
from statgpu._config import Device
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel


class PenalizedTweedieRegression(PenalizedGeneralizedLinearModel):
    """Penalized Tweedie regression with configurable power.

    Thin wrapper over ``PenalizedGeneralizedLinearModel(loss="tweedie", ...)``.

    Parameters
    ----------
    power : float, default=1.5
        Tweedie power parameter (1=Poisson, 2=Gamma, 1.5=compound Poisson-Gamma).
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
        power: float = 1.5,
        loss_kwargs: Optional[dict] = None,
    ):
        _lk = loss_kwargs or {}
        _lk["power"] = power
        super().__init__(
            loss="tweedie",
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
