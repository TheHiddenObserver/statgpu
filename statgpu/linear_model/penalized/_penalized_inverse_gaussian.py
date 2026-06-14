"""Penalized Inverse Gaussian regression wrapper."""

from __future__ import annotations

from typing import Optional, Union
from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


class PenalizedInverseGaussianRegression(PenalizedGeneralizedLinearModel):
    """Penalized Inverse Gaussian regression.

    Thin wrapper over ``PenalizedGeneralizedLinearModel(loss="inverse_gaussian", ...)``.
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
        loss_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            loss="inverse_gaussian",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            device=device,
            n_jobs=n_jobs,
            loss_kwargs=loss_kwargs,
        )
