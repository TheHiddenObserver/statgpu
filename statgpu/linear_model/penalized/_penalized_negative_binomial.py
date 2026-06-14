"""Penalized Negative Binomial regression wrapper."""

from __future__ import annotations

from typing import Optional, Union
from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


class PenalizedNegativeBinomialRegression(PenalizedGeneralizedLinearModel):
    """Penalized Negative Binomial regression with configurable dispersion.

    Thin wrapper over ``PenalizedGeneralizedLinearModel(loss="negative_binomial", ...)``.

    Parameters
    ----------
    alpha_nb : float, default=1.0
        NB dispersion parameter (larger = less overdispersion).
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
        alpha_nb: float = 1.0,
        loss_kwargs: Optional[dict] = None,
    ):
        _lk = loss_kwargs or {}
        _lk["alpha"] = alpha_nb
        super().__init__(
            loss="negative_binomial",
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
