"""Ordered probit regression."""

import numpy as np

from statgpu._config import Device
from statgpu.glm_core._family import Binomial, ProbitLink
from ._glm_base import OrderedGeneralizedLinearModel


class OrderedProbitRegression(OrderedGeneralizedLinearModel):
    """Ordered probit regression.

    Parameters
    ----------
    n_categories : int, default=3
        Number of ordinal categories.
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-4
    C : float, default=1.0
        Inverse regularization strength.
    device : str or Device, default='auto'
    """

    def __init__(
        self,
        n_categories: int = 3,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: int = None,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(
            n_categories=n_categories,
            family="binomial",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver="auto",
            gpu_memory_cleanup=gpu_memory_cleanup,
        )

    def _get_family(self):
        return Binomial(link=ProbitLink())
