"""Negative Binomial regression (GLM, log link, fixed dispersion)."""

import numpy as np

from statgpu._config import Device
from statgpu.glm_core._family import NegativeBinomial
from ._glm_base import GeneralizedLinearModel


class NegativeBinomialRegression(GeneralizedLinearModel):
    """Negative Binomial regression for overdispersed count data.

    Parameters
    ----------
    alpha : float, default=1.0
        Dispersion parameter. Var(Y) = mu + alpha * mu^2.
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-4
    C : float, default=1.0
        Inverse regularization strength.
    device : str or Device, default='auto'
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: int = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for negative binomial regression")
        self._alpha = alpha
        super().__init__(
            family="negative_binomial",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )

    def _get_family(self):
        return NegativeBinomial(alpha=self._alpha)

    def _get_loss_kwargs(self):
        return {"alpha": self._alpha}
