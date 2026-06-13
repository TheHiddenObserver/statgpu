"""Inverse Gaussian regression (GLM, log link)."""

from typing import Optional

from statgpu._config import Device
from statgpu.glm_core._family import InverseGaussian
from ._glm_base import GeneralizedLinearModel


class InverseGaussianRegression(GeneralizedLinearModel):
    """Inverse Gaussian regression for positive right-skewed outcomes.

    Parameters
    ----------
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-4
    C : float, default=1.0
        Inverse regularization strength.
    device : str or Device, default='auto'
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(
            family="inverse_gaussian",
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
        return InverseGaussian()
