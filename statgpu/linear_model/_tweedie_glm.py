"""Tweedie regression (GLM, log link, power parameter)."""

from statgpu._config import Device
from statgpu.glm_core._family import Tweedie
from ._glm_base import GeneralizedLinearModel


class TweedieRegression(GeneralizedLinearModel):
    """Tweedie regression for compound Poisson-Gamma outcomes.

    Parameters
    ----------
    power : float, default=1.5
        Tweedie power parameter. Must be in (1, 2).
        1 < power < 2: compound Poisson-Gamma.
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-4
    C : float, default=1.0
        Inverse regularization strength.
    device : str or Device, default='auto'
    """

    def __init__(
        self,
        power: float = 1.5,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: int = None,
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        if not 1.0 < power < 2.0:
            raise ValueError(f"Tweedie power must be in (1, 2), got {power}")
        self._power = power
        super().__init__(
            family="tweedie",
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
        return Tweedie(power=self._power)

    def _get_loss_kwargs(self):
        return {"power": self._power}
