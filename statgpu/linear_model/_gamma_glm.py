"""Gamma regression (GLM with Gamma family, log link)."""

from typing import Optional

from statgpu._config import Device
from statgpu.glm_core._family import Gamma, LogLink, InversePowerLink
from ._glm_base import GeneralizedLinearModel


_LINK_MAP = {
    "log": LogLink,
    "inverse_power": InversePowerLink,
}


class GammaRegression(GeneralizedLinearModel):
    """Gamma regression for positive continuous outcomes.

    Uses log link by default for numerical stability. The canonical
    inverse_power link is also supported.

    Parameters
    ----------
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-4
    C : float, default=1.0
        Inverse regularization strength.
    device : str or Device, default='auto'
    link : str, default='log'
        Link function: 'log' or 'inverse_power'.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        C: float = 1.0,
        device: Device = Device.AUTO,
        n_jobs: Optional[int] = None,
        link: str = "log",
        solver: str = "auto",
        gpu_memory_cleanup: bool = False,
    ):
        self._link_name = link
        super().__init__(
            family="gamma",
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
        if self._link_name not in _LINK_MAP:
            valid = ", ".join(sorted(_LINK_MAP))
            raise ValueError(f"GammaRegression link must be one of: {valid}")
        link_cls = _LINK_MAP[self._link_name]
        return Gamma(link=link_cls())

    def _get_loss_kwargs(self):
        return {"link": self._link_name}
