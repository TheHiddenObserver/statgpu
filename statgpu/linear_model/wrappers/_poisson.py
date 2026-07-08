"""Poisson regression with inference support (CPU)."""

from typing import Optional
from statgpu._config import Device
from statgpu.glm_core._family import Poisson
from statgpu.linear_model._glm_base import GeneralizedLinearModel


class PoissonRegression(GeneralizedLinearModel):
    """Poisson regression with GPU-accelerated fitting and inference.

    Uses IRLS, Newton, LBFGS, or FISTA solvers for coefficient estimation.
    Supports M-estimation sandwich inference (standard errors, z-values,
    p-values, confidence intervals) via ``compute_inference=True``.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    max_iter : int, default=100
        Maximum number of solver iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    C : float, default=1.0
        Inverse regularization strength (for IRLS path only).
    device : str or Device, default='auto'
        Compute device. Inference supports all three backends.
    solver : str, default='auto'
        Solver: 'auto', 'irls', 'newton', 'lbfgs', 'fista'.
        For unpenalized inference validation against statsmodels,
        use ``solver='newton'`` (IRLS with default C=1.0 adds ridge penalty).
    compute_inference : bool, default=False
        If True, compute standard errors, z-statistics, p-values, and
        95% confidence intervals after fitting. Supports all three backends.
    cov_type : str, default='nonrobust'
        Covariance type: 'nonrobust', 'hc0', or 'hc1'.
    gpu_memory_cleanup : bool, default=False
        Free GPU memory after fitting.
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
        compute_inference: bool = False,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(
            family="poisson",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            C=C,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            compute_inference=compute_inference,
            cov_type=cov_type,
            gpu_memory_cleanup=gpu_memory_cleanup,
        )

    def _get_family(self):
        return Poisson()
