"""Poisson regression with GPU-accelerated fitting and inference."""

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
        On auto/IRLS, positive C adds sum(beta**2)/(4*C) to the average
        loss, excluding the intercept; C=0 removes that term. Explicit
        Newton/L-BFGS/FISTA ignore C. Changing solvers at positive C can
        therefore change the statistical target.
    device : str or Device, default='auto'
        Compute device. Inference supports all three backends.
    n_jobs : int or None, default=None
        Shared CPU-worker configuration; no parallel-fit guarantee.
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
        Request best-effort release of reclaimable GPU cache memory.

    Notes
    -----
    fit(X=None, y=None, sample_weight=None, formula=None, data=None) returns
    self; formula and data are fit arguments, not constructor parameters.
    predict(X) returns the Poisson conditional mean. summary() returns a
    string; use print(model.summary()). There is no score or predict_proba
    method. Likelihood diagnostics and shared inference helpers are inherited
    from GeneralizedLinearModel. Positive-C IRLS inference includes penalty
    curvature; it does not remove shrinkage bias or account for selecting C.
    Failed auto/IRLS/FISTA refits can mix old estimates with new metadata;
    use a fresh estimator after such a failure. Formula prediction currently
    drops rows with missing predictors and returns a shorter unlabelled array;
    resolve missing data and check output length before associating predictions
    with the original observations.
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
