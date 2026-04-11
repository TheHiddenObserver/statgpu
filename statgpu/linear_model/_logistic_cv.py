"""
LogisticRegressionCV interface skeleton.
"""

from typing import Optional, Union

from .._config import Device
from .._cv_base import CVEstimatorBase


def _select_logistic_c_cv(
    X,
    y,
    *,
    Cs=None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-4,
    device: Union[str, Device] = Device.CPU,
):
    """Placeholder for LogisticRegression C selection via cross-validation."""
    pass


class LogisticRegressionCV(CVEstimatorBase):
    """
    Skeleton of cross-validated LogisticRegression.

    Notes
    -----
    This class currently exposes a stable interface only. The actual CV
    training logic will be implemented in a follow-up change.
    """

    def __init__(
        self,
        Cs=None,
        cv: int = 5,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.Cs = Cs
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        self.C_ = None

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError(
            "LogisticRegressionCV.fit() is not yet implemented. "
            "This class currently exposes a stable interface skeleton only."
        )
