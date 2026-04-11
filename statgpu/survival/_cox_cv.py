"""
CoxPHCV interface skeleton.
"""

from typing import Optional, Union

from .._config import Device
from .._cv_base import CVEstimatorBase


def _select_coxph_penalty_cv(
    X,
    time,
    event,
    *,
    penalties=None,
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    ties: str = "breslow",
    device: Union[str, Device] = Device.CPU,
):
    """Placeholder for CoxPH penalty selection via cross-validation."""
    pass


class CoxPHCV(CVEstimatorBase):
    """
    Skeleton of cross-validated CoxPH.

    Notes
    -----
    This class currently exposes a stable interface only. The actual CV
    training logic will be implemented in a follow-up change.
    """

    def __init__(
        self,
        penalties=None,
        cv: int = 5,
        ties: str = "breslow",
        tol: float = 1e-9,
        max_iter: int = 100,
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
        self.penalties = penalties
        self.ties = str(ties)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        self.penalty_ = None

    def fit(self, X, time, event, entry=None, cluster=None):
        raise NotImplementedError(
            "CoxPHCV.fit() is not yet implemented. "
            "This class currently exposes a stable interface skeleton only."
        )

    def predict(self, X):
        raise NotImplementedError(
            "CoxPHCV.predict() is not yet implemented. "
            "This class currently exposes a stable interface skeleton only."
        )

    def score(self, X, time, event):
        raise NotImplementedError(
            "CoxPHCV.score() is not yet implemented. "
            "This class currently exposes a stable interface skeleton only."
        )
