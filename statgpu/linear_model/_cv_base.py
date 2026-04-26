"""
Shared base class for cross-validated estimators.
"""

from typing import Optional, Union

from statgpu._base import BaseEstimator
from statgpu._config import Device


class CVEstimatorBase(BaseEstimator):
    """
    Common scaffolding for model-specific CV estimators.

    This is intentionally lightweight: each model keeps its own CV search
    routine and fitted attributes, while shared plumbing lives here.
    """

    def __init__(
        self,
        *,
        cv: int = 5,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cv = int(cv)
        self.random_state = random_state

        # Common fitted attributes for CV estimators.
        self.best_score_ = None
        self.cv_results_ = None
        self.estimator_ = None

    def predict(self, X):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        return self.estimator_.predict(X)

    def score(self, X, y):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        return self.estimator_.score(X, y)

    def summary(self):
        self._check_is_fitted()
        if self.estimator_ is None:
            raise RuntimeError("No fitted base estimator is available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(
                f"{self.estimator_.__class__.__name__} does not implement summary()."
            )
        return self.estimator_.summary()
