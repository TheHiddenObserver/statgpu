"""
Ridge regression (L2 penalty) via PenalizedLinearRegression.

The V9 ``Ridge`` class is a thin wrapper over ``PenalizedLinearRegression``
with ``penalty="l2"`` and ``solver="exact"``.

The legacy standalone implementation has been moved to ``_ridge_legacy.py``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device

from ._penalized import PenalizedLinearRegression as _PenalizedLinearRegression
from ._ridge_legacy import _RidgeLegacy  # noqa: F401 — backward compat


class Ridge(_PenalizedLinearRegression):
    """Thin sklearn-style wrapper over ``PenalizedLinearRegression`` with L2 penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = "exact",
        cpu_solver: str = "fista",
        lipschitz_L: Optional[float] = None,
    ):
        self.cov_type = str(cov_type).lower()
        self.hac_maxlags = hac_maxlags
        super().__init__(
            penalty="l2",
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
        )

    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit Ridge regression model with optimized memory-efficient path.

        Uses centering formulas to avoid allocating the full centered design matrix,
        and skips expensive inference computations when ``compute_inference=False``.
        """
        if (formula is not None
                or self._get_compute_device() != Device.CPU
                or self.solver != "exact"):
            # Fall back to parent for formula, GPU, or non-exact solver
            return super().fit(X=X, y=y, sample_weight=sample_weight, formula=formula, data=data)

        X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
        y_np = np.asarray(self._to_array(y, Device.CPU), dtype=np.float64)

        n_samples, n_features = X_np.shape
        self._nobs = n_samples
        self._fitted = False

        sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None

        if self.fit_intercept:
            if sw is not None:
                w_sum = float(sw.sum())
                X_wmean = np.average(X_np, axis=0, weights=sw)
                y_wmean = float(np.average(y_np, weights=sw))
            else:
                X_wmean = np.mean(X_np, axis=0)
                y_wmean = np.mean(y_np)

        # Build Gram matrix and RHS.
        # Weighted: X'WX, X'Wy.  Unweighted: X'X, X'y.
        # Centering for intercept: subtract weighted/unweighted outer product.
        if sw is not None:
            # Weighted normal equations: (X'WX + alpha*I) coef = X'Wy
            sw_col = sw[:, None]
            XtX = (X_np * sw_col).T @ X_np
            Xty = (X_np * sw_col).T @ y_np
            if self.fit_intercept:
                XtX -= w_sum * np.outer(X_wmean, X_wmean)
                Xty -= w_sum * X_wmean * y_wmean
                n_eff = w_sum
            else:
                n_eff = float(sw.sum())
        else:
            if self.fit_intercept:
                X_mean = np.mean(X_np, axis=0)
                y_mean = np.mean(y_np)
                XtX = X_np.T @ X_np
                XtX -= n_samples * np.outer(X_mean, X_mean)
                Xty = X_np.T @ y_np
                Xty -= n_samples * X_mean * y_mean
            else:
                XtX = X_np.T @ X_np
                Xty = X_np.T @ y_np
            n_eff = float(n_samples)

        if Xty.ndim == 0:
            Xty = Xty.reshape(1)
        if Xty.ndim == 1:
            Xty = Xty.reshape(-1, 1)

        # Solve (XtX + n_eff*alpha*I) @ coef = Xty
        # n_eff scaling matches PenalizedGeneralizedLinearModel exact ridge
        # and sklearn Ridge convention.
        A = XtX + float(self.alpha) * n_eff * np.eye(n_features, dtype=np.float64)
        try:
            coef = np.linalg.solve(A, Xty).flatten()
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(A, Xty, rcond=None)[0].flatten()

        if self.fit_intercept:
            self.intercept_ = float(y_wmean - X_wmean @ coef)
            self.coef_ = coef
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            self._params = self.coef_.copy()

        self._X_design = None
        self._resid = None
        self._scale = np.nan
        self.n_iter_ = 1
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        # Build design matrix and compute residuals only when inference is needed
        if self.compute_inference:
            if self.fit_intercept:
                self._X_design = np.column_stack([np.ones(n_samples, dtype=X_np.dtype), X_np])
            else:
                self._X_design = X_np.copy()
            y_pred = self._X_design @ self._params
            self._resid = y_np - y_pred
            if self._df_resid > 0:
                resid_sq = self._resid ** 2
                self._scale = float(np.sum(resid_sq)) / self._df_resid
            # Compute inference statistics (bse, tvalues, pvalues, conf_int).
            # For weighted fits, _compute_post_fit_gaussian_inference uses
            # sqrt(w)*X internally, producing correct weighted scale and
            # consistent inference attributes.
            self._compute_post_fit_gaussian_inference(X_np, y_np, sample_weight=sample_weight)

        self._fitted = True
        return self
