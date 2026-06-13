"""
Generalized Additive Model (GAM) with GPU support.

Implements GAM using penalized B-splines with automatic smoothing
parameter selection via Generalized Cross-Validation (GCV).
"""

__all__ = ["GAM"]

from __future__ import annotations

import numpy as np
from typing import Optional, Union

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _torch_dev, _to_numpy, xp_zeros, xp_ones, xp_asarray, xp_copy
from statgpu.nonparametric.splines._bspline_basis import bspline_basis
from statgpu.nonparametric.splines._penalized import (
    difference_penalty,
    penalized_ls,
    generalized_cross_validation,
    select_lambda_gcv,
)


class GAM(BaseEstimator):
    """
    Generalized Additive Model (GAM) using penalized B-splines.

    Fits a smooth function for each feature using B-spline basis with
    a difference penalty for smoothness. Smoothing parameters can be
    specified or automatically selected via GCV.

    The model is: y = alpha + sum_j f_j(x_j) + epsilon

    where each f_j is represented as a penalized B-spline.

    Parameters
    ----------
    n_splines : int, default=20
        Number of basis functions per feature (before penalty reduction).
    degree : int, default=3
        Degree of B-spline basis (3 = cubic).
    lam : float or None, default=None
        Smoothing parameter. If None, automatically selected via GCV.
    penalty_order : int, default=2
        Order of difference penalty (2 = second differences).
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    n_jobs : int or None, default=None
        Number of parallel jobs.

    Attributes
    ----------
    coef_ : array, shape (n_features * n_splines + 1,)
        Fitted coefficients (intercept + spline coefficients for each feature).
    intercept_ : float
        Intercept term.
    edf_ : float
        Total effective degrees of freedom.
    gcv_score_ : float
        GCV score (if lam was auto-selected).
    lam_ : float
        Smoothing parameter used (after auto-selection if applicable).
    knots_ : list of arrays
        Interior knots for each feature.
    n_features_ : int
        Number of features in training data.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.semiparametric import GAM
    >>> X = np.random.randn(100, 3)
    >>> y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + np.random.randn(100) * 0.1
    >>> gam = GAM(n_splines=15, lam=1.0)
    >>> gam.fit(X, y)
    >>> y_pred = gam.predict(X)
    """

    def __init__(
        self,
        n_splines: int = 20,
        degree: int = 3,
        lam: Optional[float] = None,
        penalty_order: int = 2,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_splines = n_splines
        self.degree = degree
        self.lam = lam
        self.penalty_order = penalty_order

        # Fitted attributes
        self.coef_ = None
        self.intercept_ = None
        self.edf_ = None
        self.gcv_score_ = None
        self.lam_ = None
        self.knots_ = None
        self.n_features_ = None

    def _get_xp(self):
        """Get the array module for computation.

        Returns ``backend.xp`` (the raw array module) so callers can use
        ``xp.asarray`` etc. directly.  Delegates to the parent's
        ``_get_backend()`` which returns a ``BackendBase`` with correct
        device/dtype handling.
        """
        backend = super()._get_backend(backend='auto')
        return backend.xp

    def _create_knots(self, x_col, n_splines, xp):
        """
        Create interior knots for a feature using quantiles.

        Parameters
        ----------
        x_col : array, shape (n,)
            Feature values.
        n_splines : int
            Number of basis functions.
        xp : module
            Array module.

        Returns
        -------
        knots : array, shape (n_splines - degree - 1,)
            Interior knots.
        """
        # Use quantiles for knot placement
        # Exclude boundary knots (they'll be added by bspline_basis)
        n_interior = n_splines - self.degree - 1

        if n_interior <= 0:
            raise ValueError(
                f"n_splines ({n_splines}) must be greater than degree ({self.degree}) + 1"
            )

        # Use percentiles from 0 to 100, excluding boundaries
        percentiles = np.linspace(0, 100, n_interior + 2)[1:-1]

        # Convert to numpy for percentile computation
        x_np = _to_numpy(x_col)

        knots = np.percentile(x_np, percentiles)

        # Remove duplicate knots (can happen with discrete data)
        knots = np.unique(knots)

        return xp_asarray(knots, dtype=xp.float64, xp=xp, ref_arr=x_col)

    def _build_basis(self, X, xp):
        """
        Build combined basis matrix for all features.

        Parameters
        ----------
        X : array, shape (n, p)
            Input features.
        xp : module
            Array module.

        Returns
        -------
        B : array, shape (n, 1 + sum(n_basis_j))
            Combined basis matrix with intercept column.
        penalty : array, shape (1 + sum(n_basis_j), 1 + sum(n_basis_j))
            Block-diagonal penalty matrix (intercept not penalized).
        """
        n, p = X.shape
        basis_blocks = []
        penalty_blocks = []
        total_basis = 0

        for j in range(p):
            x_col = X[:, j]

            # Create knots for this feature
            knots_j = self._create_knots(x_col, self.n_splines, xp)
            self.knots_.append(knots_j)

            # Store training boundary for prediction
            self._boundary_lo_.append(float(xp.min(x_col)))
            self._boundary_hi_.append(float(xp.max(x_col)))

            # Build B-spline basis
            B_j = bspline_basis(x_col, knots_j, degree=self.degree, xp=xp)
            n_basis_j = B_j.shape[1]

            # Build penalty matrix
            S_j = difference_penalty(self.penalty_order, n_basis_j, xp)

            basis_blocks.append(B_j)
            penalty_blocks.append(S_j)
            total_basis += n_basis_j

        # Combine basis matrices: [1, B_1, B_2, ..., B_p]
        intercept_col = xp_ones((n, 1), xp.float64, xp, X)
        B = xp.hstack([intercept_col] + basis_blocks)

        # Block-diagonal penalty with intercept dimension (not penalized)
        # Size: (1 + total_basis, 1 + total_basis) to match B
        full_size = 1 + total_basis
        penalty = xp_zeros((full_size, full_size), xp.float64, xp, X)
        offset = 1  # Skip intercept (row/col 0)
        for S_j in penalty_blocks:
            n_j = S_j.shape[0]
            penalty[offset:offset + n_j, offset:offset + n_j] = S_j
            offset += n_j

        return B, penalty

    def fit(self, X, y=None, **fit_params):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : GAM
            Fitted model.
        """
        xp = self._get_xp()

        # Convert to arrays on the correct device
        # For torch backend, ensure arrays land on CUDA (not CPU)
        _ref = None
        if xp.__name__ == "torch":
            import torch
            _dev = getattr(self, 'device', None)
            if _dev is not None and hasattr(_dev, 'value') and _dev.value in ('cuda', 'torch'):
                _ref = torch.empty(0, device="cuda")
            elif torch.cuda.is_available():
                _ref = torch.empty(0, device="cuda")
        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=_ref)
        y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X).ravel()

        n, p = X.shape
        self.n_features_ = p
        self.knots_ = []
        self._boundary_lo_ = []
        self._boundary_hi_ = []

        # Build basis matrix and penalty
        B, penalty = self._build_basis(X, xp)

        # Center spline basis columns (not intercept) so the intercept
        # captures the overall mean of y.  This makes the intercept
        # identifiable even though spline basis can represent constants.
        self._basis_mean_ = xp.mean(B[:, 1:], axis=0)
        B_centered = xp_copy(B)
        B_centered[:, 1:] = B[:, 1:] - self._basis_mean_

        # Select smoothing parameter
        if self.lam is None:
            # Auto-select via GCV
            best_lam, gcv_scores = select_lambda_gcv(
                B_centered, y, penalty, xp=xp
            )
            self.lam_ = best_lam
            self.gcv_score_ = float(xp.min(gcv_scores))
        else:
            self.lam_ = self.lam
            self.gcv_score_ = None

        # Fit the model with centered basis
        beta, edf = penalized_ls(B_centered, y, penalty, self.lam_, xp)

        # Store results
        self.coef_ = beta
        self.intercept_ = float(beta[0])
        self.edf_ = float(edf) if not isinstance(edf, float) else edf
        self._fitted = True

        # Store training data info for prediction
        self._xp = xp
        self._xp_asarray_ref_ = X  # device reference for xp_asarray

        return self

    def predict(self, X):
        """
        Predict using the fitted GAM model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()

        # Re-resolve backend to handle device changes since fit()
        xp = self._get_xp()
        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=self._xp_asarray_ref_)

        n, p = X.shape
        if p != self.n_features_:
            raise ValueError(
                f"X has {p} features, but model was fitted with {self.n_features_}"
            )

        # Build basis for prediction (use training boundaries to avoid
        # "knots must be strictly within boundary" errors on small batches)
        basis_blocks = []
        for j in range(p):
            x_col = X[:, j]
            knots_j = self.knots_[j]
            B_j = bspline_basis(
                x_col, knots_j, degree=self.degree, xp=xp,
                boundary_lo=self._boundary_lo_[j],
                boundary_hi=self._boundary_hi_[j],
            )
            basis_blocks.append(B_j)

        # Combine: [1, B_1, B_2, ..., B_p]
        intercept_col = xp_ones((n, 1), xp.float64, xp, X)
        B = xp.hstack([intercept_col] + basis_blocks)

        # Apply same centering as in fit
        B[:, 1:] = B[:, 1:] - self._basis_mean_

        # Predict
        y_pred = B @ self.coef_

        return _to_numpy(y_pred)

    def summary(self):
        """
        Print a summary of the fitted GAM model.

        Returns
        -------
        summary_dict : dict
            Dictionary containing model summary information.
        """
        self._check_is_fitted()

        summary_dict = {
            'n_features': self.n_features_,
            'n_splines_per_feature': self.n_splines,
            'spline_degree': self.degree,
            'penalty_order': self.penalty_order,
            'smoothing_parameter': self.lam_,
            'effective_df': self.edf_,
            'intercept': self.intercept_,
        }

        if self.gcv_score_ is not None:
            summary_dict['gcv_score'] = self.gcv_score_

        print("=" * 50)
        print("GAM Model Summary")
        print("=" * 50)
        print(f"Number of features: {self.n_features_}")
        print(f"B-splines per feature: {self.n_splines}")
        print(f"Spline degree: {self.degree}")
        print(f"Penalty order: {self.penalty_order}")
        print(f"Smoothing parameter (lambda): {self.lam_:.6g}")
        print(f"Effective degrees of freedom: {self.edf_:.2f}")
        print(f"Intercept: {self.intercept_:.6f}")
        if self.gcv_score_ is not None:
            print(f"GCV score: {self.gcv_score_:.6f}")
        print("=" * 50)

        return summary_dict

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({
            'n_splines': self.n_splines,
            'degree': self.degree,
            'lam': self.lam,
            'penalty_order': self.penalty_order,
        })
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in ('n_splines', 'degree', 'lam', 'penalty_order'):
                setattr(self, key, value)
            else:
                super().set_params(**{key: value})
        return self
