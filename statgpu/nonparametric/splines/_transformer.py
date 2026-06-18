"""sklearn-compatible SplineTransformer with GPU acceleration."""

from __future__ import annotations

__all__ = ["SplineTransformer"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy, xp_asarray
from statgpu.nonparametric.splines._bspline_basis import bspline_basis


class SplineTransformer(BaseEstimator):
    """B-spline feature transformer (sklearn-compatible API).

    Generates B-spline basis features for each input feature, suitable
    for use in pipelines and GAM-like models.

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots (including boundary knots).
    degree : int, default=3
        Spline degree (3 = cubic).
    knots : str or array-like, default='uniform'
        Knot placement strategy: ``'uniform'`` or ``'quantile'``.
        Can also be an array of shape ``(n_knots, n_features)``.
    include_bias : bool, default=True
        If True, include all basis functions (including the one that
        is redundant due to the partition-of-unity property).
    extrapolation : str, default='constant'
        How to handle extrapolation beyond boundary knots:
        ``'constant'`` (clamp to boundary values), ``'linear'``
        (linear extrapolation), or ``'continue'`` (extend with
        boundary slope).
    device : str or Device, default='auto'
        Computation device.

    Attributes
    ----------
    knots_ : list of array
        Knot positions for each feature.
    boundary_lo_ : ndarray, shape (n_features,)
        Lower boundary for each feature.
    boundary_hi_ : ndarray, shape (n_features,)
        Upper boundary for each feature.
    n_features_in_ : int
        Number of input features.
    n_features_out_ : int
        Number of output features.
    """

    def __init__(
        self,
        n_knots: int = 5,
        degree: int = 3,
        knots: str = "uniform",
        include_bias: bool = True,
        extrapolation: str = "constant",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.include_bias = include_bias
        self.extrapolation = extrapolation

    def fit(self, X, y=None, sample_weight=None):
        """Fit the spline transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored
        sample_weight : ignored

        Returns
        -------
        self
        """
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features

        if self.n_knots < 3:
            raise ValueError("n_knots must be at least 3 (boundary + at least 1 interior knot)")

        # Determine knots per feature
        if isinstance(self.knots, str):
            knots_list = []
            for j in range(n_features):
                col = X_np[:, j]
                if self.knots == "uniform":
                    kts = np.linspace(col.min(), col.max(), self.n_knots)
                elif self.knots == "quantile":
                    percentiles = np.linspace(0, 100, self.n_knots)
                    kts = np.nanpercentile(col, percentiles)
                    kts = np.unique(kts)  # handle ties
                else:
                    raise ValueError(f"Unknown knots strategy: {self.knots}")
                knots_list.append(kts)
        else:
            knots_arr = np.asarray(self.knots, dtype=np.float64)
            if knots_arr.ndim == 1:
                knots_list = [knots_arr for _ in range(n_features)]
            else:
                knots_list = [knots_arr[:, j] for j in range(n_features)]

        self.knots_ = knots_list
        self.boundary_lo_ = np.array([kts[0] for kts in knots_list])
        self.boundary_hi_ = np.array([kts[-1] for kts in knots_list])

        # Compute output dimension based on actual knot counts (handles ties)
        self._n_splines_per_feature = []
        for kts in knots_list:
            n_int = len(kts) - 2  # interior knots
            if n_int < 1:
                n_int = max(len(kts), 1)
            self._n_splines_per_feature.append(n_int + self.degree)

        # Use the minimum for consistent output dimension
        min_splines = min(self._n_splines_per_feature)
        if self.include_bias:
            self.n_features_out_ = n_features * min_splines
        else:
            self.n_features_out_ = n_features * (min_splines - 1)
        self._n_splines_per_feature = min_splines
        self._fitted = True
        return self

    def transform(self, X):
        """Transform X to B-spline basis features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features_out)
        """
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n_samples, n_features = X_np.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {n_features}"
            )

        blocks = []
        for j in range(n_features):
            col = X_np[:, j]
            kts = self.knots_[j]

            # Build augmented knot vector (Eilers & Marx style)
            # Interior knots = kts[1:-1], boundaries = kts[0], kts[-1]
            interior_knots = kts[1:-1] if len(kts) > 2 else kts

            B = bspline_basis(
                col, interior_knots, degree=self.degree,
                boundary_lo=kts[0], boundary_hi=kts[-1]
            )

            # Handle extrapolation
            if self.extrapolation == "constant":
                # Clamp values outside boundary to boundary basis values
                pass  # B-spline De Boor already handles this
            elif self.extrapolation == "error":
                mask_lo = col < kts[0]
                mask_hi = col > kts[-1]
                if np.any(mask_lo) or np.any(mask_hi):
                    raise ValueError(
                        "X contains values outside the fitted knot range "
                        "and extrapolation='error'"
                    )

            # Drop bias column if requested
            if not self.include_bias:
                B = B[:, :-1]

            blocks.append(B)

        # Concatenate across features
        X_out = np.hstack(blocks)

        # Convert to target backend
        return xp.asarray(X_out, dtype=xp.float64)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in one step."""
        return self.fit(X, y, sample_weight).transform(X)

    def predict(self, X):
        """Alias for transform (required by BaseEstimator)."""
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        self._check_is_fitted()
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = []
        n_splines = self._n_splines_per_feature if self.include_bias else self._n_splines_per_feature - 1
        for feat_name in input_features:
            for s in range(n_splines):
                names.append(f"{feat_name}_bspline{s}")
        return names

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["n_knots"] = self.n_knots
        params["degree"] = self.degree
        params["knots"] = self.knots
        params["include_bias"] = self.include_bias
        params["extrapolation"] = self.extrapolation
        return params

    def set_params(self, **params):
        for key in ["n_knots", "degree", "knots", "include_bias", "extrapolation"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
