"""sklearn-compatible SplineTransformer with GPU-compatible output."""

from __future__ import annotations

__all__ = ["SplineTransformer"]

from typing import Optional, Union

import numpy as np
from scipy.interpolate import BSpline

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy, xp_asarray


class SplineTransformer(BaseEstimator):
    """B-spline feature transformer with explicit extrapolation semantics.

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots including the two boundary knots.
    degree : int, default=3
        Spline polynomial degree.
    knots : {'uniform', 'quantile'} or array-like, default='uniform'
        Knot placement strategy or an array of shape
        ``(n_knots, n_features)``.
    include_bias : bool, default=True
        Retain all basis columns when True; otherwise drop the final column
        from each feature block.
    extrapolation : {'error', 'constant', 'linear', 'continue'}, default='constant'
        Behavior outside the fitted boundary knots.
    device : str or Device, default='auto'
        Output computation device.
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

    def _validate_parameters(self):
        if isinstance(self.n_knots, bool) or not isinstance(self.n_knots, (int, np.integer)):
            raise ValueError("n_knots must be an integer")
        if int(self.n_knots) < 3:
            raise ValueError("n_knots must be at least 3")
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)):
            raise ValueError("degree must be an integer")
        if int(self.degree) < 0:
            raise ValueError("degree must be non-negative")
        extrapolation = str(self.extrapolation).lower()
        if extrapolation not in {"error", "constant", "linear", "continue"}:
            raise ValueError(
                "extrapolation must be one of 'error', 'constant', 'linear', or 'continue'"
            )
        self._extrapolation_ = extrapolation

    @staticmethod
    def _validate_X_numpy(X, *, expected_features=None):
        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        if X_np.ndim == 1:
            if expected_features is None or expected_features == 1:
                X_np = X_np.reshape(-1, 1)
            elif X_np.size == expected_features:
                X_np = X_np.reshape(1, -1)
            else:
                raise ValueError("X shape is incompatible with fitted feature count")
        if X_np.ndim != 2 or X_np.shape[0] == 0 or X_np.shape[1] == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if expected_features is not None and X_np.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X_np.shape[1]}")
        if not np.all(np.isfinite(X_np)):
            raise ValueError("X must contain only finite values")
        return X_np

    def _build_knots(self, X_np):
        n_features = X_np.shape[1]
        if isinstance(self.knots, str):
            strategy = self.knots.lower()
            if strategy not in {"uniform", "quantile"}:
                raise ValueError("knots must be 'uniform', 'quantile', or an array")
            result = []
            for j in range(n_features):
                col = X_np[:, j]
                if strategy == "uniform":
                    values = np.linspace(col.min(), col.max(), int(self.n_knots))
                else:
                    q = np.linspace(0.0, 100.0, int(self.n_knots))
                    values = np.percentile(col, q)
                if np.unique(values).size != int(self.n_knots):
                    raise ValueError(
                        "each feature must provide n_knots distinct knot values; "
                        "quantile ties or constant features are not supported"
                    )
                result.append(values.astype(np.float64, copy=False))
            return result

        knots_arr = np.asarray(self.knots, dtype=np.float64)
        expected = (int(self.n_knots), n_features)
        if knots_arr.ndim == 1:
            if n_features != 1 or knots_arr.shape[0] != int(self.n_knots):
                raise ValueError(f"custom knots must have shape {expected}")
            knots_arr = knots_arr.reshape(-1, 1)
        if knots_arr.shape != expected:
            raise ValueError(f"custom knots must have shape {expected}")
        if not np.all(np.isfinite(knots_arr)):
            raise ValueError("custom knots must contain only finite values")
        result = []
        for j in range(n_features):
            values = knots_arr[:, j]
            if np.any(np.diff(values) <= 0):
                raise ValueError("custom knots must be strictly increasing for every feature")
            result.append(values.copy())
        return result

    def fit(self, X, y=None, sample_weight=None):
        """Learn knot locations from X."""
        self._validate_parameters()
        X_np = self._validate_X_numpy(X)
        self.n_features_in_ = int(X_np.shape[1])
        self.knots_ = self._build_knots(X_np)
        self.boundary_lo_ = np.asarray([k[0] for k in self.knots_], dtype=np.float64)
        self.boundary_hi_ = np.asarray([k[-1] for k in self.knots_], dtype=np.float64)
        self._n_splines_per_feature = int(self.n_knots) + int(self.degree) - 1
        block_width = self._n_splines_per_feature - (0 if self.include_bias else 1)
        if block_width < 1:
            raise ValueError("degree/n_knots/include_bias produce no output features")
        self.n_features_out_ = self.n_features_in_ * block_width
        self._fitted = True
        return self

    def _basis_numpy(self, values, knots):
        degree = int(self.degree)
        augmented = np.concatenate(
            [
                np.repeat(knots[0], degree + 1),
                knots[1:-1],
                np.repeat(knots[-1], degree + 1),
            ]
        )
        n_basis = len(augmented) - degree - 1
        coefficients = np.eye(n_basis, dtype=np.float64)
        spline = BSpline(augmented, coefficients, degree, extrapolate=True)
        lo, hi = float(knots[0]), float(knots[-1])
        mode = self._extrapolation_

        if mode == "error":
            if np.any(values < lo) or np.any(values > hi):
                raise ValueError(
                    "X contains values outside the fitted knot range and extrapolation='error'"
                )
            basis = spline(values)
        elif mode == "constant":
            basis = spline(np.clip(values, lo, hi))
        elif mode == "continue":
            basis = spline(values)
        else:  # linear
            clipped = np.clip(values, lo, hi)
            basis = spline(clipped)
            derivative = spline.derivative(1)
            left = values < lo
            right = values > hi
            if np.any(left):
                basis[left] = spline(lo) + (values[left] - lo)[:, None] * derivative(lo)
            if np.any(right):
                basis[right] = spline(hi) + (values[right] - hi)[:, None] * derivative(hi)

        if not self.include_bias:
            basis = basis[:, :-1]
        return np.asarray(basis, dtype=np.float64)

    def transform(self, X):
        """Transform X into concatenated B-spline basis blocks."""
        self._check_is_fitted()
        X_np = self._validate_X_numpy(X, expected_features=self.n_features_in_)
        blocks = [self._basis_numpy(X_np[:, j], self.knots_[j]) for j in range(self.n_features_in_)]
        X_out = np.hstack(blocks)
        if X_out.shape[1] != self.n_features_out_:
            raise RuntimeError(
                f"internal spline dimension mismatch: expected {self.n_features_out_}, "
                f"got {X_out.shape[1]}"
            )
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        return xp_asarray(X_out, dtype=xp.float64, xp=xp)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def predict(self, X):
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if len(input_features) != self.n_features_in_:
            raise ValueError(
                f"input_features must have length {self.n_features_in_}, got {len(input_features)}"
            )
        width = self._n_splines_per_feature - (0 if self.include_bias else 1)
        return [f"{name}_bspline{j}" for name in input_features for j in range(width)]

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            n_knots=self.n_knots,
            degree=self.degree,
            knots=self.knots,
            include_bias=self.include_bias,
            extrapolation=self.extrapolation,
        )
        return params

    def set_params(self, **params):
        for key in ["n_knots", "degree", "knots", "include_bias", "extrapolation"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
