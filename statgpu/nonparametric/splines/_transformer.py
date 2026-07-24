"""Backend-native B-spline feature transformer."""

from __future__ import annotations

__all__ = ["SplineTransformer"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import (
    _get_xp,
    _to_float_scalar,
    xp_asarray,
    xp_astype,
    xp_full,
    xp_zeros,
)
from statgpu.covariance._empirical import _detect_backend


def _copy_array(x):
    return x.clone() if hasattr(x, "clone") else x.copy()


def _clip(x, lo, hi, xp):
    if xp.__name__ == "torch":
        return xp.clamp(x, min=lo, max=hi)
    return xp.clip(x, lo, hi)


def _finite_all(x, xp):
    return bool(_to_float_scalar(xp.all(xp.isfinite(x))))


def _stack(values, xp):
    return xp.stack(values, dim=0) if xp.__name__ == "torch" else xp.stack(values, axis=0)


def _concatenate(values, xp, axis=0):
    return xp.cat(values, dim=axis) if xp.__name__ == "torch" else xp.concatenate(values, axis=axis)


class SplineTransformer(BaseEstimator):
    """B-spline feature transformer with native NumPy/CuPy/Torch evaluation."""

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
        mode = str(self.extrapolation).lower()
        if mode not in {"error", "constant", "linear", "continue"}:
            raise ValueError(
                "extrapolation must be one of 'error', 'constant', 'linear', or 'continue'"
            )
        self._extrapolation_ = mode

    def _prepare_X(self, X, expected_features=None):
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        ref = None
        if backend_name == "torch":
            import torch

            if isinstance(X, torch.Tensor):
                ref = X
            else:
                dev = self._get_compute_device()
                target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                ref = torch.empty(0, dtype=torch.float64, device=target)
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=ref)
        if X_arr.ndim == 1:
            if expected_features is None or expected_features == 1:
                X_arr = X_arr.reshape(-1, 1)
            elif int(X_arr.size) == expected_features:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise ValueError("X shape is incompatible with fitted feature count")
        if X_arr.ndim != 2 or int(X_arr.shape[0]) == 0 or int(X_arr.shape[1]) == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if expected_features is not None and int(X_arr.shape[1]) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {int(X_arr.shape[1])}")
        if not _finite_all(X_arr, xp):
            raise ValueError("X must contain only finite values")
        return backend_name, xp, X_arr

    def _linspace(self, lo, hi, count, xp, ref):
        if xp.__name__ == "torch":
            return xp.linspace(lo, hi, count, dtype=xp.float64, device=ref.device)
        return xp.linspace(lo, hi, count, dtype=xp.float64)

    def _build_knots(self, X, xp):
        n_features = int(X.shape[1])
        result = []
        if isinstance(self.knots, str):
            strategy = self.knots.lower()
            if strategy not in {"uniform", "quantile"}:
                raise ValueError("knots must be 'uniform', 'quantile', or an array")
            for j in range(n_features):
                col = X[:, j]
                if strategy == "uniform":
                    values = self._linspace(
                        _to_float_scalar(xp.min(col)),
                        _to_float_scalar(xp.max(col)),
                        int(self.n_knots),
                        xp,
                        X,
                    )
                else:
                    q = self._linspace(0.0, 1.0, int(self.n_knots), xp, X)
                    values = xp.quantile(col, q)
                n_unique = int(xp.unique(values).numel()) if xp.__name__ == "torch" else int(xp.unique(values).size)
                if n_unique != int(self.n_knots):
                    raise ValueError(
                        "each feature must provide n_knots distinct knot values; "
                        "quantile ties or constant features are not supported"
                    )
                result.append(values)
            return result

        custom = xp_asarray(self.knots, dtype=xp.float64, xp=xp, ref_arr=X)
        expected = (int(self.n_knots), n_features)
        if custom.ndim == 1:
            if n_features != 1 or int(custom.shape[0]) != int(self.n_knots):
                raise ValueError(f"custom knots must have shape {expected}")
            custom = custom.reshape(-1, 1)
        if tuple(custom.shape) != expected:
            raise ValueError(f"custom knots must have shape {expected}")
        if not _finite_all(custom, xp):
            raise ValueError("custom knots must contain only finite values")
        for j in range(n_features):
            values = custom[:, j]
            if bool(_to_float_scalar(xp.any((values[1:] - values[:-1]) <= 0))):
                raise ValueError("custom knots must be strictly increasing for every feature")
            result.append(_copy_array(values))
        return result

    def fit(self, X, y=None, sample_weight=None):
        self._validate_parameters()
        backend_name, xp, X_arr = self._prepare_X(X)
        self.n_features_in_ = int(X_arr.shape[1])
        self.knots_ = self._build_knots(X_arr, xp)
        self.boundary_lo_ = _stack([k[0] for k in self.knots_], xp)
        self.boundary_hi_ = _stack([k[-1] for k in self.knots_], xp)
        self._n_splines_per_feature = int(self.n_knots) + int(self.degree) - 1
        block_width = self._n_splines_per_feature - (0 if self.include_bias else 1)
        if block_width < 1:
            raise ValueError("degree/n_knots/include_bias produce no output features")
        self.n_features_out_ = self.n_features_in_ * block_width
        self._backend_name = backend_name
        self._xp = xp
        self._fit_ref_ = X_arr
        self._fitted = True
        return self

    def _augmented_knots(self, knots, xp, ref):
        degree = int(self.degree)
        left = xp_full(degree + 1, _to_float_scalar(knots[0]), xp.float64, xp, ref)
        right = xp_full(degree + 1, _to_float_scalar(knots[-1]), xp.float64, xp, ref)
        return _concatenate([left, knots[1:-1], right], xp, axis=0)

    def _basis_degree(self, x_weight, selector, augmented, degree, xp):
        n = int(x_weight.shape[0])
        n_intervals = int(augmented.shape[0]) - 1
        B = xp_zeros((n, n_intervals), xp.float64, xp, x_weight)
        last_nondegenerate = int(augmented.shape[0]) - int(self.degree) - 2

        for i in range(n_intervals):
            lo = augmented[i]
            hi = augmented[i + 1]
            if _to_float_scalar(hi - lo) <= 0.0:
                continue
            if i == last_nondegenerate:
                mask = (selector >= lo) & (selector <= hi)
            else:
                mask = (selector >= lo) & (selector < hi)
            B[:, i] = xp_astype(mask, xp.float64, xp)

        for current_degree in range(1, degree + 1):
            n_cur = n_intervals - current_degree
            t_lo = augmented[:n_cur]
            t_hi = augmented[current_degree: current_degree + n_cur]
            t_next = augmented[1: n_cur + 1]
            t_next_hi = augmented[current_degree + 1: current_degree + 1 + n_cur]
            denom1 = t_hi - t_lo
            denom2 = t_next_hi - t_next
            safe1 = xp.where(denom1 > 0, denom1, xp.ones_like(denom1))
            safe2 = xp.where(denom2 > 0, denom2, xp.ones_like(denom2))
            w1 = xp.where(
                (denom1 > 0)[None, :],
                (x_weight[:, None] - t_lo[None, :]) / safe1[None, :],
                xp.zeros_like(B[:, :n_cur]),
            )
            w2 = xp.where(
                (denom2 > 0)[None, :],
                (t_next_hi[None, :] - x_weight[:, None]) / safe2[None, :],
                xp.zeros_like(B[:, :n_cur]),
            )
            B = w1 * B[:, :n_cur] + w2 * B[:, 1:n_cur + 1]
        return B

    def _basis_derivative(self, x, augmented, xp):
        degree = int(self.degree)
        n_basis = int(augmented.shape[0]) - degree - 1
        if degree == 0:
            return xp_zeros((int(x.shape[0]), n_basis), xp.float64, xp, x)
        lower = self._basis_degree(x, x, augmented, degree - 1, xp)
        out = xp_zeros((int(x.shape[0]), n_basis), xp.float64, xp, x)
        for i in range(n_basis):
            denom1 = _to_float_scalar(augmented[i + degree] - augmented[i])
            denom2 = _to_float_scalar(augmented[i + degree + 1] - augmented[i + 1])
            if denom1 > 0:
                out[:, i] += degree / denom1 * lower[:, i]
            if denom2 > 0:
                out[:, i] -= degree / denom2 * lower[:, i + 1]
        return out

    def _basis(self, values, knots, xp):
        lo = _to_float_scalar(knots[0])
        hi = _to_float_scalar(knots[-1])
        mode = self._extrapolation_
        outside = (values < lo) | (values > hi)
        if mode == "error" and bool(_to_float_scalar(xp.any(outside))):
            raise ValueError(
                "X contains values outside the fitted knot range and extrapolation='error'"
            )

        augmented = self._augmented_knots(knots, xp, values)
        selector = _clip(values, lo, hi, xp)
        if mode == "continue":
            basis = self._basis_degree(values, selector, augmented, int(self.degree), xp)
        else:
            basis = self._basis_degree(selector, selector, augmented, int(self.degree), xp)

        if mode == "linear" and bool(_to_float_scalar(xp.any(outside))):
            left_x = xp_asarray([lo], dtype=xp.float64, xp=xp, ref_arr=values)
            right_x = xp_asarray([hi], dtype=xp.float64, xp=xp, ref_arr=values)
            left_basis = self._basis_degree(left_x, left_x, augmented, int(self.degree), xp)[0]
            right_basis = self._basis_degree(right_x, right_x, augmented, int(self.degree), xp)[0]
            left_derivative = self._basis_derivative(left_x, augmented, xp)[0]
            right_derivative = self._basis_derivative(right_x, augmented, xp)[0]
            left_extension = left_basis[None, :] + (values - lo)[:, None] * left_derivative[None, :]
            right_extension = right_basis[None, :] + (values - hi)[:, None] * right_derivative[None, :]
            basis = xp.where((values < lo)[:, None], left_extension, basis)
            basis = xp.where((values > hi)[:, None], right_extension, basis)

        if not self.include_bias:
            basis = basis[:, :-1]
        return basis

    def transform(self, X):
        self._check_is_fitted()
        backend_name, xp, X_arr = self._prepare_X(X, self.n_features_in_)
        if backend_name != self._backend_name:
            self.knots_ = [xp_asarray(k, dtype=xp.float64, xp=xp, ref_arr=X_arr) for k in self.knots_]
            self.boundary_lo_ = xp_asarray(self.boundary_lo_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
            self.boundary_hi_ = xp_asarray(self.boundary_hi_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
            self._backend_name = backend_name
            self._xp = xp
        blocks = [self._basis(X_arr[:, j], self.knots_[j], xp) for j in range(self.n_features_in_)]
        X_out = _concatenate(blocks, xp, axis=1)
        if int(X_out.shape[1]) != self.n_features_out_:
            raise RuntimeError(
                f"internal spline dimension mismatch: expected {self.n_features_out_}, got {int(X_out.shape[1])}"
            )
        return X_out

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
