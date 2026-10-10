"""Empirical covariance estimation with GPU support."""

from __future__ import annotations

__all__ = ["EmpiricalCovariance"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device, _get_configured_device
from statgpu.backends import (
    _LINALG_ERRORS,
    _is_cupy_array,
    _is_torch_array,
    _to_float_scalar,
    _to_numpy,
    get_backend,
    xp_asarray,
    xp_eye,
    xp_zeros,
)
from statgpu.backends._utils import _cupy_asarray_on_device
from statgpu.backends._validation import (
    _tag_finite_backend,
    _torch_cuda_device_label,
    check_finite,
)


def _torch_to_covariance_numpy(X):
    """Promote real Torch values on the CPU destination before NumPy conversion."""
    import torch

    return X.detach().cpu().to(dtype=torch.float64).resolve_neg().numpy()


def _detect_backend(X, device: Device) -> str:
    """Resolve backend from input array type, falling back to device setting."""
    if _is_torch_array(X):
        return "torch"
    if _is_cupy_array(X):
        return "cupy"
    # For numpy input, use device-based resolution
    if device == Device.TORCH:
        return "torch"
    if device == Device.CUDA:
        try:
            import cupy as cp  # noqa: F401
            return "cupy"
        except ImportError:
            raise RuntimeError(
                "CuPy is required for device='cuda' but is not installed. "
                "Use device='auto' to fall back to CPU automatically."
            )
    return "numpy"


def _torch_device_from_data(X) -> Optional[str]:
    """Extract torch device from tensor, or None for non-torch inputs."""
    try:
        import torch
        if isinstance(X, torch.Tensor):
            return str(X.device)
    except (ImportError, AttributeError):
        pass
    return None


def _validate_covariance_input(X_arr, xp, *, min_samples=1):
    """Validate shape and finiteness without transferring the full array."""
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n_samples, n_features = map(int, X_arr.shape)
    if n_samples < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples to estimate covariance, got {n_samples}"
        )
    if n_features < 1:
        raise ValueError("X must contain at least one feature")
    if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
        raise ValueError("X must contain only finite values")
    return n_samples, n_features


class EmpiricalCovariance(BaseEstimator):
    """
    Maximum likelihood covariance estimator with GPU acceleration.

    Computes the sample covariance matrix, its inverse (precision), and
    provides log-likelihood scoring and Mahalanobis distance computation.

    Parameters
    ----------
    assume_centered : bool, default=False
        If True, data is assumed to be already centered. If False, the
        mean is estimated and subtracted before computing the covariance.
    device : str or Device, default='auto'
        ``'cpu'`` selects NumPy, ``'cuda'`` selects CuPy CUDA, and
        ``'torch'`` selects Torch CUDA, regardless of input array type.
        An unavailable explicit GPU backend raises an error. ``'auto'`` uses
        the global device policy; when that is also automatic, native input
        arrays retain their backend (including Torch CPU). Other inputs use
        the available CuPy CUDA, Torch CUDA, or NumPy backend, in that order.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : array, shape (n_features, n_features)
        Estimated covariance matrix.
    location_ : array, shape (n_features,)
        Estimated location (mean) vector.
    precision_ : array, shape (n_features, n_features)
        Estimated precision matrix (inverse covariance).
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.assume_centered = assume_centered

    def _resolve_covariance_input_policy(self, X, *, fitted=False):
        """Select the same backend for prevalidation and numerical preparation."""
        if fitted:
            # Evaluation follows the fitted arrays, including their CUDA index,
            # rather than reselecting from a test array or a changed global policy.
            backend_name = self._backend_name
            ref = self.covariance_
            device = (
                Device.CPU if backend_name == "numpy"
                else Device.CUDA if backend_name == "cupy"
                else Device.TORCH if ref.is_cuda else Device.CPU
            )
        else:
            requested = self._device
            if requested == Device.AUTO:
                requested = _get_configured_device()
            # Retain native-array selection only for genuinely automatic policy.
            # In particular, an automatic Torch CPU input remains Torch CPU.
            if requested == Device.AUTO and _is_torch_array(X):
                backend_name = "torch"
                device = Device.TORCH if X.is_cuda else Device.CPU
            elif requested == Device.AUTO and _is_cupy_array(X):
                backend_name, device = "cupy", Device.CUDA
            else:
                device = self._get_compute_device()
                backend_name = {
                    Device.CPU: "numpy", Device.CUDA: "cupy", Device.TORCH: "torch"
                }[device]
            ref = None
        return backend_name, device, ref

    def _check_public_input_finite(self, value, *, name, method_name):
        # Keep shared validation and unsupported representations unchanged.
        # Only covariance's X entrypoints need its float64 working dtype before
        # the public guard: CUDA isfinite does not support float8_e5m2 on some
        # supported Torch releases, even though float64 conversion is available.
        if (
            name != "X"
            or method_name not in {"fit", "score", "predict", "mahalanobis"}
            or not _is_torch_array(value)
        ):
            return super()._check_public_input_finite(
                value, name=name, method_name=method_name
            )
        import torch

        supported_dtypes = {
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
            torch.int64, torch.float16, torch.bfloat16, torch.float32,
            torch.float64, getattr(torch, "float8_e5m2", None),
        }
        if (
            value.layout != torch.strided
            or value.is_quantized
            or value.dtype not in supported_dtypes
        ):
            return super()._check_public_input_finite(
                value, name=name, method_name=method_name
            )

        try:
            backend_name, _, _ = self._resolve_covariance_input_policy(
                value, fitted=method_name != "fit" and self._fitted
            )
            if backend_name == "numpy":
                prepared = _torch_to_covariance_numpy(value)
            elif (
                backend_name == "torch"
                and value.dtype == getattr(torch, "float8_e5m2", None)
            ):
                # Widen on the source device, preserving native CUDA ownership
                # and the Torch input type used by automatic backend selection.
                prepared = value.detach().to(dtype=torch.float64).resolve_neg()
            else:
                prepared = value
            check_finite(prepared, name=name)
            # Reuse this call-local normalization in the numerical method and
            # nested predict -> mahalanobis guards. Never cache caller arrays
            # on the estimator: a later call must validate its current values.
            return prepared
        except Exception as exc:
            # CPU-bound validation must retain the original CUDA provenance,
            # including conversion errors before the finite reduction.
            if value.is_cuda:
                _tag_finite_backend(
                    exc, "torch", device=_torch_cuda_device_label(value.device)
                )
            raise

    def _prepare_covariance_input(self, X, *, fitted=False):
        """Resolve estimator policy before converting input to numerical arrays."""
        backend_name, device, ref = self._resolve_covariance_input_policy(
            X, fitted=fitted
        )
        if backend_name == "torch" and device == Device.CPU:
            backend = get_backend(backend="torch", device="cpu")
        else:
            # Resolve a concrete library: factory auto-selection must not
            # substitute another backend for an explicit or global request.
            backend = get_backend(
                backend=backend_name,
                device="cpu" if device == Device.CPU else "cuda",
            )
        if not backend.is_available():
            label = "torch" if backend_name == "torch" else "cuda"
            requirement = "PyTorch CUDA" if backend_name == "torch" else "CuPy CUDA"
            raise RuntimeError(
                f"device='{label}' requires a working {requirement} backend. "
                "Use device='cpu' or reset the global device and use device='auto'."
            )

        # Only actual backend arrays need device transfer. Ordinary array-like
        # objects can also expose methods such as pandas' keyed ``get``; those
        # methods must not be mistaken for CuPy/Torch transfer operations.
        # Apply the covariance float64 dtype before transfer as well: numeric
        # pandas extension columns can expose an object-dtype NumPy array.
        # Native arrays retain their selected backend and device. A real Torch
        # input targeting NumPy needs normalization before Tensor.numpy():
        # NumPy cannot represent bfloat16/float8 dtypes or a lazy negative view.
        # Detach without mutating the input. Cast on the requested CPU target:
        # the source device need not support float64 (for example, Torch MPS).
        if backend_name == "numpy" and _is_torch_array(X) and not X.is_complex():
            X = _torch_to_covariance_numpy(X)
        elif not _is_cupy_array(X) and not _is_torch_array(X):
            X = np.asarray(X, dtype=np.float64)

        xp = backend.xp
        if fitted and backend_name == "cupy":
            # Allocate CPU queries directly on the fitted GPU; an already-native
            # query on another GPU needs an explicit device-to-device copy.
            with ref.device:
                X_arr = self._to_array(X, device=device, backend=backend_name)
                X_arr = _cupy_asarray_on_device(X_arr, ref.device.id, dtype=xp.float64)
        else:
            if fitted and backend_name == "torch":
                X_arr = self._to_torch(X, device=str(ref.device))
            else:
                X_arr = self._to_array(X, device=device, backend=backend_name)
            X_arr = xp_asarray(
                X_arr, dtype=xp.float64, xp=xp,
                ref_arr=ref if fitted else X_arr,
            )
        return backend_name, xp, X_arr

    def fit(self, X, y=None):
        """Fit the covariance model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API compatibility.

        Returns
        -------
        self
        """
        backend_name, xp, X_arr = self._prepare_covariance_input(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = _validate_covariance_input(
            X_arr, xp, min_samples=2
        )

        # Center if needed
        if self.assume_centered:
            location = xp_zeros(n_features, xp.float64, xp, X_arr)
        else:
            location = xp.mean(X_arr, axis=0)
            X_arr = X_arr - location

        # Sample covariance: S = X^T X / n
        S = (X_arr.T @ X_arr) / float(n_samples)

        # Compute precision (inverse) with jitter stabilization
        precision = _stable_inv(S, xp, backend_name)

        self.covariance_ = S
        self.location_ = location
        self.precision_ = precision
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        self._backend_name = backend_name
        self._fitted = True
        return self

    def predict(self, X):
        """Return Mahalanobis distances for *X* under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        distances : ndarray of shape (n_samples,)
        """
        return self.mahalanobis(X)

    def score(self, X, y=None):
        """Compute the average log-likelihood of *X* under the fitted Gaussian.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : ignored

        Returns
        -------
        ll : float
            Average log-likelihood per observation.
        """
        self._check_is_fitted()
        _backend_name, xp, X_arr = self._prepare_covariance_input(X, fitted=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, p = _validate_covariance_input(X_arr, xp, min_samples=1)
        if p != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features, got {p}")

        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        prec = xp_asarray(self.precision_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        cov = xp_asarray(self.covariance_, dtype=xp.float64, xp=xp, ref_arr=X_arr)

        X_centered = X_arr - loc

        # Mahalanobis term: sum of (x-mu)^T S^{-1} (x-mu)
        M = X_centered @ prec
        mahal_sum = _to_float_scalar(xp.sum(M * X_centered))

        # log(det(S)) via slogdet for numerical stability
        sign, logdet = xp.linalg.slogdet(cov)
        sign_val = _to_float_scalar(sign)
        if sign_val <= 0:
            return float("-inf")
        logdet_val = _to_float_scalar(logdet)

        # Average log-likelihood:
        #   LL = -(1/2) * (p * log(2*pi) + log(det(S)) + (1/n) * sum(mahal))
        ll = -0.5 * (p * np.log(2.0 * np.pi) + logdet_val + mahal_sum / n_samples)
        return float(ll)

    def mahalanobis(self, X):
        """Compute Mahalanobis distances of observations in *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Squared Mahalanobis distances.
        """
        self._check_is_fitted()
        _backend_name, xp, X_arr = self._prepare_covariance_input(X, fitted=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        _n_samples, n_features = _validate_covariance_input(
            X_arr, xp, min_samples=1
        )
        if n_features != self.n_features_:
            raise ValueError(
                f"X must have {self.n_features_} features, got {n_features}"
            )

        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        prec = xp_asarray(self.precision_, dtype=xp.float64, xp=xp, ref_arr=X_arr)

        X_centered = X_arr - loc

        # Efficient: row-wise (x-mu)^T prec (x-mu)
        M = X_centered @ prec
        mahal = xp.sum(M * X_centered, axis=1)

        return _to_numpy(mahal)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["assume_centered"] = self.assume_centered
        return params

    def set_params(self, **params):
        for key, value in list(params.items()):
            if key == "assume_centered":
                self.assume_centered = value
                del params[key]
        if params:
            super().set_params(**params)
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_inv(S, xp, backend_name: str):
    """Invert *S* with jitter-boosted diagonal for numerical stability.

    Tries the exact inverse first; if that fails or produces non-finite
    values, adds progressively larger diagonal jitter.
    """
    p = int(S.shape[0])

    trace_S = _to_float_scalar(xp.trace(S))
    base = max(abs(trace_S) / max(p, 1), 1.0) * 1e-10

    # Allocate on the covariance device, including non-default CUDA indices.
    eye = xp_eye(p, xp.float64, xp, S)

    # Preserve the exact estimator whenever the covariance is invertible.
    # Jitter is a fallback, not part of the empirical covariance definition.
    try:
        inv_S = xp.linalg.inv(S)
        test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
        if np.isfinite(test_val):
            return inv_S
    except _LINALG_ERRORS + (ValueError,):
        pass

    jitter = base
    for _ in range(12):
        try:
            inv_S = xp.linalg.inv(S + jitter * eye)
            test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
            if np.isfinite(test_val):
                return inv_S
        except _LINALG_ERRORS + (ValueError,):
            pass
        jitter *= 10.0

    raise ValueError(
        "Covariance matrix is singular and cannot be inverted even with "
        "diagonal jitter. Consider using LedoitWolf or OAS shrinkage."
    )
