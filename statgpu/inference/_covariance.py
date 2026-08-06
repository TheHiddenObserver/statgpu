"""Strict covariance-spectrum and joint-Wald inference policy.

The helpers are NumPy based because inference result containers are
backend-neutral. GPU estimators transfer only the fitted ``p x p`` covariance
matrix at this boundary; training data and score residuals remain on device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_ROBUST_COVARIANCE_TYPES = frozenset({"hc0", "hc1", "cluster"})
_POSITIVE_DEFINITE = "positive_definite"
_RANK_DEFICIENT_PSD = "rank_deficient_psd"
_MATERIALLY_INDEFINITE = "materially_indefinite"
_ROBUST_WALD_RANK_FAILURE = (
    "robust covariance is rank-deficient for the full-parameter Wald test"
)


@dataclass(frozen=True)
class CovarianceSpectrum:
    """One eigendecomposition and its scale-aware numerical classification."""

    covariance: np.ndarray
    eigenvalues: np.ndarray
    tolerance: float
    classification: str

    @property
    def minimum_eigenvalue(self) -> float:
        return float(self.eigenvalues[0])

    @property
    def maximum_absolute_eigenvalue(self) -> float:
        return float(np.max(np.abs(self.eigenvalues)))


def classify_covariance_spectrum(
    covariance,
    *,
    tolerance: Optional[float] = None,
) -> CovarianceSpectrum:
    """Classify a finite symmetric covariance as PD, singular PSD, or invalid."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or covariance.shape[0] < 1
        or not np.all(np.isfinite(covariance))
    ):
        raise RuntimeError("covariance must be a finite non-empty square matrix")

    covariance = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("covariance eigendecomposition failed") from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("covariance eigenspectrum is non-finite")

    spectral_scale = max(
        np.finfo(np.float64).tiny,
        float(np.max(np.abs(eigenvalues))),
    )
    if tolerance is None:
        tolerance = spectral_scale * max(int(covariance.shape[0]), 1) * 1e-12
    else:
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")

    minimum = float(eigenvalues[0])
    if minimum < -tolerance:
        classification = _MATERIALLY_INDEFINITE
    elif minimum <= tolerance:
        classification = _RANK_DEFICIENT_PSD
    else:
        classification = _POSITIVE_DEFINITE
    return CovarianceSpectrum(
        covariance=covariance,
        eigenvalues=eigenvalues,
        tolerance=float(tolerance),
        classification=classification,
    )


def _raise_if_materially_indefinite(
    spectrum: CovarianceSpectrum,
    *,
    cov_type: str,
) -> None:
    if spectrum.classification == _MATERIALLY_INDEFINITE:
        raise RuntimeError(
            f"{cov_type} covariance is not positive semidefinite "
            f"(minimum eigenvalue={spectrum.minimum_eigenvalue:.6g}; "
            f"tolerance={spectrum.tolerance:.6g})"
        )


def standard_errors_from_covariance(
    covariance,
    *,
    cov_type: str,
    spectrum: Optional[CovarianceSpectrum] = None,
) -> np.ndarray:
    """Return strict marginal standard errors after full-spectrum validation."""
    if spectrum is None:
        spectrum = classify_covariance_spectrum(covariance)
    diagonal = np.diag(spectrum.covariance).copy()
    if np.any(diagonal < -spectrum.tolerance):
        minimum = float(np.min(diagonal))
        raise RuntimeError(
            f"{cov_type} covariance has a materially negative diagonal "
            f"entry ({minimum:.6g}; tolerance={spectrum.tolerance:.6g})"
        )
    _raise_if_materially_indefinite(spectrum, cov_type=str(cov_type).lower())

    # Roundoff-level negative values are compatible with a singular PSD matrix.
    diagonal[diagonal < 0.0] = 0.0
    if str(cov_type).lower() in _ROBUST_COVARIANCE_TYPES and np.any(
        diagonal <= 0.0
    ):
        raise RuntimeError(
            f"{cov_type} covariance produced a non-positive marginal variance"
        )
    return np.sqrt(diagonal)


def joint_wald_from_covariance(
    coef,
    covariance,
    *,
    cov_type: str,
    tolerance: Optional[float] = None,
    spectrum: Optional[CovarianceSpectrum] = None,
):
    """Return a strict full-parameter Wald statistic and failure reason."""
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef.size < 1 or not np.all(np.isfinite(coef)):
        return np.nan, "joint Wald coefficients must be finite and non-empty"
    if spectrum is None:
        spectrum = classify_covariance_spectrum(
            covariance,
            tolerance=tolerance,
        )
    elif tolerance is not None:
        raise ValueError("tolerance cannot be supplied with a classified spectrum")
    if spectrum.covariance.shape != (coef.size, coef.size):
        return np.nan, "joint Wald inputs must be dimensionally consistent"

    cov_type = str(cov_type).lower()
    _raise_if_materially_indefinite(spectrum, cov_type=cov_type)
    if spectrum.classification == _RANK_DEFICIENT_PSD:
        reason = (
            _ROBUST_WALD_RANK_FAILURE
            if cov_type in _ROBUST_COVARIANCE_TYPES
            else "covariance is rank-deficient for the full-parameter Wald test"
        )
        return np.nan, reason

    try:
        statistic = float(
            coef @ np.linalg.solve(spectrum.covariance, coef)
        )
    except np.linalg.LinAlgError:
        return np.nan, "covariance solve failed for the full-parameter Wald test"
    if not np.isfinite(statistic) or statistic < 0.0:
        return np.nan, "full-parameter Wald statistic is non-finite or negative"
    return statistic, None


__all__ = [
    "CovarianceSpectrum",
    "classify_covariance_spectrum",
    "joint_wald_from_covariance",
    "standard_errors_from_covariance",
]
