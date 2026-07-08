"""
Base class for all penalty functions in statgpu.

The penalty framework supports:
- Convex penalties (L1, L2, Elastic Net)
- Non-convex penalties (SCAD, MCP) via LLA approximation
- Group penalties (Group Lasso, Sparse Group Lasso)
- Adaptive/weighted penalties
"""

__all__ = ["Penalty", "CompositePenalty"]


from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

from statgpu.backends._array_ops import _xp


class Penalty(ABC):
    """
    Abstract base class for regularization penalties.

    A penalty function P(w) defines the regularization term in penalized
    regression:

        minimize: (1/(2n)) * ||y - Xw||²₂ + P(w)

    Subclasses must implement:
    - value(coef): Compute P(w)
    - gradient(coef): Compute ∇P(w)
    - proximal(w, step, backend): Compute proximal operator

    For non-convex penalties (SCAD, MCP), also implement:
    - lla_weights(coef): LLA weights for local linear approximation
    """

    name: str = "base"
    is_convex: bool = True
    supports_group: bool = False
    requires_init: bool = False

    @abstractmethod
    def value(self, coef: np.ndarray) -> float:
        """
        Compute penalty value P(w).

        Parameters
        ----------
        coef : np.ndarray
            Coefficient vector.

        Returns
        -------
        float
            Penalty value.
        """
        pass

    @abstractmethod
    def gradient(self, coef: np.ndarray) -> np.ndarray:
        """
        Compute penalty gradient ∇P(w).

        Parameters
        ----------
        coef : np.ndarray
            Coefficient vector.

        Returns
        -------
        np.ndarray
            Gradient of penalty at coef.
        """
        pass

    def proximal(
        self,
        w: np.ndarray,
        step: float,
        backend: str = "numpy"
    ) -> np.ndarray:
        """
        Proximal operator: argmin_z { (1/2)||z - w||² + step * P(z) }

        Default implementation uses soft thresholding for L1-type penalties.
        Override for group penalties or non-convex penalties.

        Parameters
        ----------
        w : np.ndarray
            Input array (pre-proximal update).
        step : float
            Step size (typically 1/Lipschitz constant).
        backend : str, default='numpy'
            Backend: 'numpy', 'cupy', or 'torch'.

        Returns
        -------
        np.ndarray
            Result of proximal operator.
        """
        raise NotImplementedError(
            f"proximal() not implemented for {self.name}. "
            "Subclass must implement this method."
        )

    def lla_weights(self, coef: np.ndarray) -> np.ndarray:
        """
        Local Linear Approximation (LLA) weights for non-convex penalties.

        For a penalty P(w), the LLA approximates:
            P(w) ≈ P(coef) + Σ w_j * |w_j - coef_j|

        where w_j = P'(|coef_j|) for coef_j ≠ 0.

        This is used to solve non-convex penalties via iteratively
        reweighted L1.

        Parameters
        ----------
        coef : np.ndarray
            Current coefficient estimate.

        Returns
        -------
        array
            LLA weights (default: ones for convex L1).
        """
        xp = _xp(coef)
        return xp.ones_like(coef)

    def curvature_diag(self, coef: np.ndarray) -> np.ndarray:
        """Diagonal of second derivative P''(coef) for penalized sandwich inference.

        Returns a (p,) vector.  Default: zeros(p).
        L2 overrides: ``alpha * ones(p)``.
        SCAD/MCP: raises NotImplementedError (concave — use oracle/bootstrap).
        """
        xp = _xp(coef)
        return xp.zeros_like(coef)

    def get_params(self) -> dict:
        """
        Get penalty parameters for serialization.

        Returns
        -------
        dict
            Dictionary of penalty parameters.
        """
        return {"name": self.name}

    def _check_coef_shape(self, coef: np.ndarray) -> None:
        """Validate coefficient array shape."""
        if coef.ndim != 1:
            raise ValueError(f"coef must be 1D, got shape {coef.shape}")

    def __repr__(self) -> str:
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class CompositePenalty(Penalty):
    """
    Composite penalty combining multiple penalties.

    P(w) = Σ weight_i * P_i(w)

    This allows combining different penalty types, e.g.:
    - Group Lasso + L1 (Sparse Group Lasso)
    - Group Lasso + SCAD
    """

    name = "composite"
    is_convex = True  # Only if all component penalties are convex

    def __init__(
        self,
        penalties: list,
        weights: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        penalties : list of Penalty
            List of penalty objects.
        weights : list of float, optional
            Weight for each penalty. Default: equal weights.
        """
        self.penalties = penalties
        self.n_penalties = len(penalties)

        if weights is None:
            self.weights = [1.0 / self.n_penalties] * self.n_penalties
        else:
            if len(weights) != self.n_penalties:
                raise ValueError(
                    f"weights must have length {self.n_penalties}, "
                    f"got {len(weights)}"
                )
            self.weights = weights

        # Composite is convex only if all components are convex
        self.is_convex = all(p.is_convex for p in penalties)

        # Composite requires init if any component requires it
        self.requires_init = any(p.requires_init for p in penalties)

    def value(self, coef: np.ndarray) -> float:
        """Sum of weighted penalty values."""
        total = 0.0
        for w, pen in zip(self.weights, self.penalties):
            total += w * pen.value(coef)
        return total

    def gradient(self, coef):
        """Sum of weighted penalty gradients."""
        xp = _xp(coef)
        total = xp.zeros_like(coef)
        for w, pen in zip(self.weights, self.penalties):
            total = total + w * pen.gradient(coef)
        return total

    def proximal(
        self,
        w: np.ndarray,
        step: float,
        backend: str = "numpy"
    ) -> np.ndarray:
        """
        Proximal for composite penalty.

        Note: This is an approximation. The exact proximal for a sum
        of penalties is not the composition of individual proximals
        (unless they commute). For most practical cases (e.g., sparse
        group lasso), this approximation works well.
        """
        # Sequential application of proximal operators
        # (Dykstra-like splitting, simplified)
        result = w.clone() if hasattr(w, 'clone') else w.copy()
        for weight, pen in zip(self.weights, self.penalties):
            result = pen.proximal(result, step * weight, backend)
        return result

    def lla_weights(self, coef):
        """LLA weights: weighted sum of individual LLA weights.

        For composite penalty P(w) = sum_i w_i * P_i(w),
        the LLA weight is sum_i w_i * P_i'(|coef|).
        """
        xp = _xp(coef)
        if not any(not p.is_convex for p in self.penalties):
            return xp.ones_like(coef)

        result = xp.zeros_like(coef)
        for weight, pen in zip(self.weights, self.penalties):
            if not pen.is_convex:
                result = result + weight * pen.lla_weights(coef)
        return result

    def get_params(self) -> dict:
        params = {
            "name": "composite",
            "n_penalties": self.n_penalties,
            "penalties": [p.name for p in self.penalties],
            "weights": self.weights,
        }
        return params
