"""Host-side regression diagnostics for fitted statgpu models.

Diagnostics intentionally copy the fitted design and residual vectors to NumPy:
they are reporting/influence utilities built on SciPy statistics rather than a
model-training path. The full copy is performed once during construction.
"""

from __future__ import annotations

__all__ = ["RegressionDiagnostics", "diagnose_model"]

import numpy as np
from scipy import stats

from statgpu.backends import _to_numpy


class RegressionDiagnostics:
    """Residual, leverage, influence, and multicollinearity diagnostics.

    The fitted model must expose ``_X_design``, ``_y``, and ``_resid``. A finite
    residual variance in ``_scale`` is preferred; otherwise it is estimated from
    the residual degrees of freedom.
    """

    def __init__(self, model):
        self.model = model
        self._validate_and_snapshot_model()
        self._leverage_cache = None

    def _validate_and_snapshot_model(self) -> None:
        required = ("_resid", "_X_design", "_y")
        missing = [
            name
            for name in required
            if not hasattr(self.model, name) or getattr(self.model, name) is None
        ]
        if missing:
            raise ValueError(
                "Model is missing fitted diagnostic attributes: " + ", ".join(missing)
            )

        self._residuals = np.asarray(
            _to_numpy(self.model._resid), dtype=float
        ).reshape(-1)
        self._X_design = np.asarray(
            _to_numpy(self.model._X_design), dtype=float
        )
        self._y = np.asarray(_to_numpy(self.model._y), dtype=float).reshape(-1)

        if self._X_design.ndim != 2:
            raise ValueError("model._X_design must be a 2D matrix")
        n = self._X_design.shape[0]
        if self._residuals.size != n or self._y.size != n:
            raise ValueError(
                "model diagnostic arrays have inconsistent sample counts: "
                f"X={n}, residuals={self._residuals.size}, y={self._y.size}"
            )
        if n == 0:
            raise ValueError("regression diagnostics require at least one observation")
        if not (
            np.all(np.isfinite(self._X_design))
            and np.all(np.isfinite(self._residuals))
            and np.all(np.isfinite(self._y))
        ):
            raise ValueError("regression diagnostic arrays must be finite")

        self._rank = int(np.linalg.matrix_rank(self._X_design))
        df_default = n - self._rank
        self._df_resid = int(getattr(self.model, "_df_resid", df_default))

        scale = getattr(self.model, "_scale", None)
        if scale is not None:
            scale_array = np.asarray(_to_numpy(scale), dtype=float).reshape(-1)
            if scale_array.size != 1:
                raise ValueError(
                    "RegressionDiagnostics currently supports single-output models only"
                )
            scale_value = float(scale_array[0])
        else:
            scale_value = float("nan")

        if not np.isfinite(scale_value) or scale_value < 0.0:
            if self._df_resid <= 0:
                raise ValueError(
                    "A finite model._scale or positive residual degrees of freedom is required"
                )
            scale_value = float(
                np.dot(self._residuals, self._residuals) / self._df_resid
            )
        self._scale = scale_value
        self._fit_intercept = bool(getattr(self.model, "fit_intercept", False))

    @property
    def residuals(self):
        """Raw residuals as a NumPy copy."""
        return self._residuals.copy()

    @property
    def fitted_values(self):
        """Fitted values reconstructed as ``y - residual``."""
        return (self._y - self._residuals).copy()

    @property
    def standardized_residuals(self):
        """Residuals divided by the fitted residual standard deviation."""
        if self._scale == 0.0:
            return np.full_like(self._residuals, np.nan)
        return self._residuals / np.sqrt(self._scale)

    @property
    def studentized_residuals(self):
        """Internally studentized residuals.

        These use the common full-model residual variance estimate. Use
        :attr:`externally_studentized_residuals` for leave-one-out variances.
        """
        denominator = np.sqrt(
            self._scale * np.clip(1.0 - self.leverage, np.finfo(float).eps, None)
        )
        return np.divide(
            self._residuals,
            denominator,
            out=np.full_like(self._residuals, np.nan),
            where=denominator > 0,
        )

    @property
    def externally_studentized_residuals(self):
        """Externally studentized (deleted) residuals."""
        if self._df_resid <= 1:
            raise ValueError(
                "externally studentized residuals require df_resid greater than 1"
            )
        one_minus_h = np.clip(
            1.0 - self.leverage, np.finfo(float).eps, None
        )
        deleted_sse = (
            self._df_resid * self._scale
            - (self._residuals**2) / one_minus_h
        )
        deleted_scale = deleted_sse / float(self._df_resid - 1)
        denominator = np.sqrt(
            np.where(deleted_scale > 0.0, deleted_scale * one_minus_h, np.nan)
        )
        return self._residuals / denominator

    @property
    def leverage(self):
        """Diagonal of the projection (hat) matrix."""
        if self._leverage_cache is None:
            X = self._X_design
            if self._rank == X.shape[1]:
                q, _ = np.linalg.qr(X, mode="reduced")
                leverage = np.einsum("ij,ij->i", q, q)
            else:
                x_pinv = np.linalg.pinv(X)
                leverage = np.einsum("ij,ji->i", X, x_pinv)
            self._leverage_cache = np.clip(
                np.asarray(leverage, dtype=float), 0.0, 1.0
            )
        return self._leverage_cache.copy()

    @property
    def cooks_distance(self):
        """Cook's distance based on internally studentized residuals."""
        h = self.leverage
        p_effective = max(self._rank, 1)
        one_minus_h = np.clip(1.0 - h, np.finfo(float).eps, None)
        return (self.studentized_residuals**2 / p_effective) * (
            h / one_minus_h
        )

    def vif(self):
        """Variance inflation factor for each non-intercept design column."""
        X = self._X_design
        start_idx = 1 if self._fit_intercept else 0
        values = []
        for index in range(start_idx, X.shape[1]):
            target = X[:, index]
            target_ss = float(np.sum((target - np.mean(target)) ** 2))
            if target_ss <= np.finfo(float).eps:
                values.append(float("inf"))
                continue

            others = np.delete(X, index, axis=1)
            if others.shape[1] == 0:
                values.append(1.0)
                continue
            try:
                coef, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
            except np.linalg.LinAlgError:
                values.append(float("inf"))
                continue
            residual_ss = float(np.sum((target - others @ coef) ** 2))
            r_squared = float(np.clip(1.0 - residual_ss / target_ss, 0.0, 1.0))
            values.append(
                float("inf")
                if 1.0 - r_squared <= np.finfo(float).eps
                else 1.0 / (1.0 - r_squared)
            )
        return np.asarray(values, dtype=float)

    def summary(self):
        """Print a diagnostic summary."""
        print("=" * 60)
        print("Regression Diagnostics Summary")
        print("=" * 60)

        resid = self._residuals
        print("\n--- Residuals ---")
        print(f"Min:    {np.min(resid):10.4f}")
        print(f"Q1:     {np.percentile(resid, 25):10.4f}")
        print(f"Median: {np.median(resid):10.4f}")
        print(f"Q3:     {np.percentile(resid, 75):10.4f}")
        print(f"Max:    {np.max(resid):10.4f}")

        if resid.size >= 3:
            _, shapiro_p = stats.shapiro(resid[: min(5000, resid.size)])
            print(f"\nShapiro-Wilk normality test p-value: {shapiro_p:.4f}")
            print(
                "⚠ Residuals may not be normally distributed"
                if shapiro_p < 0.05
                else "✓ Residuals appear normally distributed"
            )
        else:
            print("\nShapiro-Wilk normality test requires at least 3 residuals")

        h = self.leverage
        threshold = 2.0 * max(self._rank, 1) / h.size
        print("\n--- Leverage ---")
        print(f"Mean leverage: {np.mean(h):.4f}")
        print(f"Max leverage:  {np.max(h):.4f}")
        print(f"High leverage points (>{threshold:.4f}): {np.sum(h > threshold)}")

        cooks = self.cooks_distance
        print("\n--- Cook's Distance ---")
        print(f"Mean: {np.mean(cooks):.4f}")
        print(f"Max:  {np.max(cooks):.4f}")
        print(f"Influential points (>1): {np.sum(cooks > 1.0)}")

        print("\n--- Variance Inflation Factor ---")
        vif_values = self.vif()
        for offset, value in enumerate(vif_values, start=1):
            status = "⚠" if value > 10.0 else "✓"
            print(f"  x{offset}: {value:.2f} {status}")
        if np.any(vif_values > 10.0):
            print("\n⚠ High multicollinearity detected (VIF > 10)")
        elif np.any(vif_values > 5.0):
            print("\n⚠ Moderate multicollinearity (VIF > 5)")
        else:
            print("\n✓ No significant multicollinearity")
        print("=" * 60)


def diagnose_model(model):
    """Construct diagnostics, print their summary, and return the object."""
    diagnostics = RegressionDiagnostics(model)
    diagnostics.summary()
    return diagnostics
