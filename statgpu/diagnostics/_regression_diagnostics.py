"""
Regression diagnostics for model validation.
Includes residual analysis, influence measures, and VIF.
"""

import numpy as np
from scipy import stats


class RegressionDiagnostics:
    """
    Diagnostics for regression models.
    
    Parameters
    ----------
    model : fitted model
        Fitted regression model with residuals_, fitted_, X_design attributes.
    """
    
    def __init__(self, model):
        self.model = model
        self._validate_model()
    
    def _validate_model(self):
        """Check model has required attributes."""
        required = ['_resid', '_X_design', '_y']
        for attr in required:
            if not hasattr(self.model, attr) or getattr(self.model, attr) is None:
                raise ValueError(f"Model missing required attribute: {attr}")
    
    @property
    def residuals(self):
        """Raw residuals."""
        return self.model._resid
    
    @property
    def fitted_values(self):
        """Fitted (predicted) values."""
        return self.model._y - self.model._resid
    
    @property
    def standardized_residuals(self):
        """Standardized residuals (divided by estimated standard deviation)."""
        sigma = np.sqrt(self.model._scale) if hasattr(self.model, '_scale') else np.std(self.residuals)
        return self.residuals / sigma
    
    @property
    def studentized_residuals(self):
        """Studentized residuals (externally studentized)."""
        n = len(self.residuals)
        h = self.leverage
        sigma = np.sqrt(self.model._scale) if hasattr(self.model, '_scale') else np.std(self.residuals)
        
        # Internally studentized
        stud = self.residuals / (sigma * np.sqrt(1 - h + 1e-10))
        return stud
    
    @property
    def leverage(self):
        """Leverage values (diagonal of hat matrix)."""
        X = self.model._X_design
        try:
            hat_matrix = X @ np.linalg.inv(X.T @ X) @ X.T
            return np.diag(hat_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            hat_matrix = X @ np.linalg.pinv(X.T @ X) @ X.T
            return np.diag(hat_matrix)
    
    @property
    def cooks_distance(self):
        """Cook's distance (influence measure)."""
        stud = self.studentized_residuals
        h = self.leverage
        p = self.model._X_design.shape[1]
        
        # Cook's D
        cooks_d = (stud**2 / p) * (h / (1 - h + 1e-10))
        return cooks_d
    
    def vif(self):
        """
        Variance Inflation Factor (multicollinearity measure).
        
        Returns
        -------
        vif : ndarray
            VIF for each feature (excluding intercept).
        """
        X = self.model._X_design
        n_features = X.shape[1]
        
        # Skip intercept
        start_idx = 1 if self.model.fit_intercept else 0
        
        vif_values = []
        for i in range(start_idx, n_features):
            # Regress feature i on all other features
            y_vif = X[:, i]
            X_vif = np.delete(X, i, axis=1)
            
            try:
                coef, _, _, _ = np.linalg.lstsq(X_vif, y_vif, rcond=None)
                y_pred = X_vif @ coef
                ss_res = np.sum((y_vif - y_pred)**2)
                ss_tot = np.sum((y_vif - np.mean(y_vif))**2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                vif = 1 / (1 - r2 + 1e-10)
            except:
                vif = np.inf
            
            vif_values.append(vif)
        
        return np.array(vif_values)
    
    def summary(self):
        """Print diagnostic summary."""
        print("=" * 60)
        print("Regression Diagnostics Summary")
        print("=" * 60)
        
        # Residuals
        print("\n--- Residuals ---")
        resid = self.residuals
        print(f"Min:    {np.min(resid):10.4f}")
        print(f"Q1:     {np.percentile(resid, 25):10.4f}")
        print(f"Median: {np.median(resid):10.4f}")
        print(f"Q3:     {np.percentile(resid, 75):10.4f}")
        print(f"Max:    {np.max(resid):10.4f}")
        
        # Normality test
        _, shapiro_p = stats.shapiro(resid[:min(5000, len(resid))])
        print(f"\nShapiro-Wilk normality test p-value: {shapiro_p:.4f}")
        if shapiro_p < 0.05:
            print("⚠ Residuals may not be normally distributed")
        else:
            print("✓ Residuals appear normally distributed")
        
        # Leverage
        h = self.leverage
        h_threshold = 2 * len(self.model._params) / len(h)
        high_leverage = np.sum(h > h_threshold)
        print(f"\n--- Leverage ---")
        print(f"Mean leverage: {np.mean(h):.4f}")
        print(f"Max leverage:  {np.max(h):.4f}")
        print(f"High leverage points (>{h_threshold:.4f}): {high_leverage}")
        
        # Cook's distance
        cooks = self.cooks_distance
        influential = np.sum(cooks > 1)
        print(f"\n--- Cook's Distance ---")
        print(f"Mean: {np.mean(cooks):.4f}")
        print(f"Max:  {np.max(cooks):.4f}")
        print(f"Influential points (>1): {influential}")
        
        # VIF
        print(f"\n--- Variance Inflation Factor ---")
        vif_values = self.vif()
        for i, v in enumerate(vif_values):
            status = "⚠" if v > 10 else "✓"
            print(f"  x{i+1}: {vif_values[i]:.2f} {status}")
        
        if np.any(vif_values > 10):
            print("\n⚠ High multicollinearity detected (VIF > 10)")
        elif np.any(vif_values > 5):
            print("\n⚠ Moderate multicollinearity (VIF > 5)")
        else:
            print("\n✓ No significant multicollinearity")
        
        print("=" * 60)


def diagnose_model(model):
    """
    Convenience function to diagnose a fitted model.
    
    Parameters
    ----------
    model : fitted model
        Fitted regression model.
    
    Returns
    -------
    diagnostics : RegressionDiagnostics
        Diagnostics object.
    """
    diag = RegressionDiagnostics(model)
    diag.summary()
    return diag
