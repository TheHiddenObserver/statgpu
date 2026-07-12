# Regression Diagnostics

> Language: English  
> Last updated: 2026-07-12  
> Switch: [Chinese](../../cn/guides/regression-diagnostics.md)

`RegressionDiagnostics(model)` consumes the fitted design, response, residuals, and
scale from a compatible regression model. It reports raw/standardized/internal and
external studentized residuals, leverage, Cook's distance, and VIF. Rank-deficient
designs use a pseudoinverse-based hat diagonal; external studentization uses deleted
residual variances.

```python
from statgpu import LinearRegression, RegressionDiagnostics

model = LinearRegression().fit(X, y)
diag = RegressionDiagnostics(model)
print(diag.leverage)
print(diag.externally_studentized_residuals)
print(diag.cooks_distance)
print(diag.vif())
```

Diagnostics are intentionally reporting-side CPU utilities: fitted arrays are copied
once to NumPy because SciPy distribution tests and human-readable summaries are used.
This is an explicit boundary, not a model-training fallback. Reference tests compare
influence quantities with `statsmodels.OLSInfluence`.
