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

`RegressionDiagnostics` runs on CPU, including when the model was fitted on GPU.
Creating it converts the required fitted arrays to NumPy, so allow for the CPU
memory and data-transfer cost. This does not refit the model.
