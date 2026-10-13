# Regression Diagnostics

> Language: English  
> Last updated: 2026-10-12  
> Switch: [Chinese](../../cn/guides/regression-diagnostics.md)

After fitting a regression model, inspect residuals, high-leverage or influential
observations, and collinearity between features. `RegressionDiagnostics(model)`
computes these diagnostics from a compatible fitted model. It reports raw and
standardized residuals, internally and externally studentized residuals, leverage,
Cook's distance, and VIF.

Leverage measures how unusual an observation is in the design space. Cook's distance measures its influence on the fitted regression, while VIF describes collinearity among predictors.

With a design matrix `X` and response `y` prepared, fit a linear regression and
read its diagnostics:

```python
from statgpu import LinearRegression, RegressionDiagnostics

model = LinearRegression().fit(X, y)
diag = RegressionDiagnostics(model)
print(diag.leverage)
print(diag.externally_studentized_residuals)
print(diag.cooks_distance)
print(diag.vif())
```

Interpret these together with sample size and model assumptions; no single cutoff is a universal rule for deleting observations.

`RegressionDiagnostics` runs on CPU, including when the model was fitted on GPU.
Creating it converts the required fitted arrays to NumPy, so allow for the CPU
memory and data-transfer cost. This does not refit the model.

The calculations use the model's saved design, response, residuals, and scale.
Rank-deficient designs use a pseudoinverse-based hat diagonal; external
studentization uses deleted residual variances.
