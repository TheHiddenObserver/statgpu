# Regression Diagnostics

> Language: English  
> Last updated: 2026-10-12  
> Switch: [Chinese](../../cn/guides/regression-diagnostics.md)

After fitting a regression model, inspect residuals, high-leverage or influential
observations, and collinearity between features. `RegressionDiagnostics(model)`
computes these diagnostics from a compatible fitted model. It reports raw and
standardized residuals, internally and externally studentized residuals, leverage,
Cook's distance, and VIF.

High leverage means a row's combination of predictor values differs from most observations. It signals potential influence, not necessarily a data error. Cook's distance combines residual size and leverage to measure a row's influence on the fitted regression. VIF (variance inflation factor) measures how well each feature can be explained linearly by the other features, helping identify coefficients that are difficult to estimate separately.

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

The value at index `i` in `leverage`, `externally_studentized_residuals`, and `cooks_distance` describes row `i` of the fitted data. `vif()` returns one value per non-intercept design column, in that column order. For example, fitting an array `X` with 100 rows and two nonconstant features produces three length-100 row diagnostics and two VIF values; the automatically added intercept has no VIF entry. After formula fitting, use the retained design rows and expanded design columns, which may differ from the raw dataframe.

Start with rows having large Cook's distances: check the original records for entry errors, measurement problems, or unusual circumstances, then compare fits with and without those rows as a sensitivity check. These diagnostics identify observations to investigate, rather than automatically discard.

Interpret these together with sample size and model assumptions; no single cutoff is a universal rule for deleting observations.

`RegressionDiagnostics` runs on CPU, including when the model was fitted on GPU.
Creating it converts the required fitted arrays to NumPy, so allow for the CPU
memory and data-transfer cost. This does not refit the model.

The calculations use the model's saved design, response, residuals, and scale.
Rank-deficient designs use a pseudoinverse-based hat diagonal; external
studentization uses deleted residual variances.
