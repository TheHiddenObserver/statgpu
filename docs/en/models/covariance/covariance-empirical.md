# Empirical Covariance: Learn the Unregularized Baseline

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../../cn/models/covariance/covariance-empirical.md)

## What does joint variation look like?

Suppose each row records several measurements from the same observation. You
want to describe their variability and how they move together, without a target
variable to predict. `EmpiricalCovariance` estimates this directly from the data.
It is a useful baseline when observations substantially outnumber features and
there are no nearly redundant columns or influential outliers.

Start with the [covariance overview](README.md) for choosing among estimators.
This tutorial focuses on what the empirical estimate and its distances mean.

## Intuition and formula

A variance measures the spread of one feature. A covariance measures whether
two features tend to depart from their means in the same direction. For $n$
observations with $p$ features, the default estimate is

$$
\bar x=\frac{1}{n}\sum_{i=1}^n x_i,
\qquad
S=\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)^\top.
$$

The divisor is **$n$, not $n-1$**. This is the Gaussian maximum-likelihood
covariance estimate, rather than the usual unbiased sample covariance. It
matches `np.cov(X, rowvar=False, ddof=0)`. Computing it does not require data to
be Gaussian; interpreting `score` as a correctly specified likelihood does.

`covariance_` contains $S$: its diagonal gives variances and its off-diagonal
entries give covariances. Covariance is in the product of the features' units,
so an entry larger than one is not an invalid correlation coefficient.
`location_` contains $\bar x$. If `assume_centered=True`, the estimator instead
uses zero as its location and computes $X^\top X/n$ without subtracting a mean.

## When to use it, and when to switch

Use this estimator for a transparent baseline, descriptive covariance, or a
well-sampled problem where regularization would add unnecessary bias.

- Choose [Ledoit–Wolf](covariance-ledoit-wolf.md) or [OAS](covariance-oas.md)
  when small eigenvalues make inversion unstable or there are many features.
- Choose [fixed shrinkage](covariance-shrunk.md) when you want to control the
  regularization strength explicitly.
- Consider [MinCovDet](covariance-min-cov-det.md) for a contaminated sample,
  rather than expecting empirical covariance to resist outliers.

## CPU example: estimate spread and measure distance

The following complete example creates 400 observations with two correlated
features. Their population mean is `[10, -3]` and population covariance is
`[[4, 1.2], [1.2, 1]]`. No response vector or manual centering is needed.

<!-- example: covariance-empirical -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)
transform = np.array([[2.0, 0.0], [0.6, 0.8]])
X = rng.normal(size=(400, 2)) @ transform.T + [10.0, -3.0]
```

Fit the covariance and location on these rows.

```python
model = EmpiricalCovariance(device="cpu").fit(X)
```

Inspect the matrix, verify the normalization, and evaluate two new points.

```python
print("location:", np.round(model.location_, 3))
print("covariance:\n", np.round(model.covariance_, 3))
print("matches divisor n:", np.allclose(
    model.covariance_, np.cov(X, rowvar=False, ddof=0)
))
queries = np.array([[10.0, -3.0], [14.0, -3.0]])
print("squared distances:", np.round(model.mahalanobis(queries), 3))
```
<!-- example-end: covariance-empirical -->

Output:

```text
location: [ 9.96  -3.042]
covariance:
 [[4.027 1.454]
 [1.454 1.104]]
matches divisor n: True
squared distances: [0.002 7.522]
```

The estimated means and variances are near their population values, but finite
samples do not recover the population covariance exactly. The positive
cross-covariance indicates that the two features tend to increase together.

`mahalanobis` returns **squared** distances,
$d^2(x)=(x-\bar x)^\top\texttt{precision\_}(x-\bar x)$.
The first query is near the estimated center. The second is less typical because
its first feature is high without a corresponding increase in the second.
`7.522` is neither a probability nor an automatically calibrated outlier label.
`predict` returns the same squared distances; it does not predict feature values.

## Choices and pitfalls

- Keep `assume_centered=False` unless zero is the intended, known center or
  preprocessing has already established it. A merely small sample mean does
  not make centering irrelevant. Apply training-derived preprocessing to new
  rows; do not independently recenter evaluation data.
- There is no shrinkage parameter to tune. Assess the condition number and
  held-out performance before using the precision matrix downstream.
- With estimated centering, the covariance rank is at most $\min(p,n-1)$.
  Thus $p\ge n$ guarantees rank deficiency, and collinearity can cause it even
  when $n$ is larger. Inversion stabilization is only a numerical safeguard:
  it leaves `covariance_` unchanged and does not create missing information.
- `score(X_test)` is average Gaussian log likelihood per observation. Compare
  higher scores on the same held-out rows and feature scale. A singular
  covariance can yield `-inf`; fitting successfully does not prove that a
  usable Gaussian density was obtained.
- At least two finite training rows and one feature are required. Handle
  missing values before fitting and keep feature order consistent at evaluation.

## API and references

See the [complete covariance API](../../reference/covariance-api.md) for all
parameters, fitted attributes, input conventions, and methods. See
[device and dtype](../../reference/covariance-api.md#backend-device-and-dtype)
for device selection and
[shared methods](../../reference/covariance-api.md#fit-score-predict-and-mahalanobis)
for score, distance, and precision behavior.

- [scikit-learn: EmpiricalCovariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html),
  a reference for Gaussian maximum-likelihood covariance estimation.
