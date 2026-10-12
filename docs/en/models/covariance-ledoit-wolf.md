# Ledoit–Wolf: Let the Data Choose Shrinkage

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance-ledoit-wolf.md)

## How much should you trust a noisy covariance matrix?

When the number of features is large relative to the number of observations,
empirical covariance can fit the observed relationships too closely. Its
inverse can magnify noise in directions with small estimated variance.
[Fixed shrinkage](covariance-shrunk.md) helps, but requires choosing a strength.

`LedoitWolf` estimates that strength from the training data in one fit. It is a
useful first comparison to [empirical covariance](covariance-empirical.md) when
you need a stable dense covariance or precision estimate without running a
shrinkage grid search. See the [covariance overview](covariance.md) for the
broader choice of robust and sparse estimators.

## Intuition: balance estimated noise against structure

The estimator blends the empirical matrix $S$ with a scaled identity:

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I,
\qquad \mu=\operatorname{tr}(S)/p.
$$

The target retains the average variance but assumes equal variances and zero
cross-covariances. Greater uncertainty in $S$ calls for more shrinkage; strong
estimated departures from the target call for retaining more of $S$.
Ledoit–Wolf derives the intensity from a squared-error covariance criterion,
not by maximizing the training Gaussian score.

For centered observations $z_i$ and $S=n^{-1}\sum_i z_i z_i^\top$, the calculation
used here can be written as

$$
\beta=\frac{\sum_i\lVert z_i\rVert_2^4-n\lVert S\rVert_F^2}{n^2},
\qquad
\delta=\lVert S\rVert_F^2-\frac{\operatorname{tr}(S)^2}{p},
\qquad
\alpha=\operatorname{clip}(\beta/\delta,0,1).
$$

The squared Frobenius norm sums the squares of all matrix entries. $\delta$ measures the
departure from the target; $\beta$ estimates sampling variability. When
$\delta\le0$, this implementation sets $\alpha=1$. With
`assume_centered=True`, $z_i=x_i$ and the fitted location is zero.

## When is another estimator preferable?

- Use [OAS](covariance-oas.md) as a Gaussian-based alternative, especially with
  limited observations. Compare them on data relevant to your application;
  neither dominates every finite sample.
- Use [ShrunkCovariance](covariance-shrunk.md) if you need a predetermined
  strength or want to select one for a downstream criterion.
- Keep [EmpiricalCovariance](covariance-empirical.md) as a baseline when data
  are plentiful. For contamination or sparse conditional relationships, see
  the robust and sparse choices in the [overview](covariance.md).

## CPU example: evaluate on new observations

We generate 20 correlated features, with correlation decreasing as columns get
farther apart. Only 40 rows are used for fitting. A separate 300-row sample
measures how the fitted Gaussian models describe new observations.

<!-- example: covariance-ledoit-wolf -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, LedoitWolf

rng = np.random.default_rng(42)
p = 20
population = 0.6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
L = np.linalg.cholesky(population)
X_train = rng.normal(size=(40, p)) @ L.T
X_test = rng.normal(size=(300, p)) @ L.T
```

Fit both models using only the training rows.

```python
empirical = EmpiricalCovariance(device="cpu").fit(X_train)
model = LedoitWolf(device="cpu").fit(X_train)
```

Inspect the learned intensity and compare conditioning and held-out scores.

```python
print(f"shrinkage: {model.shrinkage_:.3f}")
print(f"condition numbers: {np.linalg.cond(empirical.covariance_):.1f}"
      f" -> {np.linalg.cond(model.covariance_):.1f}")
print(f"held-out scores: empirical={empirical.score(X_test):.3f},"
      f" Ledoit-Wolf={model.score(X_test):.3f}")
```
<!-- example-end: covariance-ledoit-wolf -->

Output:

```text
shrinkage: 0.311
condition numbers: 162.7 -> 10.3
held-out scores: empirical=-38.391, Ledoit-Wolf=-26.268
```

About 31% of the final estimate comes from the target in this particular sample.
The smaller condition number indicates less extreme eigenvalue ratios.
The higher held-out score (`-26.268`) means a higher average Gaussian log
likelihood on these same test rows, not 26% accuracy. Here shrinkage also
improves out-of-sample fit. One simulation does not guarantee that outcome for
other covariance structures or datasets.

## What still needs choosing?

There is no user-set `shrinkage` parameter on `LedoitWolf`; inspect `shrinkage_`
after fitting. You still choose the features, units, preprocessing, centering,
and evaluation criterion. The default estimates the training mean. Do not set
`assume_centered=True` simply because the population mean is approximately
zero unless treating zero as known is intentional.

The common variance target depends on feature units. If standardization is
appropriate, fit it on training rows and reuse it on new rows. If you use
held-out scores to choose among several estimators, treat those rows as a
validation set and reserve other rows for final evaluation.

## Limitations

- This is linear shrinkage toward a scaled identity, not arbitrary structured
  covariance estimation. It can reduce real correlations and unequal variances.
- It does not require the Gaussian-specific OAS derivation, but is not an
  outlier-resistant estimator. Large observations can strongly affect the
  fourth-moment calculation.
- Shrinkage does not seek exact zeros in the precision matrix or supply
  coefficient p-values. A large `shrinkage_` is not a significance test that
  the population covariance equals the target.
- Positive shrinkage gives positive eigenvalues only when the target variance
  is positive. All-zero variation remains degenerate. Training still requires
  at least two finite observations and one feature.

## API and references

The [complete covariance API](../reference/covariance-api.md) lists parameters,
methods, attributes, and shared input rules. Use the
[covariance overview](covariance.md) for device selection and common score and
distance semantics.

- Ledoit, O., & Wolf, M. (2004).
  [A well-conditioned estimator for large-dimensional covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4).
