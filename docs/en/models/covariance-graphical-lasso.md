# GraphicalLasso: a sparse map of conditional relationships

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance-graphical-lasso.md)

## What relationships remain after accounting for the other variables?

Two measurements can move together because they share a third influence.
Ordinary covariance describes their marginal association. A precision matrix,
the inverse covariance, helps distinguish that association from a relationship
that remains after conditioning on the other measured variables.

`GraphicalLasso` estimates a precision matrix with an L1 penalty on its
off-diagonal entries. Many entries can become zero, giving a compact undirected
graph. For a multivariate Gaussian distribution, a zero precision entry means
conditional independence. This is a model-based statement about the measured
variables, not evidence of a causal direction.

Start with the [covariance overview](covariance.md) for concepts and method
selection. The [complete API reference](../reference/covariance-api.md) covers
inputs, distances, scores, public parameters, and fitted attributes; its
[backend, device, and dtype section](../reference/covariance-api.md#backend-device-and-dtype)
explains device settings.

## When is sparsity the right assumption?

Use Graphical Lasso when a relatively small number of conditional relationships
is a useful working model, or when regularizing a precision estimate matters.
The estimated covariance need not be sparse: inverting a sparse precision
matrix usually produces a dense matrix.

- Use [EmpiricalCovariance](covariance-empirical.md) for an unpenalized baseline
  when the sample is large relative to the number of features.
- Use [LedoitWolf](covariance-ledoit-wolf.md) or [OAS](covariance-oas.md) when the
  main goal is a stable covariance estimate without a sparse graph assumption.
- Use [GraphicalLassoCV](covariance-graphical-lasso-cv.md) to select the penalty
  by held-out Gaussian likelihood.
- Use [MinCovDet](covariance-min-cov-det.md) for a robust estimate under
  contamination. L1 regularization alone does not make Graphical Lasso robust
  to outlying rows.

## What is being optimized?

Let $S$ be the empirical covariance computed with divisor $n$, after subtracting
the training mean unless `assume_centered=True`. The target precision is

$$
\widehat\Theta
=\arg\min_{\Theta\succ0}
\left\{\operatorname{tr}(S\Theta)-\log\det\Theta
+\alpha\sum_{j\ne k}|\Theta_{jk}|\right\}.
$$

The first two terms fit a Gaussian covariance model. The penalty discourages
nonzero off-diagonal precision entries; it does not penalize the diagonal.
Larger `alpha` generally gives a sparser estimate. Here `alpha` is a penalty
strength, not a significance level. The covariance estimate is
$\widehat\Sigma=\widehat\Theta^{-1}$ when the matrices are nonsingular.

For distinct features $j,k$, the fitted partial correlation is

$$
\widehat\rho_{jk\mid\mathrm{others}}
=-\frac{\widehat\Theta_{jk}}
{\sqrt{\widehat\Theta_{jj}\widehat\Theta_{kk}}}.
$$

These relations explain how to read the graph; they do not provide edge
p-values or confidence intervals. See the [original Graphical Lasso paper](https://hastie.su.domains/Papers/graph.pdf)
for the statistical formulation and algorithm.

## Fit and inspect a four-variable graph on CPU

This self-contained example simulates a Gaussian chain with three population
edges. The first 300 rows train the model; the last 100 evaluate it. Features
are generated on comparable scales, so no additional scaling step is used.

<!-- example: covariance-graphical-lasso -->
Import NumPy and the estimator.

```python
import numpy as np
from statgpu.covariance import GraphicalLasso
```

Generate the chain and set aside test rows. Each row is an observation and each column is a feature.

```python
rng = np.random.default_rng(12)
true_precision = np.array([
    [1.0, -0.5, 0.0, 0.0],
    [-0.5, 1.0, -0.4, 0.0],
    [0.0, -0.4, 1.0, -0.3],
    [0.0, 0.0, -0.3, 1.0],
])
X = rng.multivariate_normal(
    np.zeros(4), np.linalg.inv(true_precision), size=400
)
X_train, X_test = X[:300], X[300:]
```

Fit only the training rows with a fixed penalty.

```python
model = GraphicalLasso(
    alpha=0.1, max_iter=200, tol=1e-6, device="cpu"
).fit(X_train)
```

Inspect the precision matrix, count numerical edges, and evaluate untouched test rows.

```python
print(np.round(model.precision_, 3))
edge_count = np.count_nonzero(
    np.triu(np.abs(model.precision_) > 1e-4, k=1)
)
print("edges:", edge_count)
print("iterations:", model.n_iter_)
print("test score:", round(model.score(X_test), 3))
print("squared distances:", np.round(model.mahalanobis(X_test[:3]), 3))
```
<!-- example-end: covariance-graphical-lasso -->

Output, rounded to three decimals:

```text
[[ 0.929 -0.34  -0.074 -0.   ]
 [-0.34   0.851 -0.219  0.   ]
 [-0.074 -0.219  0.915 -0.242]
 [-0.     0.    -0.242  0.872]]
edges: 4
iterations: 2
test score: -6.036
squared distances: [2.358 1.085 6.911]
```

The fitted graph has four edges, including one extra edge between features 0
and 2 compared with the simulated chain. Finite samples and a chosen penalty do
not guarantee recovery of the population graph. The `1e-4` threshold is an
explicit numerical display rule, not a statistical test. Rounded negative zero
also has no substantive meaning.

`score` is the mean Gaussian log-likelihood per held-out row: higher is better
when evaluating comparable models on the same rows and feature scale. It is
not accuracy. `mahalanobis` and `predict` both return squared Mahalanobis
distances, not class predictions. The third displayed row is farther from the
fitted center than the first two in this metric; the distances alone do not
establish that any row is erroneous.

## Choosing the penalty and checking the fit

- `alpha=0.01` is the default. Its effect depends on feature units. If you
  standardize, learn means and scales from training data only, and reuse them
  unchanged on evaluation rows. Constant columns should be removed.
- `alpha=0` takes the empirical covariance and pseudoinverse path. It is not a
  sparse regularized fit and does not solve rank deficiency by adding information.
- `max_iter=100` and `tol=1e-4` control iterative fitting. A smaller tolerance
  can require more work. Inspect `n_iter_` and check whether results change
  under a larger iteration budget or tighter tolerance.
- Reaching `max_iter` currently returns the last estimate without a convergence
  warning. Neither a successful return nor a small iteration count is a
  certificate that all numerical or optimality conditions hold.
- Difficult, nearly collinear data and very small penalties deserve additional
  checks for finite outputs and a positive-definite covariance. Positive
  `alpha` is not a blanket numerical guarantee for the current solver.

Inversion may use numerical stabilization when an exact inverse fails; see the
API reference's [inverse, distance, and score rules](../reference/covariance-api.md#fit-score-predict-and-mahalanobis).
For scientific graph recovery,
check sensitivity to the penalty and sampling variation. Predictive likelihood
selection and reliable edge selection are different goals.

## References and next steps

- Friedman, J., Hastie, T., & Tibshirani, R. (2008).
  [Sparse inverse covariance estimation with the graphical lasso](https://hastie.su.domains/Papers/graph.pdf).
  *Biostatistics*, 9(3), 432–441.
- [Select a penalty with GraphicalLassoCV](covariance-graphical-lasso-cv.md).
- [Covariance overview and method selection](covariance.md).
- [Complete covariance API](../reference/covariance-api.md).
