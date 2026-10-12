# Fixed Shrinkage: Trade Some Detail for Stability

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance-shrunk.md)

## Why can an estimated inverse be unreliable?

Two almost identical measurements contain little information about their
difference. Empirical covariance then has a very small eigenvalue in that
direction. Inverting the matrix magnifies small estimation errors, so distances
or downstream calculations can become unstable.

`ShrunkCovariance` lets you deliberately pull the estimate toward a simpler
matrix. You control how much detail to retain with `shrinkage`. Use it when that
choice should come from a sensitivity analysis, domain requirements, or your own
validation procedure. The [covariance overview](covariance.md) compares the
other estimators.

## What does shrinkage change?

Let $S$ be the [empirical covariance](covariance-empirical.md), computed with
divisor $n$, and let $\mu=\operatorname{tr}(S)/p$ be the average feature variance.
For a chosen $\alpha\in[0,1]$,

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I.
$$

Here $\alpha$ is `shrinkage`, whose default is `0.1`.

- At `0`, the covariance is the empirical estimate.
- Between `0` and `1`, off-diagonal covariances are multiplied by $1-\alpha$,
  while variances move toward their average.
- At `1`, the result is $\mu I$: equal variances and zero cross-covariances.

Each empirical eigenvalue $\lambda_j$ becomes
$(1-\alpha)\lambda_j+\alpha\mu$. Consequently positive shrinkage gives a positive
minimum eigenvalue when $\mu>0$, even if $S$ is rank deficient. The trace is
preserved. Better conditioning comes with bias: real differences in variances
and relationships are also reduced. This is a scaled-identity target, not a
user-supplied target or a procedure that preserves each original variance.

## Choose fixed or automatic shrinkage?

Use fixed shrinkage when you need an explicit strength or want to compare a
small set of strengths using a task-relevant criterion. Choose
[Ledoit–Wolf](covariance-ledoit-wolf.md) to estimate the strength analytically
without a validation search, or [OAS](covariance-oas.md) when a Gaussian-based
small-sample alternative is appropriate. Neither fixed nor automatic shrinkage
is designed to remove the influence of outliers.

## CPU example: stabilize nearly duplicated features

This self-contained example creates 80 rows and two features that differ only
by small noise. It compares the empirical and shrunk eigenvalues, rather than
mistaking a successful matrix inversion for a reliable estimate.

<!-- example: covariance-shrunk -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, ShrunkCovariance

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)
z = rng.normal(size=80)
X = np.column_stack([z, z + 0.05 * rng.normal(size=80)])
```

Fit both estimators on exactly the same observations.

```python
empirical = EmpiricalCovariance(device="cpu").fit(X)
model = ShrunkCovariance(shrinkage=0.2, device="cpu").fit(X)
```

Compare their eigenvalues and condition numbers.

```python
print(f"shrinkage: {model.shrinkage_:.3f}")
print("empirical eigenvalues:", np.round(
    np.linalg.eigvalsh(empirical.covariance_), 3
))
print("shrunk eigenvalues:", np.round(
    np.linalg.eigvalsh(model.covariance_), 3
))
print(f"condition numbers: {np.linalg.cond(empirical.covariance_):.1f}"
      f" -> {np.linalg.cond(model.covariance_):.1f}")
```
<!-- example-end: covariance-shrunk -->

Output:

```text
shrinkage: 0.200
empirical eigenvalues: [0.001 1.195]
shrunk eigenvalues: [0.12  1.075]
condition numbers: 1117.4 -> 8.9
```

The empirical matrix has a very narrow direction. Shrinkage raises its small
eigenvalue and lowers the large one. The condition number, the ratio of largest
to smallest eigenvalue here, drops sharply. `shrinkage_` records the supplied
`0.2`; the estimator did not learn or optimize that value.

This example demonstrates numerical stabilization, not that `0.2` is optimal.
The almost-perfect relationship is real in these simulated data, so too much
shrinkage can discard useful information even while improving conditioning.

## How to choose the strength

1. Decide whether the original feature units make a common variance target
   sensible. If you standardize, learn the transformation on training rows
   only and apply it unchanged to validation/test rows.
2. Fit candidate strengths in `[0, 1]` on the same training data. If a Gaussian
   density is your goal, compare `score` on validation rows; larger is better.
   For another downstream task, use that task's validation criterion.
3. Choose a strength without looking at the final test set, then refit on the
   available training data. There is no built-in CV selector on this class.

The default `0.1` is a starting value, not a data-dependent recommendation.
Keep `assume_centered=False` unless zero is the intended center. All candidates
must use the same centering and feature scale for a fair comparison.

## Limits and next steps

`shrinkage` must be finite and lie in `[0, 1]`; values outside this range raise
`ValueError`. At least two finite rows and one feature are needed. If all
centered observations are zero, $\mu=0$ and shrinkage cannot manufacture
variation. A well-conditioned matrix also does not guarantee a good population
estimate, robust outlier handling, or a sparse precision matrix.

For those other goals, consult the [covariance overview](covariance.md). The
[complete API reference](../reference/covariance-api.md) lists all parameters,
methods, and fitted attributes; see its
[device and dtype section](../reference/covariance-api.md#backend-device-and-dtype)
for device selection.

## References

- [scikit-learn: ShrunkCovariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html),
  the fixed scaled-identity shrinkage construction.
- Ledoit, O., & Wolf, M. (2004).
  [A well-conditioned estimator for large-dimensional covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4).
