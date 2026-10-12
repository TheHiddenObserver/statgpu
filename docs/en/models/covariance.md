# Covariance Estimation

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance.md)

## Overview

How do several measurements vary together, and what counts as an unusual
combination of them? A covariance matrix summarizes joint variation. It can be
used to describe data, construct multivariate distances, or supply a covariance
estimate to another model. There is no response variable in this task: rows are
observations and columns are measured variables.

Start here for the common ideas and a two-variable example. Each estimator has
its own tutorial below; the [complete covariance API](../reference/covariance-api.md)
collects constructors, methods, outputs, input restrictions and device behavior.

<a id="covariance-correlation-precision"></a>

<a id="objectives"></a>

## Three matrices, three interpretations

For a random vector $x$ with mean $\mu$, the population covariance is

$$
\Sigma=\mathbb E[(x-\mu)(x-\mu)^\top].
$$

Its diagonal entries are variances. An off-diagonal entry is positive when two
variables tend to move in the same direction about their means, and negative
when they tend to move in opposite directions. Covariance measures linear
co-variation; zero covariance does not rule out nonlinear dependence.

The correlation matrix removes marginal scales:

$$
R_{jk}=\frac{\Sigma_{jk}}{\sqrt{\Sigma_{jj}\Sigma_{kk}}}.
$$

A correlation is dimensionless and lies between $-1$ and $1$ when both variances
are positive. It is undefined for a constant variable. A covariance of 10 has
no universal interpretation without the two variables' units.

When $\Sigma$ is invertible, the precision matrix is $\Theta=\Sigma^{-1}$.
For a multivariate Gaussian model, $\Theta_{jk}=0$ means variables $j$ and $k$
are conditionally independent given the other variables. This is different from
zero marginal correlation. Partial correlation is
$-\Theta_{jk}/\sqrt{\Theta_{jj}\Theta_{kk}}$; the raw precision entry is not itself
a correlation. Conditional association does not establish causation, and the
Gaussian conditional-independence interpretation does not hold for every
non-Gaussian distribution.

<a id="units-and-standardization"></a>

## Units and standardization change the question

If variable $j$ is measured in meters, its variance is in square meters; a
covariance carries the product of the two variables' units. Multiplying a
column by 100 multiplies its variance by 10,000. Large matrix entries can
therefore reflect measurement units rather than stronger relationships.

Standardizing columns puts them on comparable marginal scales. That can be
helpful for a sparse graph or a scaled-identity shrinkage target, but it changes
the covariance being estimated. Fit any means and scales on the training rows
only, then reuse them for held-out rows. Do not standardize train and test
independently. Constant columns need separate handling. If $D$ holds the
training scales, a standardized estimate $\widehat\Sigma_z$ maps back to the
original units as $D\widehat\Sigma_zD$.

`assume_centered=True` only skips estimation of the mean; it does not
standardize columns. Use it only when zero is the intended known center.

<a id="sample-size-and-stability"></a>

<a id="limitations-and-failure-modes"></a>

## Why the number of variables matters

With $n$ observations and $p$ variables, estimating covariance means estimating
$p(p+1)/2$ distinct numbers. For centered observations, the empirical covariance
has rank at most $\min(p,n-1)$. It is necessarily singular when $p\ge n$, and
can be unstable well before that when columns are strongly related.

The inverse magnifies errors in small eigenvalues. Consequently a covariance
matrix that looks reasonable entry by entry may still give unreliable distances
or precision estimates. More observations, fewer redundant variables, or
appropriate regularization can help. An inversion safeguard cannot recover
information the data never contained; a stabilized precision does not guarantee
a nonsingular stored covariance or a finite Gaussian score.

<a id="choose-an-estimator"></a>

<a id="robust-and-sparse-alternatives"></a>

## Choose the question before the estimator

| Question | Tutorial | Main trade-off |
|---|---|---|
| What is the direct observed covariance? | [EmpiricalCovariance](covariance-empirical.md) | Simple baseline; sensitive to outliers and ill-conditioning |
| How much should I shrink toward equal variances and zero covariances? | [ShrunkCovariance](covariance-shrunk.md) | User chooses the bias–stability trade-off |
| Can the data estimate the shrinkage strength? | [LedoitWolf](covariance-ledoit-wolf.md) | Analytic choice toward the scaled identity |
| Can I use a Gaussian-oriented analytic shrinkage rule? | [OAS](covariance-oas.md) | Particularly motivated by small Gaussian samples |
| What describes the main data cloud despite contamination? | [MinCovDet](covariance-min-cov-det.md) | Robust subset search; requires more rows than columns |
| Which conditional relationships can be represented sparsely? | [GraphicalLasso](covariance-graphical-lasso.md) | L1 penalty on off-diagonal precision entries |
| Which Graphical Lasso penalty predicts held-out rows best? | [GraphicalLassoCV](covariance-graphical-lasso-cv.md) | Selects and refits the same model; costs multiple fits |

Shrinkage, robustness and sparsity solve different problems. Shrinkage stabilizes
an estimate by moving it toward a target; it does not prevent a gross outlier
from influencing the empirical covariance. Robust estimation limits the influence
of contamination but is not a substitute for high-dimensional regularization.
Sparse precision encodes a conditional-dependence model, not a sparse covariance
or a test of whether an edge is scientifically real.

<a id="joint-distance"></a>

## A shared example: unusual combinations, not just large values

Both query points below are equally far from the origin in ordinary Euclidean
distance. The second point goes against the positive association in the data,
so it should be much farther away in the fitted covariance geometry.

<!-- example: covariance-geometry -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance

rng = np.random.default_rng(42)
z = rng.normal(size=(400, 2))
X = np.column_stack((z[:, 0], 0.8 * z[:, 0] + 0.6 * z[:, 1]))
model = EmpiricalCovariance(device="cpu").fit(X)
```

Inspect the covariance, its rescaled correlation, and the precision:

```python
variances = np.diag(model.covariance_)
correlation = model.covariance_ / np.sqrt(np.outer(variances, variances))
print(np.round(model.covariance_, 3))
print(np.round(correlation, 3))
print(np.round(model.precision_, 3))
```

```text
[[1.007 0.898]
 [0.898 1.126]]
[[1.    0.843]
 [0.843 1.   ]]
[[ 3.434 -2.737]
 [-2.737  3.069]]
```

Now measure two new rows:

```python
queries = np.array([[2.0, 2.0], [2.0, -2.0]])
distance_squared = model.mahalanobis(queries)
print(np.round(distance_squared, 3))
print(round(model.score(X), 3))
```

```text
[ 4.227 47.515]
-2.28
```
<!-- example-end: covariance-geometry -->

The squared Mahalanobis distance is

$$
d^2(x)=(x-\widehat\mu)^\top\widehat\Theta(x-\widehat\mu).
$$

The point `(2, -2)` is much less consistent with the observed positive
association. `mahalanobis` and `predict` both return these **squared distances**,
not ordinary distances, labels, probabilities or p-values. A threshold requires
separate assumptions and calibration; fitting a covariance does not automatically
supply an outlier test.

`score` is the mean Gaussian log-likelihood per row under the fitted location,
covariance and precision. Here `-2.28` is a training score, not an accuracy
percentage. For model comparison, use the same held-out rows in the same units;
larger scores are better. Different feature scalings change density units and
make unadjusted scores incomparable.

<a id="inference-boundaries"></a>

<a id="covariance-precision-and-inference-semantics"></a>

## What fitting does and does not tell you

These estimators describe variation among observed variables. Their covariance
matrix is not automatically the covariance of regression coefficient estimates.
For regression standard errors, use the downstream model's documented inference
method. See [regression inference](../guides/inference-api.md) and
[panel covariance](../panel/covariance.md) for those different targets.

No fit here automatically produces confidence intervals, p-values or significance
tests for covariance entries, robust support membership or selected graph edges.
CV chooses a predictive penalty, not a significance level. Any resampling study
must specify the sampling unit and whether centering, scaling, tuning and model
selection are repeated; merely inspecting a fitted matrix does not account for
these sources of uncertainty.

<a id="cpu-and-gpu-examples"></a>

<a id="optional-gpu-execution"></a>

## Practical input and device choices

Numeric matrices, nested lists and numeric pandas data are accepted; there is
no automatic alignment by DataFrame column names. Keep columns in the fitted
order and handle missing/infinite values before fitting or querying. A
one-dimensional fit input is treated as one feature; query shapes and all
restrictions are in the [API reference](../reference/covariance-api.md).

The following optional blocks reuse `X` above. They require the corresponding
CUDA runtime and are alternatives to the CPU fit, not prerequisites for it.

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

model_cupy = LedoitWolf(device="cuda").fit(cp.asarray(X))
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.as_tensor(X, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

An explicit GPU request raises an error if that runtime is unavailable. Automatic
selection and native-array device retention are described in the
[device and dtype rules](../reference/covariance-api.md#backend-device-and-dtype). Fitted queries use
the fitted backend/device even if global settings change later. Array outputs,
Torch input normalization (including supported FP8), and CPU control operations
are specified there rather than assumed to be fully GPU-resident.

<a id="external-validation"></a>

## Comparing estimates

Match observations, feature order, units, centering and covariance normalization
before comparing implementations. The empirical estimate uses `1/n`, whereas
many sample-covariance functions default to `1/(n-1)`. Match the shrinkage target
and strength, or Graphical Lasso penalty and convergence controls, as applicable.
MCD implementations can differ in subset search and correction details; a common
name does not guarantee identical results. See each tutorial's references.

<a id="earlier-section-links"></a>

## Where to go next

Choose a route for your current question; you do not need to read all seven
tutorials in order.

<a id="empirical-covariance"></a>
<a id="shrinkage-estimators"></a>
<a id="covariance-basic"></a>
<a id="estimate-a-covariance-matrix-on-cpu"></a>

- **Understand stability from a baseline**: start with [empirical covariance](covariance-empirical.md),
  then inspect how [fixed shrinkage](covariance-shrunk.md) changes eigenvalues.
  Compare [Ledoit–Wolf](covariance-ledoit-wolf.md) and [OAS](covariance-oas.md)
  when you want an automatically chosen shrinkage strength.

<a id="minimum-covariance-determinant"></a>
<a id="covariance-robust"></a>
<a id="robust-support"></a>

- **Assess contamination**: use the [MCD tutorial](covariance-min-cov-det.md) to
  compare ordinary and robust estimates, and understand why exclusion from
  `support_` is not proof of an erroneous observation.

<a id="graphical-lasso"></a>
<a id="covariance-sparse"></a>
<a id="sparse-precision-with-cross-validation"></a>

- **Study sparse conditional relationships**: read the [Graphical Lasso](covariance-graphical-lasso.md)
  objective and result interpretation, then use the [CV tutorial](covariance-graphical-lasso-cv.md)
  for penalty selection and held-out evaluation.

<a id="backend-and-execution-boundaries"></a>
<a id="common-parameters"></a>
<a id="estimation-algorithms"></a>
<a id="fitted-attributes-and-outputs"></a>
<a id="strict-and-approximate-behavior"></a>
<a id="api-and-execution-details"></a>

Once you have chosen a method, use the [complete API](../reference/covariance-api.md)
for parameters, fitted attributes and execution limits. There is no shared
strict/approximate switch; each estimator has its own algorithm.

## FAQ

### Which estimator should I use when $p$ is close to $n$?

Consider [LedoitWolf](covariance-ledoit-wolf.md) or [OAS](covariance-oas.md), while
checking whether the scaled-identity target suits the variables and their units.

### Does `MinCovDet` remove observations?

It estimates a robust center and covariance using a support mask; it does not
edit your data. Inspect support and distances in the context of the application.

### Why is the Graphical Lasso precision sparse but the covariance dense?

The penalty targets off-diagonal precision entries. Matrix inversion does not
preserve zeros in general.

### Can I pass a Torch CUDA tensor with `device="cuda"`?

Yes, when CuPy CUDA is available: this explicitly requests CuPy computation.
`device="torch"` requests Torch CUDA. Input type cannot override an explicit
backend request; see the [complete API](../reference/covariance-api.md).

## Paths

```python
from statgpu.covariance import (
    EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS,
    MinCovDet, GraphicalLasso, GraphicalLassoCV,
)
```

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
  for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance
  estimation with the graphical lasso.

- [Contributor validation reference](../../../dev/reviews/pr168-model-validation-provenance.md#covariance).
