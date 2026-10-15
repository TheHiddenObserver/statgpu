# MinCovDet: estimate the center and spread despite outlying rows

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../../cn/models/covariance/covariance-min-cov-det.md)

## What if a few observations distort the whole covariance?

A small group of unusual observations can pull the ordinary mean away from the
main cloud and inflate its covariance. Their influence can even hide how
unusual they are when distances are measured using that same inflated matrix.
`MinCovDet` estimates location and covariance from a concentrated subset, then
uses robust distances to reweight the observations.

It is useful when most rows form one approximately elliptical population and
a minority may be contaminated. It is not a general clustering method: a second
legitimate population can be downweighted as readily as a recording error.
The [covariance overview](README.md) helps with method selection;
the [complete API reference](../../reference/covariance-api.md) covers inputs,
parameters, methods, and fitted attributes. See its
[backend, device, and dtype section](../../reference/covariance-api.md#backend-device-and-dtype)
for device settings.

## The smallest-volume-subset intuition

For a subset $H$ containing $h$ rows, define

$$
\overline x_H=\frac1h\sum_{i\in H}x_i,
\qquad
S_H=\frac1h\sum_{i\in H}
(x_i-\overline x_H)(x_i-\overline x_H)^\top.
$$

The minimum covariance determinant target is

$$
\widehat H\in\arg\min_{H:\,|H|=h}\det(S_H).
$$

The determinant measures multivariate spread. Searching for a small determinant
seeks a concentrated group rather than allowing distant points to stretch the
estimate. Enumerating every subset is impractical; FAST-MCD uses repeated
initial subsets and concentration steps. The search is approximate and need
not find the global minimum. See the [FAST-MCD paper](https://wis.kuleuven.be/stat/robust/papers/1999/rousseeuwvandriessen-fastalgorithmformcd-technomet.pdf)
for the algorithm's statistical background.

The implementation corrects the raw subset covariance for consistency, computes
squared robust distances, and reweights rows using a chi-square cutoff at
probability `0.975` with $p$ degrees of freedom. It then estimates the final
location and corrected covariance from that support. If too few rows remain,
it uses the raw subset with its corrected covariance instead. With
`assume_centered=True`, zero replaces the estimated center throughout.

## Compare ordinary and robust estimates on CPU

The following self-contained example creates 180 rows near `[0, 0]` and 20
clearly shifted rows near `[6, -6]`. Their labels are known only because this is
a simulation. Both estimators receive all 200 rows without labels.

<!-- example: covariance-min-cov-det -->
Import the ordinary and robust estimators.

```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, MinCovDet
```

Create the clean population and the shifted group, then combine them.

```python
rng = np.random.default_rng(42)
clean = rng.multivariate_normal(
    [0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=180
)
contamination = rng.normal(loc=[6.0, -6.0], scale=0.5, size=(20, 2))
X = np.vstack([clean, contamination])
```

Fit both models on the same input.

```python
ordinary = EmpiricalCovariance(device="cpu").fit(X)
robust = MinCovDet(random_state=42, device="cpu").fit(X)
```

Inspect their centers, the robust support, and distances for the two known simulation groups.

```python
print("ordinary center:", np.round(ordinary.location_, 3))
print("robust center:", np.round(robust.location_, 3))
print("robust covariance:")
print(np.round(robust.covariance_, 3))
print("final support:", int(robust.support_.sum()))
print("shifted rows in support:", int(robust.support_[180:].sum()))
print("median clean squared distance:", round(float(
    np.median(robust.mahalanobis(clean))
), 3))
print("median shifted squared distance:", round(float(
    np.median(robust.mahalanobis(contamination))
), 3))
```
<!-- example-end: covariance-min-cov-det -->

Output, rounded to three decimals:

```text
ordinary center: [ 0.597 -0.635]
robust center: [0.056 0.027]
robust covariance:
[[0.95  0.606]
 [0.606 0.857]]
final support: 174
shifted rows in support: 0
median clean squared distance: 1.391
median shifted squared distance: 238.589
```

The robust center remains close to the simulated clean center, and the robust
covariance retains its positive association. The final support contains 174
rows, all from the clean group. It therefore also excludes six genuinely clean
rows: `support_ == False` is not automatically a label for a bad observation.
The mask describes participation in robust fitting; it does not delete rows
from `X` or provide a calibrated outlier classifier.

The median squared distance is much larger for the shifted group. Both
`mahalanobis` and `predict` return these squared distances. `dist_` contains
squared distances for the training rows under the final estimate. Because
support is chosen using the corrected raw estimate before the final refit,
thresholding `dist_` need not reproduce `support_`.

## Choose the subset size and check assumptions

- By default, the raw subset size is
  $h=\lceil(n+p+1)/2\rceil$, capped at $n$. This is the search subset size,
  not necessarily the size of the final reweighted support.
- Set `support_fraction` in `(0, 1]` to use $\lceil n\,\text{support\_fraction}\rceil$
  rows, clipped to the range $[p+1,n]$. A larger subset generally trades
  resistance to contamination for more information from clean observations.
  Do not interpret this parameter as a requested final outlier fraction.
- `random_state` controls randomized subset initialization. For important
  analyses, check stability across seeds instead of relying on one search.
- The implementation requires `n_samples > n_features`. This is necessary
  but not sufficient for a nonsingular support covariance. Duplicate or
  linearly dependent features can still prevent a usable fit.
- Handle missing or infinite values before fitting. A single dominant
  elliptical cloud is a better match than multimodal, strongly skewed, or
  manifold-constrained data.

`raw_location_` is the selected subset's center, and `raw_covariance_` is its
uncorrected covariance with divisor $h$. Use `location_` and `covariance_` for
the final robust estimate. The final `score(X)` remains a mean Gaussian
log-likelihood; it is not a robust-loss score or an outlier detection accuracy.
A robust estimator need not maximize likelihood on contaminated data, because
it intentionally limits those rows' influence.

## Alternatives and limitations

For clean, well-sampled data, [EmpiricalCovariance](covariance-empirical.md) is
simpler and avoids randomized subset search. When $p$ approaches or exceeds
$n$, [LedoitWolf](covariance-ledoit-wolf.md) and [OAS](covariance-oas.md) offer
regularization, but do not provide MCD-style contamination resistance. For a
sparse conditional-dependence graph, use [GraphicalLasso](covariance-graphical-lasso.md);
it answers a different question from robust location and scatter.

MCD costs more than direct covariance calculation. Numerical inverse
stabilization cannot make a degenerate support statistically informative.
Inspect finite outputs, covariance conditioning, and sensitivity to the subset
size before using distances downstream. Gaussian chi-square cutoffs are useful
reference points, not automatic false-positive guarantees for fitted distances
in arbitrary real data.

## References and next steps

- Rousseeuw, P. J., & Van Driessen, K. (1999).
  [A fast algorithm for the minimum covariance determinant estimator](https://wis.kuleuven.be/stat/robust/papers/1999/rousseeuwvandriessen-fastalgorithmformcd-technomet.pdf).
  *Technometrics*, 41(3), 212–223.
- [Covariance overview and method selection](README.md).
- [Complete covariance API](../../reference/covariance-api.md).
