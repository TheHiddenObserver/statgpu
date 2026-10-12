# GraphicalLassoCV: choose a sparse-model penalty with held-out data

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance-graphical-lasso-cv.md)

## How much sparsity should you ask for?

A small Graphical Lasso penalty can preserve noisy relationships; a large one
can erase useful dependence. `GraphicalLassoCV` compares candidate penalties on
rows excluded from each fit, selects the highest mean validation score, and
refits on every supplied row. It is a selection procedure for the same
[Graphical Lasso model](covariance-graphical-lasso.md), not a different covariance
model.

Use it when observations can reasonably be split into random folds and your
selection goal is Gaussian predictive fit. For a fixed, externally chosen
penalty, use `GraphicalLasso` directly. For covariance stability without graph
sparsity, consider [LedoitWolf](covariance-ledoit-wolf.md) or [OAS](covariance-oas.md).
The [covariance overview](covariance.md) helps with method selection;
the [API reference](../reference/covariance-api.md) explains inputs and the full
interface, including [backend, device, and dtype rules](../reference/covariance-api.md#backend-device-and-dtype).

## What does cross-validation maximize?

Each candidate $\alpha$ fits a precision matrix using the same objective as
Graphical Lasso:

$$
\operatorname{tr}(S\Theta)-\log\det\Theta
+\alpha\sum_{j\ne k}|\Theta_{jk}|,
\qquad \Theta\succ0.
$$

For validation fold $V_k$, the fitted training-fold location $\widehat\mu_k$
and covariance $\widehat\Sigma_k$ define the mean Gaussian log-likelihood

$$
s_k(\alpha)=-\frac12\left[
 p\log(2\pi)+\log\det\widehat\Sigma_k
 +\frac{1}{|V_k|}\sum_{i\in V_k}
 (x_i-\widehat\mu_k)^\top\widehat\Theta_k(x_i-\widehat\mu_k)
\right].
$$

The selected penalty maximizes $K^{-1}\sum_{k=1}^K s_k(\alpha)$, an unweighted
mean of fold scores. Larger values are better. The validation score excludes
the L1 penalty: it evaluates the Gaussian fit to unseen rows. Final
`covariance_`, `precision_`, and `location_` come from refitting the selected
penalty on the complete input, not averaging fold matrices.

## Compare five penalties on CPU

This example creates a six-feature Gaussian chain. It keeps 60 rows completely
outside selection so that the final score is not the score used to choose
`alpha`. Every import and data-generation step is included.

<!-- example: covariance-graphical-lasso-cv -->
Import the tools used in this example.

```python
import numpy as np
from statgpu.covariance import GraphicalLassoCV
```

Generate the data and reserve a separate test set.

```python
rng = np.random.default_rng(23)
true_precision = np.eye(6)
for j in range(5):
    true_precision[j, j + 1] = -0.3
    true_precision[j + 1, j] = -0.3
X = rng.multivariate_normal(
    np.zeros(6), np.linalg.inv(true_precision), size=240
)
X_train, X_test = X[:180], X[180:]
```

Select the penalty using four folds of the training data. The selected model is then refitted automatically.

```python
model = GraphicalLassoCV(
    alphas=[0.01, 0.03, 0.1, 0.3, 1.0],
    cv=4,
    random_state=7,
    max_iter=200,
    tol=1e-6,
    device="cpu",
).fit(X_train)
```

Compare candidate scores and inspect the final held-out score.

```python
for result in model.cv_results_:
    print(f"alpha={result['alpha']:.2f}, mean score={result['mean_score']:.3f}")
print("selected alpha:", model.alpha_)
print("refit rows:", model.n_samples_)
print("test score:", round(model.score(X_test), 3))
```
<!-- example-end: covariance-graphical-lasso-cv -->

Output, with scores rounded to three decimals:

```text
alpha=0.01, mean score=-8.829
alpha=0.03, mean score=-8.828
alpha=0.10, mean score=-8.833
alpha=0.30, mean score=-8.940
alpha=1.00, mean score=-9.037
selected alpha: 0.03
refit rows: 180
test score: -8.782
```

The mean score is largest at `0.03`, although `0.01` is almost tied. This is a
reason to inspect fold variability and sensitivity to the split, rather than
treating the winning value as a precisely estimated constant. The final fit
uses all 180 training rows. The held-out score of `-8.782` measures predictive
fit on the remaining 60 rows; it is neither an accuracy percentage nor proof
that the selected graph is the true graph.

`cv_results_` is a list of dictionaries, one per candidate, with keys `alpha`,
`mean_score`, and `scores`. The latter contains the individual fold scores.
It is not a scikit-learn-style dictionary of result arrays.

## Choose the grid and folds deliberately

- An integer `alphas=m` creates exactly `np.logspace(-2, 0, m)`: a fixed grid
  from `0.01` to `1.0` when `m > 1`, and `[0.01]` when `m == 1`.
  It is not a data-adaptive or successively refined search. An explicit
  nonempty sequence accepts finite, nonnegative penalties, including `0`.
- If the winner is at a boundary, consider extending the grid in that
  direction and checking numerical behavior. Penalties depend on feature
  scale, so this default range is not universal.
- `cv` must be an integer from 2 through the number of rows. The implementation
  randomly permutes rows and splits them into that many folds; `random_state`
  makes those splits reproducible. Every training fold must still contain at
  least two rows for a Graphical Lasso fit.
- Custom splitter objects, grouped folds, and time-series folds are not
  accepted. For repeated measurements, related subjects, or time-dependent
  observations, construct appropriate training/validation partitions yourself
  and fit `GraphicalLasso` on each partition.
- If scaling, imputation, or another transformation is estimated from data,
  fit it separately inside each training fold. `GraphicalLassoCV` does not
  offer a fold-specific preprocessing hook. Use a manual validation loop with
  `GraphicalLasso` when that separation is required.
- `n_jobs` is reserved; increasing it does not enable parallel CV here. More
  candidates and folds require more fits, followed by one final refit.

## Limits and diagnostics

The model inherits the direct estimator's numerical limitations. Low penalties
and ill-conditioned data may be difficult. `max_iter` and `tol` apply to every
fold and the final fit; reaching the iteration budget does not currently emit
a convergence warning. `GraphicalLassoCV` does **not** expose `n_iter_`.
For iteration diagnostics, fit `GraphicalLasso(alpha=model.alpha_, ...)`
separately with the same settings and data; this checks that direct fit, not
all of the earlier fold fits.

Inspect every candidate's fold scores for non-finite values before relying on
a selected penalty. Selection is not a numerical validity certificate. Keep an
independent test set, or use a suitable outer validation scheme, when reporting
performance after tuning. `predict` and `mahalanobis` still return squared
distances; this class does not predict graph labels or outlier classes.

## References and next steps

- Friedman, J., Hastie, T., & Tibshirani, R. (2008).
  [Sparse inverse covariance estimation with the graphical lasso](https://hastie.su.domains/Papers/graph.pdf).
  *Biostatistics*, 9(3), 432–441.
- [GraphicalLasso: interpreting precision entries and partial correlations](covariance-graphical-lasso.md).
- [Covariance overview and method selection](covariance.md).
- [Complete covariance API](../reference/covariance-api.md).
