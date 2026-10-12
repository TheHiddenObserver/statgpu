# Covariance API reference

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/reference/covariance-api.md)

This is the complete covariance-estimator API inventory. For choosing a method,
start with the [covariance overview](../models/covariance.md). The seven classes
are exported from both `statgpu.covariance` and `statgpu`.

| Class | Method guide | Additional constructor controls |
|---|---|---|
| `EmpiricalCovariance` | [Empirical covariance](../models/covariance-empirical.md) | None |
| `LedoitWolf` | [Ledoit–Wolf shrinkage](../models/covariance-ledoit-wolf.md) | None |
| `OAS` | [Oracle Approximating Shrinkage](../models/covariance-oas.md) | None |
| `ShrunkCovariance` | [Fixed shrinkage](../models/covariance-shrunk.md) | `shrinkage` |
| `MinCovDet` | [Minimum covariance determinant](../models/covariance-min-cov-det.md) | `support_fraction`, `random_state` |
| `GraphicalLasso` | [Graphical Lasso](../models/covariance-graphical-lasso.md) | `alpha`, `max_iter`, `tol` |
| `GraphicalLassoCV` | [Cross-validated Graphical Lasso](../models/covariance-graphical-lasso-cv.md) | `alphas`, `cv`, `max_iter`, `tol`, `random_state` |

## Constructors and common parameters

The signatures below list every constructor parameter in order. `Device.AUTO`
is the actual default enum value and is equivalent to passing `"auto"`.
`LedoitWolf` and `OAS` inherit the `EmpiricalCovariance` constructor.

```python
from statgpu import Device
from statgpu.covariance import (
    EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance,
    MinCovDet, GraphicalLasso, GraphicalLassoCV,
)

EmpiricalCovariance(assume_centered=False, device=Device.AUTO, n_jobs=None)
LedoitWolf(assume_centered=False, device=Device.AUTO, n_jobs=None)
OAS(assume_centered=False, device=Device.AUTO, n_jobs=None)
ShrunkCovariance(shrinkage=0.1, assume_centered=False, device=Device.AUTO, n_jobs=None)
MinCovDet(support_fraction=None, random_state=None, assume_centered=False,
          device=Device.AUTO, n_jobs=None)
GraphicalLasso(alpha=0.01, max_iter=100, tol=1e-4, assume_centered=False,
               device=Device.AUTO, n_jobs=None)
GraphicalLassoCV(alphas=4, cv=5, max_iter=100, tol=1e-4, random_state=None,
                 assume_centered=False, device=Device.AUTO, n_jobs=None)
```

| Parameter | Type / default | Behavior |
|---|---|---|
| `assume_centered` | bool, `False` | Estimate the location and subtract it before covariance calculation. `True` fixes the location at zero and uses input values directly; it does not verify that their mean is zero. For `MinCovDet`, this also applies to subset and reweighted locations. |
| `device` | str or `Device`, `Device.AUTO` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"`; see [backend and dtype rules](#backend-device-and-dtype). |
| `n_jobs` | int or `None`, `None` | Accepted and returned by `get_params`, but reserved: these estimators do not use it to parallelize fits or CV folds. |

`gpu_memory_cleanup` is **not** a constructor parameter or public control of
these seven classes. Passing it to a constructor raises `TypeError`; passing it
to `set_params` raises `ValueError`. Do not infer its availability from other
statgpu estimators. There are no `sample_weight`, formula, `store_precision`, or
precomputed-covariance options; `fit` always receives observations.

## Inputs and public methods

### Data shape and validation

Use finite, real, dense numeric data. NumPy arrays, CuPy arrays, supported dense
Torch tensors, numeric pandas `DataFrame`/`Series` objects, and ordinary numeric
array-like inputs such as nested lists are accepted. Numeric pandas nullable
dtypes are accepted when they contain no missing values. Rows are observations
and columns are features. DataFrame labels are not used to align, reorder, or
remember features; keep query columns in training order.

| Call | Two-dimensional input | One-dimensional input |
|---|---|---|
| `fit(X, y=None)` | `(n, p)`, `n >= 2`, `p >= 1` | `(n,)` becomes `(n, 1)`: n observations of one feature |
| `score(X, y=None)` | `(m, p)`, `m >= 1`, p must match training | `(m,)` becomes `(m, 1)`: valid only for a one-feature fit |
| `mahalanobis(X)` | `(m, p)`, `m >= 1`, p must match training | `(p,)` becomes `(1, p)`: one observation |
| `predict(X)` | Same as `mahalanobis` | Same as `mahalanobis` |

For one observation, prefer an explicit row such as `X_query.reshape(1, -1)`.
For several observations of one feature, use `X_query.reshape(-1, 1)`, especially
with `predict` or `mahalanobis`. Scalar, empty, higher-dimensional, and
feature-mismatched inputs fail validation. `MinCovDet` additionally requires
`n > p`; every Graphical Lasso CV training fold needs at least two rows.

NaN, infinity, and pandas missing values raise `ValueError` before estimation.
`y` is unused, but a supplied `y` still passes through the public finite-value
check and can raise for NaN/Inf. Leave it as `None`. Sparse, quantized, complex,
and nonnumeric inputs are outside this dense real-valued API: do not rely on
implicit casting or a uniform exception type for them. No missing-value
imputation, categorical encoding, or automatic standardization is performed.

### fit, score, predict, and mahalanobis

These methods are available on all seven classes, with the exact signatures
listed above. `fit` returns `self`; the other three require successful fitting
and raise `RuntimeError` on an unfitted estimator. Public input validation may
raise first when both the state and input are invalid.

| Method | Return | Definition |
|---|---|---|
| `fit(X, y=None)` | The same estimator | Populate the [common](#common-fitted-attributes) and class-specific fitted attributes. |
| `score(X, y=None)` | Python `float` | Mean Gaussian log-likelihood per query observation, using the fitted location, covariance determinant, and precision. Higher is better on the same evaluation rows. |
| `mahalanobis(X)` | NumPy `ndarray`, `float64`, shape `(m,)` | Squared Mahalanobis distance for each query row. |
| `predict(X)` | NumPy `ndarray`, `float64`, shape `(m,)` | Identical to `mahalanobis(X)`, including its one-dimensional input convention; no labels or thresholds. |

Writing the fitted location as $\mu$ and precision as $P$,

$$
d_i^2=(x_i-\mu)^\top P(x_i-\mu),\qquad
\operatorname{score}(X)=-\frac12\left[
p\log(2\pi)+\log\det(\hat\Sigma)+\frac1m\sum_{i=1}^m d_i^2
\right].
$$

`score` is an average, not a sum, accuracy, or an R-squared value. It does not
recenter the query rows at their own mean. Distances are squared; take their
square root explicitly if the ordinary Mahalanobis distance is needed.

Most classes compute precision by exact inversion first and add diagonal
stabilization only if inversion fails or returns nonfinite values. This does
not change `covariance_`. If the covariance determinant has nonpositive sign,
`score` returns `-inf`, even when stabilization produced finite `precision_`
and finite distances. Graphical Lasso has the pseudoinverse special case
[described below](#graphicallasso). Numerical stabilization is not a statistical
regularization choice or a guarantee of a meaningful singular Gaussian model.
Failure to obtain a finite stabilized inverse raises `ValueError`; other
backend linear-algebra errors can also propagate.

### get_params and set_params

- `get_params(deep=True)` returns a dictionary of the constructor parameters
  above, not fitted attributes. `deep=False` omits nested-estimator expansion;
  these classes have no estimator-valued constructor controls. Returned mutable
  values, such as an `alphas` list, are not an independent serialized model.
- `set_params(**params)` returns `self`. An empty call is a no-op. Unknown names
  raise `ValueError`. Most statistical parameter validation occurs at `fit`.
- To change settings reliably, construct and fit a fresh estimator with the
  desired parameters. Do not rely on `set_params` to reset fitted state or apply
  updated `max_iter`, `tol`, or `cv` values.

### Inherited statistical helpers

The remaining four public methods are inherited generic helpers. Their full
arguments, defaults, return fields, backend behavior, and statistical
limitations are documented in the [shared estimator API](estimator-api.md#inference-helpers):

| Method | Canonical signature and return reference |
|---|---|
| `adjust_pvalues` | [P-value adjustment](estimator-api.md#adjust_pvalues), returns a dictionary |
| `combine_pvalues` | [P-value combination](estimator-api.md#combine_pvalues), returns a dictionary |
| `bootstrap_statistic` | [Scalar bootstrap](estimator-api.md#bootstrap_statistic), returns `BootstrapResult` |
| `permutation_test` | [Permutation test](estimator-api.md#permutation_test), returns `PermutationTestResult` |

Covariance fitting supplies neither coefficient p-values nor the cached
regression X/y pairs used by the generic helpers' omitted-data defaults. Pass
p-values explicitly to `adjust_pvalues`/`combine_pvalues`, and arrays explicitly
to `bootstrap_statistic`; otherwise they raise `RuntimeError` even after a
covariance fit. `permutation_test` requires its explicit statistic, X, and y.
The helpers do not automatically refit covariance estimates or compute
covariance-entry standard errors. Their backend policy is separate from the
fitted-query policy below; consult their reference when selecting a backend.

## Common fitted attributes

All seven classes expose these attributes after a successful fit. Here `n` and
`p` refer to the training shape. A “backend array” means NumPy `ndarray`, CuPy
`ndarray`, or Torch `Tensor` on the selected device.

| Attribute | Type / shape | Meaning |
|---|---|---|
| `covariance_` | `float64` backend array, `(p, p)` | Estimated covariance; estimator-specific definitions appear below. |
| `precision_` | `float64` backend array, `(p, p)` | Inverse, stabilized inverse, or Graphical Lasso pseudoinverse special case. |
| `location_` | `float64` backend array, `(p,)` | Fitted location; zero with `assume_centered=True`. |
| `n_samples_` | Python `int` | Number of training rows, not the size of a robust support or a CV fold. |
| `n_features_` | Python `int` | Number of training columns. |

No fitted `n_features_in_`, feature-name alignment, coefficient table, or
regression p-values are provided by these classes. A failed refit does not
promise to erase results from an earlier successful fit; after a fit exception,
do not treat existing attributes as results for the failed input.

## EmpiricalCovariance

No parameters or fitted attributes beyond the common inventory. For centered
data $X_c$, `covariance_ = X_c.T @ X_c / n`; the divisor is n, not n−1.
`precision_` uses the inversion/stabilization behavior above. Rank deficiency is
not always rejected at fit time and can lead to `score(X) == -inf`.

## LedoitWolf

No additional constructor parameters. Fits

$$
\hat\Sigma=(1-\alpha)S+\alpha\mu I,\qquad \mu=\operatorname{tr}(S)/p,
$$

where S uses the same centering and `1/n` normalization as empirical covariance.
The analytical shrinkage intensity is clipped to `[0, 1]`; a degenerate
nonpositive-denominator case uses complete shrinkage.

| Additional attribute | Type | Meaning |
|---|---|---|
| `shrinkage_` | Python `float` | Data-selected intensity in `[0, 1]`. |

The target is a scaled identity, with no configurable structural target or
block-size parameter.

## OAS

Same constructor and shrinkage form as `LedoitWolf`, with the Oracle
Approximating Shrinkage intensity. The implementation follows the commonly used
scikit-learn convention omitting the original formula's `2/p` correction; see
the [OAS guide](../models/covariance-oas.md) for interpretation.

| Additional attribute | Type | Meaning |
|---|---|---|
| `shrinkage_` | Python `float` | Data-selected intensity clipped to `[0, 1]`; complete shrinkage if its denominator is nonpositive. |

## ShrunkCovariance

| Additional parameter | Type / default | Validation and behavior |
|---|---|---|
| `shrinkage` | float, `0.1` | Fixed intensity in the same scaled-identity formula. `fit` raises `ValueError` for a nonfinite value or a value outside `[0, 1]`. |

| Additional attribute | Type | Meaning |
|---|---|---|
| `shrinkage_` | Numeric scalar (normally `float`) | The supplied `shrinkage` value; it is not optimized. |

Zero uses empirical covariance; one uses the scaled identity. With default
mean-centering, shrinking a dataset whose features all have zero variance
still produces a singular zero covariance.

## MinCovDet

| Additional parameter | Type / default | Validation and behavior |
|---|---|---|
| `support_fraction` | float or `None`, `None` | `None` uses `h = min(ceil((n + p + 1) / 2), n)`. A supplied finite fraction in `(0, 1]` uses `ceil(fraction * n)`, clipped to `[p + 1, n]`. Invalid numeric fractions raise `ValueError` at fit. |
| `random_state` | int or `None`, `None` | Seed for NumPy random initial subset selection. Use an integer for repeatable selection in the same numerical environment. |

`n > p` is required; otherwise `fit` raises `ValueError`. This condition alone
does not guarantee full-rank support data. If no positive-determinant support
covariance is found, fitting raises `ValueError`.

| Additional attribute | Type / shape | Meaning |
|---|---|---|
| `support_` | Boolean backend array, `(n,)` | Final training-row support mask after reweighting, or the raw selected support in the fallback below. |
| `raw_location_` | `float64` backend array, `(p,)` | Selected subset location before correction/reweighting; zero when centered input is assumed. |
| `raw_covariance_` | `float64` backend array, `(p, p)` | Selected subset covariance divided by h, **before** consistency correction. |
| `dist_` | `float64` backend array, `(n,)` | Squared distances for the training rows under the final location and covariance/precision calculation. |

The raw covariance receives a Gaussian consistency correction. Reweighting
uses the chi-square 97.5th percentile with p degrees of freedom. If fewer than
`p + 1` rows qualify, the result retains the raw support and corrected raw
covariance; otherwise it recomputes and corrects covariance on the reweighted
support. Thus `support_fraction` controls the initial support size, not the
final fraction of `True` values. `support_` is not a prediction for new rows;
use squared distances to evaluate new observations. There is no `raw_support_`
attribute or built-in outlier-label threshold method.

## GraphicalLasso

| Additional parameter | Type / default | Validation and behavior |
|---|---|---|
| `alpha` | float, `0.01` | Finite nonnegative L1 penalty on off-diagonal precision entries; invalid numeric values raise `ValueError` at fit. Diagonal entries are not penalized. |
| `max_iter` | int, `100` | Positive integer outer-iteration limit; Boolean, noninteger, and nonpositive values raise `ValueError` at fit. |
| `tol` | float, `1e-4` | Finite positive stopping tolerance for the largest absolute covariance change between outer iterations; invalid numeric values raise `ValueError` at fit. |

| Additional attribute | Type | Meaning |
|---|---|---|
| `n_iter_` | Python `int` | Number of outer iterations performed. Equals 1 when `alpha == 0` or `p == 1`. |

For `alpha == 0` **or** one feature, covariance is the empirical estimate and
precision is its pseudoinverse. Otherwise the fit uses block coordinate updates
and stabilized inversion. A nonpositive empirical variance diagonal raises
`ValueError` on the positive-alpha, multi-feature path. Reaching `max_iter`
returns the current estimate without a convergence warning or a separate
convergence flag; `n_iter_ == max_iter` does not establish convergence. There
are no public solver, warm-start, dual-gap, or objective-history controls.

## GraphicalLassoCV

| Additional parameter | Type / default | Validation and behavior |
|---|---|---|
| `alphas` | positive int or array-like, `4` | An integer k generates `np.logspace(-2, 0, k)`, a fixed grid from 0.01 to 1 (k=1 gives only 0.01). Explicit values are converted to `float64` and flattened; they must form a nonempty finite nonnegative collection. |
| `cv` | int, `5` | Integer fold count with `2 <= cv <= n`; Boolean, noninteger, and out-of-range values raise `ValueError`. Splitter objects and user-supplied split iterables are not supported. |
| `max_iter` | int, `100` | Same positive outer-iteration limit as `GraphicalLasso`, used for every fold fit and final refit. |
| `tol` | float, `1e-4` | Same finite positive covariance-change tolerance as `GraphicalLasso`. |
| `random_state` | int or `None`, `None` | Seed for the shuffled NumPy permutation used to create folds. |

Rows are always shuffled and divided into `cv` folds using approximately equal
sizes. There is no `shuffle` switch, stratification, grouped splitting, or
time-aware splitting. All training folds must have at least two rows: for
example, `n=2, cv=2` passes the fold-count check but fails when a fold is fitted.
The grid is fixed, not adapted or refined from the data. Explicit candidate
order and duplicates are retained.

Each candidate is fitted on every training fold and scored by mean Gaussian
log-likelihood on its test fold. The selection score is the unweighted mean
of fold scores, including when fold sizes differ. The highest mean wins, and
exact ties retain the earliest candidate. If every mean is `-inf`, the first
candidate remains selected; inspect scores before treating selection as useful.
A failed fold fit raises instead of assigning an `error_score` and continuing.
Finally, the selected penalty is fitted on all training rows.

| Additional attribute | Type / shape | Meaning |
|---|---|---|
| `alpha_` | Python `float` | Selected penalty from the candidate grid. |
| `cv_results_` | `list[dict]`, length = number of candidates | One record per candidate in input/grid order, with exactly `alpha` (float), `mean_score` (float), and `scores` (list of `cv` floats). |

`covariance_`, `precision_`, and `location_` describe the final full-data refit.
This class exposes neither `n_iter_` nor `best_score_`; it does not retain a
public final estimator object. In particular, `cv_results_` is not a
scikit-learn-style dictionary of arrays, and there is no fitted `alphas_` grid
attribute. Read the candidate penalties from its records.

<a id="device-and-dtype"></a>

## Backend, device, and dtype

The following applies to `fit`, `score`, `predict`, and `mahalanobis` for all
seven classes; generic inherited helpers have their own documented policy.

| Estimator setting | Computation |
|---|---|
| `device="cpu"` | NumPy CPU, converting input arrays from other libraries as needed. |
| `device="cuda"` | CuPy CUDA, including when input is a Torch tensor. Requires a working CuPy CUDA runtime. |
| `device="torch"` | Torch CUDA. A Torch CPU input is moved to CUDA; Torch CPU does not satisfy this explicit GPU request. |
| `device="auto"` | Inherit `statgpu.set_device(...)`. Only when the global setting is also automatic, native CuPy/Torch input keeps its backend and device (including Torch CPU); other inputs choose available CuPy CUDA, then Torch CUDA, then NumPy. |

An explicit estimator setting overrides global configuration. Explicit global
GPU requests used by an automatic estimator are also strict: an unavailable
requested runtime raises, without silently choosing CPU or another GPU library.
Invalid device values raise `ValueError`; estimator values do not accept indexed
strings such as `"cuda:1"`. An already-native GPU input retains its index when
that backend is selected. CPU inputs moved to GPU use the library's current
CUDA device. See the [device guide](../guides/device-and-memory.md).

Numerical covariance arrays use `float64`, regardless of the supported input
floating precision or integer dtype. Boolean support masks retain Boolean
dtype. Fitted arrays remain on the selected backend/device; they are not all
converted to NumPy. The exception is query output: `score` returns a Python
float, and `mahalanobis`/`predict` return CPU NumPy arrays.

### Torch normalization and float8

When NumPy computation is selected, supported real Torch inputs are detached
from autograd, transferred to CPU, and normalized to `float64` before the
finite-value check. The caller's tensor is not modified. Supported cases
include `bfloat16`, `float8_e5m2`, and real-valued views of dense,
non-quantized tensors, including lazy negative views. Transferring before the
cast also permits CPU computation from MPS tensors, whose source device does
not support `float64`.

For Torch computation, dense `float8_e5m2` is widened to `float64` on its
source device before checking finiteness. Native CUDA data stays on CUDA.
This avoids requiring a raw float8 finite-check kernel, but still requires the
Torch runtime to support conversion to `float64`. This handling does not add
support for other float8 formats, sparse/quantized tensors, or complex inputs,
and it is not a promise of automatic differentiation through an estimator.
Do not assume all source-device/dtype combinations are supported just because
an explicit CPU conversion is supported.

### Fitted-device ownership and execution boundaries

`score`, `mahalanobis`, and `predict` convert query input to the **fitted**
backend and device, including its CUDA index. They do not select a new backend
from query type or a subsequently changed global device policy.
`GraphicalLassoCV` keeps the backend and device chosen at the beginning of fit
through folds, scoring, and the final refit.

Matrix products, centering, covariance updates, inversion, Graphical Lasso
coordinate updates, and MCD distance/C-step/reweighting calculations use the
selected numerical backend. Random subset/fold indices, MCD chi-square cutoffs,
Python control flow, and some scalar diagnostics use CPU. The API therefore
does not promise an entirely GPU-resident workflow or asynchronous results.
There is no shared strict/approximate mode; backend selection, numerical
stabilization, and the statistical estimator are separate choices.
