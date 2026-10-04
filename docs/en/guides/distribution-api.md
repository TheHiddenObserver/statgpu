# Distribution API Guide

> Language: English
> Last updated: 2026-10-04
> This page: Guide
> Switch: [Chinese](../../cn/guides/distribution-api.md)

## What can a distribution tell you?

A probability distribution turns a model assumption into probabilities,
percentiles, densities, or simulated observations. Use this API when the
reference distribution and its parameters are already known: for example,
normal measurement errors, a Student t test statistic, or Poisson event counts.
It does not fit a distribution to data or check whether that assumption is
appropriate. For an unknown density, consider [kernel density estimation](../models/nonparametric.md).
For regression inference, start with the fitted model's inference output rather
than choosing a reference distribution without accounting for how it was fitted.

Import the object-style API from `statgpu.inference`. The examples below run
independently after installing statgpu with NumPy and SciPy; no GPU is required.

## Start with probabilities and percentiles

For a standard normal variable, `cdf(1)` is the chance of an observation at or
below 1; `ppf(0.975)` asks which value has 97.5% of the distribution below it.
These are opposite directions of the same question.

```python
# Example: normal_probabilities
import numpy as np
from statgpu.inference import norm

x = np.array([-1.0, 0.0, 1.0])
cdf = norm.cdf(x, backend="numpy")
sf = norm.sf(x, backend="numpy")
density = norm.pdf(x, backend="numpy")
quantiles = norm.ppf(np.array([0.025, 0.5, 0.975]), backend="numpy")
upper_cutoff = norm.isf(0.025, backend="numpy")
central_probability = norm.cdf(1.96, backend="numpy") - norm.cdf(-1.96, backend="numpy")
print("CDF:", np.round(cdf, 6))
print("SF:", np.round(sf, 6))
print("Density:", np.round(density, 6))
print("Quantiles:", np.round(quantiles, 6))
print("Upper cutoff:", round(float(upper_cutoff), 6))
print("Central probability:", round(float(central_probability), 6))
```

Expected rounded output:

```text
CDF: [0.158655 0.5      0.841345]
SF: [0.841345 0.5      0.158655]
Density: [0.241971 0.398942 0.241971]
Quantiles: [-1.959964  0.        1.959964]
Upper cutoff: 1.959964
Central probability: 0.950004
```

About 95% of this distribution lies between -1.96 and 1.96. The density at zero
is about 0.399, but it is **not** the probability of observing exactly zero:
that probability is zero for a continuous distribution. A density can exceed 1;
probability is the area over an interval.

### Choose the method for the question

| Method | Meaning | Input and interpretation |
|---|---|---|
| `cdf(x, ...)` | $P(X\le x)$ | Observation threshold; returns a probability. |
| `sf(x, ...)` | $P(X>x)$ | Strict upper tail. For integer counts, $P(X\ge k)=\operatorname{sf}(k-1)$. |
| `ppf(q, ...)` | Lower-tail quantile | Probability `q` in `[0, 1]`; for discrete distributions and `0 < q < 1`, the smallest supported integer whose CDF reaches `q`. |
| `isf(q, ...)` | Upper-tail quantile | Uses the upper-tail probability; corresponds to `ppf(1-q)`, with discrete quantile conventions. |
| `pdf(x, ...)` | Continuous density | Available for continuous families; not a point probability. |
| `pmf(k, ...)` | $P(X=k)$ | Available for Poisson/binomial counts; noninteger counts have mass zero. |
| `rvs(..., size=...)` | Random observations | `size` is an integer, tuple of dimensions, or `None` for a scalar-shaped result. Samples are observations, not probabilities. |

Quantile endpoints can be infinite. Poisson/binomial `ppf(0)` uses the
lower-support-minus-one convention (`-1` when `loc=0`); it is not a possible
draw. Prefer interior probabilities for numerical inverse checks. Invalid
parameters often produce `nan`, while unsupported arguments can raise; validate
inputs rather than relying on one uniform error behavior.

## Counts: mass, cumulative probability, and an upper threshold

```python
# Example: count_probabilities
import numpy as np
from statgpu.inference import poisson, binom

k = np.array([0, 1, 2, 3])
mass = poisson.pmf(k, mu=3.0, backend="numpy")
cumulative = poisson.cdf(k, mu=3.0, backend="numpy")
probability_at_least_three = poisson.sf(2, mu=3.0, backend="numpy")
cutoff_95 = poisson.ppf(0.95, mu=3.0, backend="numpy")
binomial_quantiles = binom.ppf(np.array([0.1, 0.5, 0.9]), n=20, p=0.2, backend="numpy")
print("Mass:", np.round(mass, 6))
print("CDF:", np.round(cumulative, 6))
print("P(X >= 3):", round(float(probability_at_least_three), 6))
print("95% count cutoff:", int(cutoff_95))
print("Binomial quantiles:", binomial_quantiles.astype(int))
```

Expected values are mass `[0.049787, 0.149361, 0.224042, 0.224042]`, CDF
`[0.049787, 0.199148, 0.423190, 0.647232]`, probability `0.576810`, cutoff
`6`, and binomial quantiles `[2, 4, 6]`. At least 95% of a Poisson distribution
with mean 3 is at or below 6; the cumulative probability at 5 is below 95%.
Use Poisson for an unbounded nonnegative event count under its assumptions;
use binomial for successes among a fixed number of independent trials with a
common success probability.

Pass the threshold positionally to `poisson.cdf(k, mu=...)` or use `x=k`.
The module-level proxy's CDF argument is named `x`, so `cdf(k=..., mu=...)`
is not a valid proxy call. Shape parameters such as `mu`, `df`, `n`, and `p`
must be keywords when using proxies.

## A practical two-sided test and confidence interval

Suppose these are independent observations from a normal population with unknown
variance. The one-sample t statistic is
$t=(\bar x-\mu_0)/(s/\sqrt n)$ with `df=n-1`. The assumption justifies the
reference distribution; calling a CDF alone does not establish it.

```python
# Example: student_t_inference
import numpy as np
from statgpu.inference import t

values = np.array([2.1, 2.4, 2.2, 2.6, 2.8, 2.0, 2.7, 2.3, 2.9, 2.5, 2.4])
null_mean = 2.0
alpha = 0.05
mean = values.mean()
standard_error = values.std(ddof=1) / np.sqrt(values.size)
df = values.size - 1
statistic = (mean - null_mean) / standard_error
pvalue = t.two_sided_pvalue(abs(statistic), df=df, backend="numpy", use_lut=False)
critical = t.two_sided_critical_value(alpha, df=df, backend="numpy", use_lut=False)
interval = mean + np.array([-1.0, 1.0]) * critical * standard_error
print("Mean and SE:", round(mean, 6), round(standard_error, 6))
print("t and two-sided p:", round(statistic, 6), round(float(pvalue), 6))
print("Critical value:", round(float(critical), 6))
print("95% mean interval:", np.round(interval, 6))
print("Reject at 5%:", bool(pvalue < alpha))
```

Expected output is mean `2.445455`, SE `0.086722`, statistic `5.136596`,
p-value `0.000440`, positive critical value `2.228139`, interval
`[2.252226, 2.638683]`, and `True`. The null mean 2 lies outside the interval.
The p-value is a tail probability under the null, not the probability that the
null is true. This is a confidence interval for the population mean, not a
prediction interval for another observation.

`norm` and `t` provide `two_sided_pvalue(stat_abs, ...)` and
`two_sided_critical_value(alpha, ...)`. Pass an absolute, standardized statistic,
and `0 < alpha < 1`; these helpers do not accept `loc` or `scale`. For a
right-sided test use `sf(statistic, ...)` and compare with `alpha`, with a
critical value from `isf(alpha, ...)`. Do not halve a two-sided p-value without
considering the sign and the pre-specified alternative. Multiple tests require
an appropriate adjustment; see the [inference guide](inference-api.md).

### Tail accuracy has limits

Prefer `sf` over computing `1-cdf` yourself, especially for the normal upper
tail. However, several current families implement `sf` by subtraction, and
native `isf` methods form `1-q`: very small `q` can round away and produce an
infinite or inaccurate cutoff. `use_lut=False` does not remove that cancellation.
Extreme tails, singular density endpoints, and unusual parameters need a
method-specific reference check. Use SciPy's corresponding `sf`, `logsf`, or
`isf` directly on CPU when its specialized tail implementation is needed;
`logcdf` and `logsf` are not exposed by these native objects. In particular,
several positive-support/beta density kernels return zero exactly at the
support boundary rather than the analytic boundary limit. Evaluate their PDF
in the interior when checking against a reference.

## Native families and parameter choices

This is the complete native name list. All rows provide `cdf`, `sf`, `ppf`,
and `isf`; continuous rows also provide `pdf`, discrete rows `pmf`. `rvs` is
available except for the current F sampling failure described below.
Parameters are scalars; vectorize over thresholds/probabilities, not over a
batch of different shape parameters. `loc`/`scale` default to `0`/`1` where
supported, and `scale` must be positive.

| Name(s) | Required shape parameters | Support at default `loc`/`scale` | Location/scale controls |
|---|---|---|---|
| `norm`, `t` | None for normal; `df > 0` for t | Real line | `loc`, `scale` |
| `uniform` | None | `[0, 1]` | `loc`, `scale` (interval width) |
| `expon` | None | `[0, infinity)` | `loc`, `scale` (mean above `loc`, reciprocal of rate) |
| `cauchy`, `laplace`, `logistic` | None | Real line | `loc`, `scale` |
| `chi2` | `df > 0` | `[0, infinity)` | Neither |
| `gamma` | `a > 0` | `[0, infinity)` | `loc`, `scale`; mean is `loc + a*scale` |
| `beta` | `a > 0`, `b > 0` | `[0, 1]` | `loc`, `scale` |
| `f` | `dfn > 0`, `dfd > 0` | `[0, infinity)` | Neither |
| `weibull_min` | `c > 0` | `[0, infinity)` | `loc`, `scale` |
| `lognorm` | `s > 0` | `(0, infinity)` | `loc`, `scale`; at `loc=0`, `log(X)` has mean `log(scale)` and SD `s` |
| `poisson` | `mu >= 0` | Integers `0, 1, ...` | Integer `loc` only |
| `binom` | Integer `n >= 0`, `0 <= p <= 1` | Integers `0, ..., n` | Integer `loc` only |

For normal, `scale` is a standard deviation, not a variance. Gamma's `scale`
is the reciprocal of a rate, and `a` is its shape. Supply an integer `n` to
binomial calls: the implementation's integer coercion is not input validation.
For a noninteger count threshold, the discrete CDF uses its floor.
These objects are SciPy-like, not complete `scipy.stats` replacements: they do
not expose frozen-call syntax, `fit`, `stats`, `mean`, `var`, or log-density methods.

## Backend selection, output types, and conversion

There are two different automatic-selection rules:

- **Module-level proxies**, such as `norm.cdf(x)`, inspect call arguments for
  Torch tensors or CuPy arrays. The first recognized such array selects the
  backend; Python scalars/lists and NumPy-only input use NumPy. A Torch tensor's
  device is used unless overridden. Do not mix array libraries/devices in one call.
- **`get_distribution(name, backend="auto")`** has no input array to inspect.
  It attempts construction in CuPy, Torch, then NumPy order. Torch can select
  CPU when CUDA is unavailable. Construction alone does not verify that every
  later GPU operation will work. Choose an explicit backend for predictable behavior.

Proxies accept `backend="numpy"`, `"cupy"`, `"torch"`, or `"auto"`, plus
`device` (Torch) and `use_lut`, per call. Fixed objects receive those settings
at construction instead; do not pass them again to the object's methods.
Distribution `backend="torch", device="cpu"` is valid; this functional API is
distinct from estimators' strict `device="torch"` CUDA setting.

```python
# Example: fixed_numpy_backend
import numpy as np
from statgpu.inference import norm, get_distribution, list_available_distributions

x = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
automatic = norm.cdf(x)
fixed = get_distribution("norm", backend="numpy", use_lut=False)
result = fixed.cdf(x, loc=0.0, scale=1.0)
scalar = float(fixed.cdf(0.0))
np.testing.assert_allclose(automatic, result)
print("Shape and dtype:", result.shape, result.dtype)
print("Scalar CDF:", scalar)
print("Native count:", len(list_available_distributions()))
```

Expected: shape `(2, 2)`, dtype `float64`, scalar probability `0.5`, and 15
native families. Native calculations convert numeric inputs to float64.
NumPy calls return NumPy arrays or scalar/zero-dimensional results; CuPy calls
return CuPy arrays; Torch calls return tensors on the resolved device. Array
outputs follow the threshold/probability shape. Use `float(result)` only for
one element; it can synchronize a GPU. Convert arrays deliberately with
`np.asarray(result)` for NumPy, `cupy.asnumpy(result)` for CuPy, or
`result.detach().cpu().numpy()` for Torch. Moving GPU results to CPU has a
transfer cost, and detaching a tensor removes its gradient connection.

To use a GPU, create an input with `cupy.asarray(..., dtype=cupy.float64)` and
call the proxy, or create a fixed object with
`get_distribution("norm", backend="torch", device="cuda:0")` and supply a
compatible tensor. An explicit unavailable backend/device can raise during
construction or evaluation. See [device and memory](device-and-memory.md) for
installation/device context; CPU-first examples do not establish GPU accuracy
or performance.

## Reproducible sampling

Current native `rvs` methods generate values with NumPy/SciPy **on CPU**, then
convert to the selected backend. They use NumPy's global random state, accept
no `random_state` argument, and currently do not honor the accepted `dtype`
argument. Seed NumPy for reproducibility in one environment and cast the result
explicitly if needed. A separate `np.random.default_rng(...)` does not seed
these calls. Exact sequences need not be portable across library versions.

```python
# Example: reproducible_sampling
import numpy as np
from statgpu.inference import norm, poisson

np.random.seed(7)
sample = norm.rvs(size=(2, 3), backend="numpy")
counts = poisson.rvs(mu=3.0, size=4, backend="numpy")
counts_integer = counts.astype(np.int64)
print("Normal sample:", np.round(sample, 6))
print("Count sample:", counts_integer)
print("Original dtypes:", sample.dtype, counts.dtype)
```

The normal sample rounds to `[[1.690526, -0.465937, 0.032820],
[0.407516, -0.788923, 0.002066]]`; counts are `[4, 1, 4, 5]`. Both original
arrays have dtype `float64`, even though the Poisson observations are integral.
These six normal draws are not an estimate precise enough to validate a distribution.

### Current F sampling limitation

`f.rvs(dfn=..., dfd=...)` and `rf_gpu(...)` currently raise `TypeError`
because the native sampling helper passes unsupported argument names to
NumPy. F density, CDF, and quantile methods are unaffected. For CPU sampling,
use SciPy directly; unlike native `rvs`, its sampler accepts `random_state`:

```python
# Example: f_sampling_workaround
import numpy as np
from scipy import stats

sample = stats.f.rvs(dfn=5, dfd=10, size=4, random_state=np.random.default_rng(17))
print("F sample:", np.round(sample, 6))
```

Expected rounded draws are `[2.350997, 0.207108, 0.491596, 3.343483]`.
They are positive observations from the F distribution, not tail probabilities.

## LUTs and backend-specific numerical limitations

`use_lut=True` is the default. Eligible inverse incomplete-beta/gamma paths use
lookup tables and refinement; those functions support quantiles for `t`, `f`,
`beta`, `chi2`, and `gamma`. Eligibility depends on the parameters and backend.
The flag is not a universal speed or error guarantee, and normal quantiles do
not use these LUTs.

| Backend | What `use_lut` controls / what to check |
|---|---|
| NumPy | Eligible inverse beta/gamma paths use LUT interpolation plus Newton refinement; `False` uses SciPy special-function inverses. |
| CuPy | Eligible inverse beta/gamma paths also use LUTs. `False` uses `cupyx.scipy.special` inverses; the flag does affect CuPy. LUT creation uses host SciPy and transfers tables to the device. |
| Torch | Selects eligible LUT paths versus numerical fallbacks; incomplete-beta forward evaluation also has LUT paths. Native special-function availability is checked at runtime, not guaranteed by a Torch version number. |

Torch LUT creation uses host SciPy. Without a native incomplete-beta function,
non-scalar incomplete-beta evaluation can also fall back to SciPy on CPU and
convert the result back. Consequently a GPU-shaped output is not proof that
all work stayed on GPU. Disabling LUTs is not a blanket no-CPU guarantee or a
promise of better accuracy for every Torch parameter regime. Validate the
actual method, parameters, tails, backend, and package versions you need.

```python
# Example: compare_lut_paths
import numpy as np
from scipy import stats
from statgpu.inference import get_distribution

q = np.array([0.025, 0.1, 0.9, 0.975])
fast = get_distribution("t", backend="numpy", use_lut=True)
reference_path = get_distribution("t", backend="numpy", use_lut=False)
fast_values = fast.ppf(q, df=10)
reference_values = reference_path.ppf(q, df=10)
scipy_values = stats.t.ppf(q, df=10)
np.testing.assert_allclose(reference_values, scipy_values, atol=1e-9, rtol=1e-9)
np.testing.assert_allclose(fast_values, reference_values, atol=1e-8, rtol=1e-8)
print("Quantiles:", np.round(reference_values, 6))
```

This yields `[-2.228139, -1.372184, 1.372184, 2.228139]`. The assertions check
only these moderate probabilities and `df=10`; they do not establish a global
error bound, a GPU comparison, or a speedup. Repeated fixed-object calls retain
instance-level caches; proxies construct distribution objects per call.

## Explicit SciPy fallback for additional families

`get_distribution` accepts only the native names listed above, including for
`backend="numpy"`; `get_distribution("gumbel_r", backend="numpy")` raises
`ValueError`. The compatibility factory `get_distribution_gpu` can explicitly
wrap another SciPy family with `allow_fallback=True`:

```python
# Example: explicit_scipy_fallback
import numpy as np
from scipy import stats
from statgpu.inference import get_distribution_gpu

x = np.array([0.0, 1.0, 2.0])
dist = get_distribution_gpu("gumbel_r", allow_fallback=True)
raw = dist.cdf(x)
# The wrapper may return a GPU array on a GPU-equipped machine.
if hasattr(raw, "get"):
    out = raw.get()
elif hasattr(raw, "detach"):
    out = raw.detach().cpu().numpy()
else:
    out = np.asarray(raw)
np.testing.assert_allclose(out, stats.gumbel_r.cdf(x))
print("Gumbel CDF:", np.round(out, 6))
```

Expected CDF: `[0.367879, 0.692201, 0.873423]`. Computation is always in SciPy
on CPU: GPU inputs are copied to CPU, and the wrapper may convert the output
to the globally auto-selected GPU backend, even for NumPy input. It has no
per-call `backend` control. If you require a NumPy result and no GPU conversion,
call `scipy.stats.gumbel_r.cdf(x)` directly. `allow_fallback=False` rejects
non-native SciPy families; unknown names also raise. For native names,
`get_distribution_gpu` still uses the native factory's automatic selection.

## Compatibility and migration

Prefer object-style calls in new code. The R-style names below remain available
compatibility wrappers (see the F sampling limitation); their names do not imply full R argument compatibility
(for example, `lower.tail`, `log.p`, and gamma `rate` are not added).

| Family | Density / mass | CDF | Quantile | Sampling |
|---|---|---|---|---|
| Normal | `dnorm_gpu` → `norm.pdf` | `pnorm_gpu` → `norm.cdf` | `qnorm_gpu` → `norm.ppf` | `rnorm_gpu` → `norm.rvs` |
| Student t | `dt_gpu` → `t.pdf` | `pt_gpu` → `t.cdf` | `qt_gpu` → `t.ppf` | `rt_gpu` → `t.rvs` |
| Chi-square | `dchisq_gpu` → `chi2.pdf` | `pchisq_gpu` → `chi2.cdf` | `qchisq_gpu` → `chi2.ppf` | `rchisq_gpu` → `chi2.rvs` |
| Gamma | `dgamma_gpu` → `gamma.pdf` | `pgamma_gpu` → `gamma.cdf` | `qgamma_gpu` → `gamma.ppf` | `rgamma_gpu` → `gamma.rvs` |
| Beta | `dbeta_gpu` → `beta.pdf` | `pbeta_gpu` → `beta.cdf` | `qbeta_gpu` → `beta.ppf` | `rbeta_gpu` → `beta.rvs` |
| F | `df_gpu` → `f.pdf` | `pf_gpu` → `f.cdf` | `qf_gpu` → `f.ppf` | `rf_gpu` → `f.rvs` |
| Poisson | `dpois_gpu` → `poisson.pmf` | `ppois_gpu` → `poisson.cdf` | `qpois_gpu` → `poisson.ppf` | `rpois_gpu` → `poisson.rvs` |
| Binomial | `dbinom_gpu` → `binom.pmf` | `pbinom_gpu` → `binom.cdf` | `qbinom_gpu` → `binom.ppf` | `rbinom_gpu` → `binom.rvs` |

Use keyword shape parameters for these wrappers too. Positional shape arguments
forwarded by wrappers such as `dpois_gpu(3, 4.0)` are not accepted by the proxy;
use `dpois_gpu(3, mu=4.0)` or migrate to `poisson.pmf(3, mu=4.0)`.

```python
# Example: compatibility_migration
import numpy as np
from statgpu.inference import dpois_gpu, pnorm_gpu, qt_gpu, poisson, norm, t

mass_old = dpois_gpu(3, mu=4.0)
mass_new = poisson.pmf(3, mu=4.0, backend="numpy")
np.testing.assert_allclose(mass_old, mass_new)
np.testing.assert_allclose(pnorm_gpu(1.96), norm.cdf(1.96, backend="numpy"))
np.testing.assert_allclose(qt_gpu(0.975, df=10), t.ppf(0.975, df=10, backend="numpy"))
print("Poisson mass:", round(float(mass_new), 6))
```

Expected mass is `0.195367`. In contrast, the following **non-R historical
names emit `DeprecationWarning`** and should be migrated:

- `norm_cdf_gpu`, `norm_sf_gpu`, `norm_ppf_gpu`, `norm_isf_gpu` → corresponding
  `norm.cdf`, `norm.sf`, `norm.ppf`, `norm.isf` methods.
- `norm_two_sided_pvalue_gpu`, `norm_two_sided_critical_value_gpu` →
  `norm.two_sided_pvalue`, `norm.two_sided_critical_value`.
- `t_cdf_gpu`, `t_sf_gpu`, `t_ppf_gpu` → `t.cdf`, `t.sf`, `t.ppf`.
- `t_two_sided_pvalue_gpu`, `t_two_sided_critical_value_gpu` →
  `t.two_sided_pvalue`, `t.two_sided_critical_value`.

## Complete API and references

- Factory signature: `get_distribution(name, backend="auto", device=None, *, use_lut=True)`.
  Names are case-insensitive. `list_available_distributions()` lists native names.
- Proxy methods take one positional threshold/probability (except `rvs`, which
  takes keyword arguments only), the family parameters above, and per-call
  `backend`, `device`, `use_lut` controls. Fixed objects accept their native
  method parameters only. `t.ppf`, `t.isf`, and `t.two_sided_critical_value`
  additionally accept `max_bisect_steps=60` for the bisection fallback; this is
  not an accuracy tolerance. Every native `rvs` signature includes `size=None`
  and `dtype=None`, subject to the dtype limitation above.
- Compatibility factories: `get_distribution_gpu(name, *, allow_fallback=False)`
  and `list_available_distributions_gpu(include_scipy=True)`.
- Complete signatures and method implementations: [distribution objects, proxies,
  backends, and factories](../../../statgpu/inference/_distributions_backend.py).
  Public imports: [`statgpu.inference`](../../../statgpu/inference/__init__.py).
  Compatibility signatures/warnings: [function-style wrappers](../../../statgpu/linear_model/legacy/_distributions_legacy_gpu.py).
  Student t `df=1`/`df=2` two-sided runtime handling: [low-degree reference contract](../../../statgpu/inference/_low_df_reference_contract.py).
- Probability terminology and aligned external reference: [SciPy statistical
  distributions](https://docs.scipy.org/doc/scipy/reference/stats.html),
  [normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html),
  [Student t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html),
  [Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html).
