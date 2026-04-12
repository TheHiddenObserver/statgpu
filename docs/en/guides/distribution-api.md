# Distribution API Guide

> Language: English  
> Last updated: 2026-04-12  
> This page: Guide  
> Switch: [Chinese](../../guides/distribution-api.md)

Language switch: [Chinese](../../guides/distribution-api.md)

This page documents the current, recommended distribution API usage.

## 1) Preferred entrypoint (object-style API)

Import distribution objects from `statgpu.inference` and call methods like `cdf/sf/ppf/isf/pdf/pmf/rvs`.

```python
import cupy as cp
from statgpu.inference import norm, t, chi2, gamma, beta, f, poisson, binom

x = cp.array([0.0, 1.0, 2.0], dtype=cp.float64)
q = cp.array([0.1, 0.5, 0.9], dtype=cp.float64)

# Continuous distributions
norm_cdf = norm.cdf(x)
norm_ppf = norm.ppf(q)
t_sf = t.sf(x, df=10)
chi2_ppf = chi2.ppf(q, df=8)

# Discrete distributions
poi_cdf = poisson.cdf(k=cp.array([1, 2, 3]), mu=3.5)
binom_ppf = binom.ppf(q, n=20, p=0.2)
```

Natively implemented distribution names:

- `norm`
- `t`
- `uniform`
- `expon`
- `cauchy`
- `laplace`
- `logistic`
- `chi2`
- `gamma`
- `beta`
- `f`
- `weibull_min`
- `lognorm`
- `poisson`
- `binom`

## 2) Dynamic lookup by name

Use `get_distribution_gpu(name)` when distribution names are selected dynamically.

```python
from statgpu.inference import get_distribution_gpu

dist = get_distribution_gpu("norm")
y = dist.cdf(0.0)
```

Current default policy is native-only:

- `allow_fallback=False` by default.
- Non-native names raise `ValueError` unless fallback is explicitly enabled.

## 3) Explicit SciPy fallback (optional)

For long-tail distributions that are not natively implemented yet, fallback can be enabled explicitly:

```python
import cupy as cp
from statgpu.inference import get_distribution_gpu

dist = get_distribution_gpu("gumbel_r", allow_fallback=True)
out = dist.cdf(cp.asarray([0.0, 1.0, 2.0]))
```

Notes:

- Fallback computes via `scipy.stats`, then converts outputs back to CuPy.
- This is explicit behavior only (no implicit fallback in default path).

## 4) R-style compatibility API (kept for compatibility)

The following wrappers follow R-style naming and are grouped by distribution family.

### Normal family (`norm`)

- `dnorm_gpu` -> `norm.pdf`
- `rnorm_gpu` -> `norm.rvs`
- `pnorm_gpu` -> `norm.cdf`
- `qnorm_gpu` -> `norm.ppf`

### Student t family (`t`)

- `dt_gpu` -> `t.pdf`
- `rt_gpu` -> `t.rvs`
- `pt_gpu` -> `t.cdf`
- `qt_gpu` -> `t.ppf`

### Chi-square family (`chi2`)

- `dchisq_gpu` -> `chi2.pdf`
- `pchisq_gpu` -> `chi2.cdf`
- `qchisq_gpu` -> `chi2.ppf`
- `rchisq_gpu` -> `chi2.rvs`

### Gamma family (`gamma`)

- `dgamma_gpu` -> `gamma.pdf`
- `pgamma_gpu` -> `gamma.cdf`
- `qgamma_gpu` -> `gamma.ppf`
- `rgamma_gpu` -> `gamma.rvs`

### Beta family (`beta`)

- `dbeta_gpu` -> `beta.pdf`
- `pbeta_gpu` -> `beta.cdf`
- `qbeta_gpu` -> `beta.ppf`
- `rbeta_gpu` -> `beta.rvs`

### F family (`f`)

- `df_gpu` -> `f.pdf`
- `pf_gpu` -> `f.cdf`
- `qf_gpu` -> `f.ppf`
- `rf_gpu` -> `f.rvs`

### Poisson family (`poisson`)

- `dpois_gpu` -> `poisson.pmf`
- `ppois_gpu` -> `poisson.cdf`
- `qpois_gpu` -> `poisson.ppf`
- `rpois_gpu` -> `poisson.rvs`

### Binomial family (`binom`)

- `dbinom_gpu` -> `binom.pmf`
- `pbinom_gpu` -> `binom.cdf`
- `qbinom_gpu` -> `binom.ppf`
- `rbinom_gpu` -> `binom.rvs`

Example call:

```python
from statgpu.inference import (
	dnorm_gpu, pnorm_gpu, qnorm_gpu, rnorm_gpu,
	dt_gpu, pt_gpu, qt_gpu, rt_gpu,
	dpois_gpu, ppois_gpu, qpois_gpu, rpois_gpu,
)

pdf = dnorm_gpu(0.0)
p = pnorm_gpu(1.96)
q = qnorm_gpu(0.975)
sample_norm = rnorm_gpu(size=8)
pdf_t = dt_gpu(2.0, df=10)
pt = pt_gpu(2.0, df=10)
qt = qt_gpu(0.975, df=10)
sample_t = rt_gpu(df=10, size=8)
pmf_pois = dpois_gpu(3, 4.0)
sample_pois = rpois_gpu(4.0, size=8)
```

These wrappers are recommended for compatibility use only; object-style API is still preferred for new code. Common migration examples:

- `dnorm_gpu(x)` -> `norm.pdf(x)`
- `pnorm_gpu(x)` -> `norm.cdf(x)`
- `qnorm_gpu(q)` -> `norm.ppf(q)`
- `rnorm_gpu(size=...)` -> `norm.rvs(size=...)`
- `dpois_gpu(k, mu)` -> `poisson.pmf(k, mu)`
- `rpois_gpu(mu, size=...)` -> `poisson.rvs(mu, size=...)`
- `pt_gpu(x, df)` -> `t.cdf(x, df=df)`
- `qt_gpu(q, df)` -> `t.ppf(q, df=df)`

## 5) Legacy non-R function names (soft-deprecated)

The following non-R historical names are still available but emit `DeprecationWarning`, grouped by family.

### Legacy `norm` names

- `norm_cdf_gpu` -> `norm.cdf`
- `norm_sf_gpu` -> `norm.sf`
- `norm_ppf_gpu` -> `norm.ppf`
- `norm_isf_gpu` -> `norm.isf`
- `norm_two_sided_pvalue_gpu` -> `norm.two_sided_pvalue`
- `norm_two_sided_critical_value_gpu` -> `norm.two_sided_critical_value`

### Legacy `t` names

- `t_cdf_gpu` -> `t.cdf`
- `t_sf_gpu` -> `t.sf`
- `t_ppf_gpu` -> `t.ppf`
- `t_two_sided_pvalue_gpu` -> `t.two_sided_pvalue`
- `t_two_sided_critical_value_gpu` -> `t.two_sided_critical_value`

Prefer migrating these names to object-style calls (`norm.*` / `t.*`).

## 6) CPU vs GPU behavior

- Distribution objects in this page are GPU-first (CuPy input/output).
- Native distribution kernels do not silently fall back to CPU when required `cupyx.scipy.special` functionality is unavailable; they raise explicit errors.
- CPU-side model inference (for example p-values/critical values in linear/logistic/Cox models) still directly uses `scipy.stats`; this is a separate path from distribution objects.

## 7) Common issues

1. Error: `cupyx.scipy.special.* is required for GPU backend`

- Cause: missing/unsupported `cupyx.scipy.special` in the current CUDA/CuPy setup.
- Action: verify CUDA driver + CuPy compatibility, or use CPU inference paths temporarily.

2. Listing available distributions

```python
from statgpu.inference import list_available_distributions_gpu

native_only = list_available_distributions_gpu(include_scipy=False)
all_names = list_available_distributions_gpu(include_scipy=True)
```
