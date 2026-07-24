# Distribution API Guide

> Language: English
> Last updated: 2026-04-24
> This page: Guide
> Switch: [Chinese](../../cn/guides/distribution-api.md)

Language switch: [Chinese](../../cn/guides/distribution-api.md)

This page documents the recommended distribution API usage.

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

## 2) Backend selection

The module-level proxies (`norm`, `t`, `chi2`, etc.) automatically select the best available backend (GPU > Torch > NumPy). You can override the backend per-call:

```python
from statgpu.inference import norm, t

# Auto-detect (default) — uses CuPy if available, then Torch, then NumPy
result = norm.cdf(x)

# Force Torch backend
result = norm.cdf(x, backend="torch")

# Force NumPy/scipy backend
result = norm.cdf(x, backend="numpy")

# Per-call override works for all distributions and methods
result = t.ppf(q, df=10, backend="cupy")
```

You can also create a distribution with a fixed backend:

```python
from statgpu.inference import get_distribution

# Explicit backend
norm_cupy = get_distribution("norm", backend="cupy")
norm_torch = get_distribution("norm", backend="torch")
norm_numpy = get_distribution("norm", backend="numpy")

# Torch-specific: control device
norm_torch_gpu = get_distribution("norm", backend="torch", device="cuda:0")

# Disable LUT cache for inverse special functions (full iterative precision)
norm_full_precision = get_distribution("norm", backend="numpy", use_lut=False)
```

## 3) Dynamic lookup by name

Use `get_distribution(name, backend=...)` when distribution names are selected dynamically.

```python
from statgpu.inference import get_distribution

dist = get_distribution("norm", backend="auto")
y = dist.cdf(0.0)
```

The `backend` parameter accepts: `"auto"` (default), `"numpy"`, `"cupy"`, `"torch"`.

## 4) Explicit SciPy fallback (optional)

For long-tail distributions that are not natively implemented, fallback can be enabled explicitly:

```python
import numpy as np
from statgpu.inference import get_distribution

dist = get_distribution("gumbel_r", backend="numpy")
out = dist.cdf(np.array([0.0, 1.0, 2.0]))
```

Notes:

- For non-native distribution names, `get_distribution` with `backend="numpy"` wraps the corresponding `scipy.stats` distribution.
- GPU backends only work with natively implemented distributions.

## 5) LUT acceleration for inverse functions

Inverse CDF methods (`ppf`/`isf`) for `t`, `f`, `beta`, `chi2`, `gamma` use LUT (lookup table) + 1-step Newton refinement by default. This provides 10-50x speedup with negligible precision loss (~1e-11).

```python
# Default: LUT enabled (fast)
t.ppf(q, df=10)

# Disable LUT for full iterative precision (slower)
t.ppf(q, df=10, use_lut=False)

# Or create a distribution with LUT disabled globally
t_full = get_distribution("t", backend="torch", use_lut=False)
t_full.ppf(q, df=10)  # always uses full iterative solver
```

**Precision trade-off**:

| Backend | `use_lut=True` | `use_lut=False` |
|---|---|---|
| numpy | LUT + 1 Newton (err ~1e-11) | scipy.special (full iterative) |
| cupy | Native `cupyx.scipy.special` | Same (no LUT effect) |
| torch | LUT + 1 Newton (err ~1e-5 for t/f) | Newton + 64K Chebyshev integral |

## 6) R-style compatibility API (kept for compatibility)

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

## 7) Legacy non-R function names (soft-deprecated)

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

## 8) Backend behavior

- **Auto mode (default)**: Proxies try CuPy > Torch > NumPy/scipy. Input array type is not changed.
- **NumPy backend**: Uses `scipy.stats` and `scipy.special`. Accepts numpy arrays.
- **CuPy backend**: Uses `cupyx.scipy.special`. Accepts CuPy arrays.
- **Torch backend**: Uses `torch.special` with fallbacks for missing functions. Accepts Torch tensors.

Native distribution kernels do not silently fall back to CPU when required special functions are unavailable.

## 9) Common issues

1. Error: missing special function for GPU backend

- Cause: required `cupyx.scipy.special` or `torch.special` function is unavailable.
- Action: verify CUDA driver + CuPy/Torch compatibility, or use `backend="numpy"` temporarily.

2. Listing available distributions

```python
from statgpu.inference import list_available_distributions

native_only = list_available_distributions()
```

3. Backend precision differences

- CuPy and NumPy backends match `scipy.stats` to machine epsilon for most functions.
- Torch 2.0 lacks `torch.special.betainc` — t/f/beta CDF/PPF may have ~1e-5 to 1e-7 error.
- Upgrading to Torch >= 2.1 with native `torch.special.betainc` resolves this.
