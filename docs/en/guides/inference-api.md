# Inference API Reference

> **Module:** `statgpu.inference`  
> **Last updated:** 2026-09-06  
> **Backends:** NumPy, CuPy, PyTorch

The `statgpu.inference` module provides statistical inference tools: distributions, multiple testing, permutation tests, and bootstrap. Public statgpu estimators also inherit model-bound convenience methods that call these engines using the estimator's device/backend context.

## Quick Reference

```python
from statgpu.inference import norm, poisson, t, adjust_pvalues, combine_pvalues, permutation_test
```

| Function/Class | Description |
|---|---|
| `norm`, `t`, `chi2`, `f`, `beta`, `gamma`, `poisson`, `binom`, `uniform`, `expon`, `cauchy`, `laplace`, `logistic`, `lognorm`, `weibull_min` | Distribution objects (scipy-compatible API) |
| `get_distribution(name, backend=...)` | Dynamic distribution lookup |
| `adjust_pvalues(pvals, method=...)` | Multiple testing correction |
| `combine_pvalues(pvals, method=...)` | Global p-value combination |
| `permutation_test(statistic, X, y, ...)` | Permutation-based hypothesis testing |
| `bootstrap_statistic(statistic, arrays, ...)` | Generic bootstrap engine |
| `multipletests(...)` | Alias for `adjust_pvalues` (scientific naming) |

## Estimator-bound inference helpers

Every public estimator derived from `BaseEstimator` also exposes thin model-context wrappers around the shared inference engines. These methods are inherited; they do not need to be reimplemented by `Ridge`, `Lasso`, `ElasticNet`, or other estimators.

| Estimator method | Signature | Model-context behavior |
|---|---|---|
| `adjust_pvalues` | `adjust_pvalues(pvalues=None, method="bh", alpha=0.05, axis=0, backend="auto")` | If `pvalues` is omitted, uses the fitted estimator's `_pvalues`. Returns raw/adjusted p-values, reject mask, method, alpha, axis, and resolved backend. |
| `combine_pvalues` | `combine_pvalues(pvalues=None, method="fisher", weights=None, axis=None, backend="auto")` | If `pvalues` is omitted, uses `_pvalues`; returns the combination statistic and global p-value together with backend metadata. |
| `bootstrap_statistic` | `bootstrap_statistic(statistic, *arrays, n_resamples=200, strategy="iid", strata=None, clusters=None, block_size=None, confidence_level=0.95, random_state=None, statistic_name="statistic", backend="auto")` | Uses the estimator's backend. If no arrays are supplied, it uses cached fitted design/response arrays when available. |
| `permutation_test` | `permutation_test(statistic, X, y, n_resamples=1000, strategy="iid", strata=None, groups=None, alternative="two-sided", random_state=None, statistic_name="statistic", backend="auto")` | Casts the supplied data and optional strata/groups to the resolved model-context backend before calling the shared permutation engine. |

`backend="auto"` follows the estimator's resolved device: NumPy for CPU, CuPy for `device="cuda"`, and Torch for `device="torch"`. Passing an explicit backend overrides that model-context choice where the shared engine supports it.

If `adjust_pvalues()` or `combine_pvalues()` is called without explicit p-values and the fitted estimator has no `_pvalues`, statgpu raises a `RuntimeError` rather than silently fabricating an input. Likewise, `bootstrap_statistic()` without explicit arrays requires cached training arrays from a fitted model.

A typical model-bound workflow is:

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    compute_inference=True,
).fit(X, y)

adjusted = model.adjust_pvalues(method="bh")
combined = model.combine_pvalues(method="fisher")

boot = model.bootstrap_statistic(
    lambda X_, y_: float((X_[:, 0] * y_).mean()),
    n_resamples=500,
    random_state=7,
)

perm = model.permutation_test(
    lambda X_, y_: float((X_[:, 0] * y_).mean()),
    X,
    y,
    n_resamples=999,
    random_state=7,
)
```

Use the **estimator-bound methods** when you want inference utilities to follow the fitted model's backend and, where supported, reuse fitted state. Use the **module-level functions** below when you want a standalone inference calculation that is independent of a fitted estimator.

---

## Distributions

### Direct Import (NumPy default)

```python
from statgpu.inference import norm, poisson, t

# Generate random samples
X = norm.rvs(size=1000)

# CDF, survival function, PPF
p = norm.cdf(1.96)           # 0.975
s = norm.sf(1.96)            # 0.025
q = norm.ppf(0.975)          # 1.96

# Poisson with parameter
y = poisson.rvs(mu=3.0, size=1000)

# t-distribution with degrees of freedom
p = t.cdf(2.0, df=10)
```

### GPU Backend

```python
from statgpu.inference import norm

# Torch backend
X_torch = norm.rvs(size=1000, backend="torch")    # torch tensor on CUDA
p = norm.cdf(x_torch, backend="torch")

# CuPy backend
X_cupy = norm.rvs(size=1000, backend="cupy")      # CuPy array on GPU

# Auto-detect from input type
import torch
x = torch.tensor([0.0, 1.96]).cuda()
p = norm.cdf(x)  # automatically uses torch backend
```

### Available Distributions

| Distribution | Parameters | Methods |
|---|---|---|
| `norm` | — | rvs, cdf, sf, ppf, isf, pdf |
| `t` | `df` | rvs, cdf, sf, ppf, isf, pdf |
| `chi2` | `df` | rvs, cdf, sf, ppf, isf, pdf |
| `f` | `dfn, dfd` | rvs, cdf, sf, ppf, isf, pdf |
| `beta` | `a, b` | rvs, cdf, sf, ppf, isf, pdf |
| `gamma` | `a` | rvs, cdf, sf, ppf, isf, pdf |
| `uniform` | — | rvs, cdf, sf, ppf, isf, pdf |
| `expon` | — | rvs, cdf, sf, ppf, isf, pdf |
| `cauchy` | — | rvs, cdf, sf, ppf, isf, pdf |
| `laplace` | — | rvs, cdf, sf, ppf, isf, pdf |
| `logistic` | — | rvs, cdf, sf, ppf, isf, pdf |
| `lognorm` | `s` | rvs, cdf, sf, ppf, isf, pdf |
| `weibull_min` | `c` | rvs, cdf, sf, ppf, isf, pdf |
| `poisson` | `mu` | rvs, cdf, sf, ppf, pmf |
| `binom` | `n, p` | rvs, cdf, sf, ppf, pmf |

### Dynamic Lookup

```python
from statgpu.inference import get_distribution

# Lookup by name
norm = get_distribution("norm", backend="torch")
pois = get_distribution("poisson", backend="cupy")

# List available distributions
from statgpu.inference import list_available_distributions
print(list_available_distributions())
```

---

## Multiple Testing

### adjust_pvalues (p-value correction)

```python
from statgpu.inference import adjust_pvalues
import numpy as np

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

# Benjamini-Hochberg (FDR control)
reject, pvals_adj = adjust_pvalues(pvals, method='bh')

# Other methods: 'bonferroni', 'holm', 'hochberg', 'by' (Benjamini-Yekutieli)
reject, pvals_adj = adjust_pvalues(pvals, method='bonferroni')
```

### combine_pvalues (global p-value)

```python
from statgpu.inference import combine_pvalues

pvals = np.array([0.01, 0.04, 0.03, 0.40])

# Fisher's method
stat, p_global = combine_pvalues(pvals, method='fisher')

# Cauchy combination test (ACAT)
stat, p_global = combine_pvalues(pvals, method='cauchy')

# Stouffer's method
stat, p_global = combine_pvalues(pvals, method='stouffer')
```

---

## Permutation Testing

```python
from statgpu.inference import permutation_test
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

# Test correlation between X[:,0] and y
result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X, y,
    n_resamples=999,
    random_state=42,
)
print(f"p-value: {result.pvalue:.4f}")
```

---

## Bootstrap

```python
from statgpu.inference import bootstrap_statistic
import numpy as np

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

# Bootstrap mean
result = bootstrap_statistic(
    np.mean, (data,),
    n_resamples=9999,
    random_state=42,
)
print(f"Mean: {result.statistic:.4f}")
print(f"95% CI: [{result.confidence_interval.low:.4f}, {result.confidence_interval.high:.4f}]")
```

---

## R-style Compatibility

For users migrating from R, the module provides R-compatible function names:

```python
from statgpu.inference import norm

# R-style: dnorm, pnorm, qnorm, rnorm
from statgpu.inference import dnorm_gpu, pnorm_gpu, qnorm_gpu, rnorm_gpu

# These are GPU-accelerated equivalents of R's dnorm/pnorm/qnorm/rnorm
```

---

## FAQ

**Q: When should I use `get_distribution()` vs direct import?**  
A: Use direct import (`from statgpu.inference import norm`) for numpy backend. Use `get_distribution("norm", backend="torch")` when you need to control the backend explicitly.

**Q: Can I use statgpu distributions with my existing scipy code?**  
A: Yes. The API is scipy-compatible: `rvs`, `cdf`, `sf`, `ppf`, `isf`, `pdf`/`pmf` all have the same signatures. Replace `scipy.stats.norm` with `statgpu.inference.norm`.

**Q: How do I use GPU-accelerated distributions?**  
A: Pass `backend="torch"` or `backend="cupy"` to any distribution method: `norm.rvs(size=1000, backend="torch")`.

**Q: What's the difference between `sf` and `1 - cdf`?**  
A: `sf(x)` is the survival function (1 - CDF). It's more numerically stable for extreme values where CDF approaches 1.

---

## References

- **scipy.stats**: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- **R distributions**: [https://stat.ethz.ch/R-manual/R-patched/library/stats/html/Distributions.html](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/Distributions.html)
- **Multiple testing**: Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"
- **Cauchy combination**: Liu & Xie (2020), "Cauchy Combination Test"
