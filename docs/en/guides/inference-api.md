# Inference API Reference

> Language: English  
> Last updated: 2026-09-17  
> Module: `statgpu.inference`  
> Switch: [Chinese](../../cn/guides/inference-api.md)

`statgpu.inference` collects reusable statistical utilities for probability distributions, multiple testing, p-value combination, permutation tests, and bootstrap calculations.

This page is the **module entry point**. Detailed distribution behavior is documented once in [Distribution API](distribution-api.md), and model coefficient-inference choices are documented in [Inference Modes](inference-modes.md).

## Quick reference

```python
from statgpu.inference import (
    norm,
    t,
    adjust_pvalues,
    combine_pvalues,
    permutation_test,
    bootstrap_statistic,
)
```

| API | Purpose | Detailed documentation |
|---|---|---|
| distribution objects such as `norm`, `t`, `chi2`, `poisson` | CDF/SF/PPF/PDF/PMF/random sampling | [Distribution API](distribution-api.md) |
| `get_distribution(...)` | choose a distribution/backend dynamically | [Distribution API](distribution-api.md) |
| `adjust_pvalues(...)` | multiple-testing adjustment | [Multiple Testing](multiple-testing-combine-pvalues.md) |
| `combine_pvalues(...)` | combine evidence across p-values | [Multiple Testing](multiple-testing-combine-pvalues.md) |
| `permutation_test(...)` | permutation-based hypothesis test | this page |
| `bootstrap_statistic(...)` | generic bootstrap for a user-supplied statistic | this page |

## Distribution functions

Distribution objects expose scipy-style methods such as `cdf`, `sf`, `ppf`, `isf`, `pdf`/`pmf`, and `rvs`.

```python
from statgpu.inference import norm, t

p = norm.cdf(1.96)
q = t.ppf(0.975, df=10)
```

Backend selection, supported distributions, inverse-function precision, R-style compatibility aliases, and legacy names are all documented in [Distribution API](distribution-api.md). They are intentionally not duplicated here.

## Multiple testing and p-value combination

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

reject, pvals_bh = adjust_pvalues(pvals, method="bh")
stat, p_global = combine_pvalues(pvals, method="fisher")
```

Available adjustment/combination methods and their interpretation are described in [Multiple Testing](multiple-testing-combine-pvalues.md).

## Permutation testing

`permutation_test` repeatedly permutes the supplied data according to its API and recomputes a user-provided statistic to form a permutation reference distribution.

```python
import numpy as np
from statgpu.inference import permutation_test

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X,
    y,
    n_resamples=999,
    random_state=42,
)

print(result.pvalue)
```

Choose the statistic and permutation scheme to match the null hypothesis of the application; a generic permutation engine cannot determine exchangeability assumptions on the user's behalf.

## Generic bootstrap

`bootstrap_statistic` bootstraps a statistic supplied by the caller.

```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

result = bootstrap_statistic(
    np.mean,
    (data,),
    n_resamples=9999,
    random_state=42,
)

print(result.statistic)
print(result.confidence_interval)
```

This generic utility is different from estimator-specific inference modes such as the penalized Gaussian residual bootstrap. For coefficient inference after a fitted model, start with [Inference Modes](inference-modes.md) instead of assuming the generic bootstrap reproduces a model-specific inferential procedure.

## API map

Use the documentation according to the question you are trying to answer:

- **How do I evaluate a probability distribution on NumPy/CuPy/Torch?** → [Distribution API](distribution-api.md)
- **How do I adjust or combine many p-values?** → [Multiple Testing](multiple-testing-combine-pvalues.md)
- **How do I run a generic permutation or bootstrap calculation?** → this page
- **Which inference method should a regression estimator use?** → [Inference Modes](inference-modes.md)
- **What does penalized-GLM coefficient inference target?** → [Penalized GLM inference](penalized-glm-inference.md)
