# Multiple Testing Correction

> **Module:** `statgpu.inference`  
> **Last updated:** 2026-06-14  
> **Backends:** NumPy, CuPy, PyTorch

## Overview

When testing multiple hypotheses simultaneously, the probability of at least one false discovery increases. This module provides p-value adjustment and combination methods to control the family-wise error rate (FWER) or false discovery rate (FDR).

## Mathematical Foundation

### P-value Adjustment (adjust_pvalues)

Given $m$ raw p-values $p_1, p_2, \ldots, p_m$, the adjusted p-values $\tilde{p}_i$ control the specified error rate.

**Bonferroni correction** (FWER control):
$$\tilde{p}_i = \min(m \cdot p_i, 1)$$

**Holm step-down procedure** (FWER control, uniformly more powerful than Bonferroni):
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. Reject $H_{(i)}$ if $p_{(i)} < \alpha / (m - i + 1)$
3. Adjusted: $\tilde{p}_{(i)} = \max_{j \leq i} \min((m-j+1) \cdot p_{(j)}, 1)$

**Benjamini-Hochberg (BH)** (FDR control):
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject $H_{(1)}, \ldots, H_{(k)}$
4. Adjusted: $\tilde{p}_{(i)} = \min_{j \geq i} \min(\frac{m}{j} p_{(j)}, 1)$

**Benjamini-Yekutieli (BY)** (FDR control under arbitrary dependence):
- Same as BH but with correction factor $\sum_{j=1}^{m} \frac{1}{j}$
- More conservative than BH but valid under any dependence structure

**Hochberg step-up procedure** (FWER control, assumes non-negative correlation):
1. Start from largest p-value
2. Accept $H_{(i)}$ if $p_{(i)} > \alpha / (m - i + 1)$
3. Adjusted: $\tilde{p}_{(i)} = \min_{j \geq i} \min((m-j+1) \cdot p_{(j)}, 1)$

### P-value Combination (combine_pvalues)

**Fisher's method** (chi-squared combination):
$$T = -2 \sum_{i=1}^{m} \ln(p_i) \sim \chi^2_{2m}$$
- Under $H_0$: $T \sim \chi^2_{2m}$
- Powerful when a small subset of p-values is very small

**Cauchy Combination Test (ACAT)** (robust to arbitrary dependence):
$$T = \sum_{i=1}^{m} w_i \tan\left((0.5 - p_i)\pi\right) \sim \text{Cauchy}(0, 1)$$
- Approximately distributed as Cauchy under $H_0$
- No assumption on dependence structure
- Liu & Xie (2020)

**Stouffer's method** (z-score combination):
$$T = \frac{\sum_{i=1}^{m} w_i \Phi^{-1}(1-p_i)}{\sqrt{\sum_{i=1}^{m} w_i^2}} \sim N(0, 1)$$
- Under $H_0$: $T \sim N(0, 1)$
- Most powerful when effects are in the same direction

## Parameters

### adjust_pvalues

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvalues` | array-like | — | Raw p-values |
| `method` | str | `'bh'` | `'bh'`, `'by'`, `'holm'`, `'bonferroni'`, `'hochberg'` |
| `alpha` | float | `0.05` | Significance level |
| `axis` | int or None | `None` | Axis for batch processing |
| `backend` | str | `'auto'` | `'numpy'`, `'cupy'`, `'torch'`, `'auto'` |

### combine_pvalues

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvalues` | array-like | — | Raw p-values |
| `method` | str | `'fisher'` | `'fisher'`, `'cauchy'`/`'acat'`, `'stouffer'` |
| `weights` | array-like | `None` | Non-negative weights (cauchy/stouffer) |
| `axis` | int or None | `None` | Axis for batch processing |
| `backend` | str | `'auto'` | `'numpy'`, `'cupy'`, `'torch'`, `'auto'` |

## CPU+GPU Examples

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

# Raw p-values from 5 hypothesis tests
pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])

# Benjamini-Hochberg FDR control
reject, pvals_adj = adjust_pvalues(pvals, method='bh', alpha=0.05)
print(f"Rejected: {reject}")
print(f"Adjusted p-values: {pvals_adj}")

# Fisher combination
stat, p_global = combine_pvalues(pvals, method='fisher')
print(f"Fisher statistic: {stat:.4f}, global p-value: {p_global:.6f}")

# Cauchy combination (robust to dependence)
stat, p_global = combine_pvalues(pvals, method='cauchy')

# Stouffer with weights
weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
stat, p_global = combine_pvalues(pvals, method='stouffer', weights=weights)
```

**GPU acceleration:**

```python
import torch
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals_gpu = torch.tensor([0.001, 0.01, 0.03, 0.05, 0.50], device='cuda')
reject, pvals_adj = adjust_pvalues(pvals_gpu, method='bh', backend='torch')
```

## Outputs

| Method | Returns | Description |
|---|---|---|
| `adjust_pvalues` | `(reject, pvals_adj)` | Boolean rejection array + adjusted p-values |
| `combine_pvalues` | `(statistic, p_global)` | Test statistic + global p-value |

## FAQ

**Q: Which method should I use?**  
A: Use **BH** for FDR control (most common). Use **Bonferroni/Holm** for strict FWER control. Use **Cauchy** when p-values are correlated.

**Q: Can I use these for genome-wide association studies?**  
A: Yes. BH is standard for GWAS. For correlated tests, use Cauchy combination.

**Q: What's the difference between FDR and FWER?**  
A: FDR = expected proportion of false rejections. FWER = probability of at least one false rejection. FDR is less conservative.

## External Validation

- R: `p.adjust()` for adjustment, `pchisq()` for Fisher
- statsmodels: `multipletests()` (BH, Holm, Bonferroni, BY, Hochberg)
- scipy: `combine_pvalues()` (Fisher, Stouffer, Tippett)

## References

1. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
2. Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *Annals of Statistics*, 29(4), 1165-1188.
3. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
4. Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
5. Liu, Y. & Xie, J. (2020). "Cauchy Combination Test: A Powerful Test With Analytic p-Value Calculation Under Arbitrary Dependency Structures." *Journal of the American Statistical Association*, 115(529), 393-402.
6. Stouffer, S.A. et al. (1949). *The American Soldier*. Princeton University Press.
