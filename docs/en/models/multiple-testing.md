# Multiple Testing Correction

> **Module:** `statgpu.inference`  
> **Last updated:** 2026-10-03  
> **Backends:** NumPy, CuPy, PyTorch  
> Switch: [Chinese](../../cn/models/multiple-testing.md)

## Overview

When testing multiple hypotheses simultaneously, the probability of at least one false discovery increases. P-value adjustment controls the family-wise error rate (FWER) or false discovery rate (FDR) across individual decisions. P-value combination instead tests a global null hypothesis; it does not identify which individual hypotheses to reject.

## Choose the question and error criterion

Suppose you test five candidate biomarkers. “Which biomarkers should I flag?” calls for individual p-value adjustment. “Is there evidence against the hypothesis that all five are null?” calls for one global combination test. Choose the hypothesis family before inspecting which p-values are small; neither operation repairs invalid or selection-biased input p-values.

| Your goal | Starting choice | Important condition |
|---|---|---|
| Limit the chance of any false rejection in the chosen family | `holm` | Valid individual p-values; arbitrary dependence is allowed. Bonferroni is a simpler, more conservative alternative. |
| Limit the expected false-discovery proportion among rejections | `bh`, or `by` if dependence is unrestricted | BH requires independence or suitable positive dependence; BY allows arbitrary dependence. |
| Test the global null that all component nulls hold | `fisher` or `stouffer` | Independent, null-uniform p-values; Stouffer also requires fixed weights. |
| Combine dependent evidence using a tail approximation | `cauchy` / `acat` | Check the Liu–Xie assumptions and approximation limits below; dependence alone does not guarantee calibration. |

FWER is the probability of at least one false rejection in the family. FDR is the expectation of the fraction of false rejections **among all rejections**, with that fraction defined as zero when there are no rejections. An FDR target of 0.05 is not a promise that at most 5% of this particular result list is wrong, nor a 5% probability that each individual discovery is false. Set `alpha` from the scientific error tolerance before looking at the results.

## Minimal runnable example

The following illustrative p-values represent one prespecified family. Use BH only when the input tests meet its validity and dependence conditions.

```python
import numpy as np
from statgpu.inference import adjust_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
reject, pvals_adj = adjust_pvalues(pvals, method='bh', alpha=0.05)
print(reject.tolist())
print(np.round(pvals_adj, 4).tolist())
```

Expected output:

```text
[True, True, True, False, False]
[0.005, 0.025, 0.05, 0.0625, 0.5]
```

The first three hypotheses are rejected because their adjusted p-values are **less than or equal to** 0.05. Output entries retain the input order; you do not need to sort or unsort them. `False` means “not rejected at this level,” not “proved true.” A rejection is not proof of a practically important effect or causality.

## Define the family with `axis`

For a matrix, `axis=None` (the default) pools every entry into one family. `axis=1` treats each row as a separate family; `axis=0` treats each column separately. Negative axes follow the usual array convention. Choose this from the scientific question, not from whichever option yields more rejections.

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

p_matrix = np.array([[0.01, 0.04], [0.20, 0.80]])
reject_all, adjusted_all = adjust_pvalues(p_matrix, method='holm', axis=None)
reject_rows, adjusted_rows = adjust_pvalues(p_matrix, method='holm', axis=1)
print(reject_all.tolist())
print(np.round(adjusted_all, 4).tolist())
print(reject_rows.tolist())
print(np.round(adjusted_rows, 4).tolist())

# Fisher requires independence and null-uniform p-values within each row.
row_stat, row_p_global = combine_pvalues(p_matrix, method='fisher', axis=1)
print(np.round(row_p_global, 6).tolist())
```

Expected output:

```text
[[True, False], [False, False]]
[[0.04, 0.12], [0.4, 0.8]]
[[True, True], [False, False]]
[[0.02, 0.04], [0.4, 0.8]]
[0.00353, 0.453213]
```

The rowwise calculation tests two families of size two instead of one family of size four. Separate rowwise FWER/FDR control does **not** automatically control the corresponding error rate across all entries collectively. If the intended discovery claim spans all four tests, adjust that whole family instead.

Combination reduces the selected axis: here `row_p_global.shape == (2,)`. At level 0.05, the first row provides evidence against its all-null hypothesis and the second does not; this does not identify which component null is false. `axis=None` would instead return one statistic and one p-value for all four entries. If you will make discoveries across many row-level global tests, those global p-values themselves need an appropriate multiplicity procedure.

## Mathematical Foundation

### P-value Adjustment (adjust_pvalues)

Given $m$ raw p-values $p_1, p_2, \ldots, p_m$, the adjusted p-values $\tilde{p}_i$ control the specified error rate.

**Bonferroni correction** (FWER control):
$$\tilde{p}_i = \min(m \cdot p_i, 1)$$

**Holm step-down procedure** (FWER control, uniformly more powerful than Bonferroni):
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. Starting at $i=1$, reject $H_{(i)}$ while $p_{(i)} \leq \alpha / (m - i + 1)$ and continue to the next rank. At the first failed comparison, stop and do not reject that hypothesis or any remaining hypothesis. If every comparison passes, reject all hypotheses.
3. Adjusted: $\tilde{p}_{(i)} = \max_{j \leq i} \min((m-j+1) \cdot p_{(j)}, 1)$

**Benjamini-Hochberg (BH)** (FDR control under independence or suitable positive dependence, such as PRDS on the true nulls):
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject $H_{(1)}, \ldots, H_{(k)}$; if no such $k$ exists, reject none.
4. Adjusted: $\tilde{p}_{(i)} = \min_{j \geq i} \min(\frac{m}{j} p_{(j)}, 1)$

**Benjamini-Yekutieli (BY)** (FDR control under arbitrary dependence):
- Same as BH but with correction factor $\sum_{j=1}^{m} \frac{1}{j}$
- More conservative than BH but valid under any dependence structure

**Hochberg step-up procedure** (FWER control under independence or suitable positive dependence; non-negative pairwise correlation alone is not a sufficient general condition):
1. Order the p-values increasingly and search from the largest rank downwards.
2. Find the largest $k$ such that $p_{(k)} \leq \alpha / (m-k+1)$. Reject $H_{(1)}, \ldots, H_{(k)}$ and do not reject the remaining hypotheses. If no such $k$ exists, reject none.
3. Adjusted: $\tilde{p}_{(i)} = \min_{j \geq i} \min((m-j+1) \cdot p_{(j)}, 1)$

### P-value Combination (combine_pvalues)

**Fisher's method** (chi-squared combination):
$$T = -2 \sum_{i=1}^{m} \ln(p_i) \sim \chi^2_{2m}$$
- This null distribution requires independent p-values that are uniform under the global null (continuous, correctly calibrated tests).
- Powerful when a small subset of p-values is very small

**Cauchy Combination Test (ACAT)** (Cauchy-tail approximation for dependent tests):
$$T = \sum_{i=1}^{m} w_i \tan\left((0.5 - p_i)\pi\right)$$
- Weights are non-negative and normalized to sum to one; the default is equal weights.
- The implementation uses $p_{\mathrm{global}} = 0.5 - \arctan(T)/\pi$. Under dependence, this is a tail approximation under the conditions of Liu & Xie (2020), not an exact Cauchy null distribution for all dependence structures or significance levels.
- See Liu & Xie (2020) for the statistical assumptions and scope of the approximation.

**Stouffer's method** (z-score combination):
$$T = \frac{\sum_{i=1}^{m} w_i \Phi^{-1}(1-p_i)}{\sqrt{\sum_{i=1}^{m} w_i^2}} \sim N(0, 1)$$
- This null distribution requires independent, null-uniform p-values and fixed weights.
- For a directional interpretation, use one-sided p-values with a common prespecified direction. Two-sided p-values alone do not retain effect signs.

## Parameters

### adjust_pvalues

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvalues` | array-like | — | Finite raw p-values in `[0, 1]`; NaN/inf are rejected |
| `method` | str | `'bh'` | `'bh'`, `'by'`, `'holm'`, `'bonferroni'`, `'hochberg'` |
| `alpha` | float | `0.05` | Rejection threshold in `(0, 1)` |
| `axis` | int or None | `None` | Tests within each axis slice; `None` pools all entries |
| `backend` | str | `'auto'` | `'numpy'`, `'cupy'`, `'torch'`, `'auto'` |

### combine_pvalues

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvalues` | array-like | — | Finite raw p-values in `[0, 1]`; NaN/inf are rejected |
| `method` | str | `'fisher'` | `'fisher'`, `'cauchy'`/`'acat'`, `'stouffer'` |
| `weights` | array-like | `None` | One shared weight vector for cauchy/stouffer; see constraints below |
| `axis` | int or None | `None` | Tests within each axis slice; `None` pools all entries |
| `backend` | str | `'auto'` | `'numpy'`, `'cupy'`, `'torch'`, `'auto'` |

## Combination example and interpretation

```python
import numpy as np
from statgpu.inference import combine_pvalues

# Illustrative independent, null-uniform component tests under the global null.
pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
stat, p_global = combine_pvalues(pvals, method='fisher')
print(f"{float(stat):.4f}, {float(p_global):.6f}")

# Fixed weights, chosen before looking at these p-values.
weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
z_stat, p_stouffer = combine_pvalues(pvals, method='stouffer', weights=weights)
```

The Fisher output is `37.4167, 0.000048`. Under its assumptions, the small global p-value is evidence against the hypothesis that every component null holds. It does not mean all five hypotheses should be rejected. For directional Stouffer interpretation, supply one-sided p-values with a common prespecified direction; two-sided p-values do not recover the effect signs.

## Input constraints and numerical limits

- P-values must be finite and within `[0, 1]`. NaN/inf and out-of-range values raise `ValueError`; there is no `nan_policy` option or automatic omission. Resolve missing-test handling as part of the analysis plan rather than deleting tests after seeing their results.
- Pass a finite `alpha` strictly between zero and one. `adjust_pvalues` returns `reject = pvals_adj <= alpha`; changing `alpha` does not change the adjusted p-values.
- For Cauchy/Stouffer, supply a finite, non-negative weight vector with length equal to the selected axis size (or total input size for `axis=None`) and positive sum. The same vector is used for every slice and is normalized internally; its overall positive scale does not matter. `None` means equal weights. Fisher rejects non-`None` weights. Prespecify Cauchy/Stouffer weights before inspecting the component p-values; do not choose weights to favor the observed small p-values. Numerical acceptance of weights is not evidence that their statistical choice is valid.
- Combination requires at least one p-value per combined slice. For scalar input, use `axis=None`.
- Inputs are converted to float64 for these calculations. Fisher clips zero to the smallest positive normal float64 value before taking logarithms; Cauchy/Stouffer clip to `[eps, 1-eps]`, where `eps` is float64 machine epsilon. Extreme tails and endpoints can therefore saturate or lose resolution. Do not interpret such outputs as arbitrary-precision tail probabilities. These numerical safeguards do not establish statistical validity.

## Backends and GPU use

For these module-level functions, `backend='auto'` selects the array library from `pvalues`: NumPy for Python/NumPy input, CuPy for CuPy arrays, and Torch for Torch tensors. It does not automatically send a NumPy input to a GPU. Results use the selected array library, rather than always returning NumPy.

`backend='cupy'` requires a working CuPy/CUDA installation. The functional Torch backend uses CUDA when available and Torch CPU otherwise; this differs from the strict GPU meaning of an estimator's `device='torch'`. Backend selection is not a guarantee to preserve an input tensor's concrete device ordinal. These functions do not expose a `device` argument. Check the returned tensor's device when placement matters.

After installing PyTorch with working CUDA support:

```python
import torch
from statgpu.inference import adjust_pvalues

pvals_gpu = torch.tensor([0.001, 0.01, 0.03, 0.05, 0.50],
                         dtype=torch.float64, device='cuda')
reject_gpu, adjusted_gpu = adjust_pvalues(pvals_gpu, method='bh', backend='torch')
print(reject_gpu.cpu().tolist())
```

The mask is `[True, True, True, False, False]`, as in the CPU example. GPU use changes the computation location, not the hypothesis family or statistical assumptions; small arrays need not be faster on a GPU.

## Outputs

| Method | Returns | Description |
|---|---|---|
| `adjust_pvalues` | `(reject, pvals_adj)` | Boolean mask + float64 adjusted p-values, both with input shape and order |
| `combine_pvalues` | `(statistic, p_global)` | Statistic + global p-value, with selected axis removed; scalar/zero-dimensional results for `axis=None` |

## FAQ

**Q: Which method should I use?**  
A: Use **BH** for FDR control when its dependence assumptions hold, or **BY** for arbitrary dependence. Use **Bonferroni/Holm** for FWER control with valid individual p-values. **Cauchy** combines evidence for a global test using the approximation described above.

**Q: Can I use these for genome-wide association studies?**  
A: Yes, but choose the error criterion and hypothesis family first. Per-variant FWER/FDR adjustment and a combined gene- or set-level global test answer different questions. A combined p-value does not replace adjustment across the multiple genes or sets tested.

**Q: What's the difference between FDR and FWER?**  
A: FDR is the expected false-rejection fraction among rejections (zero if none are rejected); FWER is the probability of at least one false rejection in the family. FDR is a less stringent error criterion, not a universal ranking of every method’s conservativeness.

## External Validation

- R: `p.adjust()` for adjustment, `pchisq()` for Fisher
- statsmodels: `multipletests()` (BH, Holm, Bonferroni, BY, Hochberg)
- scipy: `combine_pvalues()` for Fisher/Stouffer comparisons. SciPy also offers methods such as Tippett that this API does not implement; its axis and missing-value defaults differ.

## References

1. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
2. Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *Annals of Statistics*, 29(4), 1165-1188.
3. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
4. Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
5. Liu, Y. & Xie, J. (2020). "Cauchy Combination Test: A Powerful Test With Analytic p-Value Calculation Under Arbitrary Dependency Structures." *Journal of the American Statistical Association*, 115(529), 393-402.
6. Stouffer, S.A. et al. (1949). *The American Soldier*. Princeton University Press.

See also: [R adjustment assumptions](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html), [SciPy combination assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html), and [Liu & Xie (2020)](https://doi.org/10.1080/01621459.2018.1554485).
