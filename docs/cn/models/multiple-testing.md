# 多重检验校正

> 语言：中文  
> 最后更新：2026-09-29  
> 页面定位：`statgpu.inference` 推断模块（NumPy、CuPy、PyTorch）  
> 切换：[English](../../en/models/multiple-testing.md)

## 概述

当同时检验多个假设时，至少出现一次错误发现的概率会增加。本模块提供 P 值校正与合并方法，用于控制族错误率（FWER）或错误发现率（FDR）。

## 数学原理

### P 值校正（adjust_pvalues）

给定 $m$ 个原始 P 值 $p_1, p_2, \ldots, p_m$，校正后的 P 值 $\tilde{p}_i$ 控制指定的错误率。

**Bonferroni 校正**（FWER 控制）：
$$\tilde{p}_i = \min(m \cdot p_i, 1)$$

**Holm step-down 过程**（FWER 控制，检验功效不低于 Bonferroni）：
1. 将 P 值排序： $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. 若 $p_{(i)} < \alpha / (m - i + 1)$，则拒绝 $H_{(i)}$
3. 校正后：$\tilde{p}_{(i)} = \max_{j \leq i} \min((m-j+1) \cdot p_{(j)}, 1)$

**Benjamini-Hochberg（BH）**（FDR 控制）：
1. 将 P 值排序： $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. 找到满足 $p_{(k)} \leq \frac{k}{m} \alpha$ 的最大 $k$
3. 拒绝 $H_{(1)}, \ldots, H_{(k)}$
4. 校正后：$\tilde{p}_{(i)} = \min_{j \geq i} \min(\frac{m}{j} p_{(j)}, 1)$

**Benjamini-Yekutieli（BY）**（任意相依结构下控制 FDR）：
- 与 BH 相同，但带有校正因子 $\sum_{j=1}^{m} \frac{1}{j}$
- 比 BH 更保守，但在任意相依结构下都成立

**Hochberg step-up 过程**（FWER 控制，假设非负相关）：
1. 从最大的 P 值开始
2. 若 $p_{(i)} > \alpha / (m - i + 1)$，则接受 $H_{(i)}$
3. 校正后：$\tilde{p}_{(i)} = \min_{j \geq i} \min((m-j+1) \cdot p_{(j)}, 1)$

### P 值合并（combine_pvalues）

**Fisher 方法**（卡方合并）：
$$T = -2 \sum_{i=1}^{m} \ln(p_i) \sim \chi^2_{2m}$$
- 在 $H_0$ 下：$T \sim \chi^2_{2m}$
- 当少数 P 值极小时检验功效较高

**Cauchy 合并检验（ACAT）**（对任意相依结构稳健）：
$$T = \sum_{i=1}^{m} w_i \tan\left((0.5 - p_i)\pi\right) \sim \text{Cauchy}(0, 1)$$
- 在 $H_0$ 下近似服从 Cauchy 分布
- 无需对相依结构作任何假设
- 见 Liu & Xie (2020)

**Stouffer 方法**（z 分数合并）：
$$T = \frac{\sum_{i=1}^{m} w_i \Phi^{-1}(1-p_i)}{\sqrt{\sum_{i=1}^{m} w_i^2}} \sim N(0, 1)$$
- 在 $H_0$ 下：$T \sim N(0, 1)$
- 当各效应方向一致时功效最高

## 参数

### adjust_pvalues

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pvalues` | array-like | — | 原始 P 值 |
| `method` | str | `'bh'` | `'bh'`、`'by'`、`'holm'`、`'bonferroni'`、`'hochberg'` |
| `alpha` | float | `0.05` | 显著性水平 |
| `axis` | int 或 None | `None` | 批量处理的轴 |
| `backend` | str | `'auto'` | `'numpy'`、`'cupy'`、`'torch'`、`'auto'` |

### combine_pvalues

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pvalues` | array-like | — | 原始 P 值 |
| `method` | str | `'fisher'` | `'fisher'`、`'cauchy'`/`'acat'`、`'stouffer'` |
| `weights` | array-like | `None` | 非负权重（cauchy/stouffer） |
| `axis` | int 或 None | `None` | 批量处理的轴 |
| `backend` | str | `'auto'` | `'numpy'`、`'cupy'`、`'torch'`、`'auto'` |

## CPU 与 GPU 示例

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

# 5 个假设检验的原始 P 值
pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])

# Benjamini-Hochberg FDR 控制
reject, pvals_adj = adjust_pvalues(pvals, method='bh', alpha=0.05)
print(f"拒绝: {reject}")
print(f"校正后 P 值: {pvals_adj}")

# Fisher 合并
stat, p_global = combine_pvalues(pvals, method='fisher')
print(f"Fisher 统计量: {stat:.4f}, 全局 P 值: {p_global:.6f}")

# Cauchy 合并（对相依结构稳健）
stat, p_global = combine_pvalues(pvals, method='cauchy')

# 带权 Stouffer
weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
stat, p_global = combine_pvalues(pvals, method='stouffer', weights=weights)
```

**GPU 加速：**

```python
import torch
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals_gpu = torch.tensor([0.001, 0.01, 0.03, 0.05, 0.50], device='cuda')
reject, pvals_adj = adjust_pvalues(pvals_gpu, method='bh', backend='torch')
```

## 输出

| 方法 | 返回 | 说明 |
|---|---|---|
| `adjust_pvalues` | `(reject, pvals_adj)` | 布尔拒绝数组与校正后 P 值 |
| `combine_pvalues` | `(statistic, p_global)` | 检验统计量与全局 P 值 |

## 常见问题

**Q: 应该用哪种方法？**  
A: FDR 控制用 **BH**（最常用）。严格 FWER 控制用 **Bonferroni/Holm**。P 值相关时用 **Cauchy**。

**Q: 可以用于全基因组关联分析吗？**  
A: 可以。BH 是 GWAS 的标准方法。对于相关检验，使用 Cauchy 合并。

**Q: FDR 与 FWER 有什么区别？**  
A: FDR = 错误拒绝的期望比例。FWER = 至少一次错误拒绝的概率。FDR 更宽松。

## 外部验证

- R：`p.adjust()` 用于校正，`pchisq()` 用于 Fisher 合并
- statsmodels：`multipletests()`（BH、Holm、Bonferroni、BY、Hochberg）
- scipy：`combine_pvalues()`（Fisher、Stouffer、Tippett）

## 参考文献

1. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
2. Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *Annals of Statistics*, 29(4), 1165-1188.
3. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
4. Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
5. Liu, Y. & Xie, J. (2020). "Cauchy Combination Test: A Powerful Test With Analytic p-Value Calculation Under Arbitrary Dependency Structures." *Journal of the American Statistical Association*, 115(529), 393-402.
6. Stouffer, S.A. et al. (1949). *The American Soldier*. Princeton University Press.
