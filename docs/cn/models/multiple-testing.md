# 多重检验校正

> 语言：中文  
> 最后更新：2026-10-03  
> 页面定位：`statgpu.inference` 推断模块（NumPy、CuPy、PyTorch）  
> 切换：[English](../../en/models/multiple-testing.md)

## 概述

当同时检验多个假设时，至少出现一次错误发现的概率会增加。P 值校正控制各项检验决策的族错误率（FWER）或错误发现率（FDR）。P 值合并则检验全局原假设，并不直接判定应拒绝哪些单个假设。

## 先确定问题与错误率目标

假设要检验五个候选生物标志物。“应该标记哪些标志物？”需要对各项 P 值进行校正；“是否有证据反对五个原假设全部成立？”需要一个全局合并检验。应在查看哪些 P 值较小之前确定假设族。校正或合并都不能修复无效或受选择偏差影响的原始 P 值。

| 目标 | 起点选择 | 重要条件 |
|---|---|---|
| 限制所选假设族中出现任一次错误拒绝的概率 | `holm` | 单项 P 值有效；允许任意相依结构。Bonferroni 更简单，但更保守。 |
| 限制全部拒绝中错误拒绝所占比例的期望 | `bh`；相依结构不受限制时用 `by` | BH 要求独立或适当正相依；BY 允许任意相依结构。 |
| 检验所有单项原假设均成立的全局原假设 | `fisher` 或 `stouffer` | P 值独立且在原假设下均匀分布；Stouffer 还要求权重固定。 |
| 用尾部近似合并相依的证据 | `cauchy` / `acat` | 须检查下文的 Liu–Xie 条件和近似边界；仅有相依性并不保证校准正确。 |

FWER 是假设族中至少出现一次错误拒绝的概率。FDR 是**全部拒绝中**错误拒绝所占比例的期望；没有拒绝时，该比例定义为零。FDR 目标为 0.05，不保证本次发现列表中错误占比一定不超过 5%，也不表示每项发现都有 5% 的错误概率。应根据科学问题可接受的错误风险预先确定 `alpha`。

## 最小可运行示例

以下示意 P 值属于一个预先确定的假设族。只有输入检验满足有效性及相依条件时，才可采用 BH。

```python
import numpy as np
from statgpu.inference import adjust_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
reject, pvals_adj = adjust_pvalues(pvals, method='bh', alpha=0.05)
print(reject.tolist())
print(np.round(pvals_adj, 4).tolist())
```

预期输出：

```text
[True, True, True, False, False]
[0.005, 0.025, 0.05, 0.0625, 0.5]
```

前三个假设的校正后 P 值**小于或等于** 0.05，因此被拒绝。输出保持输入顺序，无须自行排序或还原顺序。`False` 表示“在此水平下未拒绝”，并不表示“已证明为真”。拒绝原假设也不证明效应具有实际重要性或因果关系。

## 用 `axis` 确定假设族

对于矩阵，默认的 `axis=None` 将所有元素归入同一个假设族。`axis=1` 将每一行视为独立处理的假设族；`axis=0` 则逐列处理。负轴编号遵循通常的数组约定。应根据科学问题选择轴，而非选择产生更多拒绝结果的选项。

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

# Fisher 要求每行内的 P 值独立，且在原假设下均匀分布。
row_stat, row_p_global = combine_pvalues(p_matrix, method='fisher', axis=1)
print(np.round(row_p_global, 6).tolist())
```

预期输出：

```text
[[True, False], [False, False]]
[[0.04, 0.12], [0.4, 0.8]]
[[True, True], [False, False]]
[[0.02, 0.04], [0.4, 0.8]]
[0.00353, 0.453213]
```

逐行计算处理的是两个各含两项检验的假设族，而不是一个含四项检验的假设族。各行分别控制 FWER/FDR，**不会**自动控制所有元素合在一起的相应错误率。如果发现结论覆盖全部四项检验，就应对整个假设族进行校正。

合并会移除指定的轴：此处 `row_p_global.shape == (2,)`。在 0.05 水平下，第一行有证据反对“该行所有原假设均成立”，第二行则没有；这不能指出哪一个单项原假设为假。若用 `axis=None`，则对全部四项只返回一个统计量和一个 P 值。如果要在许多行级全局检验之间作出发现结论，还需对这些全局 P 值采用适当的多重检验方法。

## 数学原理

### P 值校正（adjust_pvalues）

给定 $m$ 个原始 P 值 $p_1, p_2, \ldots, p_m$，校正后的 P 值 $\tilde{p}_i$ 控制指定的错误率。

**Bonferroni 校正**（FWER 控制）：
$$\tilde{p}_i = \min(m \cdot p_i, 1)$$

**Holm step-down 过程**（FWER 控制，检验功效不低于 Bonferroni）：
1. 将 P 值排序： $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. 从 $i=1$ 开始：若 $p_{(i)} \leq \alpha / (m - i + 1)$，则拒绝 $H_{(i)}$ 并继续下一项；首次不满足条件时立即停止，该项及后续假设均不拒绝。若全部满足条件，则拒绝全部假设。
3. 校正后：$\tilde{p}_{(i)} = \max_{j \leq i} \min((m-j+1) \cdot p_{(j)}, 1)$

**Benjamini-Hochberg（BH）**（独立或满足适当正相依条件，例如真原假设上的 PRDS 条件时控制 FDR）：
1. 将 P 值排序： $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$
2. 找到满足 $p_{(k)} \leq \frac{k}{m} \alpha$ 的最大 $k$
3. 拒绝 $H_{(1)}, \ldots, H_{(k)}$；若不存在这样的 $k$，则全部不拒绝。
4. 校正后：$\tilde{p}_{(i)} = \min_{j \geq i} \min(\frac{m}{j} p_{(j)}, 1)$

**Benjamini-Yekutieli（BY）**（任意相依结构下控制 FDR）：
- 与 BH 相同，但带有校正因子 $\sum_{j=1}^{m} \frac{1}{j}$
- 比 BH 更保守，但在任意相依结构下都成立

**Hochberg step-up 过程**（独立或满足适当正相依条件时控制 FWER；仅有两两非负相关一般不足以保证有效性）：
1. 将 P 值从小到大排序，再从最大秩向前查找。
2. 找到满足 $p_{(k)} \leq \alpha / (m-k+1)$ 的最大 $k$，拒绝 $H_{(1)}, \ldots, H_{(k)}$，其余假设不拒绝。若不存在这样的 $k$，则全部不拒绝。
3. 校正后：$\tilde{p}_{(i)} = \min_{j \geq i} \min((m-j+1) \cdot p_{(j)}, 1)$

### P 值合并（combine_pvalues）

**Fisher 方法**（卡方合并）：
$$T = -2 \sum_{i=1}^{m} \ln(p_i) \sim \chi^2_{2m}$$
- 该原假设分布要求各 P 值相互独立，且在全局原假设下服从均匀分布（连续且校准正确的检验）。
- 当少数 P 值极小时检验功效较高

**Cauchy 合并检验（ACAT）**（相依检验的 Cauchy 尾部近似）：
$$T = \sum_{i=1}^{m} w_i \tan\left((0.5 - p_i)\pi\right)$$
- 权重非负，并归一化为和等于一；默认使用等权重。
- 实现使用 $p_{\mathrm{global}} = 0.5 - \arctan(T)/\pi$。存在相依关系时，这是在 Liu & Xie (2020) 所列条件下的尾部近似，不能视为对所有相依结构和显著性水平都成立的精确 Cauchy 原假设分布。
- 统计假设与近似适用范围见 Liu & Xie (2020)。

**Stouffer 方法**（z 分数合并）：
$$T = \frac{\sum_{i=1}^{m} w_i \Phi^{-1}(1-p_i)}{\sqrt{\sum_{i=1}^{m} w_i^2}} \sim N(0, 1)$$
- 该原假设分布要求各 P 值相互独立、在原假设下服从均匀分布，且权重固定。
- 若要作方向性解释，应使用方向事先统一的单侧 P 值。仅凭双侧 P 值无法保留效应正负方向。

## 参数

### adjust_pvalues

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pvalues` | array-like | — | `[0, 1]` 内的有限原始 P 值；拒绝 NaN/inf |
| `method` | str | `'bh'` | `'bh'`、`'by'`、`'holm'`、`'bonferroni'`、`'hochberg'` |
| `alpha` | float | `0.05` | `(0, 1)` 内的拒绝阈值 |
| `axis` | int 或 None | `None` | 每个轴切片内的检验；`None` 合并考虑所有元素 |
| `backend` | str | `'auto'` | `'numpy'`、`'cupy'`、`'torch'`、`'auto'` |

### combine_pvalues

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pvalues` | array-like | — | `[0, 1]` 内的有限原始 P 值；拒绝 NaN/inf |
| `method` | str | `'fisher'` | `'fisher'`、`'cauchy'`/`'acat'`、`'stouffer'` |
| `weights` | array-like | `None` | cauchy/stouffer 共用的一组权重；约束见下文 |
| `axis` | int 或 None | `None` | 每个轴切片内的检验；`None` 合并考虑所有元素 |
| `backend` | str | `'auto'` | `'numpy'`、`'cupy'`、`'torch'`、`'auto'` |

## 合并示例与结果解释

```python
import numpy as np
from statgpu.inference import combine_pvalues

# 示意输入：各单项检验独立，P 值在全局原假设下均匀分布。
pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
stat, p_global = combine_pvalues(pvals, method='fisher')
print(f"{float(stat):.4f}, {float(p_global):.6f}")

# 在查看这些 P 值之前确定的固定权重。
weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
z_stat, p_stouffer = combine_pvalues(pvals, method='stouffer', weights=weights)
```

Fisher 的输出为 `37.4167, 0.000048`。在其假设成立时，较小的全局 P 值提供了反对“所有单项原假设均成立”的证据，并不表示五个假设都应拒绝。若要作 Stouffer 方向性解释，应输入方向事先统一的单侧 P 值；双侧 P 值不能恢复效应的正负方向。

## 输入约束与数值边界

- P 值必须有限且位于 `[0, 1]`。NaN/inf 和越界值会引发 `ValueError`；没有 `nan_policy` 参数，也不会自动略去缺失值。应在分析方案中确定缺失检验的处理方式，而不是看到结果后再删除检验。
- `alpha` 应为严格介于零和一之间的有限值。`adjust_pvalues` 返回 `reject = pvals_adj <= alpha`；改变 `alpha` 不改变校正后的 P 值。
- Cauchy/Stouffer 的权重向量须有限、非负、总和为正，长度等于所选轴的大小；`axis=None` 时，长度等于输入元素总数。所有切片共用该向量，函数会在内部归一化，因此整体乘以正常数不影响结果。`None` 表示等权重。Fisher 不接受非 `None` 的权重。Cauchy/Stouffer 权重应在查看单项 P 值之前确定，不能为了偏重已观察到的小 P 值而选择权重。权重通过数值检查不表示其统计选择有效。
- 每个待合并切片至少应有一个 P 值。标量输入应使用 `axis=None`。
- 计算时输入转换为 float64。Fisher 在取对数前将零截到 float64 的最小正规正数；Cauchy/Stouffer 则截到 `[eps, 1-eps]`，其中 `eps` 是 float64 的机器精度。因此极端尾部和端点可能出现饱和或分辨率损失，不能将结果视为任意精度的尾部概率。这些数值保护并不保证统计有效性。

## 后端与 GPU 使用

这些模块级函数的 `backend='auto'` 根据 `pvalues` 选择数组库：Python/NumPy 输入使用 NumPy，CuPy 数组使用 CuPy，Torch 张量使用 Torch；不会自动把 NumPy 输入送到 GPU。结果使用所选数组库，并非一律返回 NumPy。

`backend='cupy'` 需要可用的 CuPy/CUDA 安装。函数级 Torch 后端在 CUDA 可用时使用 CUDA，否则使用 Torch CPU；这与估计器 `device='torch'` 严格要求 GPU 的含义不同。选择后端不保证保留输入张量的具体设备编号，这些函数也没有 `device` 参数。若设备位置重要，应检查返回张量的设备。

安装并确认 PyTorch 的 CUDA 支持可用后：

```python
import torch
from statgpu.inference import adjust_pvalues

pvals_gpu = torch.tensor([0.001, 0.01, 0.03, 0.05, 0.50],
                         dtype=torch.float64, device='cuda')
reject_gpu, adjusted_gpu = adjust_pvalues(pvals_gpu, method='bh', backend='torch')
print(reject_gpu.cpu().tolist())
```

拒绝掩码为 `[True, True, True, False, False]`，与 CPU 示例相同。GPU 改变计算位置，不改变假设族或统计条件；小数组在 GPU 上不一定更快。

## 输出

| 方法 | 返回 | 说明 |
|---|---|---|
| `adjust_pvalues` | `(reject, pvals_adj)` | 布尔掩码与 float64 校正后 P 值；均保持输入形状和顺序 |
| `combine_pvalues` | `(statistic, p_global)` | 统计量与全局 P 值；移除所选轴，`axis=None` 时为标量或零维结果 |

## 常见问题

**Q: 应该用哪种方法？**  
A: 满足相依条件时，可用 **BH** 控制 FDR；任意相依结构下可用 **BY**。各单项 P 值有效时，可用 **Bonferroni/Holm** 控制 FWER。**Cauchy** 按上述近似合并证据，用于全局检验。

**Q: 可以用于全基因组关联分析吗？**  
A: 可以，但应先确定错误率目标和假设族。逐变异位点的 FWER/FDR 校正，与基因或集合层面的全局合并检验回答不同问题；合并后的 P 值不能替代多个基因或集合之间的多重检验校正。

**Q: FDR 与 FWER 有什么区别？**  
A: FDR 是全部拒绝中错误拒绝比例的期望（没有拒绝时比例为零）；FWER 是假设族中至少一次错误拒绝的概率。FDR 是更宽松的错误率标准，并非对所有方法保守程度的统一排序。

## 外部验证

- R：`p.adjust()` 用于校正，`pchisq()` 用于 Fisher 合并
- statsmodels：`multipletests()`（BH、Holm、Bonferroni、BY、Hochberg）
- scipy：`combine_pvalues()` 可用于 Fisher/Stouffer 比较。SciPy 还提供本 API 未实现的 Tippett 等方法；其轴和缺失值默认规则也不同。

## 参考文献

1. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
2. Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *Annals of Statistics*, 29(4), 1165-1188.
3. Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
4. Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
5. Liu, Y. & Xie, J. (2020). "Cauchy Combination Test: A Powerful Test With Analytic p-Value Calculation Under Arbitrary Dependency Structures." *Journal of the American Statistical Association*, 115(529), 393-402.
6. Stouffer, S.A. et al. (1949). *The American Soldier*. Princeton University Press.

另见：[R 校正方法的适用条件](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)、[SciPy 合并方法的适用条件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)及 [Liu & Xie (2020)](https://doi.org/10.1080/01621459.2018.1554485)。
