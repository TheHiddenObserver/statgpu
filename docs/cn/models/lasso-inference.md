# Lasso 推断

> 语言：中文  
> 最后更新：2026-09-06  
> 模型指南：[Lasso](lasso.md)  
> 切换：[English](../../en/models/lasso-inference.md)

本页是 learner-first [Lasso 指南](lasso.md)的统计推断 companion。它说明 statgpu 的不同拟合后推断路径实际计算什么、报告的区间应该怎样解释，以及当前实现在哪些地方没有声称更强的统计保证。

## 为什么 Lasso 推断需要单独一页？

Lasso 的拟合系数来自带惩罚的预测/选择问题，而不是“在看数据之前就已经固定模型”的普通 OLS 估计量。用同一批数据选择稀疏 active set、再估计系数之后，直接给选中模型附上普通 OLS 标准误，通常会忽略选择带来的额外不确定性。

因此，statgpu 的多个 `inference_method` 并不是“打印同一组 p 值的不同实现”，而是统计主张不同的路径。

## 根据你真正需要的统计主张选择路径

| `inference_method` | statgpu 计算什么 | 合适的解释 | 主要限制 |
|---|---|---|---|
| `debiased` | 去偏/去稀疏化系数、标准误、z 统计量、p 值和边际置信区间 | 在去偏 Lasso 假设下做逐系数高维推断 | 有效性依赖稀疏性、设计矩阵、噪声以及正则化/去偏构造 |
| `cpu_ols` | 通过当前 CPU-oriented helper 对 selected active set 做 OLS 风格重拟合 | 工程或 post-selection diagnostic | 不是一般意义上的 selective-inference 置信程序 |
| `gpu_ols` | unified wrapper 接受的兼容拼法；当前同样进入 CPU-oriented post-selection OLS helper | 与 `cpu_ols` 相同的 diagnostic 角色 | 不是 backend-native GPU OLS inference；GPU-resident 输入可能不适用 |
| `bootstrap` | 对惩罚模型做 residual-bootstrap 重拟合 | 基于重采样的不确定性 diagnostic | 计算昂贵，而且本身不是对数据驱动模型选择的普适修正 |

当前 unified 实现中，`cpu_ols` 与 `gpu_ols` **并不是两个不同的统计程序**：两者最终都进入同一个 CPU-oriented post-selection OLS helper。它们带硬件含义的名字属于兼容 surface，而不应该成为“根据设备选择统计方法”的理由。更早的 `cpu_ols_inference` / `gpu_ols_inference` 名称属于 legacy Lasso surface 和历史文档，不是当前 unified `Lasso` wrapper 的 active `inference_method` 值。

如果目标是 Lasso 之后的正式逐系数推断，`debiased` 是 statgpu 的主要路径。如果只关心预测或特征选择，可以设置 `compute_inference=False`，避免支付不需要的推断成本。

## 最小去偏推断示例

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(700, 10))
beta = np.zeros(10)
beta[[1, 4, 8]] = [1.5, -1.1, 0.7]
y = 0.4 + X @ beta + rng.normal(scale=1.0, size=X.shape[0])

model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

用于预测的惩罚系数仍保存在 `coef_`。推断报告使用去偏后的参数向量，因此 penalized coefficient 与对应的去偏推断估计不必数值相同。

## 去偏到底改变了什么？

记拟合得到的 Lasso 系数为 $\hat\beta$，残差为

$$
r = y - b - X\hat\beta.
$$

L1 惩罚会产生收缩偏差。statgpu 使用

$$
\hat\theta^{\mathrm{db}}
= \hat\beta + \frac{1}{n} M X^\top r
$$

校正系数，其中 $M$ 是由数据构造的 decorrelation matrix，用来近似特征 Gram/协方差矩阵的逆。

这并不意味着原来的稀疏 Lasso 估计“变成了 OLS”。`coef_` 仍然是预测使用的惩罚模型；去偏向量是逐系数不确定性报告使用的 inference object。

## statgpu 如何构造 decorrelation matrix？

对每个特征 $j$，statgpu 用 node-wise Lasso 把 $x_j$ 对其余列 $X_{-j}$ 做稀疏回归。在 CPU 实现中，node-wise penalty scale 为

$$
\lambda_{\mathrm{nw}}
= \hat\sigma\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

记 node-wise 系数为 $\hat\gamma_j$，定义

$$
z_j = x_j - X_{-j}\hat\gamma_j,
\qquad
C_j = \frac{z_j^\top x_j}{n}.
$$

随后用对角位置的 $1/C_j$ 与其他位置的 $-\hat\gamma_j/C_j$ 组成 $M$ 的第 $j$ 行。这正是 de-sparsified Lasso 中稀疏 precision/decorrelation 思路与 statgpu 实际计算之间的桥梁。

node-wise 数值收敛是必要条件，但不能替代去偏推断所要求的统计假设。

## 标准误、z 统计量与边际区间

令

$$
\widehat\Sigma = \frac{X^\top X}{n},
\qquad
V = M\widehat\Sigma M^\top,
$$

statgpu 使用

$$
\widehat{\mathrm{se}}_j
= \sqrt{\frac{\hat\sigma^2 V_{jj}}{n}},
\qquad
Z_j = \frac{\hat\theta_j^{\mathrm{db}}}{\widehat{\mathrm{se}}_j}.
$$

双侧 p 值使用标准正态参考分布。当前 `_conf_int` 保存的是基于正态临界值的 95% **边际**置信区间。

需要区分：

- `_conf_int[j]` 是单个参数的边际区间；
- 一组 95% 边际区间并不会自动对整个系数向量提供 95% simultaneous coverage；
- `simultaneous_alpha` 不改变这些边际区间，它只控制下面单独的 simultaneous procedure。

拟合截距时，statgpu 也会报告截距的不确定性，但 feature de-biasing 与截距计算不是同一个代数步骤；不要把截距行理解成另一个 node-wise-Lasso 坐标。

## Simultaneous max-|Z| 推断

可以这样启用 multiplier-bootstrap simultaneous calibration：

```python
model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    compute_inference=True,
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=2000,
    simultaneous_random_state=123,
).fit(X, y)

marginal = model._conf_int
simultaneous = model._conf_int_simultaneous
critical = model._simultaneous_critical_value
```

当前方法抽取独立标准正态 multiplier $\xi_i$，用拟合残差构造 bootstrap score perturbation，并记录目标特征集合上最大的标准化绝对扰动。示意地，

$$
T^*
= \max_{j\in\mathcal J}
\left|
\frac{n^{-1}\sum_i \xi_i r_i (MX_i)_j}
{\widehat{\mathrm{se}}_j}
\right|.
$$

$T^*$ 的经验 $(1-\alpha)$ 分位数给出共同临界值 $c_{1-\alpha}$，区间为

$$
\hat\theta_j^{\mathrm{db}}
\pm c_{1-\alpha}\widehat{\mathrm{se}}_j.
$$

因为同一个临界值要保护整个目标 family，simultaneous interval 通常比相应的 marginal interval 更宽。

### Simultaneous inference 控制项

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `enable_simultaneous_inference` | `False` | 成功完成 debiased inference 后再做 simultaneous calibration。 |
| `simultaneous_method` | `"maxz_bootstrap"` | 当前支持的 simultaneous calibration 方法。 |
| `simultaneous_alpha` | `0.05` | 用于选择共同临界值的 family-wise error level。 |
| `simultaneous_n_bootstrap` | `1000` | multiplier-bootstrap 抽样次数。 |
| `simultaneous_random_state` | `None` | 控制 multiplier 抽样的随机种子。 |
| `simultaneous_include_intercept` | `False` | 请求把截距加入报告的 simultaneous target；见下面的当前实现边界。 |

`enable_simultaneous_inference=True` 要求同时满足 `compute_inference=True`、`inference_method="debiased"` 和 `simultaneous_method="maxz_bootstrap"`。不支持的组合应失败，而不是静默返回普通边际区间。

### 当前截距边界

推荐/默认的 simultaneous family 是 feature vector（`simultaneous_include_intercept=False`）。当前实现中，设为 `True` 会把由**特征** max-|Z| bootstrap 校准出的临界值应用到截距行，但 bootstrap maximum 本身仍由 feature-score 坐标构成。因此，不要把该选项解释为“截距拥有单独 bootstrap score 的完整 intercept-inclusive max-|Z| family”。如果科学问题要求 family 必须包含截距，应把它视为需要额外验证的情形，而不是直接依赖默认 feature-family 保证。

## Simultaneous interval 不等于 p 值校正

下面几件事相关但并不相同：

- `_conf_int` 是逐系数 marginal interval；
- `_conf_int_simultaneous` 使用共同 max-|Z| 临界值保护一个 interval family；
- `model.adjust_pvalues(method="bh")` 或模块级 `adjust_pvalues(...)` 是对已有 p-value vector 做 multiple-testing correction；
- `model.combine_pvalues(...)` 检验一组 p 值是否包含全局证据，它不会构造 simultaneous coefficient interval。

通用 estimator-bound 与模块级 multiple-testing API 见[推断 API 参考](../guides/inference-api.md)。

## Backend 行为与 reporting boundary

去偏系数推断具有 NumPy CPU、CuPy CUDA 和 Torch CUDA 的专用数值路径。在相应路径可用时，昂贵的 node-wise/decorrelation 计算会在所选 numerical backend 上执行。

最终 reporting layer 以 NumPy 为主：`_bse`、`_pvalues`、`_conf_int` 等推断数组以及 structured result metadata 会作为 host-side reporting object 暴露。

Simultaneous calibration 还存在进一步的边界：backend-native debiased inference 完成后，max-|Z| multiplier bootstrap 所需状态会表示到 CPU/NumPy 侧，bootstrap calibration 在那里运行。因此 `device="cuda"` / `device="torch"` 并不意味着每一次 simultaneous-bootstrap draw 都留在 GPU。

显式不可用的 GPU device 应失败，而不是静默转成 CPU estimation path。这里描述的 CPU reporting/calibration boundary 是明确的 inference/reporting boundary，不是隐藏 estimator fallback。

## Sample weight 与推断范围

`Lasso.fit(..., sample_weight=...)` 是支持的拟合接口。但加权高维推断涉及比未加权公式更强的建模与归一化问题。本页公式描述的是核心去偏构造，不应被理解为对任意 analytic-weight design 自动成立的定理。如果 weighted coefficient inference 是科学结论的核心，应针对具体应用验证准确的 weighting convention 与目标 estimand，而不是默认未加权渐近理论原样迁移。

## Residual bootstrap 路径

`inference_method="bootstrap"` 会反复重采样 residual 并重新拟合惩罚模型。constructor 控制项为 `n_bootstrap`（默认 `200`）和 `bootstrap_random_state`（默认 `None`）。

当前实现用 bootstrap variability 计算标准误，用符号变化概率构造双侧 p 值，并使用 percentile confidence interval。它比一次去偏拟合昂贵得多，也没有被声明为完整的 selective-inference procedure。

## OLS 风格 post-selection 路径

当前 `cpu_ols` 与 `gpu_ols` 两个拼法都进入同一个 post-selection OLS diagnostic。它使用普通线性模型机制对 selected active set 做重拟合。这个结果适合工程对比，但不能因为模型是稀疏的就把区间描述成有效 selective-inference interval。

尤其需要把**统计方法**和**执行 backend**分开理解：`inference_method` 应描述“计算什么”，而 `device`/backend routing 描述“在哪里计算”。当前带硬件含义的两个 alias 尚未做到这种干净分离。

## 拟合后推断输出

所选推断路径成功后，可以出现：

| 属性 | 含义 |
|---|---|
| `_params` | inference reporting 使用的参数向量；`debiased` 时 feature entries 是去偏估计 |
| `_bse` | 标准误 |
| `_tvalues` | 某些路径沿用的历史/statistic storage；debiased reporting 是 z 语义 |
| `_zvalues` | structured result layer 填充时的 z-style statistic field |
| `_pvalues` | 双侧系数 p 值 |
| `_conf_int` | marginal confidence interval |
| `_conf_int_simultaneous` | 成功完成 max-|Z| calibration 后的 simultaneous interval |
| `_simultaneous_critical_value` | 校准出的共同临界值 |
| `_inference_result` | structured inference result 与 method/backend metadata |

`summary()` 使用当前 fitted model 已存在的 inference result。下划线推断数组在当前 release 中属于既有 reporting surface，但其含义仍然取决于具体 `inference_method`。

## 可复现性与计算成本

需要可复现重采样时，应分别设置 `bootstrap_random_state` 或 `simultaneous_random_state`。提高 `simultaneous_n_bootstrap` 可以减少临界值 Monte Carlo 波动，但会增加 CPU 计算与内存流量。

去偏推断通常显著比原始 Lasso 拟合昂贵，因为它需要求解许多 node-wise sparse regression。backend acceleration 会改变计算成本，但不会改变统计假设。

## 当前实现没有声称什么？

- 普通 post-selection OLS 区间不是一般 selective-inference interval；
- residual bootstrap 不是对模型选择不确定性的普适修正；
- marginal debiased interval 不经过 simultaneous calibration 时不提供联合覆盖；
- max-|Z| simultaneous interval 不能替代以 FDR 为目标的 Benjamini-Hochberg 等程序；
- 数值收敛、GPU 执行以及很小的 KKT residual 不能证明数据满足去偏推断需要的高维统计假设；
- 默认 simultaneous target 是 feature family；请求包含截距前应阅读上面的截距边界。

## 参考文献

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
