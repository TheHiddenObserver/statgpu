# Lasso 推断

> 语言：中文  
> 最后更新：2026-09-09  
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
| `post_selection_ols` | 在惩罚拟合选出的 active set 上，用拟合已解析的 backend 做未惩罚 OLS/WLS 重拟合 | 工程或 post-selection diagnostic | 不是一般意义上的 selective-inference 置信程序 |
| `bootstrap` | 对惩罚模型做 residual-bootstrap 重拟合 | 基于重采样的不确定性 diagnostic | 计算昂贵，而且本身不是对数据驱动模型选择的普适修正 |

`post_selection_ols` 是与硬件无关的 canonical 拼法。旧 unified alias `cpu_ols` 与 `gpu_ols` 已同时 deprecated，在一个兼容周期内仍可使用，但会发出 `FutureWarning`，并统一 normalize 到 `post_selection_ols`。`LassoCV` 在兼容边界上还识别更早的 `cpu_ols_inference` / `gpu_ols_inference`，同样 normalize 到这一统计方法。

统计方法名称**不选择执行设备**。`device="cpu"`、`device="cuda"`、`device="torch"` 都是 authoritative；只有真正的 `device="auto"` 才可以在自动路由时保留 backend-native 的 CuPy 或 Torch-CUDA 输入。

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

拟合截距时，prediction 与 inference 有意采用不同的参数 ownership。公开 `coef_` / `intercept_` 仍属于惩罚预测拟合；推断斜率使用去偏后的向量，而 original-coordinate 的推断截距为

$$
\hat\theta^{\mathrm{db}}_0
= \bar y_w - \bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

它的标准误和 influence representation 因而与同一个 centered debiased parameterization 保持一致，而不是把截距当成另一个 node-wise-Lasso 特征坐标。

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

方法抽取独立标准正态 multiplier $\xi_i$，用拟合残差构造 bootstrap score perturbation，并记录所请求 target family 上最大的标准化绝对扰动。示意地，

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
| `simultaneous_include_intercept` | `False` | 把 centered debiased intercept 同时纳入 bootstrap max-|Z| target 与最终 joint interval family。 |

`enable_simultaneous_inference=True` 要求同时满足 `compute_inference=True`、`inference_method="debiased"` 和 `simultaneous_method="maxz_bootstrap"`。`simultaneous_alpha` 必须严格位于 `(0, 1)`，`simultaneous_n_bootstrap` 必须为正；不支持的组合会失败，而不是静默返回普通边际区间。

### 包含截距的 simultaneous family

当 `simultaneous_include_intercept=True` 时，边际标准误所使用的同一 centered-nodewise original-coordinate intercept influence 会直接进入 bootstrap maximum。也就是说，截距不再只是“额外加一行、套用 feature-only 临界值”，而是真正属于被校准的联合 target family。

默认仍为 `False`，因此只需要 feature vector 联合覆盖的调用不会无故扩大 target family。

## Simultaneous interval 不等于 p 值校正

下面几件事相关但并不相同：

- `_conf_int` 是逐系数 marginal interval；
- `_conf_int_simultaneous` 使用共同 max-|Z| 临界值保护一个 interval family；
- `model.adjust_pvalues(method="bh")` 或模块级 `adjust_pvalues(...)` 是对已有 p-value vector 做 multiple-testing correction；
- `model.combine_pvalues(...)` 检验一组 p 值是否包含全局证据，它不会构造 simultaneous coefficient interval。

通用 estimator-bound 与模块级 multiple-testing API 见[推断 API 参考](../guides/inference-api.md)。

## Backend 行为与 reporting boundary

拟合后统计方法与 backend identity 是两个独立维度。

对于 `post_selection_ols`，推断始终复用成功惩罚拟合记录的 backend 和 concrete device。active-set OLS/WLS refit、covariance、reference-distribution inference 以及置信区间数值计算，会按照该 fit-resolved backend 在 NumPy、CuPy 或 Torch 上完成。显式 GPU 请求在 backend 不可用时 fail closed，不会静默变成 CPU OLS inference。

对于 `debiased`，维护中的 CuPy/Torch 边际路径同样把数值系数推断留在实际执行的 GPU backend。最终 reporting layer 仍以 NumPy 为主：`_bse`、`_pvalues`、`_conf_int` 等数组以及 structured result metadata 只会在数值推断完成之后 snapshot 到 host-side reporting object。

对于 centered `fit_intercept=True` 的 CuPy/Torch simultaneous inference，昂贵的 B×n multiplier draws、feature/intercept scores、max-|Z| reduction、quantile calibration 和 joint CI 数值计算都会留在同一个 concrete GPU device 上，直到最终 reporting snapshot。历史 `fit_intercept=False` simultaneous 路径仍使用已有 generic reporting-stage helper；PR #138 并没有把这条历史路径声明为 GPU-native。

Residual `bootstrap` 不同：当前 residual-refit implementation 是 CPU-native，因此 estimator 的 GPU `device` 不代表 bootstrap resampling/refit 也会变成 GPU-native。

## Sample weight 与推断范围

`Lasso.fit(..., sample_weight=...)` 是支持的拟合接口。维护中的 NumPy/CuPy/Torch `debiased` 路径使用相同的 weighted-centered average-loss working problem，因此把所有 analytic weights 同时乘上一个正数，不会改变惩罚拟合以及维护中的去偏推断结果。

这个实现 contract 本身并不等价于“对任意科学意义下的 analytic weights 都自动得到有效定理”。如果 weighted coefficient inference 是科学结论的核心，仍应针对具体应用验证 weighting convention 与目标 estimand。

## Residual bootstrap 路径

`inference_method="bootstrap"` 会反复重采样 residual 并重新拟合惩罚模型。constructor 控制项为 `n_bootstrap`（默认 `200`）和 `bootstrap_random_state`（默认 `None`）。

当前实现用 bootstrap variability 计算标准误，用符号变化概率构造双侧 p 值，并使用 percentile confidence interval。它比一次去偏拟合昂贵得多，也没有被声明为完整的 selective-inference procedure。

## OLS 风格 post-selection 路径

`inference_method="post_selection_ols"` 会在惩罚拟合选出的**精确 active set** 上重拟合一个未惩罚 OLS/WLS 模型。`coef_` / `intercept_` 仍属于惩罚预测拟合；`_params` / `_inference_result` 则拥有用于推断报告的 active-set refit。

该 refit 使用 fit-resolved NumPy/CuPy/Torch backend。active design 秩亏时使用 effective-rank Moore-Penrose/SVD 计算，而不是普通 normal equations。经典 `cov_type="nonrobust"` 报告沿用 Student-t 约定；robust/HAC covariance 在 estimator 暴露相应选项时使用共享 Gaussian robust-covariance layer 及其 normal-reference 约定。

full-space 中未被选中的坐标仍保留兼容 placeholder（`SE=0`、statistic `0`、`p=1`、区间 `[0, 0]`）。这些值不是“系数被精确知道为 0”的统计主张；应通过 selected-feature metadata 判断哪些坐标真正接受了 active-set refit。

最重要的是：同一数据先选择变量、再构造 ordinary OLS/WLS interval，仍然只是 **post-selection diagnostic**，不是一般 selective-inference confidence procedure。

## 拟合后推断输出

所选推断路径成功后，可以出现：

| 属性 | 含义 |
|---|---|
| `_params` | inference reporting 使用的参数向量；`debiased` 时 feature entries 是去偏估计；`post_selection_ols` 时 active entries 属于未惩罚 refit |
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

需要可复现重采样时，应分别设置 `bootstrap_random_state` 或 `simultaneous_random_state`。提高 `simultaneous_n_bootstrap` 可以减少临界值 Monte Carlo 波动，但会增加实际 numerical backend 上的计算和内存流量。

去偏推断通常显著比原始 Lasso 拟合昂贵，因为它需要求解许多 node-wise sparse regression。backend acceleration 会改变计算成本，但不会改变统计假设。

## 当前实现没有声称什么？

- 普通 post-selection OLS/WLS 区间不是一般 selective-inference interval；
- residual bootstrap 不是对模型选择不确定性的普适修正；
- marginal debiased interval 不经过 simultaneous calibration 时不提供联合覆盖；
- max-|Z| simultaneous interval 不能替代以 FDR 为目标的 Benjamini-Hochberg 等程序；
- 数值收敛、GPU 执行以及很小的 KKT residual 不能证明数据满足去偏推断需要的高维统计假设；
- 历史 `fit_intercept=False` simultaneous 路径不会因为惩罚/去偏拟合使用了 GPU 就自动获得“GPU-native simultaneous”声明。

## 参考文献

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
