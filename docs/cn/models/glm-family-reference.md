# GLM 分布族与完整 API

> 语言：简体中文
> 最后更新：2026-09-04
> 切换：[English](../../en/models/glm-family-reference.md)

本进阶参考页配合 [GLM 入门文档](generalized-linear-model.md)使用。下表统一采用

$$
g(\mu_i)=\beta_0+x_i^\top\beta,
\qquad
\operatorname{Var}(Y_i\mid x_i)=\phi V(\mu_i).
$$

表中的 $V(\mu)$ 不包含离散度乘数 $\phi$。

## 分布族总表

| Family | 实现接受的响应 | 默认链接 | $V(\mu)$ | 类型化估计器 |
|---|---|---|---|---|
| Gaussian | 有限实数 | identity | $1$ | `LinearRegression` 或通用 GLM |
| Binomial | $[0,1]$ 内的有限值 | logit | $\mu(1-\mu)$ | 通用 GLM；独立 `LogisticRegression` 仅接受数组 |
| Poisson | 有限非负值 | log | $\mu$ | `PoissonRegression` |
| Negative binomial | 有限非负值 | log | $\mu+\alpha\mu^2$ | `NegativeBinomialRegression` |
| Gamma | 有限严格正值 | log | $\mu^2$ | `GammaRegression` |
| Inverse Gaussian | 有限严格正值 | log | $\mu^3$ | `InverseGaussianRegression` |
| Tweedie | 有限非负值 | log | $\mu^p$ | `TweedieRegression` |

实现层面对计数型 loss 接受非整数的非负响应；实际分析仍需确认所选 family 符合数据生成机制。

## Gaussian

identity 链接满足 $\mu=\eta$。普通最小二乘及完整 HC/HAC 推断优先使用[`LinearRegression`](linear-regression.md)；`GeneralizedLinearModel(family="gaussian")` 使用统一的迭代 GLM 接口。

## Binomial

logit 链接为

$$
g(\mu)=\log\frac{\mu}{1-\mu},
\qquad
\mu=\frac{1}{1+\exp(-\eta)}.
$$

需要 `predict_proba`、阈值、ROC 和分类指标时使用独立 `LogisticRegression`，但它目前只接受数组。需要 Formula 时使用 `GeneralizedLinearModel(family="binomial")`；此时 `predict` 返回拟合均值概率。

## Poisson

log 链接保证 $\mu>0$，系数指数化后可解释为均值比。条件方差明显超过均值时应比较 negative binomial；稳健协方差不能代替正确的条件分布。

## Negative binomial

`NegativeBinomialRegression(alpha=1.0, ...)` 暴露正的过度离散参数 $\alpha$。当 $\alpha\to0$ 时，方差趋近 Poisson 方差。这里的 `alpha` 是响应分布参数，不是正则化强度。

## Gamma

`GammaRegression(link="log", ...)` 是稳定的默认值，也实现了 `link="inverse_power"`：

$$
g(\mu)=\mu^{-1}.
$$

逆链接需要格外注意，因为经过数值保护后的线性预测子必须保持为正。

## Inverse Gaussian

`InverseGaussianRegression` 使用 log 链接，适合严格为正、明显右偏且方差大致按均值三次方增长的响应。

## Tweedie

`TweedieRegression(power=1.5, ...)` 在当前 loss 中要求 $1<p<2$。该复合 Poisson–Gamma 范围可同时表示零点质量与正连续值。`power` 控制方差结构，不是正则化强度。

## 不同 family、link 与协方差类型

这三个选择负责不同的部分：

- **family** 决定响应取值范围与方差函数 $V(\mu)$；
- **link** 把线性预测子 $\eta$ 映射为 $\mu=g^{-1}(\eta)$，因此会改变求解器和协方差使用的导数；
- `cov_type` 在**同一组已拟合系数**上选择模型型或经验 score 型不确定性。

修改 `cov_type` 不会重新拟合系数；修改 family 或 link 会改变目标函数，通常也会改变系数。

### 估计方程

令 $\tilde x_i$ 在拟合截距时包含截距列，$k$ 为拟合参数总数，并记

$$
\eta_i=\tilde x_i^\top\hat\beta,
\qquad
\mu_i=g^{-1}(\eta_i).
$$

对无权重拟合，statgpu 构造每个观测的 loss-gradient 贡献

$$
s_i(\hat\beta)
=
\frac{\partial \ell_i}{\partial\eta_i}\tilde x_i,
\qquad
\bar J
=
\frac{1}{n}\sum_{i=1}^n s_i s_i^\top.
$$

family 与 link 通过 $V(\mu_i)$ 和 $d\mu_i/d\eta_i$ 一起进入信息矩阵。用标准 GLM 记号，其期望工作权重大致为

$$
w_i
=
\frac{\left(d\mu_i/d\eta_i\right)^2}{V(\mu_i)},
\qquad
\bar H_F
=
\frac{1}{n}\tilde X^\top W\tilde X.
$$

`nonrobust` 会优先使用 loss 对象提供的期望 Fisher 信息；若未实现则回退到 Hessian。稳健协方差使用观测 Hessian $\bar H_O$。若 IRLS 拟合中 `C>0`，bread 会加入对角 Ridge 曲率 $D_P$；需要普通无惩罚似然推断时，应令 `C=0` 或明确选择无惩罚光滑求解器。

### 已实现的协方差公式

下面给出实现采用的无权重约定：

$$
\widehat{\operatorname{Cov}}_{\text{nonrobust}}(\hat\beta)
=
\frac{\hat\phi}{n}
\left(\bar H_F+D_P\right)^{-1},
$$

$$
\widehat{\operatorname{Cov}}_{\text{HC0}}(\hat\beta)
=
\frac{1}{n}
\left(\bar H_O+D_P\right)^{-1}
\bar J
\left(\bar H_O+D_P\right)^{-1},
$$

$$
\widehat{\operatorname{Cov}}_{\text{HC1}}(\hat\beta)
=
\frac{n}{n-k}
\widehat{\operatorname{Cov}}_{\text{HC0}}(\hat\beta).
$$

| `cov_type` | 含义 | 适用情况 |
|---|---|---|
| `nonrobust` | 由 $\hat\phi$ 缩放的模型型期望信息协方差 | family、link、条件方差与独立性假设可信 |
| `hc0` | 经验 score 外积 sandwich | 条件方差可能设定错误且样本不小 |
| `hc1` | HC0 乘以 $n/(n-k)$ | 需要同一稳健目标并增加自由度修正 |

使用解析 `sample_weight` 时，statgpu 在平均尺度归一化中以 $n_{\mathrm{eff}}=\sum_i w_i$ 替代 $n$，在 $\bar J$ 中使用解析权重的平方，但 HC1 修正仍使用原始观测数 $n$。因此整体倍乘所有权重不会改变拟合诊断或 HC1 修正。

`hc0` 与 `hc1` 可处理条件方差设定错误，但不能建模聚类、重复测量或序列相关。`hc2`、`hc3` 和 `hac` 在当前普通 GLM 推断路径上会抛出 `NotImplementedError`。若 Gaussian identity-link 模型需要这些类型，请使用[`LinearRegression` 及其推断 API](linear-regression-inference.md)。

### 支持的 family/link 组合

| Family | 当前接受的 link | $V(\mu)$ | `nonrobust` 的离散参数 $\hat\phi$ | 稳健协方差 |
|---|---|---|---|---|
| Gaussian | identity | $1$ | $\mathrm{RSS}/(n-k)$ | `hc0`、`hc1` |
| Binomial | logit | $\mu(1-\mu)$ | `1.0` | `hc0`、`hc1` |
| Poisson | log | $\mu$ | `1.0` | `hc0`、`hc1` |
| Negative binomial | log | $\mu+\alpha\mu^2$ | `1.0` | `hc0`、`hc1` |
| Gamma | log；`GammaRegression` 还接受 `inverse_power` | $\mu^2$ | Pearson $\chi^2/(n-k)$ | `hc0`、`hc1` |
| Inverse Gaussian | log | $\mu^3$ | Pearson $\chi^2/(n-k)$ | `hc0`、`hc1` |
| Tweedie | log | $\mu^p$ | Pearson $\chi^2/(n-k)$ | `hc0`、`hc1` |

该表记录当前公开估计器暴露的组合，并不是任意 family/link 注册表。尤其是只有 Gamma 类型化普通包装器提供替代 link。`NegativeBinomialRegression.alpha` 与 `TweedieRegression.power` 也会进入 $V(\mu)$，修改它们会同时改变拟合 loss 与协方差。

## 完整 API 参考

### 通用构造器

```python
GeneralizedLinearModel(
    family="gaussian",
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    C=1.0,
    device="auto",
    n_jobs=None,
    solver="auto",
    gpu_memory_cleanup=False,
    compute_inference=False,
    cov_type="nonrobust",
)
```

| 参数 | 约定 |
|---|---|
| `family` | `gaussian`、`binomial`、`poisson`、`gamma`、`inverse_gaussian`、`negative_binomial` 或 `tweedie` |
| `fit_intercept` | 除非 Formula 明确移除截距，否则加入截距 |
| `max_iter`、`tol` | 迭代上限与收敛容差 |
| `C` | IRLS 的逆 Ridge 强度；$C>0$ 时 $\lambda=1/(2C)$，`C=0` 移除 Ridge 项 |
| `device` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA） |
| `n_jobs` | 可选估计器级并行提示 |
| `solver` | `auto`、`irls`、`fista`、`newton` 或 `lbfgs` |
| `gpu_memory_cleanup` | 开启后在公开 GPU 工作完成时释放缓存块 |
| `compute_inference` | 在支持时计算拟合后 M-estimation 推断 |
| `cov_type` | 普通 GLM 推断支持 `nonrobust`、`hc0` 与 `hc1` |

普通 GLM 表面上的 `solver="auto"` 当前选择 IRLS。若目标是无惩罚似然拟合，应在 IRLS 中设置 `C=0`，或明确选择受支持的无惩罚光滑求解器。

算法原理与停止条件见独立的[求解器算法指南](../guides/solver-algorithms.md)；惩罚 GLM 的准确 loss × penalty 调度与拒绝组合见[求解器与惩罚兼容矩阵](../guides/solver-penalty-matrix.md)。

### 分布族专属构造参数

| 估计器 | 额外参数 |
|---|---|
| `GammaRegression` | `link="log"` 或 `"inverse_power"` |
| `NegativeBinomialRegression` | `alpha=1.0`，必须是有限正离散参数 |
| `TweedieRegression` | `power=1.5`，要求 $1<p<2$ |
| 其他普通类型化 GLM | 除通用控制项外没有 family 专属构造参数 |

### `fit`

```python
fit(X=None, y=None, sample_weight=None, formula=None, data=None)
```

可以传 `X` 和 `y`，也可以传 `formula` 和 pandas `data`。`sample_weight` 必须一维、有限、非负且总和为正。Patsy 删除缺失行后，模型才会对齐并验证保留行的权重。

### 输出与方法

| 名称 | 含义 |
|---|---|
| `coef_`、`intercept_` | 线性预测子尺度上的系数 |
| `n_iter_` | 所选求解器提供的迭代次数 |
| `llf` / `loglikelihood` | 拟合伪对数似然；加性常数可能与 R 或 statsmodels 不同 |
| `aic`、`bic` | 基于该伪对数似然的信息准则 |
| `_bse`、`_zvalues`、`_pvalues`、`_conf_int` | 当前兼容推断数组 |
| `_inference_result` | 包含协方差、分布、求解器和后端元数据的结构化结果 |
| `predict(X_new)` | 响应尺度的条件均值；Formula 拟合可接收对齐的 DataFrame |
| `summary()` | 返回包含模型元数据、系数推断、log-likelihood、AIC 与 BIC 的格式化字符串；使用 `print(model.summary())` 显示，并可查看[可运行输出示例](generalized-linear-model.md#最小可运行示例) |

独立 `LogisticRegression` 还有分类专属方法，见其[模型页](logistic-regression.md)。惩罚 GLM 还包含 `penalty`、`alpha`、`l1_ratio`、`penalty_kwargs`、`cpu_solver`、`lipschitz_L`、推断方式、HAC 滞后、停止规则与 LLA 控制项，详见[求解器与惩罚矩阵](../guides/solver-penalty-matrix.md)和[推断 API](../guides/inference-api.md)。

## 文档分层边界

- 入门模型页说明问题、假设、一个可运行流程、结果解释和常见错误。
- 进阶参考页完整列出公开构造/拟合参数、拟合属性、公式、后端差异与失败条件。
- 仅用于协调求解器的私有字段不是稳定用户约定。带下划线的推断数组因已有用户使用而记录，但可复用报告更适合读取结构化推断结果。

这样既不会让初次使用者被参数淹没，也不会向进阶用户隐藏细节。

## 源码地图

- 普通 GLM 调度：`statgpu/linear_model/_glm_base.py`
- 分布族与链接：`statgpu/glm_core/_family.py`
- 类型化包装器：`statgpu/linear_model/wrappers/`
- 响应验证：`statgpu/glm_core/_base.py` 与 `_validation.py`
- M-estimation 推断：`statgpu/inference/_sandwich.py`

跨模型行为见 [Formula 接口](../guides/formula-interface.md)和 [CPU/GPU 加速实现](../guides/acceleration-internals.md)。
