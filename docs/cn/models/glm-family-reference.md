# GLM 分布族与完整 API

> 语言：简体中文
> 最后更新：2026-09-03
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
| `summary()` | 返回格式化字符串；使用 `print(model.summary())` 显示 |

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
