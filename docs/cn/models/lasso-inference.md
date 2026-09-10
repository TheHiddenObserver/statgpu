# Lasso 推断

> 语言：中文  
> 最后更新：2026-09-10  
> 模型指南：[Lasso](lasso.md)  
> 切换：[English](../../en/models/lasso-inference.md)

本页是 learner-first [Lasso 指南](lasso.md) 的统计推断配套页。主模型页回答“什么时候使用 Lasso”，这里说明 statgpu 的拟合后推断对象到底是什么、逐节点近似精度矩阵如何构造，以及软件**没有**承诺什么。

## 为什么原始 Lasso 系数不能直接套普通 Wald 推断？

在中心化 Gaussian 线性模型中，Lasso 求解

$$
\hat\beta
=\arg\min_\beta
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1
\right\}.
$$

L1 惩罚通过收缩制造稀疏性，其 KKT 关系可概括为

$$
\frac{X^\top(y-X\hat\beta)}{n}=\alpha\hat\kappa,
\qquad \hat\kappa_j\in\partial|\hat\beta_j|.
$$

因此未惩罚 loss 的 score 并不会像 OLS 那样等于 0。正则化偏差是不能直接给 raw Lasso coefficient 套 fixed-model Wald 标准误的首要原因。

**选择不确定性**是第二个、不同的问题。如果用同一份数据先由 Lasso 选择活跃集，再在该活跃集上 OLS 重拟合，普通 OLS 区间通常没有处理前面的选择步骤。因此 statgpu 把 `post_selection_ols` 定位为选择后诊断，而不是一般意义上的选择性推断（selective inference）程序。

## 根据统计主张选择推断路径

| `inference_method` | statgpu 计算什么 | 如何解释 | 主要限制 |
|---|---|---|---|
| `debiased` | 纠偏 / de-sparsified 系数、SE、z、p 值与边际置信区间 | 满足纠偏假设时的系数级高维推断 | 依赖稀疏性、设计、噪声和调参条件 |
| `post_selection_ols` | 在 fit-resolved backend 上对 Lasso 活跃集做 OLS/WLS | 工程/统计选择后诊断 | 不是一般选择性推断置信程序 |
| `bootstrap` | penalized model 的残差自助法重拟合 | 重采样不确定性诊断 | 计算重，也不是普适的模型选择修正 |

`post_selection_ols` 与硬件无关。deprecated `cpu_ols` 与 `gpu_ols` 都映射到它；它们不选择 CPU/GPU。真正的执行位置由 `device="cpu"`、`device="cuda"`、`device="torch"` 控制，只有 genuine `device="auto"` 才允许自动选择可用后端。

只做预测或变量选择时，设置 `compute_inference=False`。

## 最小纠偏推断示例

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(700, 10))
beta = np.zeros(10)
beta[[1, 4, 8]] = [1.5, -1.1, 0.7]
y = 0.4 + X @ beta + rng.normal(scale=1.0, size=X.shape[0])

model = Lasso(
    alpha=0.05,
    nodewise_alpha=None,
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model.nodewise_alpha_)
print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

`coef_` 保持为 penalized prediction coefficient vector；`_params` 是纠偏后的推断/reporting 参数，因此两者不要求数值相同。

## 纠偏一步更新做了什么？

令拟合残差为 $r=y-b-X\hat\beta$。statgpu 构造

$$
\hat\theta^{\mathrm{db}}
=\hat\beta+\frac{1}{n}MX^\top r,
$$

其中 $M$ 是设计 Gram matrix 的数据依赖近似逆/近似精度矩阵。若 $\widehat\Sigma=X^\top X/n$，则

$$
\hat\theta^{\mathrm{db}}-\beta^0
=
\frac{1}{n}MX^\top\varepsilon
+
(I-M\widehat\Sigma)(\hat\beta-\beta^0).
$$

第一项是主要噪声项，第二项是 remainder。只有当 $M\widehat\Sigma$ 足够接近单位矩阵、并且相应高维假设使 remainder 在推断尺度上足够小时，纠偏后的近似 Gaussian 推断才有相应理论依据。

## 规范的中心化/加权工作设计

维护中的 sparse-Gaussian 路径首先生成同一个规范中心化/加权 average-loss 工作设计 $X_w$。使用分析权重 `sample_weight` 时，NumPy、CuPy 与 Torch 采用相同的加权中心化和 row-rescaling 约定。所有权重同时乘以同一个正数不会改变预期统计问题，全 1 权重与 unweighted 定义一致。

逐节点调参只依赖这个设计侧问题，不使用响应残差尺度。

## 标准化与逐节点 Lasso（node-wise Lasso）

定义

$$
d_j^2=\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad D=\operatorname{diag}(d_1,\ldots,d_p),
\qquad Z=X_wD^{-1}.
$$

对每个特征 $j$，statgpu 在标准化设计上求解

$$
\hat\gamma_j
=\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{
\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\}.
$$

理论上逐节点惩罚常见的量级为

$$
\lambda_j\asymp\sqrt{\frac{\log p}{n}},
$$

具体常数取决于理论和尺度约定。statgpu 的自动规则是库层面的具体默认值，而不是声称某个定理唯一规定了这个常数。

## 公开 `nodewise_alpha` 契约

主 Lasso `alpha` 与 inference-only `nodewise_alpha` 是两个不同控制量：

- `alpha` 控制 penalized prediction/selection fit；
- `nodewise_alpha` 只控制 `debiased` inference 使用的逐节点精度矩阵回归；
- 显式有限正实数具有最高优先级；
- `None` 使用库默认值；
- bool、复数/非标量、NaN/inf、0 和负数都会被拒绝；
- `get_params`、`set_params` 与 sklearn clone 保留用户请求的 constructor value；改变它会清除陈旧 inference state。

$p\ge2$ 时自动值为

$$
\boxed{
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}
}
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重时使用 Kish 型有效样本量

$$
n_{\mathrm{nw}}
=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

该规则有意做到**与响应变量尺度无关**。被取代的历史内部实现曾使用类似 $\hat\sigma_y\sqrt{2\log(p)/n}$ 的 response-residual-scaled 形式；它已经不是当前契约，也没有作为 legacy public mode 保留。参见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

`p=1` 时没有 nuisance node-wise regression。statgpu 直接使用解析一维精度矩阵，不消费用户请求的 `nodewise_alpha`，因此 `nodewise_alpha_` 保持为 `None`。

## paper-style normalizer 与精度矩阵回变换

逐节点标准化残差为

$$
r_j=Z_j-Z_{-j}\hat\gamma_j.
$$

statgpu 使用

$$
\hat\tau_j^2
=\frac{\lVert r_j\rVert_2^2}{n}
+\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1.
$$

标准化精度矩阵第 $j$ 行的对角元素为 $1/\hat\tau_j^2$，非对角部分为 $-\hat\gamma_j/\hat\tau_j^2$。所有行完成后，通过

$$
M=D^{-1}\Theta_ZD^{-1}
$$

变换回工作特征尺度，再进入纠偏和方差计算。

## 数值 publication gate

内部 solver 的 stopping condition 不足以直接发布推断。维护中的契约使用 node-wise FISTA、`coef_delta` stopping、`1e-8` 内部容差和最多 3000 次迭代，随后**独立重新计算完整 KKT residual**。只有 residual 不超过 `1e-5`，且尺度、normalizer、precision state 和 reporting arrays 都有限且非退化时，结果才可发布。

这些是内部数值设置，不是新的公共调参。通过 KKT 只说明数值解符合声明的逐节点优化问题，不代表高维统计假设自动成立。

## 后端契约

NumPy、CuPy 与 Torch 求解同一个标准化统计问题。CuPy/Torch 路径会批量构造 Gram subproblem 来提高执行效率，但不会改变 $\lambda_{\mathrm{nw}}$、$\hat\gamma_j$、$\hat\tau_j^2$ 或 $M$ 的统计定义。

显式 CUDA/Torch inference 下，逐节点求解、KKT 检查、back-transform 与维护中的 simultaneous inference 保持在实际选中的 GPU backend/device 上。precision cache 为了**缓存身份哈希**可以把后端驻留设计的分块内容传到 host，但这不是 CPU numerical fallback。只有数值推断完成后，小型 reporting arrays 才转成 NumPy。`_inference_result.metadata` 会记录 `numerical_backend`、`numerical_device`、逐节点调参来源、KKT 与 cache provenance。

## `LassoCV`：只作用于最终重拟合推断

`LassoCV(nodewise_alpha=...)` 把该参数当成最终全数据重拟合的 inference config。改变 `nodewise_alpha` 不应改变主 alpha grid、fold MSE、最终 `alpha_` 或 penalized final-refit coefficient。成功的多特征 debiased inference 后，外层 `nodewise_alpha_` 与 final estimator 对齐。

CV 后推断仍条件于已选择的主调参值；当前实现不会额外修正 CV tuning uncertainty。

## 标准误、z 统计量与边际置信区间

令

$$
V=M\widehat\Sigma M^\top.
$$

statgpu 根据拟合残差尺度与 $V_{jj}/n$ 构造 standard error，并使用标准正态参考分布得到维护中的 debiased marginal z、p-value 与置信区间。

`_conf_int` 是**边际置信区间**。同时报告许多 95% marginal intervals，不等于整个目标参数集合具有 95% 同时覆盖。

有截距时，prediction 与 inference 有不同 ownership。公开 `coef_` / `intercept_` 仍属于 penalized prediction fit；纠偏 reporting 使用 corrected slopes 及 coherent original-coordinate intercept

$$
\hat\theta_0^{\mathrm{db}}
=\bar y_w-\bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

## 同时推断

设置 `enable_simultaneous_inference=True` 可以请求 max-|Z| Gaussian multiplier bootstrap calibration。程序生成乘子、计算目标集合上的标准化 score perturbation、取最大绝对值，并用经验分位数作为共同 critical value。

```python
model = Lasso(
    alpha=0.05,
    nodewise_alpha=None,
    inference_method="debiased",
    compute_inference=True,
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=2000,
    simultaneous_random_state=123,
    simultaneous_include_intercept=True,
).fit(X, y)

print(model._conf_int)
print(model._conf_int_simultaneous)
print(model._simultaneous_critical_value)
```

`simultaneous_include_intercept=True` 会让 coherent debiased intercept 真正进入 max-|Z| calibration family，而不只是额外添加一行输出。维护中的 centered CuPy/Torch simultaneous inference 会记录实际 numerical backend/device provenance。

## 多重检验是另一层问题

同时置信区间和 multiple-testing adjustment 相关但不相同。如果已经有有效的 marginal p-values，并希望控制 FDR/FWER，可以使用 estimator-context `adjust_pvalues` 或 [推断 API](../guides/inference-api.md) 中的函数。多重检验校正无法修复本身无效的 p-value，也不会消除 debiased inference 的统计假设。

## 主要输出字段

- `coef_`, `intercept_`：penalized prediction fit；
- `nodewise_alpha_`：成功多特征 debiased inference 后的逐节点调参解析值；
- `_params`：纠偏 reporting 参数；
- `_bse`, `_zvalues`, `_pvalues`, `_conf_int`：边际推断数组；
- `_conf_int_simultaneous`：请求 simultaneous inference 后的同时置信区间；
- `_simultaneous_critical_value`：共同 max-|Z| critical value；
- `_inference_result`：包含 method、precision、backend/device、node-wise、KKT 和 cache provenance 的结构化结果。

## 解释检查表

- 先区分目标是预测/选择、系数级纠偏推断还是选择后诊断。
- 不要混淆主 `alpha` 与 `nodewise_alpha`。
- 自动逐节点规则是设计侧的，并且 response-scale independent。
- `post_selection_ols` 应解释为选择后诊断。
- CV final-refit inference 条件于已选调参值。
- 同时推断或 multiple-testing correction 都不能替代底层 inferential validity。
- backend 加速不会改变统计假设。

## 参考文献

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional linear models. *JASA*, 112(518), 757–768.
