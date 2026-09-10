# Lasso 推断

> 语言：中文  
> 最后更新：2026-09-10  
> 模型指南：[Lasso](lasso.md)  
> 切换：[English](../../en/models/lasso-inference.md)

> **版本提示：** 当前正式发布版为 **0.2.5**。本页所述公开参数 `nodewise_alpha` 及新的自动逐节点调参规则已经进入当前 `master`，计划随 **0.2.6** 发布；0.2.5 尚不包含这一公开接口和新默认规则。

本页是 [Lasso 指南](lasso.md) 的统计推断配套页。主模型页侧重“什么时候用 Lasso、如何拟合”，这里进一步说明：为什么原始 Lasso 系数不能直接套普通 OLS 推断，纠偏推断如何构造近似精度矩阵，以及这些结果在什么条件下才可以解释。

## 为什么原始 Lasso 系数不能直接套普通 Wald 推断？

在中心化的高斯线性模型中，Lasso 求解

$$
\hat\beta
=\arg\min_\beta
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1
\right\}.
$$

L1 惩罚通过收缩产生稀疏性，其 KKT 条件可以写成

$$
\frac{X^\top(y-X\hat\beta)}{n}=\alpha\hat\kappa,
\qquad \hat\kappa_j\in\partial|\hat\beta_j|.
$$

因此，无惩罚损失的得分并不会像 OLS 那样等于 0。**正则化偏差**是不能直接给原始 Lasso 系数套用“模型预先给定”时 Wald 标准误的首要原因。

**选择不确定性**是另一个不同的问题。如果用同一份数据先由 Lasso 选择活跃集，再在这个活跃集上做 OLS 重拟合，普通 OLS 区间通常没有计入前面的变量选择步骤。因此 statgpu 把 `post_selection_ols` 定位为选择后诊断，而不是一般意义上的选择性推断（selective inference）程序。

## 根据统计目标选择推断方法

| `inference_method` | statgpu 计算什么 | 如何解释 | 主要限制 |
|---|---|---|---|
| `debiased` | 纠偏（debiased / de-sparsified）系数、标准误、z 统计量、p 值和边际置信区间 | 满足相应高维条件时的系数级推断 | 依赖稀疏性、设计、噪声和调参条件 |
| `post_selection_ols` | 在本次拟合实际采用的后端上，对 Lasso 选出的活跃集做 OLS/WLS 重拟合 | 选择后的工程/统计诊断 | 不是一般的选择性推断置信程序 |
| `bootstrap` | 对惩罚模型进行残差自助法重拟合 | 重采样不确定性诊断 | 计算开销较大，也不是对模型选择不确定性的普适修正 |

`post_selection_ols` 的统计含义与硬件无关。历史别名 `cpu_ols` 和 `gpu_ols` 都会映射到同一个方法，它们并不决定计算设备。真正的执行位置由 `device="cpu"`、`device="cuda"`、`device="torch"` 控制；只有 `device="auto"` 时才允许自动路由到可用后端。

如果只关心预测或变量选择，可以设置 `compute_inference=False`，避免额外的推断计算。

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

`coef_` 仍然是用于预测的惩罚估计系数；`_params` 是纠偏后的推断参数，因此两者本来就不要求数值相同。

## 一步纠偏做了什么？

令拟合残差为 $r=y-b-X\hat\beta$。statgpu 构造

$$
\hat\theta^{\mathrm{db}}
=\hat\beta+\frac{1}{n}MX^\top r,
$$

其中 $M$ 是设计 Gram 矩阵的近似逆，也可理解为近似精度矩阵。若记 $\widehat\Sigma=X^\top X/n$，则

$$
\hat\theta^{\mathrm{db}}-\beta^0
=
\frac{1}{n}MX^\top\varepsilon
+
(I-M\widehat\Sigma)(\hat\beta-\beta^0).
$$

第一项是主要的随机噪声项，第二项是余项。只有当 $M\widehat\Sigma$ 足够接近单位矩阵，并且稀疏性、设计和噪声条件使这个余项在推断尺度上足够小时，纠偏后的近似正态推断才有理论依据。

## 中心化和加权后的工作设计

维护中的稀疏高斯线性模型推断路径首先构造统一的工作设计 $X_w$。如果给出分析权重 `sample_weight`，NumPy、CuPy 和 Torch 使用相同的加权中心化与按行缩放规则，并保持平均损失的尺度约定。

把所有权重同时乘以同一个正数，不会改变预期的统计问题；全 1 权重与不加权情形一致。

逐节点调参只由这个**设计矩阵侧的问题**决定，不再使用响应变量残差的尺度。

## 标准化与逐节点 Lasso（node-wise Lasso）

定义

$$
d_j^2=\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad D=\operatorname{diag}(d_1,\ldots,d_p),
\qquad Z=X_wD^{-1}.
$$

对每个特征 $j$，statgpu 在标准化设计 `Z` 上求解逐节点 Lasso：

$$
\hat\gamma_j
=\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{
\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\}.
$$

理论上，逐节点惩罚常见的量级是

$$
\lambda_j\asymp\sqrt{\frac{\log p}{n}},
$$

但具体常数取决于理论条件和尺度约定。statgpu 的自动公式是一个明确的库默认值，不应理解为某篇论文中的定理唯一指定了这个常数。

## `nodewise_alpha` 的公开参数约定

主模型的 `alpha` 与 `nodewise_alpha` 是两个不同参数：

- `alpha` 控制用于预测和变量选择的惩罚拟合；
- `nodewise_alpha` 只控制 `debiased` 纠偏推断中估计近似精度矩阵的逐节点 Lasso；
- 显式给出的有限正实数直接生效；
- `None` 使用 statgpu 的自动规则；
- `bool`、复数、非标量、NaN、无穷值、0 和负数都会被拒绝；
- `get_params`、`set_params` 和 sklearn clone 会保留用户指定的构造参数值；修改 `nodewise_alpha` 会使旧的推断结果失效。

当 $p\ge2$ 时，自动值为

$$
\boxed{
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}
}
$$

无分析权重时 $n_{\mathrm{nw}}=n$；存在非均匀分析权重时，使用 Kish 型有效样本量

$$
n_{\mathrm{nw}}
=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

**版本变化：** 这套与响应变量尺度无关的自动规则计划自 **0.2.6** 起成为公开行为。0.2.5 及更早的内部实现曾使用类似 $\hat\sigma_y\sqrt{2\log(p)/n}$ 的响应残差尺度修正；旧规则不会作为公开兼容模式继续保留。详见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

当 `p=1` 时没有需要拟合的辅助逐节点回归，statgpu 直接使用一维解析精度矩阵，因此 `nodewise_alpha_` 保持为 `None`。

## 归一化与精度矩阵的尺度变换

令第 $j$ 个标准化逐节点残差为

$$
r_j=Z_j-Z_{-j}\hat\gamma_j.
$$

statgpu 使用与经典逐节点 Lasso 文献记号一致的归一化量

$$
\hat\tau_j^2
=\frac{\lVert r_j\rVert_2^2}{n}
+\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1.
$$

标准化尺度上的近似精度矩阵第 $j$ 行，对角元素为 $1/\hat\tau_j^2$，其余元素由 $-\hat\gamma_j/\hat\tau_j^2$ 给出。所有行构造完成后，再通过

$$
M=D^{-1}\Theta_ZD^{-1}
$$

变换回原工作特征尺度，并进入后续纠偏与方差计算。

## 结果采用前的 KKT 数值检查

逐节点求解器自己的停止条件并不足以决定推断结果是否可以采用。当前实现使用 FISTA，按 `coef_delta` 停止，内部容差为 `1e-8`、最多迭代 3000 次；求解结束后还会**独立重新计算一次完整的 KKT 残差**。

只有最大 KKT 残差不超过 `1e-5`，并且特征尺度、归一化量、精度矩阵和最终推断数组都为有限且非退化值时，statgpu 才继续写入推断结果。若检查失败，会直接中止推断并报错，而不是返回可能误导的占位结果。

这些都是内部数值设置，不是新增的公开调参参数。通过 KKT 检查只说明逐节点优化问题在数值上求解充分，并不意味着高维推断所需的统计假设自动成立。

## 后端计算约定

NumPy、CuPy 与 Torch 求解的是同一个标准化统计问题。CuPy/Torch 可以批量构造 Gram 子问题来减少开销，但不会改变 $\lambda_{\mathrm{nw}}$、$\hat\gamma_j$、$\hat\tau_j^2$ 或 $M$ 的统计定义。

显式选择 CUDA 或 Torch 后端时，逐节点求解、KKT 检查、精度矩阵尺度变换以及维护中的同时推断都在实际选中的 GPU 后端和设备上完成，不会把这些数值计算静默转到 CPU。

精度矩阵缓存为了判断“当前设计是否与已缓存问题相同”，可以把后端上的工作设计分块传到主机内存计算哈希；这只是缓存身份识别，不是 CPU 数值求解。数值推断完成后，少量用于展示和汇总的结果数组才转换为 NumPy。`_inference_result.metadata` 会记录 `numerical_backend`、`numerical_device`、逐节点调参来源、KKT 检查和缓存溯源信息。

## `LassoCV`：只影响最终重拟合后的推断

`LassoCV(nodewise_alpha=...)` 把该参数作为最终全数据重拟合后的推断配置。改变 `nodewise_alpha` 不应改变主模型候选 `alpha` 网格、各折均方误差、最终 `alpha_`，也不应改变用于预测的最终惩罚系数。多特征纠偏推断成功后，外层的 `nodewise_alpha_` 与最终重拟合模型保持一致。

交叉验证之后的推断仍然是在“主调参值已经选定”的条件下进行；当前实现不会额外修正交叉验证调参带来的不确定性。

## 标准误、z 统计量与边际置信区间

令

$$
V=M\widehat\Sigma M^\top.
$$

statgpu 根据拟合残差尺度和 $V_{jj}/n$ 构造标准误，并使用标准正态参考分布得到纠偏后的 z 统计量、p 值和边际置信区间。

`_conf_int` 保存的是**边际置信区间**。对很多参数分别报告 95% 边际区间，并不意味着整个参数集合同时具有 95% 的覆盖概率。

如果模型包含截距，用于预测的参数与用于纠偏推断的参数有不同归属。公开的 `coef_` / `intercept_` 仍是惩罚拟合结果；纠偏推断使用纠偏后的斜率，以及与中心化参数化一致的原坐标截距

$$
\hat\theta_0^{\mathrm{db}}
=\bar y_w-\bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

## 同时推断

设置 `enable_simultaneous_inference=True` 可以启用 **max-|Z| 高斯乘子自助法**校准。程序生成高斯乘子，计算目标参数集合上的标准化得分扰动，取其中最大绝对值，再用经验分位数作为共同临界值。

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

当 `simultaneous_include_intercept=True` 时，纠偏后的截距会真正进入 max-|Z| 的校准目标集合，而不只是额外显示一行结果。CuPy/Torch 上的同时推断也会记录实际执行数值计算的后端和设备。

## 多重检验是另一层问题

同时置信区间与多重检验校正相关，但并不是同一件事。如果已经得到有效的边际 p 值，并希望控制 FDR 或 FWER，可以使用估计器上的 `adjust_pvalues`，或 [推断 API](../guides/inference-api.md) 中的相应函数。

多重检验校正无法把本身无效的 p 值“修复”为有效 p 值，也不能替代纠偏推断本身所需的稀疏性、设计和噪声条件。

## 主要输出字段

- `coef_`, `intercept_`：用于预测的惩罚拟合结果；
- `nodewise_alpha_`：多特征纠偏推断成功后实际采用的逐节点调参值；
- `_params`：纠偏后的推断参数；
- `_bse`, `_zvalues`, `_pvalues`, `_conf_int`：标准误、z 统计量、p 值和边际置信区间；
- `_conf_int_simultaneous`：启用同时推断后得到的同时置信区间；
- `_simultaneous_critical_value`：max-|Z| 校准得到的共同临界值；
- `_inference_result`：包含推断方法、近似精度矩阵、后端/设备、逐节点调参、KKT 检查和缓存溯源信息的结构化结果。

## 解读要点

- 先区分你的目标是预测/变量选择、系数级纠偏推断，还是选择后诊断。
- 不要混淆主模型 `alpha` 与 `nodewise_alpha`。
- 计划自 0.2.6 起使用的自动逐节点规则只由设计侧问题决定，与响应变量的计量尺度无关。
- `post_selection_ols` 应解释为选择后诊断，而不是一般选择性推断。
- `LassoCV` 最终重拟合后的推断条件于已经选定的主调参值。
- 同时推断和多重检验校正都不能替代底层统计假设。
- NumPy、CuPy、Torch 改变计算位置和性能，不改变统计有效性所需的条件。

## 参考文献

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional linear models. *JASA*, 112(518), 757–768.
