# Lasso 推断

> 语言：中文  
> 最后更新：2026-09-09  
> 模型指南：[Lasso](lasso.md)  
> 切换：[English](../../en/models/lasso-inference.md)

本页是面向学习者的 [Lasso 指南](lasso.md)的统计推断配套页。它说明 statgpu 的不同拟合后推断方法究竟计算什么、应当如何解释所报告的区间，以及当前实现在哪些地方有意不作更强的统计保证。

## 为什么不能直接对普通 Lasso 系数套用经典推断？

Lasso 首先是一个**带正则化的估计量**。以中心化后的高斯线性模型（Gaussian linear model）为例，它求解

$$
\hat\beta
= \arg\min_\beta
\left\{
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
+ \alpha\lVert\beta\rVert_1
\right\}.
$$

L1 惩罚正是产生稀疏性的来源，但它也会把拟合系数向 0 收缩。在最优解处，KKT 条件可以示意写成

$$
\frac{X^\top(y-X\hat\beta)}{n}
= \alpha\hat\kappa,
\qquad
\hat\kappa_j\in\partial|\hat\beta_j|,
$$

因此，无惩罚损失部分的得分（score）并不像未惩罚 OLS 那样等于 0。由惩罚引入的**收缩偏差（shrinkage bias）**，才是不能直接拿原始 Lasso 系数配上 OLS 风格标准误、并期待得到以 0 为中心的近似正态统计量的核心原因。在高维情形中，正则化尺度通常不会小到使这一偏差在 $n^{-1/2}$ 的推断尺度上自动消失。此外，Lasso 这类稀疏估计量具有非光滑性，其极限分布依赖未知参数，并不具有固定模型 OLS 那种简单、统一的渐近分布（van de Geer et al., 2014；Javanmard & Montanari, 2014）。

**选择不确定性是第二个、不同的问题。** 当我们先用 Lasso 选择活跃集（active set），再用同一批数据在该活跃集上重拟合 OLS 时，这个问题尤其突出：普通 OLS 置信区间一般没有把“模型本身由数据选择出来”这一层不确定性计入。因此 statgpu 把 `post_selection_ols` 定位为一种选择后诊断方法，而不是一般的选择性推断（selective inference）程序。

所以，对 `debiased` 路径来说，主要动机是**消除主要的正则化偏差，并重新构造一个近似正态的逐系数推断估计量**；而“选择不确定性”的警告主要解释为什么活跃集 OLS 路径只能作较弱的选择后推断主张。两者都属于高维推断的难点，但并不是同一个问题。

因此，statgpu 的多个 `inference_method` 并不是“打印同一组 p 值的不同实现”，而是统计目标和解释范围不同的推断方法。

## 根据你真正需要的统计主张选择路径

| `inference_method` | statgpu 计算什么 | 合适的解释 | 主要限制 |
|---|---|---|---|
| `debiased` | 纠偏系数、标准误、z 统计量、p 值和边际置信区间 | 在纠偏 Lasso 的理论条件下进行逐系数高维推断 | 有效性依赖稀疏性、设计矩阵、噪声以及正则化/纠偏构造 |
| `post_selection_ols` | 在惩罚拟合选出的活跃集上，使用拟合阶段确定的后端进行未惩罚 OLS/WLS 重拟合 | 选择后诊断 | 不是一般意义上的选择性推断置信程序 |
| `bootstrap` | 对惩罚模型进行残差自助法重拟合 | 基于重采样的不确定性诊断 | 计算昂贵，而且本身不是对数据驱动模型选择的普适修正 |

这里把 debiased/de-sparsified Lasso 统一称为**纠偏 Lasso**；中文文献中也常见“去偏 Lasso”的译法。本文后续以“纠偏”为主，以便与国内高维统计推断文献的用语保持一致。

`post_selection_ols` 是与硬件无关的规范名称。统一接口中的旧别名 `cpu_ols` 与 `gpu_ols` 已同时进入弃用期；在一个兼容周期内仍可使用，但会发出 `FutureWarning`，并统一映射到 `post_selection_ols`。`LassoCV` 在兼容接口层还识别更早的 `cpu_ols_inference` / `gpu_ols_inference` 写法，同样映射到这一统计方法。

统计方法名称**不负责选择执行设备**。显式设置 `device="cpu"`、`device="cuda"` 或 `device="torch"` 时，该设置具有决定作用；只有真正的 `device="auto"` 才可能在自动路由时保留原生的 CuPy 或 Torch-CUDA 输入后端。

如果目标是 Lasso 之后较正式的逐系数推断，`debiased` 是 statgpu 的主要路径。如果只关心预测或特征选择，可以设置 `compute_inference=False`，避免承担不需要的推断计算成本。

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
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

用于预测的惩罚系数仍保存在 `coef_`。推断报告使用纠偏后的参数向量，因此惩罚系数与对应的纠偏推断估计不必在数值上相同。

如果使用 `LassoCV` 选择 `alpha`，statgpu 会先完成交叉验证，再只在选定 `alpha` 的最终全数据重拟合上计算推断。因此这些区间和 p 值**条件于已经选出的调参值**；当前实现不会再额外校正“使用同一份数据选择 `alpha`”带来的调参不确定性。

## 纠偏到底改变了什么？

记拟合得到的 Lasso 系数为 $\hat\beta$，残差为

$$
r = y - b - X\hat\beta.
$$

L1 惩罚会产生收缩偏差。statgpu 使用

$$
\hat\theta^{\mathrm{db}}
= \hat\beta + \frac{1}{n} M X^\top r
$$

校正系数，其中 $M$ 是由数据构造的**近似精度矩阵**，用于近似特征 Gram 矩阵/协方差矩阵的逆，并在纠偏公式中起到去相关校正作用。这正是 Zhang & Zhang (2014)、van de Geer et al. (2014) 与 Javanmard & Montanari (2014) 等工作中纠偏 Lasso（debiased/de-sparsified Lasso）的核心构造。

为什么这一校正能够消除主要偏差，可以从误差分解直接看出来。写 $y=X\beta^0+\varepsilon$，并令 $\widehat\Sigma=X^\top X/n$。暂时省略截距记号，则

$$
\hat\theta^{\mathrm{db}}-\beta^0
=
\frac{1}{n}MX^\top\varepsilon
+
\left(I-M\widehat\Sigma\right)
\left(\hat\beta-\beta^0\right).
$$

第一项是由噪声驱动、可近似正态的主导项；第二项是由于 $M\widehat\Sigma$ 不能精确等于单位阵而留下的余项。如果 $M\widehat\Sigma$ 足够接近单位阵，并且稀疏性、设计矩阵等条件使这一余项足够小，那么 Lasso 的主要收缩偏差就会在推断尺度上被抵消。这也是为什么纠偏估计量能够获得渐近正态的逐系数推断，而原始 Lasso 估计量一般不能直接这样处理。

这并不意味着原来的稀疏 Lasso 估计“变成了 OLS”。`coef_` 仍然保存用于预测的惩罚系数；$\hat\theta^{\mathrm{db}}$ 则是用于统计推断的纠偏估计量。

## statgpu 如何构造近似精度矩阵？

这里采用的是纠偏 Lasso 中通过**逐节点 Lasso（node-wise Lasso；文献中也常写作 nodewise LASSO）**构造近似精度矩阵的思路；可参见 van de Geer et al. (2014)，以及 Zhang & Zhang (2014) 的相关低维投影（low-dimensional projection）/偏差校正构造。

对每个特征 $j$，statgpu 用逐节点 Lasso 把 $x_j$ 对其余列 $X_{-j}$ 做稀疏回归：

$$
\hat\gamma_j
=
\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{
\frac{1}{2n}
\left\lVert x_j-X_{-j}\gamma\right\rVert_2^2
+
\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\}.
$$

在相关理论中，逐节点惩罚本身是需要选择的调参量，典型量级可写成

$$
\lambda_j \asymp \sqrt{\frac{\log p}{n}},
$$

其中常数以及是否还带尺度因子取决于目标函数和标准化约定。原始纠偏/去稀疏化 Lasso 理论**并没有规定必须采用下面这一条 statgpu 公式**。statgpu 当前的具体实现规则是

$$
\lambda_{\mathrm{nw}}
= \hat\sigma\sqrt{\frac{2\log(\max(p,2))}{n}},
$$

其中 $\hat\sigma$ 来自当前拟合路径的噪声尺度估计。因此这里应把它理解为：**statgpu 在理论要求的高维量级基础上采用的一条实现级调参规则**，而不是 van de Geer et al. 或 Zhang & Zhang 原文中的唯一标准公式。

把 $\hat\gamma_j$ 嵌回 $p$ 维空间：定义 $\tilde\gamma_j\in\mathbb R^p$，令 $(\tilde\gamma_j)_j=0$，其余位置放入逐节点 Lasso 系数。再定义

$$
\hat a_j=e_j-\tilde\gamma_j,
\qquad
z_j=X\hat a_j=x_j-X_{-j}\hat\gamma_j,
$$

并使用逐行归一化常数

$$
C_j
=\frac{x_j^\top z_j}{n}.
$$

于是 $M$ 的第 $j$ 行可以紧凑地写成

$$
\boxed{
\hat m_j^\top
=
\frac{\hat a_j^\top}{C_j}
=
\frac{(e_j-\tilde\gamma_j)^\top}{x_j^\top z_j/n}
}
$$

从而

$$
M
=
\begin{bmatrix}
\hat m_1^\top\\
\vdots\\
\hat m_p^\top
\end{bmatrix}.
$$

这与“第 $j$ 个位置放 $1/C_j$、其他位置放 $-\hat\gamma_{j,k}/C_j$”完全是同一个实现规则，但矩阵表达更能直接看出统计含义：每一行都是一个经过归一化的**残差化方向**，目标是让 $M\widehat\Sigma$ 尽可能接近单位阵。van de Geer et al. (2014) 的常见记号会把这一对象写成由逐节点 Lasso 得到、按行归一化的近似逆矩阵 $\widehat\Theta$；国内综述通常称其为近似精度矩阵。不同论文对归一化常数的记号会随目标函数尺度略有不同。

### CPU 与 GPU 的统计构造相同

上面的公式**不是 CPU 专属算法**。当前维护的 NumPy、CuPy 和 Torch 纠偏路径使用相同的 statgpu 逐节点惩罚规则、相同的 $\hat\gamma_j$、$C_j$ 与 $M$ 定义，以及相同的后续纠偏与方差公式。

差别主要在计算实现：

- CPU 路径逐个特征求解逐节点 Lasso；
- CuPy 与 Torch 路径在具体 GPU 设备上构造相应的 Gram 子问题，并把多个逐节点 FISTA 求解批量执行；
- 批处理改变的是执行方式与内存流量，不改变近似精度矩阵的统计定义。

逐节点优化的数值收敛是必要条件，但不能替代纠偏推断所要求的稀疏性、设计矩阵、噪声和调参量级条件。

## 标准误、z 统计量与边际置信区间

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

双侧 p 值使用标准正态参考分布。当前 `_conf_int` 保存的是基于正态临界值的 95% **边际置信区间**。

需要区分：

- `_conf_int[j]` 是单个参数的边际置信区间；
- 一组 95% 边际置信区间并不会自动对整个系数向量提供 95% 的同时覆盖；
- `simultaneous_alpha` 不改变这些边际置信区间，它只控制下面单独的同时推断程序。

拟合截距时，statgpu 会把**用于预测的参数**与**用于推断的参数**分开保存。公开 `coef_` / `intercept_` 仍属于惩罚预测拟合；推断斜率使用纠偏后的向量，而原始坐标系中的推断截距为

$$
\hat\theta^{\mathrm{db}}_0
= \bar y_w - \bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

它的标准误和影响函数表示因而与同一个中心化纠偏参数化保持一致，而不是把截距当成另一个逐节点 Lasso 特征坐标。

## 从边际置信区间到同时覆盖

95% 的边际置信区间控制的是**单个系数**的覆盖误差。如果一次报告很多坐标，那么“至少有一个区间没有覆盖真值”的概率可能远高于 5%。例如在独立的理想情形下，$m$ 个各自 95% 的区间，其同时覆盖率是 $(0.95)^m$，而不是 0.95。Bonferroni 校正可以简单地恢复族错误率控制，但通常偏保守，因为它没有利用纠偏系数统计量之间的相关结构。

高维同时推断的思路则是直接校准所关心参数集合上**最大标准化误差**的分布。Zhang & Cheng (2017) 以及 Dezeure、Bühlmann & Zhang (2017) 都研究了基于纠偏 Lasso 的自助法同时推断；这种最大统计量构造能够把不同系数之间的依赖关系纳入共同临界值，而不是逐个坐标分别校准。

### max-|Z| 高斯乘子自助法校准

statgpu 可以用**高斯乘子自助法（Gaussian multiplier bootstrap）**校准一个共同临界值：

```python
model = Lasso(
    alpha=0.05,
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

该方法抽取独立标准正态乘子 $\xi_i$，利用拟合残差构造自助得分扰动，并记录所请求参数集合上最大的标准化绝对扰动。示意地，

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

因为这个临界值是针对整个目标参数集合中的最大值来校准的，所以这些区间的目标是对整个集合提供同时覆盖，通常会比对应的边际置信区间更宽。和边际纠偏推断一样，理论保证仍然依赖相应的高维统计假设；自助法并不会让这些假设自动消失。

### 同时推断控制参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `enable_simultaneous_inference` | `False` | 成功完成纠偏推断后，再执行同时覆盖校准。 |
| `simultaneous_method` | `"maxz_bootstrap"` | 当前支持的同时推断校准方法。 |
| `simultaneous_alpha` | `0.05` | 用于选择共同临界值的族错误率水平。 |
| `simultaneous_n_bootstrap` | `1000` | 乘子自助法抽样次数。 |
| `simultaneous_random_state` | `None` | 控制乘子抽样的随机种子。 |
| `simultaneous_include_intercept` | `False` | 是否把中心化纠偏截距纳入 max-|Z| 目标参数集合和最终同时置信区间。 |

`enable_simultaneous_inference=True` 要求同时满足 `compute_inference=True`、`inference_method="debiased"` 和 `simultaneous_method="maxz_bootstrap"`。`simultaneous_alpha` 必须严格位于 `(0, 1)`，`simultaneous_n_bootstrap` 必须为正；不支持的组合会明确失败，而不是静默返回普通边际置信区间。

### 将截距纳入同时推断

当 `simultaneous_include_intercept=True` 时，边际标准误所使用的同一个、基于中心化逐节点构造的原始坐标系截距影响量会直接进入自助最大统计量。也就是说，截距不再只是“额外加一行、套用仅由特征校准出的临界值”，而是真正属于被校准的目标参数集合。

默认值仍为 `False`，因此只需要对特征系数向量进行同时覆盖的调用不会无故扩大目标参数集合。

## 同时置信区间不等于 p 值校正

下面几件事相关，但并不相同：

- `_conf_int` 给出逐系数边际置信区间；
- `_conf_int_simultaneous` 使用共同的 max-|Z| 临界值保护一组同时置信区间；
- `model.adjust_pvalues(method="bh")` 或模块级 `adjust_pvalues(...)` 对已有 p 值向量进行多重检验校正；
- `model.combine_pvalues(...)` 检验一组 p 值是否包含全局证据，它不会构造同时系数置信区间。

通用的估计器绑定方法和模块级多重检验 API 见[推断 API 参考](../guides/inference-api.md)。

## 后端行为与数值计算—结果报告边界

拟合后统计方法与数值后端是两个独立维度。

对于 `post_selection_ols`，推断始终复用成功惩罚拟合所记录的后端和具体设备。活跃集 OLS/WLS 重拟合、协方差计算、参考分布推断以及置信区间数值计算，都会按照拟合阶段确定的后端在 NumPy、CuPy 或 Torch 上完成。显式 GPU 请求在相应后端不可用时会明确报错，不会静默回退为 CPU OLS 推断。

对于 `debiased`，当前维护的 CuPy/Torch 边际纠偏推断路径同样把数值计算留在实际执行的 GPU 后端。最终的结果报告层仍以 NumPy 为主：`_bse`、`_pvalues`、`_conf_int` 等数组以及结构化结果元数据，只有在数值推断全部完成后才会复制为主机端结果对象。

对于中心化且 `fit_intercept=True` 的 CuPy/Torch 同时推断，计算量较大的 $B\times n$ 乘子抽样、特征/截距得分、max-|Z| 最大值归约、分位数校准和同时置信区间计算，都会留在同一个具体 GPU 设备上，直到最后才复制到结果报告层。历史的 `fit_intercept=False` 同时推断路径仍使用已有的通用结果报告阶段辅助函数；PR #138 并没有把这条历史路径声明为 GPU 原生实现。

残差自助法 `bootstrap` 不同：当前残差重拟合实现只在 CPU 上执行，因此估计器使用 GPU `device` 并不意味着自助重采样和重拟合也会在 GPU 上执行。

## 样本权重与推断范围

`Lasso.fit(..., sample_weight=...)` 支持样本权重。当前维护的 NumPy/CuPy/Torch `debiased` 路径使用相同的**加权中心化平均损失问题**，因此把所有分析权重同时乘以同一个正数，不会改变惩罚拟合以及当前维护的纠偏推断结果。

这一实现约定本身并不意味着“对任意科学含义下的分析权重都自动得到有效的统计定理”。如果加权系数推断是科学结论的核心，仍应针对具体应用验证权重约定以及目标量（estimand）的含义。

## 残差自助法路径

`inference_method="bootstrap"` 会反复重采样残差并重新拟合惩罚模型。对应的构造参数为 `n_bootstrap`（默认 `200`）和 `bootstrap_random_state`（默认 `None`）。

当前实现用自助样本之间的变异计算标准误，用符号变化概率构造双侧 p 值，并使用百分位数置信区间。它比一次纠偏拟合昂贵得多，也没有被声明为完整的选择性推断程序。

## OLS 风格的选择后重拟合诊断

`inference_method="post_selection_ols"` 会在惩罚拟合选出的**精确活跃集**上重拟合一个未惩罚 OLS/WLS 模型。`coef_` / `intercept_` 仍属于惩罚预测拟合；`_params` / `_inference_result` 则保存用于推断报告的活跃集重拟合结果。

该重拟合使用拟合阶段确定的 NumPy/CuPy/Torch 后端。活跃设计矩阵秩亏时使用基于有效秩的 Moore-Penrose 广义逆/SVD 计算，而不是普通正规方程。经典 `cov_type="nonrobust"` 报告沿用 Student-t 约定；当估计器暴露相应选项时，稳健/HAC 协方差使用共享的高斯稳健协方差层及其正态参考分布约定。

完整参数空间中未被选中的坐标仍保留兼容占位值（`SE=0`、统计量 `0`、`p=1`、区间 `[0, 0]`）。这些值不是“系数被精确知道为 0”的统计主张；应通过已选特征元数据判断哪些坐标真正接受了活跃集重拟合。

最重要的是：使用同一份数据先选择变量、再构造普通 OLS/WLS 置信区间，仍然只能视为**选择后诊断**，而不是一般的选择性推断置信程序。

## 拟合后推断输出

所选推断路径成功后，可能出现以下结果属性：

| 属性 | 含义 |
|---|---|
| `_params` | 推断报告所使用的参数向量；`debiased` 时特征项是纠偏估计；`post_selection_ols` 时活跃坐标属于未惩罚重拟合 |
| `_bse` | 标准误 |
| `_tvalues` | 某些路径沿用的历史统计量字段；纠偏推断时具有 z 统计量语义 |
| `_zvalues` | 结构化推断结果层填充时的 z 统计量字段 |
| `_pvalues` | 双侧系数 p 值 |
| `_conf_int` | 边际置信区间 |
| `_conf_int_simultaneous` | 成功完成 max-|Z| 校准后的同时置信区间 |
| `_simultaneous_critical_value` | 校准得到的共同临界值 |
| `_inference_result` | 结构化推断结果及方法/后端元数据 |

`summary()` 使用当前已拟合模型中可用的推断结果。下划线开头的推断数组是当前版本既有的结果接口，但其统计含义仍然取决于具体 `inference_method`。

## 可复现性与计算成本

需要可复现的重采样时，应分别设置 `bootstrap_random_state` 或 `simultaneous_random_state`。提高 `simultaneous_n_bootstrap` 可以减少共同临界值的蒙特卡洛波动，但会增加实际数值后端上的计算量与内存流量。

纠偏推断通常显著比原始 Lasso 拟合昂贵，因为它需要求解许多逐节点稀疏回归。后端加速会改变计算成本，但不会改变统计假设。

## 当前实现没有声称什么？

- 普通选择后 OLS/WLS 区间不是一般的选择性推断区间；
- 残差自助法不是对模型选择不确定性的普适修正；
- `LassoCV` 最终重拟合推断不会自动计入通过交叉验证选择 `alpha` 的调参不确定性；
- 边际纠偏区间未经同时校准时不提供同时覆盖；
- max-|Z| 同时置信区间不能替代以错误发现率（FDR）为目标的 Benjamini-Hochberg 等程序；
- statgpu 的精确逐节点惩罚公式是一条实现级调参规则，并不是所有有效纠偏 Lasso 理论都必须使用同一常数或同一噪声尺度估计的定理；
- 数值收敛、GPU 执行以及很小的 KKT 残差不能证明数据满足纠偏推断所需的高维统计假设；
- 历史 `fit_intercept=False` 同时推断路径不会因为惩罚/纠偏拟合使用了 GPU，就自动获得“GPU 原生同时推断”的统计或实现声明。

## 参考文献

- 储嘉诚，唐炎林（2023）. 高维线性模型中的纠偏 LASSO 综述. *应用概率统计*, 39(3), 455–474. [doi:10.3969/j.issn.1001-4268.2023.03.010](https://doi.org/10.3969/j.issn.1001-4268.2023.03.010)
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242. [doi:10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202. [doi:10.1214/14-AOS1221](https://doi.org/10.1214/14-AOS1221)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909. [JMLR](https://jmlr.org/papers/v15/javanmard14a.html)
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional linear models. *Journal of the American Statistical Association*, 112(518), 757–768. [doi:10.1080/01621459.2016.1166114](https://doi.org/10.1080/01621459.2016.1166114)
- Dezeure, R., Bühlmann, P., & Zhang, C.-H. (2017). High-dimensional simultaneous inference with the bootstrap. *TEST*, 26(4), 685–719. [doi:10.1007/s11749-017-0554-2](https://doi.org/10.1007/s11749-017-0554-2)
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.