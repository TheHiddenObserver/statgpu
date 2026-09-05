# 线性回归推断与完整 API

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/models/linear-regression-inference.md)

本页是[线性回归入门文档](linear-regression.md)的进阶配套页。第一次使用模型时先看入门页；需要核对协方差定义、全部公开输入输出或后端实现边界时再查本页。

## 记号

令 $Z$ 为实际拟合的设计矩阵；存在截距时，它包含截距列。$n$ 是观测数，$r=\operatorname{rank}(Z)$，$k$ 是设计矩阵列数，$\hat e_i$ 是第 $i$ 个残差，并记

$$
B=(Z^\top Z)^{-1},
\qquad
h_{ii}=z_i^\top Bz_i.
$$

秩亏时，statgpu 会在需要时使用最小二乘或广义逆路径，残差自由度为 $n-r$。下列公式写成普通满秩形式。

若指定 `sample_weight=w`，模型拟合的是加权最小二乘。此时应把下面的 $Z$ 与 $\hat e$ 理解为变换后的 $\widetilde Z=W^{1/2}Z$ 与 $\widetilde e=W^{1/2}e$。

## 六种协方差估计

### `nonrobust`：经典同方差协方差

$$
\hat\sigma^2=\frac{\hat e^\top\hat e}{n-r},
\qquad
\widehat{\operatorname{Var}}(\hat\beta)=\hat\sigma^2B.
$$

它假设误差条件同方差且不相关。系数检验使用自由度为 $n-r$ 的 Student $t$ 分布。

### `hc0`：White 三明治协方差

$$
\widehat{\operatorname{Var}}_{\mathrm{HC0}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\hat e_i^2z_iz_i^\top\right)B.
$$

HC0 允许未知异方差，但没有有限样本自由度修正。

### `hc1`：带自由度修正的 HC0

$$
\widehat{\operatorname{Var}}_{\mathrm{HC1}}(\hat\beta)
=\frac{n}{n-r}\widehat{\operatorname{Var}}_{\mathrm{HC0}}(\hat\beta).
$$

HC1 是常见的通用稳健起点，也接近许多计量软件的约定。

### `hc2`：杠杆值修正

$$
\widehat{\operatorname{Var}}_{\mathrm{HC2}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\frac{\hat e_i^2}{1-h_{ii}}z_iz_i^\top\right)B.
$$

HC2 会提高高杠杆观测对协方差的贡献。

### `hc3`：更强的杠杆值修正

$$
\widehat{\operatorname{Var}}_{\mathrm{HC3}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\frac{\hat e_i^2}{(1-h_{ii})^2}z_iz_i^\top\right)B.
$$

小样本或存在明显高杠杆点时，HC3 常是较保守的起点。statgpu 会把数值杠杆值截断到 1 以下以避免除零；这个保护并不能让接近奇异的设计在统计上变得可靠。

### `hac`：Bartlett/Newey–West 协方差

定义逐观测得分 $s_t=z_t\hat e_t$ 与滞后乘积

$$
\Gamma_\ell=\sum_{t=\ell+1}^{n}s_ts_{t-\ell}^\top.
$$

最大滞后为 $L$ 时，statgpu 使用 Bartlett 权重 $w_\ell=1-\ell/(L+1)$：

$$
\Omega
=\Gamma_0+\sum_{\ell=1}^{L}w_\ell
\left(\Gamma_\ell+\Gamma_\ell^\top\right),
\qquad
\widehat{\operatorname{Var}}_{\mathrm{HAC}}(\hat\beta)=B\Omega B.
$$

若 `hac_maxlags=None`，

$$
L=\left\lfloor4\left(\frac{n}{100}\right)^{2/9}\right\rfloor,
$$

并限制在 $0\le L\le n-1$。HAC 直接使用数据行顺序，不会从 DataFrame 中自动识别时间变量；拟合前必须按有意义的时间顺序排序。

## 从协方差到结果表

第 $j$ 个系数满足

$$
\operatorname{se}(\hat\beta_j)
=\sqrt{\widehat{\operatorname{Var}}(\hat\beta)_{jj}},
\qquad
q_j=\frac{\hat\beta_j}{\operatorname{se}(\hat\beta_j)}.
$$

`nonrobust` 使用 $t_{n-r}$ 标定；HC 与 HAC 的 p 值和区间使用标准正态分布。为保持兼容，存储数组目前仍命名为 `_tvalues`，`summary()` 也统一把列标题写成 `t`；在 `nonrobust` 之外，应把它理解为一般的系数检验统计量。

## 应该选择哪一种？

| 误差结构 | 推荐起点 | 主要限制 |
|---|---|---|
| 独立且近似同方差 | `nonrobust` | 异方差时失效 |
| 独立、大样本、可能异方差 | `hc1` | 不能处理序列或聚类依赖 |
| 较小样本或存在明显杠杆点 | `hc3` | 极端杠杆值会使方差很大 |
| 有序观测且存在短程序列相关 | `hac` | 依赖正确行顺序与滞后选择 |

改变 `cov_type` 只改变不确定性估计，不改变 `coef_`。它不能修复非线性均值、遗漏混杂因素、聚类依赖或错误的设计矩阵。

## 完整 API 参考

### 构造器

```python
LinearRegression(
    fit_intercept=True,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    gpu_memory_cleanup=False,
    cov_type="nonrobust",
    hac_maxlags=None,
)
```

| 参数 | 可接受值与作用 |
|---|---|
| `fit_intercept` | 布尔值；除非 Formula 明确移除截距，否则加入截距 |
| `device` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）、`"torch"`（Torch CUDA）或 `Device` |
| `n_jobs` | 可选的估计器级并行提示；并非所有线性代数后端都会使用 |
| `compute_inference` | 为 `True` 时计算协方差、检验、区间与 summary 所需结果 |
| `gpu_memory_cleanup` | 为 `True` 时在公开 GPU 操作后释放后端缓存；可能降低重复拟合吞吐量 |
| `cov_type` | `"nonrobust"`、`"hc0"`、`"hc1"`、`"hc2"`、`"hc3"` 或 `"hac"` |
| `hac_maxlags` | 非负整数；`None` 使用上面的样本量规则 |

### `fit`

```python
fit(X=None, y=None, sample_weight=None, formula=None, data=None)
```

可以传 `X` 与 `y`，也可以传 `formula` 与 pandas `data`。`sample_weight` 必须是一维、有限、非负且总和为正。Patsy 会按公式规则删除缺失行；权重会按保留的行位置重新对齐。

### 拟合后输出

| 名称 | 含义 |
|---|---|
| `coef_` | 特征系数；形状为 `(p,)` 或 `(n_targets, p)` |
| `intercept_` | 标量或每个目标一个值 |
| `rank_` | 拟合设计矩阵的数值秩 |
| `rsquared`、`rsquared_adj` | $R^2$ 与调整 $R^2$ |
| `fvalue`、`f_pvalue` | 总体回归 F 统计量及 p 值 |
| `llf`、`aic`、`bic` | 高斯对数似然与信息准则；多目标不报告信息准则 |
| `_bse` | 系数标准误；有截距时截距在第一项 |
| `_tvalues` | 存储的系数统计量；标定方式见上文 |
| `_pvalues` | 双侧系数 p 值 |
| `_conf_int` | 95% 系数区间，形状 `(k, 2)` |
| `_inference_result` | 结构化推断结果，包含参数、统计量元数据、特征名与 DataFrame 转换 |

带下划线的数组属于当前兼容表面，但仍是下划线命名。编写可复用报告代码时优先使用 `_inference_result`，因为它还记录协方差与参考分布元数据。

### 方法

| 方法 | 约定 |
|---|---|
| `predict(X_new)` | 返回拟合条件均值；DataFrame 会复用存储的 Formula 编码 |
| `score(X, y)` | 返回 $R^2$；多目标时对各目标分数取平均 |
| `summary()` | 打印单目标系数表与拟合统计量；要求已经计算推断 |
| `get_params()` / `set_params()` | 从基础估计器继承的 sklearn 风格配置 |

## 后端实现细节

- CPU 使用 NumPy 最小二乘。
- CuPy 与 Torch 在指定 CUDA 后端上构建设计矩阵，优先用 Cholesky 求解；只有确实出现秩失败时才回退到对应后端的最小二乘。
- 经典与稳健协方差在所选数值后端计算；面向用户的系数和推断元数据会在报告边界转换为 NumPy。
- 大型 CPU HAC 任务可能先对小块得分矩阵计时；只有混合精度确实更快时才使用并缓存该路径。
- 显式请求 `"cuda"` 或 `"torch"` 而后端不可用时会报错，不会静默改用 CPU。

生命周期、数据传输、dtype、同步与可复现性见 [CPU/GPU 加速实现](../guides/acceleration-internals.md)。

## 验证与源码地图

- 估计器：`statgpu/linear_model/wrappers/_linear.py`
- 统一推断结果：`statgpu/inference/_results.py`
- 外部框架一致性：`dev/tests/test_external_consistency.py`
- Formula 与推断约定：`dev/tests/test_gaussian_inference_formula_cleanup_contract.py`

这些路径用于解释当前实现；私有辅助函数名不是公开 API。
