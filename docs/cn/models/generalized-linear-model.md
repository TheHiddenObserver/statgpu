# GeneralizedLinearModel 与惩罚 GLM

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/models/generalized-linear-model.md)

## GLM 解决什么问题？

广义线性模型（GLM）把线性回归扩展到取值范围或方差结构不适合高斯连续模型的响应变量。它保留可解释的线性预测子，再通过链接函数把预测子映射到响应的条件均值。

在 statgpu 中，`GeneralizedLinearModel` 是统一入口；`PoissonRegression` 等类型化估计器会明确响应分布，惩罚版本则增加系数收缩或特征选择。

## 什么时候使用 GLM？

应根据响应类型与均值—方差关系选择 family：

| 响应 | Family / 默认链接 | 推荐起点 |
|---|---|---|
| 连续且大致对称 | Gaussian / identity | `LinearRegression` 或 `GeneralizedLinearModel(family="gaussian")` |
| 0/1 响应 | Binomial / logit | `LogisticRegression` |
| 非负计数，方差接近均值 | Poisson / log | `PoissonRegression` |
| 过度离散计数 | Negative binomial / log | `NegativeBinomialRegression` |
| 正值、右偏连续响应 | Gamma / log | `GammaRegression` |
| 正值且方差大致按 $\mu^3$ 增长 | Inverse Gaussian / log | `InverseGaussianRegression` |
| 同时包含零和正连续值，或符合幂方差 | Tweedie / log | `TweedieRegression` |

不要只根据响应变量的名字选择 family。还应检查取值范围、方差模式、过多零值、观测依赖性，以及链接函数是否具有合理的科学解释。

## 直观理解

每个 GLM 都包含三部分：

1. 响应分布，即 **family**；
2. 线性预测子 $\eta_i=\beta_0+x_i^\top\beta$；
3. 把均值连接到预测子的 **link function**：

$$
g(\mu_i)=\eta_i,
\qquad
\mu_i=\mathbb E(Y_i\mid x_i).
$$

log 链接满足 $\mu_i=\exp(\eta_i)$，因此预测保持为正；logit 链接的逆函数则把任意线性预测子映射为 0 到 1 之间的概率。

## 模型、目标函数与假设

普通 GLM 最小化对应 family 的平均负对数似然：

$$
\hat\beta
=
\arg\min_\beta
\frac{1}{n}\sum_{i=1}^{n}
\ell(y_i,x_i^\top\beta).
$$

惩罚 GLM 在此基础上加入正则项：

$$
\hat\beta
=
\arg\min_\beta
\left[
\frac{1}{n}\sum_{i=1}^{n}
\ell(y_i,x_i^\top\beta)
+\alpha P(\beta)
\right],
$$

其中截距不被惩罚。

系数解释与推断要求 family、链接函数和线性预测子设定合理，观测独立或依赖结构被正确处理，并且没有严重秩亏。GLM 不会自动修复遗漏变量、过多零值或聚类观测。

## 最小可运行示例

下面生成计数数据，并用 Poisson 模型进行无惩罚拟合和稳健推断。

```python
import numpy as np
from statgpu.linear_model import PoissonRegression

rng = np.random.default_rng(1)
X = rng.normal(size=(800, 2))
true_coef = np.array([0.45, -0.25])
mean_count = np.exp(0.30 + X @ true_coef)
y = rng.poisson(mean_count)

model = PoissonRegression(
    solver="newton",
    device="cpu",
    compute_inference=True,
    cov_type="hc1",
).fit(X, y)

print("截距：", model.intercept_)
print("对数率系数：", model.coef_)
print("率比：", np.exp(model.coef_))
print("p 值：", model._pvalues)
print("期望计数：", model.predict(X[:3]))
```

这里显式设置 `solver="newton"`，因为 IRLS 路径中的 `C` 参数会加入 L2 正则化。估计系数应接近 `[0.45, -0.25]`。

## 如何读取结果？

对 log 链接模型：

$$
\log(\mu_i)=\beta_0+x_i^\top\beta.
$$

- `coef_[j]` 是对数均值尺度上的变化。
- `exp(coef_[j])` 是均值的乘法比。例如系数 `0.45` 表示该特征增加一个单位时，控制其他特征不变后的期望计数约为原来的 `exp(0.45)=1.57` 倍。
- `predict(X_new)` 返回响应自然尺度上的条件期望。
- 开启推断后，`_bse`、`_zvalues`、`_pvalues` 和 `_conf_int` 描述系数不确定性。

对 logit 模型，`exp(coef_[j])` 是优势比（odds ratio），并不是概率的直接变化量。

## 关键参数怎么选？

| 参数 | 选择建议 |
|---|---|
| `family` | 根据响应取值范围与方差模式选择；支持 `gaussian`、`binomial`、`poisson`、`gamma`、`inverse_gaussian`、`negative_binomial` 和 `tweedie` |
| `fit_intercept` | 通常保留 `True`；仅当设计中已有截距或理论要求截距为零时关闭 |
| `solver` | 先使用 `"auto"`；需要固定调度或明确无惩罚路径时显式指定 |
| `device` | 根据估计器、数据规模和已安装后端选择 `cpu`、`cuda`、`torch` 或 `auto` |
| `compute_inference` | 需要普通模型的标准误和检验时开启；许多 GLM 包装器默认关闭 |
| `cov_type` | 普通 GLM 支持 `nonrobust`、`hc0` 和 `hc1` |
| `alpha` | 在惩罚模型中，值越大收缩越强；应通过交叉验证选择 |
| `l1_ratio` | Elastic Net 中接近 1 更偏向稀疏，接近 0 更像 L2 收缩 |

## 普通模型还是惩罚模型？

| 目标 | 推荐 API |
|---|---|
| 估计并解释少量预先指定的变量 | `PoissonRegression` 等类型化普通估计器 |
| 收缩高度相关的系数 | 类型化惩罚估计器配合 `penalty="l2"` |
| 自动选择特征 | `penalty="l1"`、`"elasticnet"`、SCAD 或 MCP |
| 调整正则化强度 | `PenalizedGLM_CV` |
| 使用公式和 DataFrame | 安装 `statgpu[formula]`，同时传入 `formula=...` 与 `data=...` |

惩罚类型化包装器现已覆盖 Gaussian、logistic、Poisson、Gamma、inverse Gaussian、negative binomial 与 Tweedie。

## CPU、GPU、求解器与推断

受支持的路径中，显式 `device="cuda"` 使用 CuPy，`device="torch"` 使用 Torch CUDA；显式请求 GPU 时不会静默回退到 CPU。公式解析属于 CPU 预处理，大规模 GPU 任务更适合直接传入数组。

`solver="auto"` 会根据 family、penalty 和设备选择路径。光滑目标可使用 IRLS、Newton 或 L-BFGS；非光滑惩罚使用 FISTA 等近端算法。不合法的求解器与惩罚组合会直接报错。

普通类型化 GLM 可通过 `compute_inference=True` 进行拟合后推断。惩罚模型的推断能力取决于模型和惩罚类型；请查看[求解器与惩罚兼容矩阵](../guides/solver-penalty-matrix.md)，不要假定特征选择后仍可直接使用普通模型 p 值。

`PenalizedGLM_CV` 默认使用严格交叉验证。`cv_strategy="two_stage"` 是显式的近似筛选模式，未确认接受近似时会发出 `ApproximateCVWarning`。

## 常见误区

- Poisson 数据的方差明显大于均值时可能存在过度离散，应比较 negative binomial 或稳健推断。
- log 链接系数不是响应值的加法变化量。
- 计数中零值过多时可能需要零膨胀或 hurdle 模型，选择 Poisson 并不会自动处理。
- 正则化系数有意带偏；特征选择后不能简单套用普通标准误。
- 不同库的目标函数尺度不同，不能在未核对定义时直接复制 `alpha` 或 `C`。
- 始终检查收敛状态、样本外表现、残差或校准诊断。

## API 与验证

主要导入：

```python
from statgpu.linear_model import (
    GeneralizedLinearModel,
    LogisticRegression,
    PoissonRegression,
    GammaRegression,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)
```

常见拟合输出包括 `coef_`、`intercept_`、`n_iter_`、`predict`，以及不同 family 提供的 `predict_proba` 或 `score`。推断输出为 `_bse`、`_zvalues`、`_pvalues` 和 `_conf_int`。

外部框架与多后端验证覆盖系数一致性、目标函数差异、KKT 残差、推断和 CPU/CuPy/Torch 行为。入口见[已实现方法指南](../guides/implemented-methods.md)中的验证链接。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.
