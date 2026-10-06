# PoissonRegression

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[English](../../en/models/poisson-regression.md)

## 什么时候使用 Poisson 回归？

`PoissonRegression` 通过对数链接建模非负计数的条件均值，适合观测暴露时长
相当的事件计数。斜率反映期望计数的乘法关联，不代表因果效应。

Poisson 模型假设条件方差等于条件均值。明显过度离散时，可以考虑
`NegativeBinomialRegression`；零值过多或观测相关时，可能需要其他模型或
协方差分析。严格为正的连续响应通常更适合 Gamma 等分布族，不能直接当作
事件计数。选择依据见[广义线性模型](generalized-linear-model.md)。

该专用类固定 Poisson 分布族，其余行为沿用普通 GLM。默认 `C=1` 配合
`solver="auto"` 会施加岭惩罚；显式无惩罚 IRLS 应使用 `C=0`，也可按下文选择
不使用 C 的显式求解器。需要 alpha 惩罚接口时，用 `PenalizedPoissonRegression`。
此 API 没有 offset/exposure 参数：把暴露量放进普通预测变量，会估计它的系数，
而不是将该系数固定为一。

## 完整 CPU 示例

下面用 200 行拟合，再在另外生成的 50 行上评价。`C=0` 明确表达无惩罚意图，
Newton 本身不使用 C。示例不需要 GPU 或可选公式依赖。

<!-- learner-example: poisson-unpenalized -->
```python
import numpy as np
from statgpu import PoissonRegression

rng = np.random.default_rng(8)
X = rng.normal(size=(200, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
model = PoissonRegression(
    C=0, solver="newton", device="cpu", compute_inference=True,
    cov_type="nonrobust", max_iter=200, tol=1e-9,
).fit(X, y)

# New observations are generated separately and never used for fitting.
X_test = rng.normal(size=(50, 2))
y_test = rng.poisson(np.exp(0.3 + X_test @ np.array([0.4, -0.2])))
mean_prediction = model.predict(X_test)
heldout_loss = np.mean(mean_prediction - y_test * np.log(mean_prediction))
print("Slopes:", np.round(model.coef_, 3))
print("Mean multipliers:", np.round(np.exp(model.coef_), 3))
print("Predicted means:", np.round(mean_prediction[:3], 3))
print("Held-out Poisson loss:", round(float(heldout_loss), 3))
print("Interval shape:", model._conf_int.shape)
```

斜率约为 `[0.377, -0.154]`，指数化后约为 `[1.459, 0.857]`。保持另一个
预测变量不变，第一个变量每增加一单位，期望计数约增加 45.9%；第二个变量
每增加一单位，期望计数约降低 14.3%。斜率的指数不是事件发生概率。

预测数组形状为 `(50,)`，取值为正；观测是整数计数，预测均值仍可为小数。
留出集 Poisson 损失省略仅与响应有关的对数阶乘项，只有在相同留出响应和
权重上比较时，越小才表示越好。它不是准确率或普通 R²，该类也没有 `score` 方法。

区间数组形状为 `(3, 2)`，首行是截距，其后对应两个输入列。这些是渐近、边际
系数区间，不是未来计数的预测区间。`summary()` 返回字符串，使用
`print(model.summary())` 显示。

## 参数选择与结果检查

- 无惩罚拟合使用 `C=0`。若用正 C 的 IRLS 做预测，应在训练数据内通过验证
  选择 C；正 C 越大，收缩越弱。零是完全取消惩罚的特殊值，不是更强的惩罚。
- 每个训练折内学习缩放参数，并将相同变换用于评价数据；变量单位会影响岭
  收缩。最终测试集应独立于全部调参过程。
- 检查实际与预测计数、暴露量可比性和过度离散。有限结果或较低训练损失不能
  证明 Poisson 方差假设成立。HC 协方差只改变不确定性，不改变均值模型或预测。
- 可调整 `max_iter` 和 `tol` 检查数值稳定性。`n_iter_` 只是迭代次数，不是
  收敛证明；增加预算未必能解决尺度不良或共线性问题。

## 统计模型与目标函数

在对数链接下，条件均值为

$$
\mu_i = \exp(x_i^\top\beta).
$$

忽略与参数无关的常数项后，模型最小化平均 Poisson 负对数似然：

$$
\min_\beta
\frac{1}{n}\sum_i
\left[\mu_i-y_i\log(\mu_i)\right].
$$

有截距时，以 $b+x_i^\top\beta$ 替换线性预测子。传入分析权重
`sample_weight=w` 时，数据损失改为按 `sum(w)` 归一化的加权平均。
正 C 下，普通 auto/IRLS 路径增加 $\|\beta\|_2^2/(4C)$，截距不受惩罚。
`C=0` 去掉该项；显式普通 `newton`、`lbfgs` 和 `fista` 路径忽略 C，优化无惩罚
目标。因此，正 C 下改变求解器可能改变统计模型。该 C 尺度不同于独立的
LogisticRegression，详见[普通 GLM 目标函数](generalized-linear-model.md)。

非惩罚 Poisson GLM 的得分方程为

$$
\sum_i x_i(y_i-\mu_i)=0.
$$

正 C 下，auto/IRLS 的平均损失斜率方程还包含岭惩罚梯度 `beta/(2*C)`。

`solver="auto"` 当前选择 IRLS。对光滑的 Poisson GLM 目标，也可以显式使用 `solver="newton"` 或 `solver="lbfgs"`；这些求解器在受支持组合下运行于所选数值后端。

## 协方差与统计推断

如 CPU 示例所示，设置 `compute_inference=True` 可获得系数不确定性。
若研究问题适合相应的得分稳健协方差假设，可使用 `hc0` 或 `hc1`，这不会改变
系数估计。

当前协方差选项包括：

- `cov_type="nonrobust"`：无惩罚模型使用期望 Fisher 信息；正 C 的 IRLS 推断还包含惩罚曲率；
- `cov_type="hc0"`：基于观测 Hessian 的夹心协方差；
- `cov_type="hc1"`：在 HC0 基础上加入自由度修正；
- `hc2`、`hc3` 与 `hac`：当前 Poisson 路径不支持，显式请求会报错。

这些是边际、渐近正态参考区间。正 C 下的 IRLS 协方差围绕惩罚拟合计算，
不会消除收缩偏差，也不校正选择 C 的不确定性。与无惩罚 `statsmodels.GLM`
比较时，需对齐 C/求解器、设计矩阵、权重、协方差和收敛设置；单个数据集的
比较不能保证所有情形下的数值精度。

Poisson 推断使用渐近正态参考分布，因此报告 z 统计量和双侧 p 值。Poisson 离散参数固定为 1；相关拟合信息可通过模型元数据查看。

在 CuPy 或 Torch CUDA 上成功拟合后，受支持的推断计算继续使用同一数值后端和具体设备；不会为了计算协方差或参考分布而静默切换到 CPU。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `100` | 最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `C` | `1.0` | 正 C 下 auto/IRLS 增加 `sum(beta**2)/(4*C)`；零取消惩罚。显式 Newton/L-BFGS/FISTA 忽略 C。 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs`；实际可用性取决于完整模型路径 |
| `n_jobs` | `None` | 共享配置；该类不保证并行拟合 |
| `gpu_memory_cleanup` | `False` | 拟合后尽可能释放可回收的 GPU 缓存 |
| `compute_inference` | `False` | 是否计算受支持的系数推断 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`hc0` 或 `hc1` |

`formula` 与 `data` 是
`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` 的参数，
不是构造函数参数。该类固定 Poisson 分布族，其余方法与属性继承自
[普通 GLM](../reference/linear-model-api.md#generalizedlinearmodel)，包括 `predict`、
`summary`、似然诊断以及共享推断工具。它没有 `score` 或 `predict_proba` 方法。
`summary()` 返回字符串，应使用 `print(model.summary())`。当前 auto/IRLS/FISTA
路径在重拟合失败后应改用新估计器，详见[失败重拟合警告](../reference/linear-model-api.md#failed-ordinary-glm-refits)。

求解器组合的完整支持范围见 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)。

## 可选 GPU 与公式输入

运行 CPU 示例后，可以用 CuPy CUDA 拟合同一个无惩罚模型：

```python
model_gpu = PoissonRegression(
    C=0, solver="newton", device="cuda", max_iter=200, tol=1e-9,
).fit(X, y)
mean_prediction_gpu = model_gpu.predict(X_test)
```

Torch CUDA 使用 `device="torch"`。显式 GPU 请求需要已经安装且可用的对应
CUDA 后端，不可用时会报错。预测返回已解析后端的原生数组，系数报告数组
则是 NumPy。公式解析本身在 CPU 上完成。

安装 `statgpu[formula]` 后，下面的公式示例可以独立运行。公式语法决定截距
和分类变量编码：

<!-- learner-example: poisson-formula -->
```python
import numpy as np
import pandas as pd
from statgpu import PoissonRegression

rng = np.random.default_rng(18)
df = pd.DataFrame({"x": rng.normal(size=80), "group": ["a", "b"] * 40})
df["count"] = rng.poisson(np.exp(0.2 + 0.3 * df["x"]))
model = PoissonRegression(C=0, device="cpu")
model.fit(formula="count ~ x + C(group)", data=df)
prediction = model.predict(df.iloc[:5])
assert prediction.shape == (5,)
```

使用 formula/data 时不要同时传数组 X/y。预测 DataFrame 会重建训练列及分类
水平，未知水平会报错。预测变量缺失行目前会被删除，返回较短且无行标签的
数组。应处理缺失并核对输出长度，再将预测配给观测；详见[普通 GLM 缺失行警告](../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms)。
权重可对应原始行或保留行，按位置
匹配。详见[完整公式约定](../reference/linear-model-api.md#formula-inputs)。
大规模 GPU 任务直接使用已准备好的数组可减少公式解析开销。

## 输出与解释

常用结果包括：

- `intercept_`、`coef_`：拟合参数；
- `n_iter_`：迭代次数；
- `_bse`、`_zvalues`、`_pvalues`、`_conf_int`：启用推断时的统计结果；
- `fit()`、`predict()`：拟合与预测接口。

`predict()` 返回逆链接后的条件均值。对 Poisson 模型而言，这是估计的计数或发生率均值 $\widehat\mu$，而不是线性预测子 $X\widehat\beta$。

## 常见问题

**什么时候使用 `PoissonRegression`，什么时候直接使用通用 GLM？**  
当希望接口明确表达“这是 Poisson 回归”时使用 `PoissonRegression`；它与通用 GLM 的 Poisson 配置共享核心实现。

**什么时候使用 `PenalizedPoissonRegression`？**  
需要 L1、L2、ElasticNet、分组、自适应或其他惩罚路径时使用。

**是否支持 GPU 上的统计推断？**  
支持的 Poisson 推断路径可以在 CuPy 或 Torch CUDA 上执行。显式 GPU 请求只有在对应路径和后端可用时才成功，否则直接报错。

**为什么不支持 HC2、HC3 或 HAC？**  
这些协方差形式目前没有在 Poisson 模型的公开推断路径中实现；应使用已支持的 `nonrobust`、`hc0` 或 `hc1`，或者根据研究问题选择其他模型/推断方案。

## 与其他实现比较

与无惩罚 `statsmodels.GLM` 比较时，对齐对数链接、截距和设计矩阵、响应、
权重、协方差及收敛设置，并使用 C=0 的 IRLS 或显式无惩罚求解器。若将
正 C 的 IRLS 与 sklearn `PoissonRegressor` 比较，平均损失惩罚应换算为
`sklearn_alpha = 1 / (2*C)`；参数名称相同不代表目标函数相同。

除系数外，还应比较预测均值和得分方程残差。单个数据集或 CPU 上的比较不能
证明 GPU 精度，也不构成普适精度保证。更多说明见[GLM 对照比较](generalized-linear-model.md)。

## 相关文档

- [广义线性模型](generalized-linear-model.md)
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)
- [推断模式](../guides/inference-modes.md)
- [设备与 GPU 内存](../guides/device-and-memory.md)

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
