# 广义线性模型与带惩罚 GLM

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[英文版](../../en/models/generalized-linear-model.md)

## 如何选择响应模型

GLM 通过链接函数，将线性预测子与响应的条件均值联系起来。连续响应且均值关系近似线性时可用 Gaussian/identity；0/1 响应用 binomial/logit；计数响应用 Poisson/log。Log 链接使预测均值保持为正，系数取指数后表示条件均值的倍数变化；若没有额外假设，不能把它解释为因果效应。

Poisson 假设条件方差等于均值。计数数据明显过度离散时可考虑负二项模型；严格为正的连续响应可考虑 Gamma 或逆高斯模型。分布族应根据响应类型和研究假设选择，不能只看训练误差。这些接口没有 offset/exposure 参数；把暴露量作为普通特征加入公式，会估计它的系数，而不是把系数固定为 1。

## 完整 CPU 示例

下面模拟 log 链接的 Poisson 数据，用前 180 行和分析权重拟合无惩罚模型，最后 60 行留作评估。`C=0` 明确表达无惩罚意图；Newton 本身不使用 C。示例不需要 GPU 或可选的公式依赖。

<!-- learner-example: glm-poisson -->
```python
import numpy as np
from statgpu import GeneralizedLinearModel

rng = np.random.default_rng(59)
X = rng.normal(size=(240, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
weights = np.linspace(0.5, 2.0, 180)
model = GeneralizedLinearModel(
    family="poisson", C=0, solver="newton", device="cpu",
    max_iter=1000, tol=1e-8, compute_inference=True, cov_type="hc1",
).fit(X[:180], y[:180], sample_weight=weights)
mean_prediction = model.predict(X[180:])
heldout_loss = np.mean(mean_prediction - y[180:] * np.log(mean_prediction))
print("Slopes:", np.round(model.coef_, 3))
print("Mean multipliers:", np.round(np.exp(model.coef_), 3))
print("Predicted means:", np.round(mean_prediction[:3], 3))
print("Held-out Poisson loss:", round(float(heldout_loss), 3))
print("Interval shape:", model._conf_int.shape)
```

该随机种子对应的 CPU 四舍五入输出：

```text
Slopes: [ 0.402 -0.327]
Mean multipliers: [1.495 0.721]
Predicted means: [1.139 1.24  2.107]
Held-out Poisson loss: 0.556
Interval shape: (3, 2)
```

系数取指数后表示固定其他特征时，条件均值的倍数变化。 第一项表示该特征增加一个单位时，条件均值约增加 49.5%；第二项表示约降低 27.9%。即使观测值是整数计数，预测均值也可以是小数。留出集损失省略了只与响应有关的对数阶乘常数；只能在相同留出响应和权重下比较，数值越小越好。它不是概率、准确率或普通 R²。

这里的系数推断数组有三个位置：截距在前，随后是两列斜率；`_conf_int` 为 `(3,2)`。HC1 改变协方差估计，不改变拟合均值。这些是边际系数区间，不是未来计数的预测区间。通用普通 GLM 在 binomial 下返回均值概率，但没有 `predict_proba` 或 `score`；应按响应分布选择评估指标。用 `print(model.summary())` 显示它返回的报告字符串。

## 普通与带惩罚模型入口

`GeneralizedLinearModel` 是普通广义线性模型（GLM）的统一入口，适用于高斯、二项、Poisson 等常见分布族。`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` 等具体模型类在同一套 GLM 基础设施上提供各自的分布族和链接函数行为。

需要正则化时，可以使用 `PenalizedGeneralizedLinearModel`，或 `PenalizedLinearRegression`、`PenalizedLogisticRegression`、`PenalizedPoissonRegression` 等具体封装类。`Ridge`、`Lasso`、`ElasticNet` 是遵循 sklearn 风格的便捷封装，底层仍使用带惩罚的高斯回归。

如果主要关心**带权重的 GLM**，先记住四点即可：

- `sample_weight` 在受支持的 GLM 路径中表示**目标函数中的分析权重**；
- 加权损失按 `sum(sample_weight)` 归一化，因此把全部权重同时乘以同一个正数不会改变最优解；
- `sample_weight` 不会改变显式指定的 `solver`；如果组合不受支持，则直接报错；
- 显式 `device="cuda"` 与 `device="torch"` 在受支持的路径中分别使用 CuPy CUDA 与 Torch CUDA。

带惩罚 GLM 的系数推断见 [带惩罚 GLM 推断](../guides/penalized-glm-inference.md)；完整求解器表见 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)。

[GLM 专用类完整构造参数与方法](../reference/linear-model-api.md#typed-glm-constructors)
列出分布族的链接、离散参数、幂次，以及不同惩罚封装的默认值。
名称相近不代表可以互换构造参数。

## 公开入口

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.GammaRegression`
- `statgpu.linear_model.InverseGaussianRegression`
- `statgpu.linear_model.NegativeBinomialRegression`
- `statgpu.linear_model.TweedieRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`

内部 GLM 目标函数层为 `statgpu.glm_core`。

## 目标函数与 `sample_weight`

设 n 为观测数，$x_i$ 为特征向量，$\beta$ 为斜率向量，b 为不受惩罚的截距。记 $\eta_i=b+x_i^\top\beta$；`fit_intercept=False` 时令 b=0。**无惩罚**的数据拟合目标为相应分布族的平均负对数似然，采用该分布族的离散度约定：

$$
L(b,\beta)=\frac{1}{n}\sum_{i=1}^n\ell(y_i,\eta_i).
$$

受支持的分析权重 `sample_weight=w` 将其改为

$$
L_w(b,\beta)=\frac{\sum_i w_i\ell(y_i,\eta_i)}{\sum_i w_i}.
$$

权重必须有限、非负且总和为正。零权重观测不贡献数据拟合损失；所有权重同乘一个正常数，不改变该目标。

**普通 GLM 不等于自动采用无惩罚拟合。** 对本页列出的普通分布族，默认 `solver="auto"` 使用 IRLS。普通 IRLS 在 C 为正时加入斜率的岭惩罚：

$$
\min_{b,\beta} L_w(b,\beta)+\frac{1}{4C}\lVert\beta\rVert_2^2.
$$

无权重时以 L 代替 L_w。岭惩罚的梯度为 $\beta/(2C)$，不包含截距；默认 `C=1` 因而会收缩斜率。当前普通 IRLS 对非正 C 不施加此项，建议以 `C=0` 明确请求无惩罚 IRLS。普通 GLM 显式选择 `newton`、`lbfgs` 或 `fista` 时使用无惩罚损失，不使用 C。因此，若原来使用了正 C，更换普通 GLM 求解器可能改变统计问题。这里 C 的含义不同于独立的 LogisticRegression，也不同于下文的 alpha 接口。

**带惩罚 GLM** 则使用所声明的惩罚项：

$$
\min_{b,\beta}L(b,\beta)+\alpha P(\beta),\qquad
\text{或}\quad\min_{b,\beta}L_w(b,\beta)+\alpha P(\beta).
$$

截距仍不受惩罚。`statgpu.glm_core` 专门处理 GLM；Cox 偏似然、稳健损失和分位数损失保留各自的统计定义与文档。

## 求解器选择

对大多数用户，推荐先使用 `solver="auto"`。当前直接拟合的稀疏 Gaussian 模型中，`stopping="kkt"` 并不会切换停止条件，详见[直接拟合的控制参数限制](../reference/linear-model-api.md#elasticnet)。下表仅描述统一的**带惩罚估计器**调度；普通 GLM 默认使用上文说明的 IRLS，不会因为 C 加入岭惩罚就自动采用下表。

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` + NumPy/CPU | `exact` 闭式 L2 路径 |
| 平方误差 + L1/ElasticNet | FISTA/FISTA-BB 稀疏路径 |
| 逻辑回归/Poisson/Gamma/逆高斯/Tweedie/负二项 + L2 | Newton |
| 非凸 SCAD/MCP | FISTA + LLA 延续路径 |

`solver="auto"` 表示使用当前模型的自动调度规则。若显式指定 `solver`，该请求不会因 `sample_weight` 而改变；如果对应组合不受支持，则拟合会报错。

设备选择也遵循同样原则：显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA。公式解析可以在 CPU 上进行，但拟合与预测的数值计算仍使用选定的计算后端。

### 显式 Newton / L-BFGS 的加权支持

对于受支持的普通 GLM 分布族与链接函数组合，显式 `solver="newton"` 和
`solver="lbfgs"` 接受非均匀分析权重，并优化前面的归一化加权目标。
传入 `sample_weight` 不会更换指定的求解器或显式请求的 CUDA/Torch 设备；
不支持的组合会报错。均匀权重以及数值上等效的权重使用无权重目标。
加权更新和线搜索的数学说明见[求解器算法](../guides/solver-algorithms.md)。

#### `inverse_power` Gamma 的初始化

`GammaRegression(link="inverse_power")` 使用逆链接，因此拟合要求

$$
\eta_i=b+x_i^\top\beta>0.
$$

显式 Newton/L-BFGS 需要合法域内的初值；若无法找到数值上有效的初值，会在
优化开始前报错。没有截距时，正权重训练行的设计矩阵必须允许某个系数向量使
线性预测子全部为正。初始化失败时，应检查设计矩阵和链接函数的选择；增加
`max_iter` 不能解决没有可行初值的问题。后续迭代也必须保持在逆链接的合法域内。

#### 带权 L-BFGS 的适用范围

上述加权支持适用于受支持的 GLM 组合，不代表所有提供 L-BFGS 的模型都支持
非均匀权重。有序 GLM 与非 GLM 模型族有各自的权重和求解器规则，应查阅对应
模型页及[兼容性矩阵](../guides/solver-penalty-matrix.md)。

带惩罚的光滑 GLM 使用同一套分析权重定义。受支持的 L2 拟合使用 Newton 还是
L-BFGS，仍由直接拟合或交叉验证的调度规则决定，而不是仅由是否提供权重决定。

## 协方差与推断

对于通用的 `PenalizedGeneralizedLinearModel` 及各类专用的带惩罚 GLM 封装，推荐使用 `inference_method="auto"`。受支持的 M-估计、去偏与残差自助法结果会记录推断方法、目标及调参或选择条件。当前 `post_selection_ols` 即使返回了推断结果，`inference_method_` 和 `inference_target_` 仍可能为空；请读取 `_inference_result.method` 及其元数据，详见[选择后结果报告](../guides/inference-modes.md#post_selection_ols)。

对于受支持的光滑非高斯 L2/无惩罚模型，`auto` 解析为固定惩罚的 `m_estimation`。正 L2 惩罚对应带惩罚的估计方程；无惩罚别名会规范为惩罚强度为 0 的 L2，对应无惩罚总体参数。当前协方差支持 `nonrobust`、`hc0`、`hc1`；HC2/HC3/HAC 在这一路径中不可用，并会明确报错。

受支持的 M-估计与高斯残差自助法复用拟合的后端和设备；普通 GLM 的推断与拟合使用同一套分析权重定义，但是否支持权重仍取决于推断方法。非高斯 L1/ElasticNet 系数推断不受支持。SCAD/MCP `oracle` 需要显式请求，但当前非高斯子模型重建会把分布族设置和正则化重置为默认值，并自动选择子模型设备。不要把这些结果解释为预期的无惩罚活跃集推断，详见 [oracle 限制及诊断性重拟合替代方案](../guides/penalized-glm-inference.md#current-non-gaussian-oracle-limitation)。分组惩罚目前只提供估计。

对于受支持的高斯稀疏惩罚，`inference_method="bootstrap"` 表示 `cov_type="nonrobust"` 下的**无权重残差自助法**。设计矩阵和拟合后的调参配置保持固定：每次从残差中有放回抽样，把抽到的残差加到拟合值上，构造新的自助响应，再用同一个带惩罚模型重新拟合。`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制随机数可复现性。

CPU 拟合使用 NumPy；CuPy/Torch CUDA 拟合会把自助法重拟合留在同一个 GPU 设备。最终推断数组统一返回 NumPy。带权残差自助法、稳健/HC 协方差对应的自助法、HAC/分块自助法与非高斯自助法当前都不支持。

完整支持矩阵、重采样边界和统计解释见 [带惩罚 GLM 推断](../guides/penalized-glm-inference.md) 与 [推断模式](../guides/inference-modes.md)。

## 参数

下表精选了不同模型的控制参数，不代表一个共同构造器。`family` 属于普通 GLM，通用带惩罚类使用 `loss`；`formula` 与 `data` 是 fit 参数。完整的实际构造器与方法见[普通 GLM](../reference/linear-model-api.md#generalizedlinearmodel)、[通用带惩罚 GLM](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel)及[通用 CV](../reference/linear-model-api.md#penalizedglm_cv)参考。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `family` | 模型相关 | GLM 分布族，例如 `"gaussian"`、`"binomial"`、`"poisson"` |
| `penalty` | 通用直接模型为 `"l1"`；通用 CV 为 `"l2"` | `none`、`l1`、`l2`、`elasticnet` 以及结构化惩罚 |
| `alpha` | `1.0` 或模型默认 | statgpu 目标函数尺度下的惩罚强度 |
| `C` | 普通 GLM 为 `1.0` | 正 C 对应普通 IRLS 的斜率惩罚 `sum(beta**2)/(4*C)`；C=0 取消此项。普通显式 Newton/L-BFGS/FISTA 不使用 C。它不是带惩罚 GLM 的 alpha 参数。 |
| `l1_ratio` | 接受该参数的类为 `0.5` | 通用及专用带惩罚 GLM 的 ElasticNet 混合参数；部分专用封装不暴露此参数。 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `solver` | `"auto"` | 求解器；见上文“求解器选择” |
| `device` | `"auto"` | 根据模型支持情况使用 `cpu`、`cuda`、`torch` 或 `auto` |
| `max_iter` | 模型相关 | 最大迭代次数 |
| `tol` | 模型相关 | 收敛阈值 |
| `compute_inference` | 通常为 `False` | 是否计算系数推断 |
| `inference_method` | 通常为 `"auto"` | 推断方法请求 |
| `cov_type` | `"nonrobust"` | 协方差类型；非高斯带惩罚 M-估计当前支持 nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | HAC 相关控制；不代表上述路径已支持 HAC |
| `formula` | `None` | 可选 Patsy 风格公式，需要与 `data` 一起使用 |
| `data` | `None` | 公式模式下的数据表 |

不同框架中的同名正则化参数未必采用同一尺度。常见换算包括：

- Ridge：`sklearn_alpha = n_samples * statgpu_alpha`
- 逻辑回归 L2：`sklearn_C = 1 / (n_samples * statgpu_alpha)`
- Poisson L2：与 sklearn `PoissonRegressor(alpha=...)` 对齐
- Poisson L1/ElasticNet：与 statsmodels `fit_regularized` 对齐

## 可选 GPU 与公式输入

运行 CPU 示例中的数据准备后，可按下面的方式要求 CuPy CUDA 完成同一带权 Poisson 拟合。`device="torch"` 对应 Torch CUDA；显式请求的后端不可用时会报错。

```python
model_gpu = GeneralizedLinearModel(
    family="poisson", C=0, solver="newton", device="cuda",
    max_iter=1000, tol=1e-8, compute_inference=False,
).fit(X[:180], y[:180], sample_weight=weights)
```

公式接口需先安装 `pip install statgpu[formula]`。安装可选依赖后，下面的示例可独立运行：

<!-- learner-example: glm-formula -->
```python
import numpy as np
import pandas as pd
from statgpu import GeneralizedLinearModel

rng = np.random.default_rng(18)
df = pd.DataFrame({"x": rng.normal(size=80), "group": ["a", "b"] * 40})
df["count"] = rng.poisson(np.exp(0.2 + 0.3 * df["x"]))
model = GeneralizedLinearModel(family="poisson", C=0, device="cpu")
model.fit(formula="count ~ x + C(group)", data=df)
prediction = model.predict(df.iloc[:5])
assert prediction.shape == (5,)
```

公式在 CPU 上解析。应单独传入 formula/data，不要同时传入数组 X/y。权重可以对应原始全部行，也可以对应公式及缺失值处理后保留的行，按位置对齐。预测会重建训练时的列和类别水平。大数据可直接使用数组以避免公式解析开销。

普通 GLM 公式预测目前会删除预测变量缺失的行，并返回较短且无行标签的数组。
应处理缺失并核对输出长度，再将预测配给观测；详见[缺失行限制](../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms)。

### 重拟合失败后的限制

普通 auto/IRLS/FISTA 路径在重拟合失败后，可能混用旧系数和新的观测数、公式或截距设置，对象却仍显示为已拟合；预测和似然诊断也可能随之改变。遇到这类失败后应新建估计器，成功拟合后再读取结果，详见[失败后的完整约定](../reference/linear-model-api.md#failed-ordinary-glm-refits)。显式 Newton/L-BFGS 目前会保留上一次拟合，但这仍不代表新数据拟合成功。

## 严格与近似交叉验证

`PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`。严格模式下，每个交叉验证折和每个 `alpha` 都使用用户给定的 `max_iter` 与 `tol`；GPU 优化只改变实现效率，不改变候选模型的求解标准。

可选的 `cv_strategy="two_stage"` 会先用较宽松的求解条件筛选 `alpha` 网格，再严格复核候选 `alpha`，并进行严格的最终重拟合。由于第一阶段可能在非常接近的交叉验证曲线上改变 `alpha` 排名，该模式默认发出 `ApproximateCVWarning`；如果用户明确接受这种近似，可以传入 `acknowledge_approx=True`。

```python
from statgpu.linear_model import PenalizedGLM_CV

strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

<a id="reading-cv-inference-results"></a>

## 读取交叉验证后的推断结果

通用 `PenalizedGLM_CV` 在固定 l1_ratio 下选择 alpha，再用全部训练行重拟合。它与还能搜索 l1_ratio 的 `ElasticNetCV` 不同。通用 CV 的标量响应路径拟合截距，但不提供公开 fit_intercept 选项。最终 score 是响应尺度 R²，best_score_ 则是验证损失的负值。

当前 `summary()` 无法显示通用最终重拟合的推断报告：它调用的最终估计器没有该方法，因此会抛出 AttributeError。可以直接读取结果容器：

<!-- learner-example: glm-cv-inference -->
```python
import numpy as np
from statgpu import PenalizedGLM_CV

rng = np.random.default_rng(9)
X = rng.normal(size=(60, 2))
y = rng.poisson(np.exp(0.2 + 0.3 * X[:, 0]))
model = PenalizedGLM_CV(
    loss="poisson", penalty="l2", alpha_grid=[0.05, 0.2],
    cv=2, device="cpu", compute_inference=True,
).fit(X, y)
report = model.estimator_._inference_result.to_dict()
print("Selected alpha:", model.alpha_)
print("Method:", report["method"])
print("Standard errors:", np.round(model.estimator_._bse, 3))
assert model.cv_results_["all_scores"].shape == (2, 2)
assert np.isclose(model.best_score_, -np.min(model.cv_results_["mean_score"]))
```

该例返回 `m_estimation` 及三个有限标准误。推断以所选 alpha 为条件，不校正调参不确定性。最终通用估计器没有 predict_proba 或 summary；其 logistic 预测返回 0/1 标签，不是均值概率。若需要概率方法，应使用适当的专用模型。完整默认值、数组形状及 CV 结果键见 [API 参考](../reference/linear-model-api.md#penalizedglm_cv)。

## 相关文档

- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) — 损失函数 × 惩罚项 × 求解器的完整调度表、交叉验证路径和推断支持状态。
- [求解器算法](../guides/solver-algorithms.md) — Newton、L-BFGS、FISTA 等求解器的算法说明。
- [损失函数](losses.md) — 底层损失函数接口及其能力边界。

## 常见问题

- **这里的 `sample_weight` 表示什么？** 在受支持的普通或带惩罚 GLM 路径中，它表示目标函数中的分析权重，并按 `sum(weights)` 归一化。它不是调查抽样权重，也不是用于自助法抽样的权重；同时不会自动启用带权残差自助法。
- **加入 `sample_weight` 会改变求解器吗？** 不会。显式指定的 `solver` 保持不变；`solver="auto"` 继续按照对应模型的自动调度规则选择求解器。
- **`device="cuda"` 是否保证 GLM 求解器在 GPU 上运行？** 对受支持的路径，是的：数值计算使用 CuPy；若组合不受支持或设备不可用，则报错。
- **大规模 GPU 任务是否建议使用 `formula`？** 通常不建议。公式接口主要用于便利建模，大规模任务更适合显式数组。
- **`Ridge`、`Lasso`、`ElasticNet` 是别名吗？** 不是。它们是保留 sklearn 风格构造器语义的薄封装类。

## 与其他实现比较

比较系数或不确定性时，应对齐响应分布族与链接、截距、特征列、分析权重、惩罚定义、求解器和收敛设置。若比较无惩罚的普通 Poisson 模型，可用 `C=0` 的 IRLS 或显式无惩罚求解器；普通 GLM 默认的 `C=1` IRLS 对应另一统计目标。带惩罚拟合还需对齐平均损失与总损失以及惩罚尺度，不能仅凭参数同名就认为目标一致。

除系数外，还应比较留出数据预测及目标函数或最优性诊断。推断比较须使用同一协方差约定，并区分是否以既定调参值或已选择变量为条件。某一后端上的数值对照不能证明另一设备上的结果。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [DOI](https://doi.org/10.18637/jss.v033.i01)
