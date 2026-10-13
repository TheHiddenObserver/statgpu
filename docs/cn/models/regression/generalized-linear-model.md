# 广义线性模型与带惩罚 GLM

> 语言：中文  
> 最后更新：2026-10-13<br>
> 页面定位：模型文档  
> 切换：[英文版](../../../en/models/regression/generalized-linear-model.md)

广义线性模型（GLM）用于研究响应变量与特征之间的关系，并预测给定特征时响应的平均水平。例如，可以用它预测某类事件的发生次数、二分类事件的发生概率，或始终为正的连续测量值。不同类型的响应需要不同的分布假设，GLM 为这些问题提供了统一的建模方式。

## 如何选择响应模型

建立 GLM 时，需要区分两个概念：**分布族**描述响应的分布及其均值与方差的关系；**链接函数**把响应的条件均值与特征的线性组合联系起来。这里的条件均值，就是给定特征时响应的平均水平；线性组合由截距加上各特征与其系数的乘积组成，也称为线性预测子。

可以先根据响应类型选择模型：

- 连续响应的均值与特征近似呈线性关系时，可使用高斯分布族和恒等链接（Gaussian/identity），即直接对条件均值建模。
- 对 0/1 响应，可使用二项分布族和 logit 链接（binomial/logit），对事件发生概率建模。
- 对非负整数计数，可使用 Poisson 分布族和对数链接（Poisson/log），对期望计数建模。Poisson 假设条件方差等于条件均值；计数数据明显过度离散时，可考虑负二项模型。
- 对严格为正的连续响应，可考虑 Gamma 或逆高斯模型。

分布族应根据响应类型和研究假设选择，不能只看训练误差。使用对数链接时，预测均值保持为正；其他特征不变时，一个特征增加一单位，其系数的指数表示条件均值的倍数变化。若没有额外假设，不能把这种关系解释为因果效应。

## 普通与带惩罚模型入口

`GeneralizedLinearModel` 是普通广义线性模型（GLM）的统一入口，适用于高斯、二项、Poisson 等常见分布族。`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` 等专用模型类用于选择相应的分布族，并设置该模型支持的链接函数等选项。

当需要通过惩罚项收缩系数或进行变量选择时，可以使用 `PenalizedGeneralizedLinearModel`，或 `PenalizedLinearRegression`、`PenalizedLogisticRegression`、`PenalizedPoissonRegression` 等具体封装类。`Ridge`、`Lasso`、`ElasticNet` 是遵循 sklearn 风格的便捷封装，适合希望沿用这类接口习惯的用户；它们拟合的仍是带惩罚的高斯回归。

本页用“普通 GLM”区分上述两类接口，并不表示它默认执行无惩罚拟合：普通 GLM 默认的 `solver="auto"` 使用 IRLS，配合默认 `C=1` 会收缩斜率。若要进行无惩罚拟合，可设 `C=0`；不同求解器对 `C` 的处理见后文“目标函数与 `sample_weight`”。下面的 CPU 示例显式使用无惩罚 Newton 拟合。

<a id="cpu-example"></a>

## 完整 CPU 示例

下面用 Poisson 模型学习两个特征与期望计数之间的关系。依次准备计数数据、拟合模型、评价留出数据，最后按需计算系数区间。示例不需要 GPU 或公式依赖。

### 1. 导入

<!-- learner-example: glm-poisson -->
```python
import numpy as np
from statgpu import GeneralizedLinearModel
```

### 2. 准备输入和留出数据

`X` 的形状为 `(240, 2)`：行对应观测，列对应两个数值特征。`y` 是长度 240 的非负整数计数。对数链接使条件均值 `exp(0.3 + X @ beta)` 为正；它不是事件概率。实际数据应先处理缺失值，预测时保持训练列顺序。

```python
rng = np.random.default_rng(59)
X = rng.normal(size=(240, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
```

前 180 行用于训练，最后 60 行留出。这个加权示例还为每个训练观测指定一个有限、非负分析权重；它调整观测对拟合目标的贡献，不是重复行数或重抽样概率。无权重数据可省略 `sample_weight`。

```python
weights = np.linspace(0.5, 2.0, 180)
```

### 3. 拟合无惩罚模型

这里选择 Newton 求解器进行无惩罚拟合。Newton 不使用 `C`，示例仍写出 `C=0`，以便改用 IRLS 时也保持无惩罚目标；普通 GLM 默认的 auto/IRLS 配合正 `C` 会加入岭惩罚。

```python
model = GeneralizedLinearModel(
    family="poisson", C=0, solver="newton", device="cpu",
    max_iter=1000, tol=1e-8, compute_inference=False,
).fit(X[:180], y[:180], sample_weight=weights)
print("Slopes:", np.round(model.coef_, 3))
print("Mean multipliers:", np.round(np.exp(model.coef_), 3))
```

该种子下斜率约为 `[0.402, -0.327]`，均值倍数约为 `[1.495, 0.721]`。保持另一特征不变，第一个特征增加一单位对应期望计数约增加 49.5%；第二个特征增加一单位对应期望计数约降低 27.9%。这些是关联，不是因果效应。

### 4. 预测并评价留出数据

```python
mean_prediction = model.predict(X[180:])
heldout_loss = np.mean(mean_prediction - y[180:] * np.log(mean_prediction))
print("Predicted means:", np.round(mean_prediction[:3], 3))
print("Held-out Poisson loss:", round(float(heldout_loss), 3))
```

预测数组的形状为 `(60,)`，每个值都是相应观测的预测均值，为正且可为小数；它不必像观测计数一样是整数。留出集损失约为 `0.556`，省略只与响应有关的对数阶乘项。比较不同模型时，应使用相同的留出响应和权重，损失越小表示预测越好；它不是准确率或普通 R²。该普通 GLM 类没有 `score` 或 `predict_proba` 方法。

### 5. 可选：系数区间

继续使用第 3 步的训练数据，开启推断后重新拟合。`cov_type="hc1"` 使用 HC1 得分稳健协方差。协方差选择不改变拟合均值；适用条件与其他选项见后文推断一节。

```python
model.set_params(compute_inference=True, cov_type="hc1")
model.fit(X[:180], y[:180], sample_weight=weights)
print("Interval shape:", model._conf_int.shape)
```

区间数组的形状为 `(3, 2)`，三行依次对应截距和两个特征的斜率，两列分别为区间下限与上限。这些是渐近边际系数区间，不是未来计数的预测区间。`summary()` 返回字符串，可用 `print(model.summary())` 显示。
<!-- example-end: glm-poisson -->

## 目标函数与 `sample_weight`

拟合模型时，需要先明确优化的是无惩罚损失，还是加入正则化项的目标；权重则决定各条观测对数据拟合损失的贡献。

设 n 为观测数，$x_i$ 为特征向量，$\beta$ 为斜率向量，b 为不受惩罚的截距。记 $\eta_i=b+x_i^\top\beta$；`fit_intercept=False` 时令 b=0。**无惩罚**的数据拟合目标为相应分布族的平均负对数似然，采用该分布族的离散度约定：

$$
L(b,\beta)=\frac{1}{n}\sum_{i=1}^n\ell(y_i,\eta_i).
$$

受支持的分析权重 `sample_weight=w` 将其改为

$$
L_w(b,\beta)=\frac{\sum_i w_i\ell(y_i,\eta_i)}{\sum_i w_i}.
$$

权重必须有限、非负且总和为正。零权重观测不贡献数据拟合损失；所有权重同乘一个正常数，不改变该目标。这里的 `sample_weight` 表示分析权重，而不是重复行数或重抽样概率。传入权重不会改变显式指定的求解器；不支持的组合会直接报错。

**普通 GLM 不等于自动采用无惩罚拟合。** 对本页列出的普通分布族，默认 `solver="auto"` 使用 IRLS。普通 IRLS 在 C 为正时加入斜率的岭惩罚：

$$
\min_{b,\beta} L_w(b,\beta)+\frac{1}{4C}\lVert\beta\rVert_2^2.
$$

无权重时以 L 代替 L_w。岭惩罚的梯度为 $\beta/(2C)$，不包含截距；默认 `C=1` 因而会收缩斜率。当前普通 IRLS 对非正 C 不施加此项，建议以 `C=0` 明确请求无惩罚 IRLS。普通 GLM 显式选择 `newton`、`lbfgs` 或 `fista` 时使用无惩罚损失，不使用 C。因此，若原来使用了正 C，更换普通 GLM 求解器可能改变统计问题。这里 `C` 的含义不同于独立的 `LogisticRegression`，也不同于下文的 `alpha` 接口。

**带惩罚 GLM** 则使用所声明的惩罚项：

$$
\min_{b,\beta}L(b,\beta)+\alpha P(\beta),\qquad
\text{或}\quad\min_{b,\beta}L_w(b,\beta)+\alpha P(\beta).
$$

截距仍不受惩罚。不同惩罚项由 `penalty` 选择，`alpha` 控制惩罚强度。Cox 偏似然、稳健损失和分位数损失的统计定义见各自模型页。

## 求解器选择

确定模型和惩罚方式后，一般可以先使用 `solver="auto"`，让模型选择求解器。普通 GLM 默认使用上文说明的 IRLS；下表描述的是**带惩罚估计器**的自动选择规则。两者是不同接口，普通 GLM 不会因为正 `C` 加入岭惩罚就改用下表的规则。

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` + NumPy/CPU | `exact` 闭式 L2 路径 |
| 平方误差 + L1/ElasticNet | FISTA/FISTA-BB 稀疏路径 |
| 逻辑回归/Poisson/Gamma/逆高斯/Tweedie/负二项 + L2 | Newton |
| 非凸 SCAD/MCP | FISTA + LLA 延续路径 |

若显式指定 `solver`，该请求不会因 `sample_weight` 而改变；如果对应组合不受支持，则拟合会报错。直接拟合稀疏高斯模型时，还需注意 `stopping="kkt"` 当前并不会切换停止条件，详见[直接拟合的控制参数限制](../../reference/linear-model-api.md#elasticnet)。

设备选择也遵循同样原则：显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA。公式解析可以在 CPU 上进行，但拟合与预测的数值计算仍使用选定的计算后端。

### 显式 Newton / L-BFGS 的加权支持

对于受支持的普通 GLM 分布族与链接函数组合，显式 `solver="newton"` 和
`solver="lbfgs"` 接受非均匀分析权重，并优化前面的归一化加权目标。
传入 `sample_weight` 不会更换指定的求解器或显式请求的 CUDA/Torch 设备；
不支持的组合会报错。均匀权重以及数值上等效的权重使用无权重目标。
加权更新和线搜索的数学说明见[求解器算法](../../guides/solver-algorithms.md)。

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
模型页及[兼容性矩阵](../../guides/solver-penalty-matrix.md)。

带惩罚的光滑 GLM 使用同一套分析权重定义。受支持的 L2 拟合使用 Newton 还是
L-BFGS，仍由直接拟合或交叉验证的调度规则决定，而不是仅由是否提供权重决定。

## 协方差与推断

系数推断用于量化估计的不确定性，与前面评价留出预测是不同的问题。普通 GLM 可通过 `compute_inference=True` 开启支持的系数推断，并用 `cov_type` 选择 `nonrobust`、`hc0` 或 `hc1`。推断与拟合使用相同的分析权重定义；是否支持权重仍取决于具体推断方法。CPU 示例展示了 HC1 系数区间的读取方式。

### 带惩罚模型的推断

对于通用的 `PenalizedGeneralizedLinearModel` 及各类专用的带惩罚 GLM 封装，推荐使用 `inference_method="auto"`。受支持的 M-估计、去偏与残差自助法结果会记录推断方法、目标及调参或选择条件。当前 `post_selection_ols` 即使返回了推断结果，`inference_method_` 和 `inference_target_` 仍可能为空；请读取 `_inference_result.method` 及其元数据，详见[选择后结果报告](../../guides/inference-modes.md#post_selection_ols)。

对于受支持的光滑非高斯 L2/无惩罚模型，`auto` 会选择固定惩罚强度下的 `m_estimation`。正 L2 惩罚对应带惩罚的估计方程；无惩罚设置按惩罚强度为 0 的 L2 处理，对应无惩罚总体参数。当前协方差支持 `nonrobust`、`hc0`、`hc1`；HC2/HC3/HAC 在这一路径中不可用，并会明确报错。

受支持的 M-估计与高斯残差自助法复用拟合的后端和设备。非高斯 L1/ElasticNet 系数推断不受支持。SCAD/MCP `oracle` 需要显式请求，但当前非高斯子模型重建会把分布族设置和正则化重置为默认值，并自动选择子模型设备。不要把这些结果解释为预期的无惩罚活跃集推断，详见 [oracle 限制及诊断性重拟合替代方案](../../guides/penalized-glm-inference.md#current-non-gaussian-oracle-limitation)。分组惩罚目前只提供估计。

### 高斯稀疏模型的残差自助法

对于受支持的高斯稀疏惩罚，`inference_method="bootstrap"` 表示 `cov_type="nonrobust"` 下的**无权重残差自助法**。设计矩阵和拟合后的调参配置保持固定：每次从残差中有放回抽样，把抽到的残差加到拟合值上，构造新的自助响应，再用同一个带惩罚模型重新拟合。`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制随机数可复现性。

CPU 拟合使用 NumPy；CuPy/Torch CUDA 拟合会把自助法重拟合留在同一个 GPU 设备。最终推断数组统一返回 NumPy。带权残差自助法、稳健/HC 协方差对应的自助法、HAC/分块自助法与非高斯自助法当前都不支持。

完整支持矩阵、重采样边界和统计解释见 [带惩罚 GLM 推断](../../guides/penalized-glm-inference.md) 与 [推断模式](../../guides/inference-modes.md)。

## 参数

下表精选了不同模型的控制参数，不代表一个共同构造器。`family` 属于普通 GLM，通用带惩罚类使用 `loss`；`formula` 与 `data` 是 `fit` 的参数。完整的实际构造器与方法见[普通 GLM](../../reference/linear-model-api.md#generalizedlinearmodel)、[通用带惩罚 GLM](../../reference/linear-model-api.md#penalizedgeneralizedlinearmodel)及[通用 CV](../../reference/linear-model-api.md#penalizedglm_cv)参考。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `family` | 模型相关 | GLM 分布族，例如 `"gaussian"`、`"binomial"`、`"poisson"` |
| `penalty` | 通用直接模型为 `"l1"`；通用 CV 为 `"l2"` | `none`、`l1`、`l2`、`elasticnet` 以及受支持的结构化/分组惩罚；可用组合见[兼容性矩阵](../../guides/solver-penalty-matrix.md) |
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

先完成 [CPU 示例](#cpu-example)，再复用其中的导入、`X`、`y` 和 `weights`，可按下面的方式要求 CuPy CUDA 完成同一带权 Poisson 拟合。`device="torch"` 对应 Torch CUDA；显式请求的后端不可用时会报错。

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
```

这个独立示例中，`df` 的每行是一条观测，`count` 为响应，`x` 为连续特征，`group` 为分类特征。

```python
rng = np.random.default_rng(18)
df = pd.DataFrame({"x": rng.normal(size=80), "group": ["a", "b"] * 40})
df["count"] = rng.poisson(np.exp(0.2 + 0.3 * df["x"]))
```

公式负责生成截距和分类编码；预测时传同样列名和类别水平的 DataFrame。

```python
model = GeneralizedLinearModel(family="poisson", C=0, device="cpu")
model.fit(formula="count ~ x + C(group)", data=df)
prediction = model.predict(df.iloc[:5])
assert prediction.shape == (5,)
```
<!-- example-end: glm-formula -->


公式在 CPU 上解析。应使用 `formula`/`data` 传入数据，不要同时传入数组 `X`/`y`。权重可以对应原始全部行，也可以对应公式及缺失值处理后保留的行，按位置对齐。预测会重建训练时的列和类别水平。数据量较大时，可直接使用数组以避免公式解析开销。

这些 GLM 接口没有 `offset`/`exposure` 参数。如果不同观测的计数来自不同暴露量，不能把暴露量作为普通特征加入公式来代替 offset：普通特征的系数会被估计，而 offset 的系数应固定为 1。

普通 GLM 公式预测目前会删除预测变量缺失的行，并返回较短且无行标签的数组。
应处理缺失并核对输出长度，再将预测配给观测；详见[缺失行限制](../../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms)。

### 重拟合失败后的限制

普通 auto/IRLS/FISTA 路径在重拟合失败后，可能混用旧系数和新的观测数、公式或截距设置，对象却仍显示为已拟合；预测和似然诊断也可能随之改变。遇到这类失败后应新建估计器，成功拟合后再读取结果，详见[失败后的完整约定](../../reference/linear-model-api.md#failed-ordinary-glm-refits)。显式 Newton/L-BFGS 目前会保留上一次拟合，但这仍不代表新数据拟合成功。

## 严格与近似交叉验证

使用带惩罚模型时，可以通过交叉验证选择惩罚强度 `alpha`，以验证损失比较候选值，而不是只比较训练误差。

`PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`。严格模式下，每个交叉验证折和每个 `alpha` 都使用用户给定的 `max_iter` 与 `tol`。

可选的 `cv_strategy="two_stage"` 会先用较宽松的求解条件筛选 `alpha` 网格，再严格复核候选 `alpha`，并进行严格的最终重拟合。由于第一阶段可能在非常接近的交叉验证曲线上改变 `alpha` 排名，该模式默认发出 `ApproximateCVWarning`；如果用户明确接受这种近似，可以传入 `acknowledge_approx=True`。

下面只创建配置对象，尚未拟合；下一节将用同一份训练数据演示完整的 CPU 交叉验证。

```python
from statgpu.linear_model import PenalizedGLM_CV

strict_cv = PenalizedGLM_CV(
    loss="poisson", penalty="elasticnet",
    cv_strategy="strict", device="cpu",
)
```

保留上一代码块的导入。仅在接受初筛可能影响候选排名后，选择两阶段配置：

```python
fast_cv = PenalizedGLM_CV(
    loss="poisson", penalty="elasticnet",
    cv_strategy="two_stage", acknowledge_approx=True,
    refine_top_k=3, device="cpu",
)
```


<a id="reading-cv-inference-results"></a>

## 读取交叉验证后的推断结果

通用 `PenalizedGLM_CV` 在固定 `l1_ratio` 下选择 `alpha`，再用全部训练行重拟合。它与还能搜索 `l1_ratio` 的 `ElasticNetCV` 不同。通用 CV 对标量响应拟合截距，但不提供公开的 `fit_intercept` 选项。最终模型的 `score` 返回响应尺度上的 R²，`best_score_` 则是交叉验证损失的负值，两者含义不同。

交叉验证完成后，可以从 `estimator_` 读取最终重拟合的推断结果。当前不要用 `summary()` 获取这一报告：它会调用最终估计器不存在的方法，从而抛出 `AttributeError`。下面演示直接读取结果容器。

先完成 [CPU 示例](#cpu-example)，再复用其中的 NumPy 导入、`X`、`y` 与 `weights`；不再生成另一份数据。

<!-- example-requires: glm-poisson -->
<!-- learner-example: glm-cv-inference -->
```python
from statgpu import PenalizedGLM_CV
```

候选网格包含两个 L2 强度，每个使用两折验证。仅把原 CPU 示例的前 180 行交给 CV，最后 60 行继续留出。同时传入这 180 条训练观测对应的分析权重；留出行不参与调参或最终重拟合。

```python
model = PenalizedGLM_CV(
    loss="poisson", penalty="l2", alpha_grid=[0.05, 0.2],
    cv=2, device="cpu", compute_inference=True,
).fit(X[:180], y[:180], sample_weight=weights)
print("Selected alpha:", model.alpha_)
```

选择完成后，系数推断来自所选 `alpha` 下的全训练集重拟合。读取结果容器，并区分“验证损失”和它的负值 `best_score_`：

```python
report = model.estimator_._inference_result.to_dict()
print("Method:", report["method"])
print("Standard errors:", np.round(model.estimator_._bse, 3))
assert model.cv_results_["all_scores"].shape == (2, 2)
assert np.isclose(model.best_score_, -np.min(model.cv_results_["mean_score"]))
```
<!-- example-end: glm-cv-inference -->


该例返回 `m_estimation` 及三个有限标准误。推断以所选 `alpha` 为条件，不校正调参不确定性。最终的通用估计器没有 `predict_proba` 或 `summary` 方法；使用 `loss="logistic"` 时，预测返回 0/1 标签，不是均值概率。若需要概率方法，应使用适当的专用模型。完整默认值、数组形状及 CV 结果键见 [API 参考](../../reference/linear-model-api.md#penalizedglm_cv)。

## 公开入口

以下名称可从 `statgpu.linear_model` 导入。

带惩罚 GLM 的系数推断见 [带惩罚 GLM 推断](../../guides/penalized-glm-inference.md)；完整求解器表见 [求解器 × 惩罚项兼容性矩阵](../../guides/solver-penalty-matrix.md)。

[GLM 专用类完整构造参数与方法](../../reference/linear-model-api.md#typed-glm-constructors)
列出分布族的链接、离散参数、幂次，以及不同惩罚封装的默认值。
名称相近不代表可以互换构造参数。

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

## 相关文档

- [求解器 × 惩罚项兼容性矩阵](../../guides/solver-penalty-matrix.md) — 损失函数 × 惩罚项 × 求解器的完整调度表、交叉验证路径和推断支持状态。
- [求解器算法](../../guides/solver-algorithms.md) — Newton、L-BFGS、FISTA 等求解器的算法说明。
- [损失函数](../../reference/losses.md) — 底层损失函数接口及其能力边界。

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
