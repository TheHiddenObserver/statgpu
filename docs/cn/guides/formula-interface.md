# Formula 接口与支持矩阵

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/guides/formula-interface.md)

Formula 是建立在 pandas 与 Patsy 上的便利层。它先在 CPU 上构建带名称的设计矩阵并保存编码信息，再把数值数组交给所选 statgpu 后端。

## 安装与第一次拟合

```bash
pip install "statgpu[formula]"
```

```python
import pandas as pd
from statgpu.linear_model import LinearRegression

data = pd.DataFrame({
    "y": [1.2, 2.1, 2.8, 4.2, 5.0],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0],
    "group": ["a", "a", "b", "b", "b"],
})

model = LinearRegression(device="cpu").fit(
    formula="y ~ x + C(group)",
    data=data,
)
prediction = model.predict(data.iloc[:2])
```

常规代码中应选择 `formula + data` 或 `X + y`。Formula 默认包含截距；`y ~ x - 1` 或 `y ~ 0 + x` 可移除截距。

## 常用语法

| 语法 | 含义 |
|---|---|
| `y ~ x1 + x2` | 加性主效应 |
| `y ~ C(group)` | 分类变量编码 |
| `y ~ x1:x2` | 仅交互项 |
| `y ~ x1 * x2` | 两个主效应及其交互项 |
| `y ~ np.log(x)` | Patsy 可求值的变换 |
| `y ~ x - 1` | 不含截距 |

`predict` 接收 DataFrame 时会复用拟合时的设计信息，使分类编码与交互项列保持一致。新数据必须包含兼容的源列与分类水平。

## 已核对的支持矩阵

下表来自当前 `fit` 签名与继承关系，不表示所有估计器都支持 Formula。

| 领域 | 支持 Formula 的估计器 | 特殊语法或说明 |
|---|---|---|
| 普通线性模型 | `LinearRegression` | 标准 Patsy Formula |
| 正则化线性模型 | `Ridge`、`Lasso`、`ElasticNet`、`AdaptiveLasso`、`SCADRegression`、`MCPRegression`、`PenalizedLinearRegression` | 标准 Patsy Formula |
| 普通 GLM | `GeneralizedLinearModel`、`PoissonRegression`、`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` | 标准 Patsy Formula |
| 惩罚 GLM/loss | `PenalizedGeneralizedLinearModel` 以及类型化 linear、logistic、Poisson、Gamma、inverse-Gaussian、negative-binomial、Tweedie、quantile 与 robust 包装器 | 标准 Patsy Formula |
| 面板模型 | `PooledOLS`、`PanelOLS`、`BetweenOLS`、`RandomEffects`、`FirstDifferenceOLS`、`FamaMacBeth` | 支持 `y ~ x \| entity + time` 和面板 token |
| 生存模型 | `CoxPH` 与 `PenalizedCoxPHModel` | `Surv(time, event) ~ x` 或 `Surv(start, stop, event) ~ x` |

重要的仅数组接口包括独立 `LogisticRegression`、`QuantileRegression`、有序 GLM、`PenalizedGLM_CV`、大多数非参数/协方差/无监督估计器和预处理工具。某个估计器未出现在上表时，应检查其实际 `fit` 签名。

### Logistic Formula 替代入口

```python
from statgpu.linear_model import GeneralizedLinearModel

model = GeneralizedLinearModel(
    family="binomial",
    solver="newton",
    device="cpu",
).fit(
    formula="outcome ~ age + C(group)",
    data=data,
)
probability = model.predict(data)
```

若需要分类专属方法且可以显式提供设计矩阵，再使用独立 `LogisticRegression`。

### 面板 Formula 示例

```python
from statgpu.panel import PanelOLS

model = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=panel_frame,
)
```

竖线左侧是回归变量，右侧是固定效应标识。相应面板页还记录了标准 Formula、`EntityEffects` 与 `TimeEffects` 语法。

### 生存 Formula 示例

```python
from statgpu.survival import CoxPH

model = CoxPH().fit(
    formula="Surv(time, event) ~ age + C(treatment)",
    data=survival_frame,
)
```

Cox 偏似然不能识别截距，因此模型会移除截距列。

## 缺失值与旁路数组

Patsy 会删除公式所引用项中存在缺失值的行。statgpu 记录保留行的位置，并在验证前对齐支持的 `sample_weight` 等旁路数组。面板、Cox 和模型专属数组还有额外对齐规则；不要分别预删不同的行，应查阅相应模型页。

调试 Formula 展开时可直接查看解析结果：

```python
from statgpu.core.formula import FormulaParser

parser = FormulaParser("y ~ x + C(group)")
y_array, X_array, design_info = parser.eval(data)
print(parser.column_names)
print(parser.summary())
```

## CPU/GPU 边界

Formula 解析、DataFrame 和分类变量处理发生在 CPU。生成的稠密数组随后转换到显式 `cpu`、`cuda` 或 `torch` 后端进行模型计算。这属于预处理，不是静默设备回退。对反复运行的超大 GPU 拟合，预先构建并复用后端数组可避免重复解析和主机到设备传输。

## 常见失败

- 使用 `formula=` 却缺少 `data=` 会报错。
- 缺列、未见过的分类水平或不兼容变换会在 Patsy 求值或预测时失败。
- Formula 的截距语法优先于构造器中的 `fit_intercept`。
- 支持 Formula 不等于交叉验证也支持；`PenalizedGLM_CV` 仍是数组接口。
- 不要因为同模块另一个估计器支持 Formula，就推断当前估计器也支持。

## 实现与测试

- 解析器：`statgpu/core/formula/_parser.py`
- 旁路数组对齐：`statgpu/core/formula/_alignment.py`
- 面板扩展：`statgpu/panel/_formula.py`
- 核心测试：`statgpu/core/formula/tests/`
- 集成测试：`dev/tests/test_panel_formula.py` 与 `dev/tests/test_cox_phase1_completion.py`
