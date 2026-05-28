# statgpu 自动统计分析 Agent 设计笔记

> 日期：2026-05-28  
> 相关文件：`nature_agent_report.md`、`statgpu/statgpu/agent/`、`statgpu/`  
> 当前定位：第一版可运行的 deterministic statistical agent，不是完整 LLM 自进化科学家系统。

## 1. 为什么要做这个 agent

`statgpu` 已经有一批 GPU/CPU 后端统一的统计模型：线性回归、岭回归、Logistic 回归、Poisson 回归、CoxPH、生存分析、PCA、KMeans 等。它的问题不是“缺一个模型”，而是用户拿到一份数据时仍然需要手动完成一串判断：

- 这份数据是回归、二分类、计数、生存分析，还是无监督探索？
- 哪些列能作为特征，哪些列是 target、time、event？
- 缺失值、分类变量、二值事件变量该怎样变成模型能吃的矩阵？
- 应该调用 `statgpu` 的哪个 estimator？
- 拟合后哪些统计量要被抽取出来？
- 结果是否可信，有没有 rank deficiency、样本量过小、类别不平衡、事件数不足等风险？
- 最后怎样形成一个可审计的报告？

这个 agent 的目标就是把这些重复工作变成一个可追踪的自动流水线：

```text
输入数据
  -> profile / preprocess
  -> infer task
  -> plan statgpu methods
  -> run models
  -> validate outputs and assumptions
  -> produce dict / markdown report
```

它不是要替代统计判断，而是要先给出一份“可运行、可追踪、可复查”的初始分析。用户仍然应该确认任务定义、变量含义、模型假设和结论边界。

## 2. 从 Nature agent 报告中抽象出的设计原则

`nature_agent_report.md` 里列出的系统应用领域不同，但它们有一些稳定共性。我没有照搬某个系统，而是把这些共性翻译成适合统计分析库的工程结构。

| 报告中的 agent 思想 | 代表系统 | 在 `statgpu.agent` 中的落地 |
| --- | --- | --- |
| 多智能体分工，而不是一个大模型包办全部事情 | Robin、Co-Scientist、BioMedAgent、PrimeGen | 当前实现为逻辑多 agent：`profiler`、`planner`、`statgpu_runner`、`self_validator`、`reporter` |
| 目标驱动循环：规划、执行、解释 | CellVoyager | `analyze()` 统一执行 profile -> plan -> run -> validate -> report |
| 工具感知，把专业工具串成 workflow | BioMedAgent、PrimeGen | 不重新实现统计模型，而是调度 `statgpu.linear_model`、`statgpu.survival`、`statgpu.unsupervised` |
| 自验证 / 自反思，减少幻觉或错误输出 | GeneAgent、DeepRare | `_validate()` 检查样本量、秩、条件数、类别不平衡、事件数、模型失败和缺失 p 值 |
| 可追溯推理 | DeepRare | 用 `DataProfile`、`AnalysisPlan`、`ModelResult`、`AnalysisResult` 记录每一步的输入摘要、计划、模型输出、警告和建议 |
| LLM 无关设计 | DeepRare | 当前 agent 不依赖远程 LLM；结果由 deterministic rules + statgpu estimators 产生 |
| Human-in-the-loop | Robin、CRISPR-GPT、PrimeGen | 输出 `warnings` 和 `recommendations`，提醒用户确认任务和模型假设，而不是自动声称科学发现 |
| 中央协调器 + 专业工具层 + 数据源 | DeepRare 的三层架构 | `StatGPUAnalysisAgent` 是中央 host；`statgpu` estimators 是工具层；用户数据 / CSV 是外部数据源 |

最重要的一点是：Nature agent 的核心不是“写一个 prompt”，而是“用清晰的角色分工，把专业工具组织成可执行闭环，并且对结果进行验证”。当前 `statgpu.agent` 正是按这个思路写的。

## 3. 当前 agent 的总体架构

代码入口在：

- `statgpu/statgpu/agent/_analysis.py`
- `statgpu/statgpu/agent/__init__.py`
- `statgpu/statgpu/agent/__main__.py`

核心类是：

```python
from statgpu.agent import StatGPUAnalysisAgent

agent = StatGPUAnalysisAgent(device="auto")
result = agent.analyze(X=X, y=y, task="auto")
print(result.to_markdown())
```

对外暴露的对象包括：

- `StatGPUAnalysisAgent` / `AutoAnalysisAgent`
- `DataProfile`
- `AnalysisPlan`
- `ModelResult`
- `AnalysisResult`

其中 `StatGPUAnalysisAgent` 负责组织流程，其他 dataclass 负责保存可审计产物。

## 4. 逻辑多 agent 分工

当前没有把每个 agent 拆成独立进程或独立 LLM，而是在一个 deterministic coordinator 里实现多 agent 角色。这是第一版最稳妥的工程形态，因为它：

- 不引入额外 LLM 依赖；
- 不破坏 `statgpu` 现有核心包依赖；
- 输出可复现；
- 方便后续把某个阶段替换成 LLM planner、RAG validator 或外部 benchmark agent。

### 4.1 `profiler`

对应代码：

- `_prepare_data()`
- `_prepare_table()`
- `_build_profile()`

职责：

- 接受数组、dict/list-of-dicts、DataFrame-like 对象或 CSV；
- 识别 numeric feature 和 categorical feature；
- 对 numeric feature 做 median imputation；
- 对 categorical feature 做 one-hot encoding；
- 记录 reference level；
- 处理常见 missing string，例如 `NA`、`nan`、`null`、空字符串；
- 处理 survival 的 `time` / `event`；
- 形成 `DataProfile`。

示例 profile 内容：

```text
n_samples
n_features
task_type
feature_names
target_name
dropped_rows
imputed_values
encoded_features
target_summary
notes
```

这一步相当于 Nature agent 中的“信息收集阶段”。区别是这里收集的是统计建模所需的数据结构信息，而不是医学知识库或文献。

### 4.2 `planner`

对应代码：

- `_infer_task()`
- `_build_plan()`

任务推断规则：

```text
如果提供 time + event -> survival
如果没有 y -> unsupervised
如果 y 只有两个取值 -> binary_classification
如果 y 是非负整数计数，并且 unique 数量符合 count 数据特征 -> poisson
否则 -> regression
```

方法规划规则：

| task | 规划方法 |
| --- | --- |
| `regression` | `LinearRegression`，可选 `Ridge`，加 `PCA(diagnostic)` |
| `binary_classification` | `LogisticRegression`，加 `PCA(diagnostic)` |
| `poisson` | `PoissonRegression`，加 `PCA(diagnostic)` |
| `survival` | `CoxPH`，加 `PCA(diagnostic)` |
| `unsupervised` | `PCA` + `KMeans` |

`AnalysisPlan` 会记录：

- 任务类型；
- 本轮逻辑 agent 列表；
- 将要使用的 `statgpu` 方法；
- 为什么这么做。

这对应 CellVoyager 的“规划 -> 执行 -> 解读”，也对应 BioMedAgent 的 tool-aware workflow construction。

### 4.3 `statgpu_runner`

对应代码：

- `_run_regression()`
- `_run_binary_classification()`
- `_run_poisson()`
- `_run_survival()`
- `_run_unsupervised()`
- `_run_pca_diagnostic()`

这一层真正调用 `statgpu` 项目的 estimator：

```python
from statgpu.linear_model import LinearRegression, LogisticRegression, PoissonRegression, Ridge
from statgpu.survival import CoxPH
from statgpu.unsupervised import KMeans, PCA
```

它不会绕过 `statgpu` 自己写一套统计模型。这样做有几个好处：

- 复用 `statgpu` 现有 CPU / CuPy / Torch 后端；
- 复用 `statgpu` 现有 `device="auto" / "cpu" / "cuda" / "torch"` 语义；
- 复用模型已有的 inference 输出，例如 `_bse`、`_pvalues`、`_conf_int`；
- 复用已有模型的 `score()`、`aic()`、`bic()`、`roc_auc_score()`、`classification_table()`、C-index 等能力；
- 避免 agent 层变成另一个统计库。

### 4.4 `self_validator`

对应代码：

- `_validate()`
- Logistic 回归里的 fallback retry 逻辑

目前验证内容包括：

- 样本量是否小于 30；
- 特征数是否接近或超过样本数；
- 设计矩阵 rank 是否低于 feature 数；
- 条件数是否过高；
- 二分类 target 是否严重不平衡；
- 生存分析 observed events 是否相对模型规模过少；
- 是否所有模型都拟合失败；
- 某些模型是否没有返回 coefficient p-values；
- 提醒训练集 metric 不是 holdout/cross-validation 结果。

另外，`_run_binary_classification()` 有一个轻量自修正：

```text
先尝试近似无正则 LogisticRegression(C=1e10)
如果失败，再尝试 LogisticRegression(C=1.0)
```

这不是 GeneAgent 那种完整的“生成 -> 验证 -> 修改 -> 总结”多轮机制，但已经把自验证思想落成了第一版可运行逻辑：发现模型失败后自动换一个更稳健的拟合配置，并把 warning 写入结果。

### 4.5 `reporter`

对应代码：

- `AnalysisResult.to_dict()`
- `AnalysisResult.to_markdown()`
- `AnalysisResult.save_markdown()`

报告输出包括：

- data profile；
- agent plan；
- 每个模型的 metrics；
- coefficient table；
- confidence interval；
- Logistic 的 odds ratio；
- CoxPH 的 hazard ratio；
- PCA explained variance；
- KMeans cluster sizes；
- validation warnings；
- recommended next checks。

这对应 Nature agent 里的“总结模块”和“traceable reasoning”。重点不是生成漂亮文字，而是把统计分析过程压缩成用户能审阅的结构化证据。

## 5. 它如何结合 `statgpu` 项目

### 5.1 复用 `statgpu` 的核心模型

当前 agent 对不同任务的模型映射如下：

| 自动识别任务 | 使用的 `statgpu` 模型 | 主要输出 |
| --- | --- | --- |
| 连续型回归 | `LinearRegression` | R2/score、AIC/BIC、coef、SE、p-value、CI |
| 连续型回归的稳健备选 | `Ridge(alpha=1.0)` | score、coef、可用时 inference |
| 二分类 | `LogisticRegression` | accuracy、ROC-AUC、precision、recall、F1、coef、OR、p-value |
| 计数模型 | `PoissonRegression` | score、mean Poisson deviance、coef |
| 生存分析 | `CoxPH` | C-index、log likelihood、coef、HR、p-value |
| 无监督探索 | `PCA` | explained variance ratio、top loading |
| 无监督聚类 | `KMeans` | inertia、cluster size |

### 5.2 继承 `statgpu` 的 device 语义

agent 初始化时接受：

```python
StatGPUAnalysisAgent(device="auto")
StatGPUAnalysisAgent(device="cpu")
StatGPUAnalysisAgent(device="cuda")
StatGPUAnalysisAgent(device="torch")
```

这个参数直接传给底层 estimator。也就是说 agent 不自己决定 GPU 细节，而是遵守 `statgpu` 已经定义的规则：

- `device="cpu"` 使用 NumPy；
- `device="cuda"` 使用 CuPy，并且显式 CUDA 不可用时应报错；
- `device="torch"` 使用 Torch CUDA；
- `device="auto"` 才允许自动选择可用后端。

这符合 `AGENTS.md` 里的核心约束：新增功能应复用 `BaseEstimator`、后端抽象和现有 device 语义，不应该在业务层散落新的 backend 逻辑。

### 5.3 复用 robust inference 配置

agent 初始化有：

```python
StatGPUAnalysisAgent(cov_type="hc3")
```

对支持 robust covariance 的模型，agent 会传入 `cov_type`。当前默认是 `hc3`，原因是自动分析场景通常缺乏用户对异方差结构的确认，`hc3` 比 classical nonrobust covariance 更保守。支持范围仍由底层 `statgpu` 模型决定。

### 5.4 抽取 `statgpu` 的 inference 产物

agent 不重新计算标准误和 p 值，而是从 estimator 中读取已有字段：

```text
_bse
_pvalues
_conf_int
_zvalues
_tvalues
coef_
intercept_
```

然后统一转成 `ModelResult.coefficients`。这样报告层可以用统一表格展示不同模型：

```text
term
estimate
std_error
statistic
p_value
ci_low
ci_high
odds_ratio / hazard_ratio
```

### 5.5 不污染核心统计模型

第一版 agent 是新增包：

```text
statgpu/statgpu/agent/
```

顶层导出只是在 `statgpu/__init__.py` 加了：

```python
from .agent import StatGPUAnalysisAgent, AutoAnalysisAgent
```

这意味着 agent 是一层 orchestration，不会改变现有模型的行为，也不会改变 `LinearRegression`、`CoxPH` 等核心 estimator 的 API。

## 6. 运行方式

### 6.1 Python API

数组输入：

```python
import numpy as np
from statgpu.agent import StatGPUAnalysisAgent

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5))
y = X[:, 0] - 2 * X[:, 1] + rng.normal(size=1000)

agent = StatGPUAnalysisAgent(device="auto")
result = agent.analyze(
    X=X,
    y=y,
    target="outcome",
    feature_names=["x1", "x2", "x3", "x4", "x5"],
)

print(result.to_markdown())
```

表格输入：

```python
from statgpu.agent import StatGPUAnalysisAgent

rows = [
    {"age": 63, "sex": "M", "marker": 1.2, "case": 1},
    {"age": 58, "sex": "F", "marker": 0.7, "case": 0},
]

result = StatGPUAnalysisAgent(device="cpu").analyze(
    data=rows,
    target="case",
    task="auto",
)
```

生存分析：

```python
result = StatGPUAnalysisAgent(device="auto").analyze(
    data=df,
    time="time",
    event="event",
    task="survival",
)
```

### 6.2 CLI

`pyproject.toml` 中加了脚本入口：

```text
statgpu-agent = "statgpu.agent.__main__:main"
```

安装后可以运行：

```bash
statgpu-agent data.csv --target outcome --device auto --output report.md
```

或者直接：

```bash
python -m statgpu.agent data.csv --target outcome --device cpu
```

## 7. 为什么这一版没有直接做成 LLM agent

从 Nature 报告看，很多系统都用 LLM 做 planner 或 controller。但对 `statgpu` 来说，第一版直接上 LLM 不一定是正确顺序，原因有三点：

1. **统计分析首先需要可复现。**  
   同一份数据、同一参数应该给出同样的模型计划、同样的警告和同样的报告结构。

2. **`statgpu` 是底层统计库，不应强制依赖远程模型服务。**  
   项目核心依赖目前是 NumPy/SciPy/joblib，pandas 也是 optional extra。把 LLM 作为硬依赖会改变包定位。

3. **LLM 更适合放在 planner/report 层，而不是 estimator 层。**  
   当前 deterministic agent 已经把流程边界划清楚。后续如果要接 LLM，可以只替换 `_build_plan()` 或增加自然语言解释层，不需要改模型执行层。

所以当前 agent 是一个“agentic workflow engine”，不是“LLM scientist”。这也是有意设计的边界。

## 8. 当前边界和不足

这版 agent 能跑通自动分析骨架，但还不是完整科学发现系统。主要限制包括：

- 没有自然语言任务解析，用户仍然需要通过 `target`、`time`、`event` 或 `task` 给出基本意图；
- 没有持久化 memory，也没有 BioMedAgent 那种自进化记忆检索；
- 没有自动外部基线验证，例如自动调用 statsmodels、sklearn 或 R 做 strict parity check；
- 默认 metric 是训练集诊断，不是交叉验证或外部测试集表现；
- 分类变量只做基础 one-hot，高基数变量会截断到 top categories；
- 没有因果推断、实验设计或假设生成模块；
- 没有自动画图；
- 没有把每个逻辑 agent 拆成独立插件或服务；
- 尚未实现多轮“如果诊断失败则重新规划整个 workflow”的完整闭环。

这些限制都应该在报告的 `warnings` / `recommendations` 中明确提示，而不是隐藏起来。

## 9. 下一步可以怎么升级

如果继续沿着 Nature agent 的方向推进，建议按下面顺序做。

### 9.1 增强 self-verification

目标：接近 GeneAgent 的“生成 -> 验证 -> 修改 -> 总结”。

可做内容：

- 对回归自动检测 rank deficiency 后切换 Ridge/Lasso；
- 对 Logistic separation 自动切换正则化或 penalized GLM；
- 对 CoxPH 事件数不足自动降低模型复杂度；
- 对高维数据自动触发 feature selection 或 PCA regression；
- 增加 `validation_trace`，记录每次失败、修正和重跑。

### 9.2 增加 external baseline agent

目标：贴合 `statgpu` 项目一贯的 statsmodels/sklearn/R 对齐风格。

可做内容：

- 小数据自动用 statsmodels 对齐 OLS/Logit/Cox；
- 聚类和 PCA 自动用 sklearn 做 sanity check；
- 输出 coef diff、p-value diff、metric diff；
- 保留 JSON artifact。

### 9.3 增加自然语言 planner

目标：接近 BioMedAgent 的自然语言接口，但不破坏核心可复现性。

建议设计：

```text
user natural language
  -> LLM parses task spec
  -> deterministic StatGPUAnalysisAgent executes spec
  -> validator audits outputs
  -> LLM only summarizes with citations to structured result
```

LLM 不直接改数据、不直接编造结果，只能生成受 schema 约束的 task spec。

### 9.4 增加 notebook / report agent

目标：接近 CellVoyager 的 notebook 风格自动分析。

可做内容：

- 自动生成可运行 notebook；
- 每个 cell 对应一个 agent stage；
- 保存代码、输出、图和解释；
- 用户可以逐步审阅和修改。

### 9.5 增加 memory

目标：借鉴 BioMedAgent 的 self-evolving 能力，但先做轻量版。

可做内容：

- 记录某类数据集上成功的 workflow；
- 记录模型失败原因和修正策略；
- 记录用户确认过的 column role；
- 下次同结构数据自动复用。

## 10. 总结

这个 `statgpu.agent` 的核心思想是：

```text
把 Nature agent 中的“多角色分工、工具感知、自验证、可追溯报告”
落到 statgpu 的统计建模场景中。
```

它不是新造统计模型，而是调度 `statgpu` 已有 estimator；不是远程 LLM agent，而是 deterministic orchestration agent；不是最终科学结论生成器，而是自动产出第一版统计分析、诊断和报告的工具。

从工程上看，它给 `statgpu` 增加了一层“自动数据分析入口”：

```text
用户数据
  -> StatGPUAnalysisAgent
  -> statgpu estimators
  -> structured result
  -> markdown report
```

这层入口后续可以继续接入 LLM planner、外部 baseline validator、notebook generator 和 persistent memory，但第一版最重要的是把统计自动化的边界先做清楚：可运行、可复现、可审计、可扩展。
