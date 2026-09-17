# statgpu 的交叉验证如何工作

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：交叉验证的公开设计与执行模型  
> 切换：[English](../../en/guides/cross-validation-design.md)

## 为什么需要这一页

[交叉验证](cross-validation.md) 主要回答“**怎样配置和使用** statgpu 的 CV estimator”。本页则解释这套接口背后的设计：为什么选择阶段与最终重拟合要分开，pathwise / GPU 加速可以在哪些位置进入，selection cache 可以安全复用什么，以及实现选择更快的执行方式时哪些统计语义必须保持不变。

这是一份面向用户的设计说明，不是源码实现文档。private helper 名称、cache key 的精确字段、backend 阈值、benchmark 得出的 cutoff 与 validation artifact 都属于仓库内部 design / validation 文档。

## 1. CV 的执行模型

可以把一个 CV estimator 理解为包在基础 estimator 外层的模型选择过程：

```text
模型 + tuning space + folds
          |
          v
      构造候选
          |
          v
   在训练折上拟合
          |
          v
    held-out scoring
          |
          v
        选择
          |
          v
   全数据 final refit
          |
          v
预测 / diagnostics / 可选 inference
```

最重要的区别是：**selection work** 与最终对外发布的 **fitted model** 不是同一件事。

选择过程中 statgpu 可能拟合很多临时模型，它们只用于比较候选 tuning configuration。选定候选后，所选配置会在全部观测上重新拟合。公开的 fitted coefficient、prediction、普通模型诊断和受支持的 coefficient inference 都属于这个 final refit，而不是某个任意 fold 中的临时拟合。

## 2. 统计不变量与执行自由度

加速可以改变同一个 CV 问题“**怎样计算**”，但不能静默改变“**在计算什么统计问题**”。

对一个固定的用户请求，下列内容属于公开的统计设计：

- 经过公开输入验证后的 candidate tuning values；
- train / validation folds；
- 与该模型对应的 loss 或 validation criterion；
- 在支持权重的路径上所采用的 analytic-weight 语义；
- candidate selection rule；
- 选定后用于 full-data refit 的配置；
- 在组合受支持时，用户显式指定的 solver 与 device 请求。

而实现可以在保持这些语义不变的前提下选择更快的执行方式，例如：

- 在多个候选之间复用 factorization 或 sufficient statistics；
- 用相邻 tuning value 的结果作为 warm start；
- 在 accelerator 上联合执行多个 fold / candidate；
- 通过 cache 复用已经计算过的 selection evidence；
- 当用户指定 `device="auto"` 时，根据 workload 自动选择 backend。

这些都是性能实现选择。应用代码不应依赖某个具体 batching layout、cache 实现、private fast-path 名称或自动路由阈值。

## 3. 为什么 selection 与 final refit 必须分开

交叉验证利用 held-out data 回答“该选哪个 tuning configuration”；最终 estimator 则在 tuning 完成后利用全部观测回答“在所选配置下如何拟合最终模型”。

这一分工解释了多个公开行为：

- `alpha_` 等属性描述的是**选择出来的 tuning configuration**；
- `coef_`、`intercept_`、prediction 和普通 final diagnostics 描述的是**全数据 final refit**；
- 支持 inference 时，推断发生在选择完成后的 final refit，而不是每个 fold 内单独发布；
- 某个 CV estimator 可以合理地对 selection path 和 final refit 分别暴露 solver 控制。

`LassoCV` 是最清楚的例子：`cv_solver` 控制 CV path，`solver` 控制最终 full-data `Lasso` 拟合。只要两阶段都遵循各自公开的 objective 与 solver contract，它们使用不同算法并不会改变 selected `alpha` 的统计含义。

## 4. Candidate grid 与 fold 是一等设计输入

CV 结果只能相对于“实际评估过的候选集合”和“实际采用的 folds”来理解。

若 estimator 自动生成 tuning grid，该 grid 会依赖数据和模型；若用户显式提供 grid，则经过公开验证后的 grid 就是请求的候选集合。

fold 也不仅仅是实现里的循环索引，它定义了 resampling design。时间序列、分组、cluster 或 survival data 可能需要不同的 splitting rule。statgpu 会检查 supplied split 的结构以及 estimator-specific 要求，但某个 split 是否符合具体科学问题仍然是用户的建模判断。

从设计上看，一旦本次 fit 的 folds 与 candidates 已经确定，后续加速应围绕**同一个 selection problem**展开，而不是因为更换执行路径就重新定义候选或 folds。

## 5. Pathwise reuse 与 warm start

许多正则化问题会沿一组有序 tuning values 求解。相邻候选的解往往也比较接近，因此每个候选都从完全无关的初值重新开始会重复很多工作。

CV 实现可以在 path 上安全复用信息，例如：

- 将前一个 candidate 的 coefficient 作为下一个 candidate 的初值；
- 在同一个 fold 内复用不随 candidate 改变的 decomposition 或 cross-product；
- 复用与同一统计 objective 一致的其他 fold-local numerical state。

这只是减少重复计算，并不是另一种 CV 方法。每个 candidate 的 score 和最终 selection 仍必须对应声明的 candidate configuration。

对非凸 penalty 或某些 model-specific loss，适合的 continuation/path strategy 可能与普通 L1/L2 不同。具体数值路线应查看相应模型页与 solver 文档；本页只描述共同的设计原则。

## 6. GPU batching：加速重复计算，而不是重新定义 CV

最直接的实现会对每个 `(fold, candidate)` 分别执行很多小操作。在 GPU 上，这可能导致设备利用率不足，并产生过多 host/device synchronization。

当数值结构允许时，statgpu 可以把重复工作组织成更大的 backend-native operation，例如同时处理多个 fold、多个 candidate，或二者的组合。

用户不需要依赖某个具体 batching layout。真正需要保持的是：

- effective candidate set 不变；
- fold membership 不变；
- model objective 与 validation criterion 不变；
- weight semantics 不变；
- selection rule 不变；
- full-data final refit 的解释不变。

因此未来版本可以改变 batching strategy，而不要求用户代码同步修改。

## 7. Selection cache

有些 CV workload 会反复提出完全相同的**选择问题**。当 inputs 与所有影响 selection 的控制项都没有变化时，重复计算整个 fold × candidate scoring 过程通常没有必要。

selection cache 可以在这种情况下复用已经计算好的 selection evidence。它的公开语义刻意保持很窄：

- cache 用来加速 candidate selection；
- cache **不会**用一个缓存的最终 estimator 取代新的 full-data final refit；
- 如果 data、folds、tuning values、weighting 或影响 selection 的 numerical control 发生变化，就不能错误复用不兼容的 evidence；
- 修改一次返回的结果不能污染后续 cache hit；
- 是否命中 cache 不能改变 selected configuration 的统计解释。

cache identity 的具体字段、容量、hashing strategy 与 private helper 都属于实现细节，不是公开 API 保证。

## 8. CV 中的 device 选择

显式 device 请求与自动 device 选择承担不同职责。

当用户显式指定 `device="cpu"`、`"cuda"` 或 `"torch"` 时，estimator 按公开 backend contract 执行；若对应 accelerator 路径不可用或该组合不支持，会直接报错。CV 加速不能仅因为 CPU 更容易实现，就把一个显式 GPU 请求静默改成 CPU fit。

当 `device="auto"` 时，statgpu 可以根据 backend availability 与 workload 特征自动选择执行位置。具体 crossover threshold 属于 performance tuning parameter，因此可能随 kernel 与硬件支持变化。

核心设计原则是：backend 选择可以改变“在哪里/如何”执行一个合法的 CV 问题，但不应静默替换 loss、penalty、candidate set 或 validation criterion。

## 9. 权重必须贯穿整个 CV lifecycle

在支持 `sample_weight` 的路径上，weighting 是统计问题的一部分，而不是最后才附加的 post-processing option。

一致的 weighted CV lifecycle 可以表示为：

```text
full-data weights
      |
      +--> training-fold weights -> weighted candidate fit
      |
      +--> validation-fold weights -> weighted held-out score
      |
      `--> full-data weights -> selected final refit
```

具体 normalization 由对应 estimator/loss contract 决定。任何加速路径只有在保持同一权重约定时才是等价的。不支持 weighted objective 的组合应明确报告限制，而不是静默丢弃 weights。

## 10. 为什么 Cox CV 在结构上不同

Cox cross-validation 不能总被看成“换了一个 loss 名称的普通 scalar-response regression”。held-out partial-likelihood evidence 与 survival target、event support 以及 risk-set semantics 有关。

因此 Cox path 往往需要不同的 preparation order，并在 candidate scoring 之前额外验证 folds。selection → final refit 的通用模型仍然成立，但 survival-specific 的合法 fold 与 score 定义应以 [Cox 比例风险模型](../models/coxph.md) 为准。

这也体现了更一般的原则：共享 CV infrastructure 不应为了强行统一实现而抹掉 model-specific statistical structure。

## 11. Tuning 完成后的 inference

当 coefficient inference 受支持时，statgpu 会在 tuning 完成后，对 selected final refit 做 inference。

这样可以避免为每个临时 fold fit 发布没有实际意义的 coefficient-inference result。同时也意味着，普通 CV 后 confidence interval 和 p-value 通常是**以已经选定的 tuning configuration 为条件**的；除非某个具体 inference method 明确校正了 tuning / selection uncertainty。

具体 inferential target 与限制见 [推断模式](inference-modes.md)和 [Penalized GLM 推断](penalized-glm-inference.md)。

## 12. 哪些内容用户可以依赖

| 稳定的公开设计 | 可能变化的实现细节 |
|---|---|
| selection 完成后进行 full-data refit | private call graph |
| 受支持的显式 solver/device 请求保持权威 | `auto` 的 device threshold |
| folds/candidates/weights 定义统计 selection problem | batching 维度与 kernel fusion |
| `cv_solver` 与 final-refit `solver` 可以表示不同阶段 | 内部 warm-start storage |
| cache 可以复用 selection evidence，但不能替代 final refit | cache key 字段、容量、hash 方法 |
| model-specific CV 语义继续属于对应模型 | 具体由哪个内部 fast path 执行 |

这种分层允许 statgpu 持续优化性能，而不把每一个 optimization choice 都变成永久的 public API 承诺。

## 13. CV 文档怎么分工

根据问题选择对应文档：

- **怎样配置和使用 CV？** → [交叉验证](cross-validation.md)
- **statgpu 的 CV 在概念上怎样组织、为什么能加速？** → 本页
- **哪些 loss × penalty × solver 组合可用？** → [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)
- **某个 solver 本身怎样计算？** → [求解器算法](solver-algorithms.md)
- **selection 后 inference 应如何解释？** → [推断模式](inference-modes.md)
- **survival CV 有什么特殊之处？** → [Cox 比例风险模型](../models/coxph.md)

仓库内部 call graph、private cache contract、heuristic threshold 和 validation/evidence 仍放在 `dev/` 下，不属于本公开设计页。
