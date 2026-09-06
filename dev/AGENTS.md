# AGENTS.md

本文件用于帮助后续 coding agent 快速理解 `statgpu` 项目。它是项目导览，不是用户手册；修改代码前仍应阅读相关源码和文档。

自动开发、review 与 benchmark 的 canonical agent 协议位于：

- `.claude/skills/new-module-dev/SKILL.md`
- `.claude/skills/code-review/SKILL.md`
- `.claude/skills/benchmark/SKILL.md`

各 skill 的 supporting reference 按需加载。旧的 `.claude/skills/*.md` 平铺文件仅作为历史链接兼容指针。旧 Markdown-era workflow 的说明已移到 `.claude/legacy/new-module-dev-workflow.md`；`.claude/workflows/` 保留给 Claude Code Dynamic Workflow scripts，不作为 policy/document archive。若本文和 canonical skill 有冲突，开发任务中优先执行 canonical skill，并在结果中说明差异。

根目录 `CLAUDE.md` 是 Claude Code 的小型 bootstrap：它把 forked/general-purpose agent 引导到本文件和 canonical skills，而不是复制这里的规则。

`code-review` 使用 `context: fork`；Claude Code >= 2.1.218 可用 `background: false` 显式保持阻塞式 review。更早版本的 forked skill 本来就默认阻塞 invoking turn，因此**不存在“为了 blocking 必须升级到 2.1.218”**的要求。

## 项目概览

`statgpu` 是一个 GPU 加速的统计计算 Python 库，提供接近 scikit-learn 的 `fit` / `predict` / `score` API。项目重点是统计建模、推断、特征选择、生存分析和非参数方法，并通过 NumPy、CuPy、PyTorch 后端在 CPU/GPU 间切换。

包版本和顶层导出位于 `statgpu/__init__.py`。项目元数据、依赖和 pytest 配置位于 `pyproject.toml`。

## 主要代码入口

- `statgpu/__init__.py`: 顶层公共 API 汇总，包括模型、后端、推断、特征选择和非参数方法。
- `statgpu/_base.py`: estimator 基类，包含设备解析、后端获取和数组转换逻辑。
- `statgpu/_config.py`: 全局设备管理，支持 `cpu`、`cuda`、`torch`、`auto`。
- `statgpu/backends/`: NumPy、CuPy、Torch 后端抽象和自动选择逻辑。
- `statgpu/solvers/`: 通用优化求解器（FISTA、FISTA-BB、FISTA-LLA、Newton、L-BFGS、ADMM）。
- `statgpu/penalties/`: L1、L2、ElasticNet、SCAD、MCP、adaptive/group penalty 注册表。
- `statgpu/glm_core/`: GLM loss、family/link、IRLS、融合 kernel。
- `statgpu/cross_validation/`: 通用 CV 框架（CVEstimatorBase、kfold_indices、hash_cv_data）。
- `statgpu/linear_model/`: 线性模型 API 层，含 `wrappers/`、`penalized/`（mixin 架构）、`cv/`（CV wrappers）、`legacy/`。
- `statgpu/survival/`: CoxPH 和 CoxPHCV，含 Breslow/Efron ties 及 GPU 相关实现。
- `statgpu/inference/`: 分布 API、多重检验、p 值合并、bootstrap、permutation test。
- `statgpu/nonparametric/`: KDE、核回归、带宽选择、核岭回归、样条。
- `statgpu/feature_selection/`: knockoff 特征选择相关实现。
- `statgpu/core/formula/`: 可选 formula/dataframe 接口，依赖 `patsy`/`pandas` extra。

## 设备和后端规则

所有当前 estimator 以 `device` 参数控制计算路径：

- `device="cpu"`: 使用 NumPy CPU 路径。
- `device="cuda"`: 使用 CuPy CUDA 路径；CuPy/CUDA 不可用时应报错，不静默回退 CPU。
- `device="torch"`: 使用 Torch CUDA 路径；Torch CUDA 不可用时应报错，不使用 Torch CPU 作为隐式回退。
- `device="auto"`: 唯一允许自动选择可用后端的模式，通常优先 CuPy，其次 Torch CUDA，最后 NumPy。

保持“显式设备不静默回退”的约束很重要。新增功能时应复用 `BaseEstimator._get_backend()`、`_to_array()`、后端抽象和现有 device 语义。

## GPU 内存管理规则

当 estimator 自己持有可回收的 GPU cache/buffer，并公开 `gpu_memory_cleanup` 能力时，应沿用现有生命周期，而不是为所有 GPU-capable estimator 强行新增同一套析构逻辑：

- 构造函数/基类按该 estimator 的现有 public contract 暴露 `gpu_memory_cleanup`；
- 复用已有 `_cleanup_cuda_memory()` / `_cleanup_torch_memory()` 或共享 cleanup hook；
- 不在需要保留 fitted cache 供后续 `predict()` / `score()` / inference 使用的边界过早清理；
- 如果实现 `__del__`，保持 best-effort、无异常逃逸，并确认不会重复释放或依赖解释器退出顺序；
- 新增 cleanup 行为属于 memory/device contract，必须测试不会改变数值结果或 fitted-state 生命周期。

当前参考实现可查看 `statgpu/linear_model/wrappers/_logistic.py`、`statgpu/linear_model/penalized/_fit_mixin.py`、`statgpu/survival/_cox.py` 等实际维护路径；不要引用已删除的 `statgpu/linear_model/_logistic.py`。

## 通用开发硬规则

以下规则适用于后续功能开发和重构，但由 impact classification 决定哪些 gate 真正激活。**Impact-driven 不等于 capability 自己缩窄默认 DoD**：

- **新增或实质改变 shared numerical/statistical capability** 时，statgpu 的默认 backend contract 是 **NumPy + CuPy + Torch**。除非 task/issue 已明确给出合理的更窄能力，或用户明确批准 backend deferral，否则三后端实现/验证都是 completion gate；CPU-only 可以是中间阶段，但不能被称为完整新 capability。deferral 必须记录原因、用户可见失败行为、确定性测试/skip 条件和后续 scope。
- **新增 tunable loss x penalty capability** 时，默认 completion contract 同时包含 **direct fit + CV path/grid/folds/scoring/selection/final refit**。只有该能力统计上确实 non-tunable，或用户明确批准 CV deferral，才可以不做 CV。不能通过“不声明 CV”来绕过默认 closure。
- **API-only、deprecation-only、docs-only 或不改变 numerical dispatch 的窄 refactor** 不要求重新实现或扩展无关 backend/CV/inference 能力；应验证已有 support claim 与相关 dispatch/clone/refit 没有被破坏。这是 impact-driven scope 收敛的主要用途。
- 显式 `device="cuda"` 或 `device="torch"` 不允许静默回退 CPU；fallback、approximate inference、dtype/device 变化必须是公开契约的一部分，并在错误、warning 或结果字段中可见。
- inference 是 blocking gate 当且仅当当前 public API/docs 声明 inference，或本次变更触及 `compute_inference`、`summary()`、covariance、SE、p-value、CI、inference result/backend behavior。一个没有 inference public contract 的 prediction-oriented estimator 可以合法 estimation-only；不要仅因为外部包提供 inference 就自动扩大 statgpu scope。
- Formula-facing 方法必须验证 R-style/patsy 兼容，包括 intercept、categorical reference level、interaction/transform、missing data、列名和列顺序；暂不支持的语法要有明确失败模式和文档说明。
- 新增 inference、stopping rule、solver dispatch、影响数值的 memory behavior 或 estimator 行为时，外部/analytic baseline 应覆盖本次 active contract；能覆盖 coef/bse/p/CI/AIC/BIC/LLF、预测、目标函数或 KKT 的场景按需覆盖。
- 外部比较必须使用显式对齐设置，包括相同特征集、ties/solver、正则化参数和收敛参数，如 `alpha`、`C`、`max_iter`、`tol`。
- 外部比较前必须确认 objective normalization 和 penalty scale。若 `statgpu` 优化 `n^{-1} sum_i loss_i + lambda * penalty`，而外部框架优化 `sum_i loss_i + lambda * penalty`，测试应使用等价 penalty 映射，不要为了强行对齐外部框架而修改 `statgpu` 的 loss 定义。
- strict inference/fallback 的默认策略以具体 estimator 的 public contract 为准；strict path 失败时不得静默降级成看似成功的 approximate result。
- 历史 external-alignment 阈值（如 `coef <= 1e-6`、`bse <= 1e-3`、`p-value <= 5e-2`）只在对应维护测试/validator 明确采用时才是 acceptance contract，不应当作所有统计方法的普适阈值。
- 关键 inference 输出字段在声明 backend parity 时应保持语义一致，例如 `coef`、`bse`、`t/z`、`p`、`CI`、`AIC/BIC/LLF` 等。
- CUDA 分层规则保持有效：模型层不要无目的散落直接 `cupy` import，优先复用 `statgpu/backends/`、`BaseEstimator` 和已有后端工具；确有 kernel/device-specific 需求时可保留显式 backend code。
- 新 numerical method 的准入标准通常是 implementation、targeted tests、external/analytic comparison 或 benchmark（按 active gate）、docs update 同步完成；API/docs-only change 不应为了仪式性 completion 被强制生成无意义 benchmark。
- 用户可见能力变更后，要保持 README、USAGE、英文/中文 docs 和 changelog 中相关 capability claim 一致；只更新真正受影响的表面。
- 本地改动至少要说明已跑或未跑的相关验证，不要把 skipped GPU test 表述成 physical GPU evidence。

## 文档写作约束

新增或修改用户可见能力时，文档不能只写一句 API 说明，但页面结构要按受众分层，而不是强制所有模型页复制同一 reference-first 模板：

- 文档更新顺序默认 EN-first、CN-follow；维护中英文对应页时保持 capability claim 和概念结构一致，但不要求逐字翻译。
- README、USAGE、EN/CN docs、changelog 中的能力声明必须一致，不能出现某处宣称支持、另一处仍写未支持的状态。
- **learner-facing model page** 优先按“问题/动机 → 直觉 → 何时使用/替代方案 → 必要模型/目标函数 → 自包含示例 → 结果解释 → 参数选择 → pitfalls → advanced/reference”组织；solver/backend 实现细节通常放后面。
- learner page 的 `Key parameters` 可以是教学精选，但 public API reference/inventory 必须完整，或明确链接到 canonical complete API reference；不能以 learner-first 为由静默丢参数、方法或重要 fitted attributes。
- reference/implementation page 可以保持 reference-first，不需要强行套 learner 叙事。
- 涉及 strict/approx 两条路径时，必须明确默认策略、fallback 条件、适用边界和用户可见状态，避免让用户误以为 approximate 是严格推断。
- 涉及设备或后端时，示例只覆盖对该页面真正有帮助的路径；完整 backend capability 可链接到统一 device/backend reference，暂不支持的后端要写清限制。
- 涉及外部基线时，文档应说明对齐对象和关键设置，例如 `statsmodels`、`sklearn`、R 包、solver/ties/regularization/tolerance 以及 objective-scale mapping。
- benchmark 或远程实验结果进入文档时，应附可审计产物路径/commit/environment，而不是只写口头速度结论。
- 引用统计方法、论文、R 包或外部 API 时，应保留 References 或链接，避免只写实现细节不写统计来源。

## 常复用代码

后续开发中较常被复用的两块代码是：

- `statgpu/backends/`: 后端抽象、数组转换、可用性检测和 NumPy/CuPy/Torch 选择逻辑。新增模型或算法时，优先通过这里的接口处理跨后端计算，不要在业务模块里重复写 backend 分支。
- `statgpu/inference/` 中和 distribution 有关的代码，尤其是 `_distributions_backend.py` 及 `get_distribution()` 等导出接口。新增统计推断、p 值、置信区间或分布函数需求时，先检查这里是否已有统一实现。

## 文档入口

- `README.md`: 项目总体说明、安装方式、快速示例和功能概览。
- `docs/en/usage.md`: 英文使用入口。
- `docs/cn/usage.md`: 中文使用入口。
- `docs/en/`: 英文文档，包括 quickstart、guides 和各模型说明。
- `docs/cn/`: 中文文档目录。
- `docs/en/models/README.md`: 模型覆盖范围和当前限制的较新摘要。
- `docs/en/guides/device-and-memory.md`: 设备选择、GPU 内存和后端规则。

新增或修改用户可见能力时，通常需要同步更新受影响的 `docs/en/models/*.md`、`docs/cn/models/*.md`、changelog 或入口文档；不要机械修改与本次 capability 无关的页面。

## Changelog 写作规范

项目有三份 changelog，定位不同：

| 文件 | 定位 | 读者 | 详细程度 |
|---|---|---|---|
| `CHANGELOG.md`（根目录） | PR 级摘要 | 开发者、贡献者 | 简洁，每个 PR 1-5 行 |
| `docs/en/changelog.md` | 用户级详述 | 用户、研究者 | 详细，含性能数据、代码示例、验证产物 |
| `docs/cn/changelog.md` | 用户级中文详述 | 中文用户、研究者 | 与英文详细版结构一致，中文自然表述 |

三份文件按变更类型同步更新，内容不机械重复但能力声明必须一致。

### 根目录 `CHANGELOG.md` 格式

```markdown
# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## YYYY-MM-DD

### PR #NN — 简短标题（不超过 60 字符）
- 变更描述 1（英文，句首动词过去式）
- 变更描述 2

### PR #MM, #LL — 合并标题
- 变更描述

---

## GPU Performance Milestones

### vX.X — 标题
- 关键指标
```

**规则**：
- 按日期分组，同一日期多个 PR 按编号倒序（最新在前）
- 每个 PR 1-5 行描述，只写"做了什么"，不写"为什么"和"怎么做"
- 合并相关 PR：`PR #61, #60, #59 — 标题`
- GPU Performance Milestones 单独 section，记录重大性能版本

### 详细版 `docs/en/changelog.md` 格式

采用 [Keep a Changelog](https://keepachangelog.com/) 风格，按**类别**组织：

```markdown
# Changelog

> Language: English
> Last updated: YYYY-MM-DD
> Switch: [Chinese](../changelog.md)

## YYYY-MM

### Added (YYYY-MM-DD)

- **功能标题**:
  - 详细描述（what + why）
  - 支持的后端/场景
  - 文件变更列表
  - 代码示例（如适用）

### Fixed (YYYY-MM-DD ~ YYYY-MM-DD)

- **Bug 标题**:
  - Root cause 分析
  - 影响范围
  - 修复方式
  - 验证结果

### Optimized (YYYY-MM-DD)

- **优化标题**:
  - 优化内容
  - 性能数据（硬件型号、数据规模、提速倍数）
  - 验证产物路径（`results/*.json`）

### Improved (YYYY-MM-DD)

- **改进标题**:
  - 改进内容
  - 对比数据

### Validation (YYYY-MM-DD)

- 验证覆盖范围
- 测试结果（X/Y passed）
- 远程环境信息
```

**规则**：
- 按月份分组（`## YYYY-MM`），月内按日期和类别分
- 五个类别：`Added`、`Fixed`、`Optimized`、`Improved`、`Validation`
- `Added`：新功能、新模型、新 API。必须包含：功能描述、支持的后端、文件列表、代码示例
- `Fixed`：Bug 修复。必须包含：root cause、影响范围、修复方式、验证结果
- `Optimized`：性能优化。必须包含：优化内容、性能数据（含硬件型号和数据规模）、验证产物路径
- `Improved`：非功能性改进（代码质量、文档、测试覆盖等）
- `Validation`：验证覆盖范围扩展（新测试、新 benchmark、远程验证）
- 性能数据必须附硬件信息（如 `Tesla P100`、`n=5000, p=500`）
- 验证产物路径用 `results/*.json` 或 `dev/tests/*.py` 格式
- 破坏性变更在 `Fixed` 或 `Added` 中用 `**breaking**` 前缀标注

### 中文版 `docs/cn/changelog.md` 格式

结构与英文版完全相同，类别名使用中文：

| 英文 | 中文 |
|---|---|
| `### Added` | `### 新增` |
| `### Fixed` | `### 修复` |
| `### Optimized` | `### 优化` |
| `### Improved` | `### 改进` |
| `### Validation` | `### 验证` |

**规则**：
- 类别名**统一用中文**，不要出现 `### Fixed` 和 `### 修复` 混用的情况
- 技术术语保持英文（如 `FISTA`、`CuPy`、`L-BFGS`、`PenalizedGLM_CV`）
- 文件路径、代码、数值、硬件型号保持英文
- 描述用中文，句式自然，不要逐字翻译英文版
- 头部元信息使用中文：
  ```markdown
  > 语言：中文
  > 最后更新：YYYY-MM-DD
  > 切换：[English](en/changelog.md)
  ```

### 三份文件的同步规则

| 变更类型 | `CHANGELOG.md` | `docs/en/changelog.md` | `docs/cn/changelog.md` |
|---|---|---|---|
| 新功能 | 1 行 PR 标题 + 1-2 行描述 | `Added` 下完整描述 + 代码示例 + 文件列表 | `新增` 下对应内容（中文描述） |
| Bug 修复 | 1 行 PR 标题 + 1 行描述 | `Fixed` 下 root cause + 影响 + 修复 + 验证 | `修复` 下对应内容（中文描述） |
| 性能优化 | 1 行 PR 标题 + 提速倍数 | `Optimized` 下完整数据 + 硬件 + 产物路径 | `优化` 下对应内容（中文描述） |
| 重构 | 1 行 PR 标题 | `Improved` 下简述（或不写） | `改进` 下对应内容（或不写） |
| 文档 | 1 行 PR 标题 | `Improved` 或不写 | `改进` 或不写 |
| 破坏性变更 | `**breaking**` 前缀 + 迁移路径 | `Fixed` 或 `Added` 中标注 + 迁移路径 | `修复` 或 `新增` 中标注 + 迁移路径 |

### 更新时机

- **PR 合并时**：立即更新 `CHANGELOG.md`（根目录）
- **版本发布前**：从 `CHANGELOG.md` 汇总到 `docs/en/changelog.md`（英文详细版），同步更新 `docs/cn/changelog.md`（中文详细版）
- **性能里程碑**：benchmark 全量通过后，同步更新三份文件的 GPU Performance Milestones
- **中文版更新顺序**：先写英文版，再写中文版（避免翻译返工）

### 示例对比

**根目录 `CHANGELOG.md`**（简洁）：
```markdown
### PR #55 — Core GLM solver, backends, penalties, inference (PR-A)
- 7 GLM families: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
- 10 penalties: none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
- 6 solvers: irls, fista, fista_bb, admm, lbfgs, newton
```

**详细版 `docs/en/changelog.md`**（完整）：
```markdown
### Fixed (2026-05-20)

- **v23c: L-BFGS fused penalty gradient fix**:
  - Root cause: `lbfgs_solver` fused GLM path computed loss-only gradient, missing penalty gradient
  - L-BFGS converged to unregularized solution (`loss_grad ≈ 0`) instead of `loss_grad + α·coef = 0`
  - Fix: add `_smooth_penalty_gradient(penalty, coef)` after each `_fused_glm_value_and_gradient` call
  - Affected: all GLM families + smooth penalties (L2, ElasticNet)
  - Impact: 9 MISMATCH cases fixed (max|diff| from 1e-01~1e-02 down to 1e-04~1e-08)
  - Full benchmark: 1043/1043 ALL PASS
  - Files modified: `statgpu/glm_core/_solver.py`
```

## 测试和验证

`pyproject.toml` 将 pytest 的 `testpaths` 配置为 `dev/tests`。优先运行与改动相关的针对性测试，例如：

```bash
pytest dev/tests/test_linear.py
pytest dev/tests/test_penalties_and_exports.py
pytest dev/tests/test_distributions_backend.py
```

本地 conda `base` 环境只用于导入测试和轻量 smoke test，例如确认 `import statgpu`、顶层 API 导出和纯 CPU 基础路径没有明显断裂。精度测试、运行时间测试、GPU 数值一致性、benchmark、R 对比和远程环境相关验证，应在通过 `dev/scripts/remote_config.py` 配置的远程 GPU 环境中进行；当前常用环境是 Matpool 远程服务器的 `myconda`。

GPU、远程、benchmark、R 对比类脚本较多，通常不应在本地无目的地全量运行。涉及远程测试时，优先使用 `dev/scripts/remote_config.py` 或未跟踪的 local config/环境变量提供连接信息，不要把服务器凭据写入代码、文档、测试、memory、`.claude/settings.json` 或提交记录。

## 仓库布局和协作注意事项

- `dev/tests/` 是主要测试目录，但其中也包含大量远程、debug、benchmark、runner 脚本；选择测试时要看文件名和内容。
- `dev/benchmarks/`、`results/`、`tmp/`、`statgpu.egg-info/`、`__pycache__/` 等多为实验产物、构建产物或临时文件，不要把它们当作核心库代码。
- `.gitignore` 已排除很多本地、远程和 benchmark 产物；新增临时脚本或大结果文件前先检查是否应被跟踪。
- 不要提交远程服务器凭据、密码、token 或本地环境配置。远程配置应使用 `dev/scripts/remote_config.py`、环境变量或未跟踪的 local config。
- 当前仓库可能存在未提交改动和未跟踪调试文件。修改前先看 `git status`，只触碰任务相关文件，不要回退他人改动。

## 实现建议

- 优先沿用现有 sklearn 风格 API、后端抽象、solver/penalty 注册表和文档结构。
- 新增 estimator 时尽量继承或模仿现有 `BaseEstimator`、`linear_model` 和 `glm_core` 模式。
- 新增后端相关逻辑时，要同时考虑 NumPy、CuPy、Torch 的数组类型、设备纯度和结果转回 NumPy 的边界。
- 新增统计方法时，至少补充针对性单元测试；新 shared numerical capability 的三后端 closure 仍是默认 DoD；若涉及性能，再补充相应 benchmark，而不是无条件跑与任务无关的远程矩阵。
