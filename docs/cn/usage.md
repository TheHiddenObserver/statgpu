# statgpu 文档入口（中文）

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../en/usage.md)

该入口只链接维护中的能力清单，避免重复保存容易过期的支持矩阵。

## 快速开始

- [快速入门](getting-started/quickstart.md)
- [已实现方法](guides/implemented-methods.md)
- [设备与 GPU 内存](guides/device-and-memory.md)
- [PyTorch 后端](guides/pytorch-backend.md)
- [交叉验证](guides/cross-validation.md)
- [推断 API](guides/inference-api.md)
- [变更记录](changelog.md)

CuPy 请按 CUDA 主版本安装 `statgpu[gpu11]` 或 `statgpu[gpu12]`；
PyTorch 后端使用 `statgpu[torch]`。维护中的 Cox 实现已包含在基础安装中；
`statgpu[survival]` 仅增加 statsmodels，用于可选外部验证与比较。

## 模型族

- [模型总览](models/README.md)
- [广义线性模型](models/generalized-linear-model.md)
- [Cox 比例风险模型](models/coxph.md)
- [面板模型](models/panel.md)
- [ANOVA](models/anova.md)
- [协方差估计](models/covariance.md)
- [非参数方法](models/nonparametric.md)
- [无监督学习](models/unsupervised.md)
- [特征选择](models/feature-selection.md)
- [回归诊断](guides/regression-diagnostics.md)

`RidgeCV`、`LassoCV`、`ElasticNetCV`、`LogisticRegressionCV`、
`PenalizedGLM_CV` 与 `CoxPHCV` 均已实现。具体 loss、penalty、推断与后端覆盖见
[已实现方法](guides/implemented-methods.md)及对应模型页。

## 验证与证据

所有验证结论都应限定到实际测试的模型、后端、硬件与 commit。托管 CI、物理 GPU
测试、历史 benchmark 与发布证据分别记录在对应 workflow、模型页、changelog、
`results/` 或 `dev/` artifact 中；GPU 测试被跳过不等同于完成物理 GPU 验证。

## 贡献者检查

修改代码时遵循 [`dev/AGENTS.md`](../../dev/AGENTS.md) 与 canonical Claude Code
[`new-module-dev` skill](../../.claude/skills/new-module-dev/SKILL.md)。独立 review 使用
[`code-review` skill](../../.claude/skills/code-review/SKILL.md)，性能/证据工作激活时使用
[`benchmark` skill](../../.claude/skills/benchmark/SKILL.md)。先按影响范围激活必要 gate，
保持显式设备语义，外部比较前确认 objective normalization，并同步受影响的中英文
public capability claim。
