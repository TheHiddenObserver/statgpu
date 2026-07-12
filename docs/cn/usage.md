# statgpu 文档入口（中文）

> 语言：中文  
> 最后更新：2026-07-12  
> 切换：[English](../en/usage.md)

该入口只链接维护中的能力清单，避免重复保存容易过期的支持状态。

## 快速开始

- [快速入门](getting-started/quickstart.md)
- [已实现方法](guides/implemented-methods.md)
- [设备与 GPU 内存](guides/device-and-memory.md)
- [PyTorch 后端](guides/pytorch-backend.md)
- [交叉验证](guides/cross-validation.md)
- [推断 API](guides/inference-api.md)
- [变更记录](changelog.md)

CuPy 请按 CUDA 主版本安装 `statgpu[gpu11]` 或 `statgpu[gpu12]`；
PyTorch 后端使用 `statgpu[torch]`。

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
`PenalizedGLM_CV` 与 `CoxPHCV` 均已实现；具体 loss、penalty 与后端覆盖见
[已实现方法](guides/implemented-methods.md)。

## 验证边界

托管 CI 覆盖 Python 3.9–3.12、完整 CPU 测试、静态契约，以及受影响原生后端
路径的 NumPy/Torch-CPU 一致性。真实 CuPy CUDA 与 Torch CUDA 的收敛、传输、
显存、运行时间和重复拟合验证仍为 `PARTIAL_REMOTE_PENDING`，文档不作超出证据的声明。

## 贡献者检查

修改代码时遵循 `dev/AGENTS.md` 与 `.claude/workflows/new-module-dev.md`：显式设备
不得静默回退，外部比较前确认目标函数归一化，补齐架构相关测试，并同步 README、
中英文文档及三份 changelog。
