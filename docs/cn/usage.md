# statgpu 文档入口（中文）

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../en/usage.md)

这是一条完成首次拟合的简短路径。完整指南与参考文档按层次整理在[文档索引](README.md)中。

## 快速开始

1. 按[快速入门](getting-started/quickstart.md)完成安装、拟合与预测。
2. 选择 CPU、CUDA 或 Torch 前，先核对[设备指南](guides/device-and-memory.md)。CuPy 请按 CUDA 主版本安装 `statgpu[gpu11]` 或 `statgpu[gpu12]`；PyTorch 后端使用 `statgpu[torch]`。基础安装已经包含 Cox 实现，`statgpu[survival]` 增加的是可选的外部比较依赖。

<a id="模型族"></a>

3. 在[模型与统计方法](models/README.md)中按问题选择方法族，再运行对应教程。修改示例前，先检查输入假设、输出含义与限制。
4. 调参时参考[交叉验证](guides/cross-validation.md)；需要系数不确定性时参考[推断模式](guides/inference-modes.md)。拟合回归后，可通过[回归诊断](guides/regression-diagnostics.md)检查结果。

## 优化与兼容性参考

需要进一步配置拟合时，查阅[求解器 × 惩罚项兼容性矩阵](guides/solver-penalty-matrix.md)
与[损失函数 × 惩罚项 × 求解器框架](guides/loss-penalty-solver-framework.md)。
更新旧代码时，参阅[惩罚模型求解器迁移指南](guides/penalized-solver-api-migration.md)。
[交叉验证设计指南](guides/cross-validation-design.md)解释执行方式与加速语义。

## 统计工具

组间比较或多个 P 值从[统计检验](models/testing/README.md)开始。
分布函数、重抽样与结果对象分别见[分布 API](guides/distribution-api.md)和[推断 API](guides/inference-api.md)。

通过[已实现方法](guides/implemented-methods.md)核对支持范围，通过
[完整 API 参考](README.md#完整-api-参考)核对函数签名与拟合属性。
更新说明见[变更记录](changelog.md)。如需贡献代码、文档或测试，请参阅[贡献指南](../../CONTRIBUTING.md)。
