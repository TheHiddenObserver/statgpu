# 设备与 GPU 内存

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：设备选择与用户可见 GPU memory control  
> 切换：[English](../../en/guides/device-and-memory.md)

## 设备选择

对于暴露 `device` 参数的 estimator，statgpu 使用以下统一含义：

- `device="cpu"`：请求 NumPy CPU 计算；
- `device="cuda"`：请求 CuPy CUDA 计算；
- `device="torch"`：请求 Torch CUDA 计算；
- `device="auto"`：允许 statgpu 在该 estimator 支持且当前可用的 backend 中自动选择。

显式 accelerator 请求保持权威。如果请求的 CuPy/Torch CUDA backend 不可用，statgpu 会报错，而不是静默替换成 CPU fit。

具体模型的 backend 范围可能比通用 device vocabulary 更窄；如果应用必须使用某个 backend，请查看对应模型页或 [已实现方法](implemented-methods.md)。

## 输入转换与预处理

Formula/DataFrame 解析和其他 metadata preparation 可以先在 CPU 上进行。这并不改变模型数值计算所选择的 backend：进入受支持的数值路径前，模型数组会按照请求转换到对应 backend。

NumPy、CuPy 与 Torch 之间的传输可以在内部采用优化机制。应用代码应依赖最终的 device 语义，而不要依赖某一种具体传输实现，例如 DLPack 或 pinned memory。

## 自动设备选择

`device="auto"` 可以根据 backend availability、输入/工作量特征以及 estimator-specific performance heuristic 自动选择执行 backend。具体 size threshold 属于实现细节，可能随着 kernel 与 benchmark 改进而变化。

如果应用需要可复现的硬件 placement，应显式指定 device，而不是依赖内部 `auto` 阈值。

## Solver 兼容性单独维护

设备支持与 solver 兼容性是两个不同问题。某个 backend 可以正常使用，但特定 loss × penalty × solver 组合仍可能不受支持。

请分别查看：

- [Solver × Penalty 矩阵](solver-penalty-matrix.md)：组合兼容性；
- [求解器算法](solver-algorithms.md)：算法定义；
- 对应模型页：model-specific 限制。

本页不再重复 solver coverage matrix。

## GPU memory cleanup

部分 GPU-capable estimator 暴露 `gpu_memory_cleanup`。

- `gpu_memory_cleanup=False`（暴露该参数时的默认值）更偏向 repeated-fit throughput，允许 backend memory pool/cache 保留可复用 allocation。
- `gpu_memory_cleanup=True` 会在 estimator 公开的 cleanup point 请求释放可回收的 GPU cached memory，从而降低常驻显存，但可能减少后续复用带来的速度收益。

示例：

```python
from statgpu.linear_model import Ridge

model = Ridge(
    alpha=1.0,
    device="cuda",
    gpu_memory_cleanup=True,
)
model.fit(X, y)
```

这一参数不表示会丢弃 `predict()`、`score()` 或受支持 inference 所需的 fitted state；estimator 只会在与其 fitted-state contract 兼容的位置执行 cleanup。

## 什么时候开启

`gpu_memory_cleanup=True` 适合：

- 多个模型共享同一 GPU；
- 显存压力比 repeated-fit latency 更重要；
- 长时间运行的进程希望在 fit 之间归还可回收 pool memory。

如果持续重复拟合同类模型并优先追求吞吐，通常保留默认关闭更合适。

## 相关文档

- [已实现方法](implemented-methods.md) — model/backend 清单
- [交叉验证](cross-validation.md) — selection 与 refit 中的 device 行为
- [Solver × Penalty 矩阵](solver-penalty-matrix.md) — solver 兼容性
- [PyTorch 后端](pytorch-backend.md) — PyTorch-specific 使用
