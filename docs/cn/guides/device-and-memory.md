# 设备与 GPU 内存

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：设备选择与用户可见的 GPU 内存控制  
> 切换：[English](../../en/guides/device-and-memory.md)

## 设备选择

对于提供 `device` 参数的估计器，statgpu 统一采用以下含义：

- `device="cpu"`：请求使用 NumPy 在 CPU 上计算；
- `device="cuda"`：请求使用 CuPy 在 CUDA GPU 上计算；
- `device="torch"`：请求使用 Torch CUDA 计算；
- `device="auto"`：允许 statgpu 在该估计器支持且当前可用的后端中自动选择。

用户显式指定的加速器请求具有优先权。如果请求的 CuPy/Torch CUDA 后端不可用，statgpu 会报错，而不是静默改为 CPU 拟合。

具体模型支持的后端范围可能比通用的 `device` 取值更窄；如果应用必须使用某个后端，请查看对应模型页或 [已实现方法](implemented-methods.md)。

## 输入转换与预处理

公式（Formula）/DataFrame 解析和其他元数据准备可以先在 CPU 上完成。这并不改变模型数值计算所使用的后端：进入受支持的数值路径之前，模型数组会按照请求转换到相应后端。

NumPy、CuPy 与 Torch 之间的数据传输可以在内部采用优化机制。应用代码应依赖最终的设备语义，而不要依赖某一种具体传输实现，例如 DLPack 或固定页内存（pinned memory）。

## 自动设备选择

`device="auto"` 可以根据后端可用性、输入规模、工作量特征以及估计器专属的性能规则自动选择执行后端。具体的规模阈值属于实现细节，可能随着数值内核和基准测试结果的改进而变化。

如果应用需要可复现的硬件放置，应显式指定 `device`，而不是依赖内部的自动切换阈值。

## 求解器兼容性另见专门文档

设备支持与求解器兼容性是两个不同问题。某个后端本身可以正常使用，并不表示任意“损失函数 × 惩罚项 × 求解器”组合都受支持。

请分别查看：

- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)：组合兼容性；
- [求解器算法](solver-algorithms.md)：算法定义；
- 对应模型页：模型专属限制。

本页不再重复求解器支持矩阵。

## GPU 内存清理

部分支持 GPU 的估计器提供 `gpu_memory_cleanup` 参数。

- `gpu_memory_cleanup=False`（提供该参数时的默认值）更偏向重复拟合时的吞吐量，允许后端的内存池或缓存保留可复用的内存分配；
- `gpu_memory_cleanup=True` 会在估计器公开约定的清理时点请求释放可回收的 GPU 缓存内存，从而降低常驻显存，但也可能减少后续重复使用已有分配所带来的速度收益。

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

这一参数不会丢弃 `predict()`、`score()` 或受支持推断所需要的已拟合状态；估计器只会在不破坏已拟合模型状态的位置执行内存清理。

## 什么时候开启

`gpu_memory_cleanup=True` 适合以下场景：

- 多个模型共享同一块 GPU；
- 显存压力比重复拟合的延迟更重要；
- 长时间运行的进程希望在多次拟合之间归还可回收的内存池空间。

如果持续重复拟合同类模型，并且优先追求吞吐量，通常保留默认的关闭状态更合适。

## 相关文档

- [已实现方法](implemented-methods.md) — 模型与后端清单
- [交叉验证](cross-validation.md) — 选择与最终重拟合中的设备行为
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 求解器兼容性
- [PyTorch 后端](pytorch-backend.md) — PyTorch 专属使用说明
