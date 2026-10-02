# PyTorch 后端指南

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/guides/pytorch-backend.md)

## 概览

StatGPU 支持三个主要执行后端：

| `device` 值 | 数值后端 | 典型执行位置 |
|---|---|---|
| `"cpu"` | NumPy | CPU |
| `"cuda"` | CuPy | NVIDIA CUDA |
| `"torch"` | PyTorch | NVIDIA CUDA |
| `"auto"` | 自动选择 | 根据可用性与工作负载选择 CuPy、Torch CUDA 或 NumPy |

`device="torch"` 是显式的 PyTorch 请求；`device="cuda"` 选择 CuPy，并不是 Torch 的别名。显式请求在对应后端不可用时会报错，不会静默切换到其他后端。

不同模型、求解器、交叉验证和推断方法的后端覆盖范围可能不同。请查看 [已实现方法](implemented-methods.md)、[设备与 GPU 内存](device-and-memory.md) 和对应模型页，而不要假定每个公开估计器都有完全相同的 Torch 路径。

## 安装

通过可选依赖安装 Torch：

```bash
pip install "statgpu[torch]"
```

GPU 执行需要兼容的 PyTorch CUDA 构建和 NVIDIA 驱动。拟合前可以检查：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

仅安装基础包不会自动安装 PyTorch：

```bash
pip install statgpu
```

## 基本用法

### NumPy 输入并显式使用 Torch

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 20))
y = 1.0 + X @ rng.normal(size=20) + rng.normal(size=1000)

model = LinearRegression(device="torch")
model.fit(X, y)
print(model.score(X, y))
```

估计器会把兼容的 NumPy 输入转换到所选的 Torch CUDA 后端。如果 Torch CUDA 不可用，显式 Torch 请求会报错。

### 直接使用 Torch CUDA tensor

```python
import torch
from statgpu.linear_model import LinearRegression

X = torch.randn(1000, 20, device="cuda", dtype=torch.float64)
y = torch.randn(1000, device="cuda", dtype=torch.float64)

model = LinearRegression(device="torch")
model.fit(X, y)
prediction = model.predict(X)
```

输出是否保持为 Torch tensor 取决于具体方法。请查看模型页，确认输出是 Torch 数组，还是有意转换为 CPU 元数据、NumPy 数组或 Python 标量统计摘要。

## 设备选择

### 估计器级选择

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device="torch")
```

### 全局默认值

```python
import statgpu as sg

sg.set_device("torch")
```

如果估计器提供 `device=` 参数，估计器级设置优先。只有明确需要自动选择时才使用 `"auto"`。

## 统计推断

使用 Torch 执行并不意味着每一种推断方法都可用。推断支持范围取决于估计器、协方差类型、求解器、数据约定和可选依赖。对于支持推断的模型，应在其文档中确认：

- 支持哪些协方差估计；
- 是否提供标准误、检验统计量、p 值和置信区间；
- 某些严格或近似计算路径是否需要显式请求；
- 延迟进入、聚类、并列事件、秩亏或公式接口是否有限制；
- 最终结果是 Torch 数组、NumPy 数组还是标量元数据。

不支持的推断组合应明确报错，或在模型文档明确说明只提供参数估计；不应静默产生另一种近似推断结果。

## 执行边界

当某个方法支持 Torch 时，核心数值数组应保留在 Torch 后端。合理的 CPU 边界可能包括：

- 公式、标签、特征名和小型索引元数据；
- 数据折定义、收敛判定和标量控制逻辑；
- Torch 中缺失的标量分布函数；
- 有意表示为 NumPy 数组或 Python 标量的用户结果摘要；
- 只接受 CPU 数组的外部比较库。

这些边界取决于具体模型。声称“所有中间量始终位于 GPU”并不准确；相反，完整设计矩阵被搬回 CPU 或数值后端被静默切换，也不应作为未经说明的替代路径出现。

## 数据类型与数值精度

统计推断通常更适合使用 `float64`：

```python
X = torch.randn(2000, 50, device="cuda", dtype=torch.float64)
```

预测变量、响应变量、权重、`offset` 和初始化数组应使用相容的数据类型。比较 NumPy、CuPy 与 Torch 的结果时，应结合算法、条件数、停止规则和数据类型设置合理容差，而不是要求逐位完全一致。

L-BFGS 的 float32 跨后端差异另见 [L-BFGS Float32 数值行为](lbfgs-float32-precision-contract.md)。

## 随机性与可复现性

算法包含随机性时，可以同时设置模型的 `random_state` 与 Torch 随机种子：

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

交叉验证的数据折、landmark 抽样、随机分解和随机初始化还可能使用估计器自己的 `random_state`。要获得可复现结果，应同时确认模型文档中的随机性来源。

## 显存管理

显存需求取决于估计器和工作负载。精确核方法、稠密 Hessian 或协方差计算可能需要二次或更高阶的中间存储。模型文档提供批处理或近似方法时，可以根据问题规模选择相应选项。

排查问题时可以释放 Torch 缓存：

```python
import torch

torch.cuda.empty_cache()
```

部分估计器提供 `gpu_memory_cleanup=True`。该选项控制可回收缓存的清理，不改变统计目标，也不会允许显式 Torch 请求静默退回 CPU。更多说明见 [设备与 GPU 内存](device-and-memory.md)。

## 性能应如何理解

GPU 性能依赖样本量、特征维度、数据类型、数值内核或求解器、硬件、同步开销和显存压力。小型任务可能在 CPU 上更快，因此不应把某个模型或某一张 GPU 上的加速比当成所有工作负载的统一保证。

对于用户而言，稳定的公开语义是显式设备请求、模型目标和结果解释；具体批处理方式、自动切换阈值和某次硬件测量属于会随实现优化而变化的性能细节。

## 故障排查

### Torch CUDA 不可用

```python
import torch
print(torch.cuda.is_available())
```

检查 NVIDIA 驱动、安装的 Torch 构建及其自带 CUDA runtime。系统 CUDA toolkit 的版本本身并不能决定某个 Torch wheel 是否可用。

### 显式 Torch 执行报错

当 Torch CUDA 或必要的 Torch 运算不可用时，显式请求报错是预期行为。只有在符合应用意图时才改用 `device="cpu"` 或 `device="auto"`；不能期待 `device="torch"` 静默切换到其他后端。

### 显存不足

可以减小问题规模，使用文档明确提供的批处理或近似方法，减少 CV 候选网格或数据折数量，或选择内存复杂度更低的方法。`torch.cuda.empty_cache()` 无法减少算法当前正在使用的 tensor 所必需的内存。

### 与其他框架结果不同

首先对齐：

- 目标函数归一化；
- 正则化尺度；
- 截距与特征编码；
- 求解器与停止容差；
- 样本权重、`offset`、并列事件和协方差选项；
- 数据类型与随机种子。

如果其他框架优化求和形式的损失，而 StatGPU 优化平均损失，惩罚参数可能需要相应缩放。

## 相关文档

- [设备与 GPU 内存](device-and-memory.md)
- [已实现方法](implemented-methods.md)
- [交叉验证](cross-validation.md)
- [推断 API](inference-api.md)
- [模型总览](../models/README.md)
- [快速开始](../getting-started/quickstart.md)

## 参考资料

- [PyTorch 文档](https://pytorch.org/docs/)
