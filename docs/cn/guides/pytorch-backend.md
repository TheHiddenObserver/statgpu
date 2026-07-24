# PyTorch 后端指南

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../en/guides/pytorch-backend.md)

## 概览

StatGPU 支持三个执行后端：

| `device` 值 | 数值后端 | 典型执行位置 |
|---|---|---|
| `"cpu"` | NumPy | CPU |
| `"cuda"` | CuPy | NVIDIA CUDA |
| `"torch"` | PyTorch | NVIDIA CUDA |
| `"auto"` | 自动选择 | 根据可用性与输入选择 CuPy、Torch CUDA 或 NumPy |

`device="torch"` 是显式 PyTorch 请求；`device="cuda"` 选择 CuPy，不是 Torch 的
别名。显式请求在对应后端不可用时会报错，不会静默切换到其他后端。

不同模型、solver、交叉验证和推断选项的覆盖范围可能不同。应查看
[已实现方法](implemented-methods.md)和对应模型页，而不是假设每个公共估计器都有
完全相同的 Torch 路径。

## 安装

通过可选依赖安装 Torch：

```bash
pip install "statgpu[torch]"
```

GPU 执行需要兼容的 PyTorch CUDA build 和 NVIDIA 驱动。拟合前可检查：

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

估计器会将兼容的 NumPy 输入转换到所选 Torch CUDA 后端。若 Torch CUDA
不可用，显式 Torch 请求会报错。

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

输出是否保持为 Torch tensor 取决于具体方法。应查看模型页，确认输出是 Torch
数组，还是有意暴露为 CPU 元数据或标量统计摘要。

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

估计器公开 `device=` 参数时，估计器级设置优先。只有明确需要自动选择时才使用
`"auto"`。

## 统计推断

使用 Torch 执行并不意味着每一种推断选项都可用。推断覆盖取决于估计器、协方差
类型、solver、数据合同和可选依赖。对于支持推断的模型，应在其文档中确认：

- 支持的协方差估计；
- 标准误、检验统计量、p 值和置信区间；
- strict 路径与显式请求的 approximate 行为；
- delayed entry、cluster、ties、秩亏或 formula 限制；
- 最终摘要是 Torch 数组、NumPy 数组还是标量元数据。

不支持的推断组合应显式失败，或进入文档化的 estimation-only 模式；不应静默
产生近似结果。

## 执行边界

方法支持 Torch 时，核心数值数组应保留在 Torch 后端。合理的 CPU 边界可能包括：

- formula、标签、特征名和小型索引元数据；
- fold 定义、收敛决策和标量控制流；
- Torch 中缺失的标量分布函数；
- 有意表示为 NumPy 或 Python 标量的用户摘要；
- 只接受 CPU 数组的外部验证库。

这些边界取决于具体模型。声称所有中间量都始终位于 GPU 并不准确。完整设计矩阵
转移或后端切换不能作为静默 fallback 出现。

## Dtype 与数值精度

统计推断通常更适合使用 `float64`：

```python
X = torch.randn(2000, 50, device="cuda", dtype=torch.float64)
```

预测变量、响应、权重、offset 和初始化数组应使用相容 dtype。NumPy、CuPy 与
Torch 的结果差异应结合算法、条件数、停止规则和 dtype 设定容差，而不是要求
bitwise 相同。

## 随机性与可复现性

算法含随机性时，同时设置模型的 `random_state` 与 Torch seed：

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

交叉验证 folds、landmark 抽样、随机分解和随机初始化还可能使用估计器自己的
`random_state`。

## 显存管理

显存需求取决于估计器和工作负载。精确核方法、稠密 Hessian 或协方差计算可能需要
二次或更高阶的中间存储。应在文档支持时使用适合问题的 batching 或近似方法。

排查问题时可释放 Torch 缓存：

```python
import torch

torch.cuda.empty_cache()
```

部分估计器公开 `gpu_memory_cleanup=True`。该选项控制缓存清理，不改变统计目标，
也不允许 CPU fallback。

## 性能与验证证据

GPU 性能依赖样本量、特征维度、dtype、kernel 或 solver、硬件、同步和显存压力。
小型任务可能在 CPU 上更快。不能把一个模型或一张 GPU 上的 benchmark 当成全局
加速保证。

维护的证据应记录：

- 精确 commit SHA；
- Python、Torch、CUDA 和驱动版本；
- GPU 型号；
- 包含同步的计时方法；
- 准确性或统计一致性指标；
- passed、failed 和 skipped 测试数量。

当前与历史 benchmark 位于 `results/` 和 `dev/benchmarks/`。保留的
[Torch 后端报告](../../../dev/docs/torch_backend_final_report.md)是带日期的证据快照，
不是当前支持矩阵。

## 故障排查

### Torch CUDA 不可用

```python
import torch
print(torch.cuda.is_available())
```

检查 NVIDIA 驱动、安装的 Torch build 及其自带 CUDA runtime。系统 CUDA toolkit
版本本身不能决定哪个 Torch wheel 可用。

### 显式 Torch 执行报错

当 Torch CUDA 或必要 Torch 运算不可用时，这是预期行为。只有在符合预期合同的
情况下才改用 `device="cpu"` 或 `device="auto"`；不能期待 `device="torch"`
静默 fallback。

### 显存不足

减小问题规模，使用文档化的 batching 或近似方法，减少 CV grid 或 fold 数，或选择
内存复杂度更低的方法。`torch.cuda.empty_cache()` 无法减少算法当前活跃 tensor
必须占用的内存。

### 与其他框架结果不同

首先对齐：

- 目标函数归一化；
- 正则化尺度；
- 截距与特征编码；
- solver 与停止容差；
- sample weights、offset、ties 和协方差选项；
- dtype 与随机种子。

若其他框架优化求和损失，而 StatGPU 优化平均损失，惩罚参数可能需要相应缩放。

## 相关文档

- [设备与显存管理](device-and-memory.md)
- [已实现方法](implemented-methods.md)
- [交叉验证](cross-validation.md)
- [推断 API](inference-api.md)
- [模型总览](../models/README.md)
- [快速开始](../getting-started/quickstart.md)

## 参考资料

- [PyTorch 文档](https://pytorch.org/docs/)
- [StatGPU Torch 后端证据快照](../../../dev/docs/torch_backend_final_report.md)
