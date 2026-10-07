# 快速上手

> 语言: 中文  
> 最后更新: 2026-10-07  
> 页面定位: 快速开始  
> 切换: [English](../../en/getting-started/quickstart.md)

语言切换：[English](../../en/getting-started/quickstart.md)

## 安装

使用 Python 3.9 或更高版本，从已获取的仓库副本安装：

```bash
cd statgpu
python -m pip install -e .
```

这会安装 NumPy、SciPy 和 joblib 基础依赖。下面的 CPU 示例不需要 GPU、CuPy 或 PyTorch。
GPU 的可选安装步骤见[后文](#optional-gpu-execution)。

## 最小示例

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 20))
y = 1.0 + X @ rng.normal(size=20) + 0.1 * rng.normal(size=1000)

model = LinearRegression(device="cpu")
model.fit(X, y)
print(model.predict(X[:3]))
print(model.score(X, y))
```

第一项输出是三行数据的预测值，第二项是训练集上的 R 平方。在这个低噪声模拟示例中，
R 平方应接近 1；它不能衡量对新数据的预测能力，后者应通过留出数据或交叉验证评估。

## 常用设备控制

```python
import statgpu as sg

sg.set_device("cpu")
print(sg.get_device().value)  # cpu
sg.set_device("auto")        # 恢复自动选择
```

首个示例显式选择 CPU，不受全局设置影响。`sg.set_device("auto")` 恢复全局自动策略，
解析时优先选择可用的 CuPy CUDA，其次是 Torch CUDA，最后是 NumPy CPU。
本页的 GPU 示例会在估计器上显式指定设备，见下文。
各模型的支持范围与例外见[设备选择指南](../guides/device-and-memory.md)。

<a id="optional-gpu-execution"></a>

## 可选的 GPU 执行

如果不需要 GPU，保留 `device="cpu"` 即可。以下两种 GPU 方案都需要 NVIDIA GPU 和兼容的
NVIDIA 驱动；仅安装可选依赖，并不意味着 CUDA 已经可用。

### CuPy

在仓库目录中，根据已能正常工作的 CUDA Toolkit/运行时，选择**一种**对应的可选依赖：

```bash
# CUDA 12.x：
python -m pip install -e ".[gpu12]"
```

CUDA 11.2–11.8 则改用 `python -m pip install -e ".[gpu11]"`，并使用兼容的
[CuPy 13 版本](https://docs.cupy.dev/en/v13.6.0/install.html)。
这两个选项分别安装 `cupy-cuda12x>=13.0` 和 `cupy-cuda11x>=13.0`，不会安装 CUDA Toolkit。
同一环境不要安装多个 CuPy 发行包。Python、CuPy、CUDA 和驱动的具体兼容组合见
[CuPy 安装要求](https://docs.cupy.dev/en/stable/install.html)。

确认 CuPy 能在 GPU 上分配数组后，可复用 CPU 示例中的 `X` 和 `y`：

```python
model = LinearRegression(device="cuda")  # CuPy CUDA
model.fit(X, y)
print(model.score(X, y))
```

### PyTorch

先通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装适合当前平台和驱动的
CUDA 构建，再从仓库目录安装可选依赖：

```bash
python -m pip install -e ".[torch]"
```

该选项要求 PyTorch 2.0 或更高版本，但不会替你选择合适的 CUDA 构建。确认
`torch.cuda.is_available()` 为 `True` 后，可复用前面的 `X` 和 `y`：

```python
model = LinearRegression(device="torch")  # PyTorch CUDA
model.fit(X, y)
print(model.score(X, y))
```

对本例的估计器，显式请求 `"cuda"` 或 `"torch"` 而对应后端不可用时会报错，不会退回 CPU。
`"cuda"` 选择的是 CuPy，不是 PyTorch。更多安装说明和模型支持范围见
[PyTorch 后端指南](../guides/pytorch-backend.md)。

## 推荐下一步

- [设备与显存管理](../guides/device-and-memory.md)
- [推断配置（Lasso）](../guides/inference-modes.md)
- [模型总览](../models/README.md)
- [基准脚本索引](../guides/benchmarks.md)
- [变更记录](../changelog.md)
