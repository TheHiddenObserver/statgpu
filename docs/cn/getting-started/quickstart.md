# 快速上手

> 语言：中文\
> 最后更新：2026-10-11\
> 页面定位：快速开始\
> 切换：[English](../../en/getting-started/quickstart.md)

## 安装

需要 Python 3.9 或更高版本。下载或克隆仓库后，在终端中进入 `statgpu` 目录并安装：

```bash
cd statgpu
python -m pip install -e .
```

这会安装 NumPy、SciPy 和 joblib 基础依赖。下面的 CPU 示例不需要 GPU、CuPy 或 PyTorch。
如果需要使用 GPU，请参照[下方的安装步骤](#optional-gpu-execution)。

## 最小示例

下面生成 1000 条模拟观测。`X` 每行是一个样本，20 列是输入特征；`y` 保存每个样本对应的连续数值。线性回归用这些数据学习从 `X` 预测 `y`，随后预测前 3 行，并检查训练数据上的拟合程度。

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

第一个输出是前 3 个样本的预测值，第二个是训练集上的 R²（决定系数）。
这组模拟数据的噪声较小，因此 R² 应接近 1。这里的得分反映的是对训练数据的拟合程度；
要评估模型对新数据的预测效果，请使用未参与训练的数据，或进行交叉验证。

## 选择计算设备

除了在创建模型时指定 `device`，也可以设置全局默认设备：

```python
import statgpu as sg

sg.set_device("cpu")
print(sg.get_device().value)  # cpu
sg.set_device("auto")        # 恢复自动选择
```

第一个示例通过 `device="cpu"` 指定使用 NumPy 在 CPU 上计算，不受全局设置影响。
`sg.set_device("auto")` 则恢复全局自动选择，按 CuPy CUDA、Torch CUDA、NumPy CPU
的顺序选择可用后端。下方的 GPU 示例会直接在模型中指定设备。
各模型的支持范围和自动选择的例外见[设备选择指南](../guides/device-and-memory.md)。

<a id="optional-gpu-execution"></a>

## 使用 GPU（可选）

如果不需要 GPU，保留 `device="cpu"` 即可。下面的两种方式都需要 NVIDIA GPU 和兼容的
NVIDIA 驱动。安装可选依赖后，还需要确认 CUDA 能正常工作。

### CuPy

先确认 CUDA Toolkit 或 CUDA 运行时已安装且能正常工作，再根据 CUDA 版本选择**一种**
可选依赖。例如，使用 CUDA 12.x 时，在仓库目录中运行：

```bash
# CUDA 12.x：
python -m pip install -e ".[gpu12]"
```

CUDA 11.2–11.8 则改用 `python -m pip install -e ".[gpu11]"`，并使用兼容的
[CuPy 13 版本](https://docs.cupy.dev/en/v13.6.0/install.html)。
这两个安装选项分别对应 `cupy-cuda12x>=13.0` 和 `cupy-cuda11x>=13.0`，不会安装 CUDA Toolkit。
同一环境中不要安装多个 CuPy 发行包。Python、CuPy、CUDA 和驱动的版本兼容要求见
[CuPy 安装要求](https://docs.cupy.dev/en/stable/install.html)。

确认 CuPy 能在 GPU 上创建数组后，就可以用前面 CPU 示例中的 `X` 和 `y` 在 GPU 上拟合：

```python
model = LinearRegression(device="cuda")  # CuPy CUDA
model.fit(X, y)
print(model.score(X, y))
```

### PyTorch

先按照 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)的指引，选择并安装适合当前平台和驱动、
支持 CUDA 的 PyTorch 版本，再在仓库目录中安装可选依赖：

```bash
python -m pip install -e ".[torch]"
```

这个安装选项要求 PyTorch 2.0 或更高版本，但不会替你选择合适的 CUDA 版 PyTorch。
确认 `torch.cuda.is_available()` 返回 `True` 后，就可以用前面的 `X` 和 `y` 在 GPU 上拟合：

```python
model = LinearRegression(device="torch")  # PyTorch CUDA
model.fit(X, y)
print(model.score(X, y))
```

本例中，`device="cuda"` 使用 CuPy，`device="torch"` 使用 PyTorch，两者都在 CUDA GPU 上计算。
如果指定的后端不可用，模型会报错，不会自动改用 CPU。更多安装说明和模型支持范围见
[PyTorch 后端指南](../guides/pytorch-backend.md)。

## 推荐下一步

- [设备与显存管理](../guides/device-and-memory.md)
- [选择推断方法](../guides/inference-modes.md)
- [模型总览](../models/README.md)
- [基准面板与结果解读](../guides/benchmarks.md)
- [变更记录](../changelog.md)
