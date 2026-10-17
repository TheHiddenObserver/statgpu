# 什么时候值得用 GPU？

> 语言：中文\
> 最后更新：2026-10-17\
> 页面定位：先理解工作量，再选择设备，最后运行一个小示例\
> 切换：[English](../../en/guides/when-to-use-gpu.md)

当大部分时间花在规模较大、形式规则的数值计算上，尤其是同一份数据要反复参与计算时，值得试试 GPU。小任务、逐条预测，或主要耗时在数据准备上的流程，通常先用 CPU 更合适。模型支持 GPU，并不代表它处理你的数据一定更快。

使用 statgpu 不需要自己编写 CUDA 代码。先判断时间花在哪里，再用 `device` 选择受支持的后端。

<a id="why-matrix-work-can-benefit"></a>

## 矩阵计算为什么可能受益？

把数据表看作一个矩阵：每行是一条观测，每列是一个特征。线性模型对每行使用同一组学到的权重，将各个特征乘以对应权重后相加，得到预测值。

例如，权重为 `[2, -1]`、没有截距时：

- `[3, 1]` 这一行得到 `3 × 2 + 1 × (-1) = 5`。
- `[0, 4]` 这一行得到 `0 × 2 + 4 × (-1) = -4`。
- `[2, 2]` 这一行得到 `2 × 2 + 2 × (-1) = 2`。

这些行可以独立计算。矩阵乘法将大量类似运算组织起来，适合 GPU 并行处理。拟合时，相关运算会反复组合预测值、误差和特征，逐步求出模型参数。

![CPU 用少量示意通道处理灵活任务，GPU 用较多示意通道并行处理形式相同的数值任务。通道表示并行工作，不代表实际核心数量。](/images/gpu-work-cn.svg)

*这是概念示意图，不是硬件结构图或计时结果。CPU 同样支持并行和向量化计算；GPU 则适合同时处理大量相似操作。图中通道的多少不能直接判断某个模型在哪种设备上更快。*

只有一小行数据时，启动 GPU 计算和搬运数据的开销，可能超过算术本身。矩阵足够大时，计算量才可能抵消这些开销。拟合仍有前后依赖：即使每轮内部可以并行，求解器也可能需要等上一轮完成，才能开始下一轮。

<a id="when-gpu-is-worth-trying"></a>

## 哪些任务值得试试 GPU？

重点看四件事：**数值计算量是否够大、数据是否反复使用、操作是否受支持、显存是否足够**。下面是可以尝试的具体任务，不是实测提速，也不是最低规模要求。

| 示例任务 | GPU 可能加速什么 | 哪些因素可能限制收益 |
|---|---|---|
| 对 100,000 行、200 个特征的稠密数据做 [Ridge 回归](../models/regression/ridge.md) | 大规模矩阵乘法与线性方程求解 | 高效的 CPU 实现仍可能更快；拟合与推断的耗时不同 |
| 对 50,000 行、500 个数值特征做 [PCA](../unsupervised/pca.md) | 通过矩阵乘法和分解，提取相关特征中的主要变化 | 求解器影响计算量和显存；随机化 PCA 是近似方法 |
| 对 200,000 行、40 个特征，用 20 个中心做 [KMeans](../unsupervised/kmeans.md) | 反复计算样本到中心的距离，并更新中心 | 初始化、收敛过程和中间数组也有开销 |
| 在较大的数据集上做[交叉验证](cross-validation.md) | 多次拟合，以及某些模型提供的计算复用 | 各折和候选参数并非自动全部同时计算；大量微小拟合仍可能更慢 |

这些行数和特征数只是帮助理解问题的示例，不是基准测试结果，也不表示统一的速度分界线。改变特征数量、算法、数据类型、CPU 线程数或 GPU，都可能改变比较结果。

反复计算最有利的情况，是实现能够复用数据或已分配的内存。在 Python 循环里调用很多次小拟合，并不会自动把它们合成一次 GPU 并行计算。

<a id="when-cpu-is-the-better-start"></a>

## 哪些情况先用 CPU 更合适？

- **任务小，或只偶尔运行。** 一次小回归或几个预测，可能在 GPU 启动与传输开销收回之前就完成了。
- **要求单次请求尽快返回。** 一次预测一行，与一次处理一大批数据，是不同的工作量。
- **数据频繁往返 CPU 和 GPU。** 反复转换数组、打印中间值，或把大批结果转回 NumPy，可能抵消计算收益。
- **主要耗时是解析、Python 控制流程或不规则判断。** GPU 不会自动加速文件读取、DataFrame/公式准备、Python 循环或分支密集的逻辑。
- **操作不受支持，或显存不足。** 先检查模型、求解器、输入格式和推断的支持范围；CPU 可能是更简单且受支持的选择。

不存在“超过 N 行就应该用 GPU”的通用规则。先得到正确的 CPU 结果，再比较自己实际需要的任务。

<a id="count-the-whole-journey"></a>

## 要算整段流程的时间

你等待的时间可能包含读取与准备数据、把输入复制到 GPU、运行数值计算，以及把结果送回 CPU 内存。即使中间的计算变快，整个流程也不一定明显变快。

![数据从 CPU 内存传到 GPU，在 GPU 上完成并可能重复数值计算，最后按需把结果送回 CPU。复用 GPU 上的数据可以减少重复的大批量传输。](/images/gpu-journey-cn.svg)

*这是数据流概念图，不是实测时间轴。方框宽度和箭头不表示耗时。具体方法也可能在各阶段之间使用 CPU，或传输较小的报告数组。*

“数据驻留 GPU”是指数组在相邻操作之间继续保存在 GPU 显存中。如果后续方法接受这种数组，就可以避免反复复制同一份大数据。这不代表整个流程完全不使用 CPU。例如，`Ridge` 即使在 GPU 上拟合，`coef_` 仍以 NumPy 数组保存。

### 显存要容纳全部工作数据，不只是输入

显存需要容纳输入、输出、模型状态和临时工作数组。上面 Ridge 示例中的输入矩阵，使用 float64 时仅本身就占 `100,000 × 200 × 8 = 160,000,000` 字节，约 160 MB。这个数值**不是**拟合峰值显存的估计。中心化副本、矩阵乘积、求解器工作空间、推断和多次拟合都可能额外占用显存。

Float32 与 float64 的内存和精度要求不同，不同 GPU 的双精度计算性能也不同。请检查估计器实际采用的计算类型，不要认为传入 float32 就一定全程使用 float32。先根据数值问题确定所需精度，再比较速度。

在支持该参数的方法中，`gpu_memory_cleanup=True` 可以释放内存池中未使用的分配，但不能释放仍被引用的数组，也不能让超出显存的任务自动装得下。详见[设备与显存控制](device-and-memory.md#gpu-memory-cleanup)。

<a id="try-it-with-an-explicit-device"></a>

## 用一个小例子明确选择设备

以下示例用于理解计算位置与输出处理。数据刻意保持很小，**不是性能演示**。请先完成[快速上手中的安装步骤](../getting-started/quickstart.md)。

### 1. 先得到一个小型 CPU 结果

每行包含 20 个特征，并对应一个连续响应值。`Ridge` 拟合带正则化的线性预测模型。这里仅需预测，所以用 `compute_inference=False` 跳过系数不确定性的计算。

<!-- learner-example: gpu-decision-cpu -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 20))  # NumPy float64，保存在 CPU 内存中
weights = rng.normal(size=20)
y = 1.0 + X @ weights + 0.1 * rng.normal(size=2000)

cpu_model = Ridge(
    alpha=0.1, solver="exact", device="cpu", compute_inference=False
).fit(X, y)
cpu_predictions = cpu_model.predict(X[:3])
print(cpu_predictions.shape)  # (3,)
print(isinstance(cpu_model.coef_, np.ndarray))  # True
```
<!-- example-end: gpu-decision-cpu -->

`cpu_predictions` 保存前三行的预测值。打印的 `(3,)` 表示其形状，`True` 表示拟合系数保存在 NumPy 数组中。这是在训练数据上检查用法，不是在未参与训练的数据上评估预测准确性。模型解释与数值注意事项见 [Ridge 回归](../models/regression/ridge.md)。

### 2. 选择一种可用的 CUDA 后端

对 `Ridge` 而言，`device="cuda"` 表示 CuPy CUDA，`device="torch"` 表示 PyTorch CUDA。两者都需要可用的 NVIDIA GPU 环境。显式指定的后端不可用时会报错；`device="torch"` 不会悄悄改用 Torch CPU。

**CuPy：**先运行上面的 CPU 数据准备代码，再在可用的 CuPy/CUDA 环境中运行以下示例。显式复制 `X` 和 `y`，便于看清数据所在位置。

<!-- example-requires: gpu-decision-cpu -->
<!-- learner-example: gpu-decision-cupy -->
```python
import cupy as cp

X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)
gpu_model = Ridge(
    alpha=0.1, solver="exact", device="cuda", compute_inference=False
).fit(X_gpu, y_gpu)

predictions_gpu = gpu_model.predict(X_gpu, return_cpu=False)
print(isinstance(predictions_gpu, cp.ndarray))  # True
print(predictions_gpu.device)

# 大批量预测值与残差继续留在 GPU 上，只取回一个标量。
training_mse = cp.mean((predictions_gpu - y_gpu) ** 2)
print(float(training_mse.get()))
```
<!-- example-end: gpu-decision-cupy -->

最后一个数值是训练集均方误差，展示如何衔接计算，不是速度比较或泛化能力评估。`Ridge.predict` 默认返回 NumPy 数组；`return_cpu=False` 使预测保留在已拟合模型的数值后端中。它不会改变 `coef_` 的 NumPy 存储方式，也不代表消除了所有内部传输。

**PyTorch 替代方式：**同样先运行 CPU 数据准备代码。下面直接接收 NumPy 输入，并将其转换后用于 CUDA 拟合。运行前需要确认 `torch.cuda.is_available()` 为 `True`。

<!-- example-requires: gpu-decision-cpu -->
<!-- learner-example: gpu-decision-torch -->
```python
torch_model = Ridge(
    alpha=0.1, solver="exact", device="torch", compute_inference=False
).fit(X, y)
torch_predictions = torch_model.predict(X[:3], return_cpu=False)
print(torch_predictions.is_cuda)  # True
print(torch_predictions.device)
```
<!-- example-end: gpu-decision-torch -->

### 3. 确认具体模型的行为

- `device="auto"` 是选择策略，不是速度保证。采用通用设备处理的估计器会读取全局设置；调用 `set_device("torch")` 后，自动选择的估计器仍可能要求 Torch CUDA。部分模型还有按工作量决定设备的逻辑，详见[全局与模型设备设置](device-and-memory.md#全局设置与估计器设置)。
- 函数的 `backend="torch"` 与估计器的 `device="torch"` 含义不同，部分函数支持 Torch CPU。
- 核平滑、核特征和样条有[计算位置方面的例外](device-and-memory.md#current-smoothing-and-spline-exceptions)。不要仅凭 GPU 设置，就认定这些方法在 GPU 上执行。
- 拟合、预测、交叉验证和推断的后端支持范围可能不同。请查看[已实现方法](implemented-methods.md)与对应模型页。

<a id="measure-your-own-workload"></a>

## 怎样比较自己的任务？

1. 保持**数据与统计问题一致**：数据形状、预处理、数据类型、目标函数、惩罚尺度、求解器设置、收敛条件和是否计算推断都要对齐。同时比较数值结果与耗时。
2. 先明确关心的是**完整运行一次**，还是**数据已经位于 GPU 上时的重复计算**。前者要包含传输和输出转换；后者要说明排除了哪些准备工作。
3. 各条路径都先预热，再重复测量，报告代表耗时及波动。如果关心首次调用延迟，请单独测量，不要用预热后的结果代替。
4. 在计时区间前后，同步**实际执行工作的 CUDA 设备**。GPU 操作可以异步执行，只在 CPU 上启动和停止计时，可能在计算尚未完成时就读到了耗时。
5. 记录 CPU/GPU 型号、软件包版本、源码版本、数据形状、数据类型与计时范围。没有收敛，或数值结果明显不同的较快运行，不能视为完成了同一个问题。

[基准测试面板](../../en/guides/statgpu_benchmark_dashboard.md#open-dashboard)提供已有测量记录。比较前请对齐环境、工作量、后端与指标范围。相对另一个库的提速，不等于 statgpu 自身的 GPU/CPU 提速；历史结果也不保证当前性能。详见[基准结果的解读与复现](../benchmarks.md)。

## 延伸阅读

- [设备与 GPU 内存](device-and-memory.md)：设备设置、输出数组与清理行为
- [交叉验证](cross-validation.md)：参数选择与最终重拟合
- [NVIDIA CUDA 最佳实践：哪些工作适合 GPU](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device)：并行计算与传输成本
- [CuPy 性能最佳实践](https://docs.cupy.dev/en/stable/user_guide/performance.html)：异步计时、预热与一次性开销
