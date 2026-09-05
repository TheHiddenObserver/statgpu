# CPU/GPU 加速实现

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/guides/acceleration-internals.md)

这是面向进阶用户的实现指南。模型页继续关注统计方法怎么使用；本页说明计算发生在哪里、GPU 为什么可能更快，以及数据传输、同步、精度与显存何时会成为主要成本。

## 执行约定

| `device` | 数值后端 | 约定 |
|---|---|---|
| `"cpu"` | NumPy 与 CPU 科学计算栈 | 模型数值工作留在 CPU |
| `"cuda"` | NVIDIA CUDA 上的 CuPy | CuPy/CUDA 不可用时直接报错 |
| `"torch"` | NVIDIA CUDA 上的 PyTorch | Torch CUDA 不可用时直接报错 |
| `"auto"` | 根据能力和输入自动选择 | 唯一允许选择其他可用后端的模式 |

后端覆盖取决于估计器与求解器。应查看[已实现方法](implemented-methods.md)和具体模型页，不要假定每个方法都有完全相同的 CPU、CuPy 与 Torch 路径。

## 数据生命周期

1. 公开估计器验证形状、响应范围、权重与控制参数。
2. Formula/DataFrame、标签和分类变量展开在 CPU 上处理。
3. 方法支持时，数值数组一次性转换到所选后端。
4. 拟合、推断或预测使用后端原生线性代数与 kernel。
5. 大数组应留在后端；少量收敛判断可能只同步一个标量。
6. 公开系数、表格或元数据可能在报告边界有意转换回 NumPy/Python。

有文档记录的元数据传输不等于静默设备回退。若用户明确请求 GPU，却把完整设计矩阵移到 CPU 继续拟合，则属于不允许的回退。

## 主要加速模式

| 模式 | 代表方法 | 主要计算 | GPU 注意事项 |
|---|---|---|---|
| 稠密线性代数 | 线性回归、Ridge、PCA/SVD | 矩阵乘法、分解与求解 | 矩阵足够大时才能抵消传输与启动成本 |
| 光滑似然迭代 | GLM IRLS、Newton、L-BFGS | 重复梯度、Hessian 与加权求解 | 整个迭代留在设备上比单个 kernel 很快更重要 |
| 近端与非凸优化 | Lasso、Elastic Net、SCAD、MCP | 梯度/近端步骤、LLA 外循环 | 融合操作和减少主机收敛检查可降低同步 |
| 距离与图构建 | DBSCAN、聚类 | 距离分块、邻接关系、连通分量 | `batch_size` 在峰值显存和更多启动之间权衡 |
| 重采样与模型网格 | 交叉验证 | 大量相关拟合 | 数据复用和避免重复传输可能比单次拟合更关键 |

算法选择也是性能的一部分。CPU 与 GPU 路径应优化同一个公开目标函数，但可以使用不同的分解、搜索结构、kernel 或分批方式。

## Formula 与预处理

Patsy Formula 解析属于 CPU 预处理：

```python
model.fit(formula="y ~ x1 + C(group)", data=frame)
```

生成的稠密设计矩阵随后传到所请求的数值后端。这保留了便利性和特征名，但不是零拷贝。对反复运行的大型 GPU 拟合，可解析一次或在估计器支持时直接提供 CuPy/Torch 数组。详见 [Formula 支持矩阵](formula-interface.md)。

## 数据传输与互操作

- NumPy 输入在 CuPy 或 Torch 执行前必须复制到 CUDA。
- 公开路径接受时，已有 CuPy/Torch CUDA 数组应保持设备驻留。
- CuPy 与 Torch CUDA 之间优先使用可用的 DLPack 共享。
- NumPy 到 Torch CUDA 在支持时可使用 pinned host memory。
- 即使推断数值计算发生在 GPU，系数和结构化推断摘要也常以 NumPy 形式作为报告对象。

测速应包含必要的数据传输与同步。只测异步 kernel 而不同步，会低估真实耗时。

## 精度与数值一致性

病态设计与统计推断更适合 `float64`。`float32` 可以降低显存并提高吞吐，但会改变舍入、收敛和边界判断。不同后端的标签数字、阈值附近入选变量或临界 p 值可能不同，但不一定表示优化目标不同。

比较容差应考虑：

- 设计矩阵条件数与特征尺度；
- dtype；
- 求解器停止规则与迭代上限；
- 随机种子；
- 结果是否对标签编号不变。

线性回归 HAC 还有专门的大型 CPU 优化：先对小块得分矩阵计时，再决定并缓存混合精度累加偏好；最终协方差仍以双精度存储。见[线性推断参考](../models/linear-regression-inference.md)。

## 显存行为

GPU 显存包括仍存活的数组和可复用分配器缓存。`gpu_memory_cleanup=False` 通常提高重复拟合速度；`True` 会让支持的估计器在公开操作后释放缓存块，但不能减少算法本身所需的存活内存。

需要关注：

- $O(np)$ 设计矩阵；
- $O(p^2)$ Gram、Hessian 或协方差矩阵；
- $O(n^2)$ 两两距离或 kernel 矩阵；
- 交叉验证中的 fold 与网格倍增；
- dtype 或内存布局转换产生的临时副本。

优先使用模型记录的 `batch_size`、随机/增量版本、更小网格或低内存算法。清理缓存不能改变算法固有的二次复杂度。

## 推断边界

估计与推断的成本结构可能不同：

- OLS 协方差增加 $p\times p$ 逆/求解和三明治乘积；
- HC2/HC3 还需要计算杠杆值；
- HAC 增加指定滞后内的得分自协方差；
- GLM M-estimation 在优化后增加 bread/meat 矩阵；
- SCAD oracle 推断目前会在 CPU 上重新拟合已选 active set，即使筛选开始于 GPU。

因此应分别报告“仅估计”和“估计加推断”基准，不能假定关闭 `compute_inference` 只会减少很小的常数成本。

## CPU 何时更快

小 $n$ 或 $p$、从 NumPy 开始的一次性拟合、分支很多的算法，或包含大量小同步的 GPU 路径，常常仍是 CPU 更快。数值工作足够大、能保持设备驻留且重复次数足以摊销启动成本时，GPU 加速才最可信。

应结合记录了硬件与环境的[基准面板](/dashboard/)判断，不能把单次测速写成普遍速度结论。

## 可复现的后端验证

记录：

- commit SHA 与 statgpu 版本；
- CPU/GPU 型号、驱动、CUDA、CuPy、Torch、Python 与 NumPy 版本；
- 估计器、求解器、dtype、维度和全部容差；
- 输入最初位于主机还是设备；
- warm-up 规则和同步点；
- 统计一致性指标以及耗时。

随机算法还应同时设置估计器 random state 与后端种子。应比较目标函数、预测、KKT/梯度残差或标签不变的聚类指标，而不只比较系数 bit 或簇编号。

## 源码地图

- 后端选择与转换：`statgpu/backends/` 与 `statgpu/_base.py`
- 线性 CPU/CuPy/Torch 路径：`statgpu/linear_model/wrappers/_linear.py`
- 普通 GLM 调度：`statgpu/linear_model/_glm_base.py`
- 惩罚模型调度与求解器：`statgpu/linear_model/penalized/_fit_mixin.py` 与 `statgpu/solvers/`
- DBSCAN 路径：`statgpu/unsupervised/_dbscan.py`
- 基准生成：`dev/benchmarks/`

私有辅助函数可能变化；稳定的用户约定是各模型页记录的公开估计器行为。
