# 多重检验：全局 p 值合并（Fisher/Cauchy/ACAT）

> 语言: 中文  
> 最后更新: 2026-04-06  
> 页面定位: 指南文档  
> 切换: [English](../en/guides/multiple-testing-combine-pvalues.md)

语言切换：[English](../en/guides/multiple-testing-combine-pvalues.md)

## API 概览

使用 `statgpu.combine_pvalues` 将多个 p 值合并为一个全局 p 值：

```python
statistic, pvalue = statgpu.combine_pvalues(
    pvalues,
    method="fisher",      # "fisher" | "cauchy" | "acat"
    weights=None,          # 仅 cauchy/acat 使用
    axis=None,             # None 表示先展平再合并
    backend="auto",       # "auto" | "numpy" | "cupy"
)
```

相关接口：
- `statgpu.adjust_pvalues(...)` / `statgpu.multipletests(...)`：对多假设控制 FDR/FWER。
- `statgpu.combine_pvalues(...)`：将一组 p 值压缩为单个全局检验信号。

## 方法语义

1. Fisher
- 统计量：`-2 * sum(log(p_i))`
- 参考分布：自由度 `2m` 的卡方分布。
- 不支持权重。

2. Cauchy（ACAT 别名）
- 统计量：对 p 值做加权切线变换后求和。
- p 值：基于 Cauchy 尾部变换得到。
- 若提供 `weights`，需为非负权重。
- `method="acat"` 是 `method="cauchy"` 的别名。

## 形状规则

- `axis=None`：展平后合并，返回标量统计量和标量 p 值。
- `axis=k`：沿第 `k` 维合并，返回该维被约简后的数组。
- `weights` 长度必须等于合并轴长度。

## 示例

### 1) 向量合并为一个全局 p 值

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.01, 0.07, 0.03, 0.40])
stat, p_global = combine_pvalues(p, method="fisher", backend="numpy")
```

### 2) 按行合并（`axis=1`）

```python
import numpy as np
from statgpu import combine_pvalues

p_matrix = np.random.default_rng(0).uniform(1e-8, 1 - 1e-8, size=(100, 16))
stat_row, p_row = combine_pvalues(p_matrix, method="fisher", axis=1)
```

### 3) 加权 Cauchy / ACAT

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_c, p_c = combine_pvalues(p, method="cauchy", weights=w)
stat_a, p_a = combine_pvalues(p, method="acat", weights=w)  # 别名
```

### 4) CuPy GPU 路径

```python
import cupy as cp
from statgpu import combine_pvalues

p_cp = cp.random.uniform(1e-8, 1 - 1e-8, size=(4000, 64), dtype=cp.float64)
stat_cp, p_cp_out = combine_pvalues(p_cp, method="fisher", axis=1, backend="cupy")
```

## 基准结果解读（远端补充）

远端补充产物：
- JSON：`results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- 摘要：`results/remote_fisher_cauchy_benchmark_2026-04-05.md`

该产物的工作负载：
- `n_groups=4000`、`group_size=64`、`axis=1`、`warmup=1`、`repeats=5`

关键平均耗时：

| 方法 | statgpu NumPy (ms) | statgpu CuPy (ms) | SciPy (ms) |
|---|---:|---:|---:|
| Fisher | 3.979 | 0.816 | 350.721 |
| Cauchy | 4.052 | 0.874 | N/A |
| ACAT 别名 | 3.971 | N/A | N/A |

建议解读口径：
- Fisher 在该负载下，SciPy 相比 statgpu NumPy 约慢 `88.15x`。
- statgpu NumPy 到 CuPy 的加速约为：Fisher `4.88x`、Cauchy `4.63x`。
- Cauchy 与 ACAT 别名在该基准中数值一致（`max abs p-value diff = 0.0`）。
- NumPy/CuPy 差异处于浮点噪声级（p 值量级约 `1e-15`）。

## 复现实验

本地主脚本：
- `dev/benchmarks/benchmark_inference_backends.py`

示例命令：

```bash
python dev/benchmarks/benchmark_inference_backends.py --output-tag local_check
```

输出文件：
- `results/inference_backend_benchmark_<date>_local_check.json`
