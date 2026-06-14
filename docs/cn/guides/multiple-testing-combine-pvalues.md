# 多重检验：P值校正与全局合并（BH/BY/Holm/Bonferroni/Hochberg + Fisher/Cauchy/Stouffer）

> 语言: 中文  
> 最后更新: 2026-04-26  
> 页面定位: 指南文档  
> 切换: [English](../en/guides/multiple-testing-combine-pvalues.md)

语言切换：[English](../en/guides/multiple-testing-combine-pvalues.md)

## API 概览

### P 值校正（多重假设检验）

使用 `statgpu.adjust_pvalues` 进行 FDR/FWER 校正：

```python
reject, pvals_adj = statgpu.adjust_pvalues(
    pvalues,
    method="bh",           # "bh" | "by" | "holm" | "bonferroni" | "hochberg"
    alpha=0.05,              # FWER/FDR 水平
    axis=None,               # None 表示先展平再校正
    backend="auto",         # "auto" | "numpy" | "cupy" | "torch"
)
```

等价接口：`statgpu.multipletests(...)` 与 `statgpu.adjust_pvalues(...)` 参数完全相同。

### 全局 P 值合并

使用 `statgpu.combine_pvalues` 将多个 p 值合并为一个全局 p 值：

```python
statistic, pvalue = statgpu.combine_pvalues(
    pvalues,
    method="fisher",        # "fisher" | "cauchy" | "stouffer"
    weights=None,            # 仅 cauchy/stouffer 使用
    axis=None,               # None 表示先展平再合并
    backend="auto",         # "auto" | "numpy" | "cupy" | "torch"
)
```

## 方法语义

### P 值校正方法

1. **Bonferroni**: `p_i * m`，FWER 控制，最保守。
2. **Holm**: Step-down FWER，比 Bonferroni 更高效。
3. **BH** (Benjamini-Hochberg): Step-up FDR，别名 `fdr_bh`。
4. **BY** (Benjamini-Yekutieli): FDR 控制，对任意依赖结构稳健。
5. **Hochberg**: Step-up FWER，别名 `fdr_hochberg`/`step_up`/`stepup`。

所有校正方法的结果被截断到 [0, 1]。

### 全局 P 值合并方法

1. **Fisher**
- 统计量：`-2 * sum(log(p_i))`
- 参考分布：自由度 `2m` 的卡方分布。
- 不支持权重。

2. **Cauchy**（ACAT 别名）
- 统计量：对 p 值做加权切线变换后求和。
- p 值：基于 Cauchy 尾部变换得到。
- 若提供 `weights`，需为非负权重。
- `method="acat"` 是 `method="cauchy"` 的别名。

3. **Stouffer**（加权 Z 检验）
- 统计量：`sum(w_i * Z_i) / sqrt(sum(w_i^2))`，其中 `Z_i = norm.ppf(1 - p_i)`。
- p 值：`norm.sf(statistic)`。
- 支持权重，与 cauchy 权重接口一致。
- 别名：`ztest`/`weighted_z`。

## 形状规则

- `axis=None`：展平后合并，返回标量统计量和标量 p 值。
- `axis=k`：沿第 `k` 维合并，返回该维被约简后的数组。
- `weights` 长度必须等于合并轴长度。

## 示例

### P 值校正

#### 1) 向量校正

```python
import numpy as np
from statgpu import adjust_pvalues

p = np.array([0.003, 0.02, 0.50, 0.10, 0.001])
reject, pvals_adj = adjust_pvalues(p, method='bh', alpha=0.05)
```

#### 2) Hochberg step-up FWER

```python
import numpy as np
from statgpu import adjust_pvalues

p = np.array([0.003, 0.02, 0.50, 0.10, 0.001])
reject, pvals_adj = adjust_pvalues(p, method='hochberg', alpha=0.05)
# 比 Holm 更高效（power 更高），控制 FWER
```

#### 3) 按行校正（`axis=1`）

```python
import numpy as np
from statgpu import adjust_pvalues

p_matrix = np.random.default_rng(0).uniform(0, 1, size=(100, 16))
reject_row, adj_row = adjust_pvalues(p_matrix, method='bh', axis=1)
```

### 全局 P 值合并

#### 1) 向量合并为一个全局 p 值

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.01, 0.07, 0.03, 0.40])
stat, p_global = combine_pvalues(p, method="fisher", backend="numpy")
```

#### 2) 按行合并（`axis=1`）

```python
import numpy as np
from statgpu import combine_pvalues

p_matrix = np.random.default_rng(0).uniform(1e-8, 1 - 1e-8, size=(100, 16))
stat_row, p_row = combine_pvalues(p_matrix, method="fisher", axis=1)
```

#### 3) 加权 Cauchy / ACAT

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_c, p_c = combine_pvalues(p, method="cauchy", weights=w)
stat_a, p_a = combine_pvalues(p, method="acat", weights=w)  # 别名
```

#### 4) 加权 Stouffer Z 检验

```python
import numpy as np
from statgpu import combine_pvalues

p = np.array([0.04, 0.15, 0.20, 0.01])
w = np.array([1.0, 1.0, 0.5, 2.0])

stat_s, p_s = combine_pvalues(p, method='stouffer', weights=w)
stat_z, p_z = combine_pvalues(p, method='ztest', weights=w)  # 别名
```

#### 5) CuPy GPU 路径

```python
import cupy as cp
from statgpu import combine_pvalues

p_cp = cp.random.uniform(1e-8, 1 - 1e-8, size=(4000, 64), dtype=cp.float64)
stat_cp, p_cp_out = combine_pvalues(p_cp, method="fisher", axis=1, backend="cupy")
```

#### 6) Torch GPU 路径

```python
import torch
from statgpu import combine_pvalues, adjust_pvalues

p_torch = torch.rand(5000, 64, dtype=torch.float64, device='cuda')
reject, adj = adjust_pvalues(p_torch, method='bh', backend='torch')
stat, p_global = combine_pvalues(p_torch, method='stouffer', axis=1, backend='torch')
```

## 大规模性能基准（p=50k-1M, Tesla P100）

基于 `dev/benchmarks/_bench_inference_timing_large.py`：

### adjust_pvalues BH (sort + cummin, O(n log n))

| p | NumPy | CuPy | Torch | CuPy vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 33.3 ms | 1.75 ms | 114 ms | 19.0x |
| 100,000 | 69.7 ms | 3.49 ms | 232 ms | 20.0x |
| 500,000 | 374 ms | 28.5 ms | 1.01 s | 13.1x |
| 1,000,000 | 799 ms | 76.7 ms | 1.96 s | **10.4x** |

CuPy 在 sort 密集型操作上表现最佳（adjust 类）。

### combine_pvalues Stouffer (norm.ppf + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 2.01 ms | 1.47 ms | 0.51 ms | 3.9x |
| 100,000 | 3.88 ms | 2.93 ms | 0.65 ms | 6.0x |
| 500,000 | 17.8 ms | 17.6 ms | 1.67 ms | 10.7x |
| 1,000,000 | 36.8 ms | 34.0 ms | 3.09 ms | **11.9x** |

Torch 在计算密集型操作（norm.ppf）上表现最佳。

### combine_pvalues Fisher (sum+log, O(n) bandwidth-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 5.43 ms | 6.93 ms | 2.04 ms | **2.7x** |

带宽受限操作（sum+log）GPU 加速比较温和（~2-3x）。

### combine_pvalues Cauchy (tan + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 49.5 ms | 42.6 ms | 11.2 ms | **4.4x** |

### GPU 加速总结

- **p < 10,000**: GPU kernel launch 开销 (>300us) 占主导，CPU 可能更快
- **p > 50,000**: GPU 优势开始显现
- **sort 密集型** (adjust BH): CuPy 表现最佳，10x+ 加速
- **计算密集型** (Stouffer norm.ppf): Torch 表现最佳，12x 加速
- **带宽受限** (Fisher sum+log): GPU 加速温和，2-3x
- **混合型** (Cauchy tan+sum): GPU 加速中等，4-5x

## 基准结果解读（远端补充——旧版，p=4000×64）

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
