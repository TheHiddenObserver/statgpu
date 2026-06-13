# Knockoff 滤波器 50 次实验完整对比指南

本指南介绍如何运行 statgpu 与 R knockoff 包的完整对比实验。

---

## 概述

对比包含以下实现：

| 实现 | 框架 | 后端 | 说明 |
|------|------|------|------|
| statgpu NumPy | Python | CPU | 纯 NumPy 实现 |
| statgpu CuPy | Python | GPU | CUDA GPU 加速 |
| statgpu Torch | Python | GPU | PyTorch GPU 加速 |
| R knockoff SDP | R | CPU | R knockoff 包 (SDP 方法) |
| R knockoff Equi | R | CPU | R knockoff 包 (等相关方法) |

---

## 前置要求

### Python 环境

```bash
# 基础依赖
pip install numpy scipy

# GPU 加速 (可选)
pip install cupy-cuda11x  # 或 cupy-cuda12x
pip install torch --index-url https://download.pytorch.org/whl/cu118

# statgpu
pip install -e .
```

### R 环境

```bash
# 安装 R 包
install.packages(c("knockoff", "glmnet", "jsonlite"))
```

---

## 运行步骤

### 步骤 1: 运行 Python statgpu 对比

```bash
# 完整 50 次实验
python dev/scripts/knockoff_50exp_comparison.py 50 0.5

# 快速测试 (3 次实验)
python dev/scripts/knockoff_50exp_comparison.py 3 0.5
```

**参数说明:**
- 第一个参数：实验次数 (默认 50)
- 第二个参数：beta_scale (信号强度，默认 0.5)

**输出:**
- `dev/docs/knockoff_comparison_results_*.json` - 原始结果
- `dev/docs/knockoff_comparison_report_*.md` - Markdown 报告
- `dev/docs/knockoff_50exp_r.R` - 生成的 R 脚本

### 步骤 2: 运行 R knockoff 对比

```bash
Rscript dev/docs/knockoff_50exp_r.R
```

**输出:**
- `dev/docs/knockoff_50exp_r_results.json` - R 结果

### 步骤 3: 合并结果并生成对比报告

```bash
# 自动检测最新文件
python dev/scripts/merge_knockoff_r_comparison.py

# 或指定文件
python dev/scripts/merge_knockoff_r_comparison.py \
    dev/docs/knockoff_comparison_results_20260419_120000.json \
    dev/docs/knockoff_50exp_r_results.json
```

**输出:**
- `dev/docs/knockoff_py_vs_r_report_*.md` - 完整对比报告
- `dev/docs/knockoff_py_vs_r_merged_*.json` - 合并后的数据

---

## 实验配置

### 默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_samples | 400 | 样本量 |
| n_features | 80 | 特征数 |
| n_signal | 12 | 真实信号数 (特征 0-11) |
| beta_scale | 0.5 | 信号强度系数 |
| noise_scale | 0.5 | 噪声水平 |
| q | 0.10 | 目标 FDR |

### 信号强度

真实信号的系数按以下规律递增：

| 特征 | β 值 | 强度 |
|------|------|------|
| 0 | 0.5 | 弱信号 |
| 1 | 1.0 | 弱信号 |
| 2 | 1.5 | 弱信号 |
| 3 | 2.0 | 弱信号 |
| 4 | 2.5 | 中等信号 |
| ... | ... | ... |
| 11 | 6.0 | 强信号 |

---

## 评估指标

### 主要指标

| 指标 | 说明 | 理想值 |
|------|------|--------|
| Power (检出率) | TP / 真实信号数 | 100% |
| FDR | FP / 选中数 | ≤ 10% |
| 分离度 | 信号 W 均值 / 噪声 W 最大值 | 越高越好 |
| 时间 | 单次实验运行时间 | 越短越好 |

### 弱信号检测

特别关注特征 0-3 的检出率，这些信号的β值最小 (0.5-2.0)，最难检测。

---

## 解读结果

### 典型输出示例

```
================================================================================
SUMMARY REPORT
================================================================================

Experiments: 50
True signals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

| Metric | NUMPY | CUPY | TORCH |
|--------|-------|------|-------|
| Selected | 13.1 ± 1.9 | 13.2 ± 1.7 | 12.9 ± 1.2 |
| Power | 100.0% | 100.0% | 100.0% |
| FDR | 6.7% | 7.9% | 6.4% |
| Time (s) | 0.109 ± 0.007 | 0.126 ± 0.124 | 0.119 ± 0.015 |

STATISTICAL COMPARISON (paired t-test): NumPy vs CuPy

| Metric | t 统计量 | p 值 | 显著性 |
|--------|----------|------|--------|
| power | - | - | 无方差 |
| fdr | -0.737 | 0.4649 | ns |
| time | -1.120 | 0.2681 | ns |

WEAK SIGNAL DETECTION (Features 0-3)

NUMPY: Feat0=100%, Feat1=100%, Feat2=100%, Feat3=100%
CUPY: Feat0=100%, Feat1=100%, Feat2=100%, Feat3=100%

CONCLUSIONS

- NUMPY: 100% power achieved (all true signals detected)
- CUPY: 100% power achieved (all true signals detected)
- NUMPY: FDR well-controlled (6.7% <= 10%)
- CUPY: FDR well-controlled (7.9% <= 10%)
```

### 统计显著性解读

| p 值范围 | 显著性标记 | 说明 |
|----------|------------|------|
| p < 0.001 | *** | 极显著差异 |
| p < 0.01 | ** | 高度显著差异 |
| p < 0.05 | * | 显著差异 |
| p ≥ 0.05 | ns | 无显著差异 |

---

## 常见问题

### Q: CuPy 运行时出现 CUDA error

**A:** 检查 CUDA 驱动版本与 CuPy 版本是否匹配：
```bash
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

### Q: R 脚本运行时 glmnet 报错

**A:** 确保已安装 glmnet 包：
```r
install.packages("glmnet")
```

### Q: Torch 后端运行很慢

**A:** 可能是 GPU 未被充分利用，检查：
```bash
nvidia-smi  # 查看 GPU 使用情况
```

### Q: 如何更改实验次数

**A:** 运行脚本时指定第一个参数：
```bash
python dev/scripts/knockoff_50exp_comparison.py 100 0.5  # 100 次实验
```

---

## 文件结构

```
statgpu/
├── dev/
│   ├── scripts/
│   │   ├── knockoff_50exp_comparison.py    # 主对比脚本
│   │   └── merge_knockoff_r_comparison.py  # 结果合并脚本
│   └── docs/
│       ├── knockoff_comparison_results_*.json   # Python 结果
│       ├── knockoff_comparison_report_*.md      # Python 报告
│       ├── knockoff_50exp_r.R                   # R 脚本
│       ├── knockoff_50exp_r_results.json        # R 结果
│       └── knockoff_py_vs_r_report_*.md         # 对比报告
└── statgpu/
    └── feature_selection/
        └── _knockoff_utils.py    # Knockoff 核心实现
```

---

## 参考文档

- [Knockoff 滤波器原理](https://arxiv.org/abs/1410.1671)
- [R knockoff 包文档](https://cran.r-project.org/package=knockoff)
- [statgpu 文档](../README.md)

---

*最后更新：2026-04-19*
