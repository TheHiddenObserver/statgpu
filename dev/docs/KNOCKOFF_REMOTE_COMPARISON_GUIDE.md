# Knockoff 滤波器 50 次实验完整对比（远程执行）

本指南介绍如何在远程服务器上运行 statgpu 与 R knockoff 包的完整对比实验。

---

## 概述

对比包含以下实现：

| 实现 | 框架 | 后端 | 运行环境 |
|------|------|------|----------|
| statgpu NumPy | Python | CPU | myconda |
| statgpu CuPy | Python | GPU | myconda |
| statgpu Torch | Python | GPU | myconda |
| R knockoff SDP | R | CPU | r-build |
| R knockoff Equi | R | CPU | r-build |

---

## 运行方式

### 方式一：自动执行（推荐）

运行本地脚本，自动上传代码到远程服务器并执行：

```bash
python dev/scripts/run_knockoff_full_remote_comparison.py
```

该脚本会：
1. SSH 连接到远程服务器
2. 上传 statgpu 源代码
3. 在 myconda 环境中运行 Python 对比（50 次实验）
4. 在 r-build 环境中运行 R 对比（50 次实验）
5. 下载结果到本地

### 方式二：手动执行

#### 步骤 1：登录远程服务器

```bash
ssh -p 29052 root@hz-4.matpool.com
# 密码：,@w@LTi[z,D,Vsn#
```

#### 步骤 2：运行 Python 对比（myconda 环境）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myconda

# 上传并安装 statgpu（如果需要）
pip install -e /root/cross_lang_test/statgpu_source

# 运行对比脚本
python /root/cross_lang_test/knockoff_full_comparison/run_python.py
```

#### 步骤 3：运行 R 对比（r-build 环境）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate r-build

# 运行 R 脚本
Rscript /root/cross_lang_test/knockoff_full_comparison/run_r.R
```

#### 步骤 4：下载结果

```bash
# 在本地机器上执行
scp -P 29052 root@hz-4.matpool.com:/root/cross_lang_test/knockoff_full_comparison/python_results.json dev/docs/knockoff_remote_python_results.json
scp -P 29052 root@hz-4.matpool.com:/root/cross_lang_test/knockoff_full_comparison/r_results.json dev/docs/knockoff_remote_r_results.json
```

#### 步骤 5：合并结果

```bash
python dev/scripts/merge_knockoff_r_comparison.py
```

---

## 实验配置

### 默认参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 样本量 (n) | 400 | 样本数量 |
| 特征数 (p) | 80 | 特征数量 |
| 真实信号数 | 12 | 特征 0-11 |
| 信号强度 | β = 0.5, 1.0, ..., 6.0 | 递增 |
| 噪声水平 | σ = 0.5 | - |
| 目标 FDR | q = 0.10 | - |
| 实验次数 | 50 | 独立随机种子 |

### 随机种子

50 次实验使用 seeds: 20260406, 20260407, ..., 20260455

---

## 输出文件

运行完成后，`dev/docs/` 目录下会生成以下文件：

| 文件 | 说明 |
|------|------|
| `knockoff_remote_python_results.json` | Python (NumPy/CuPy/Torch) 结果 |
| `knockoff_remote_r_results.json` | R (SDP/Equi) 结果 |
| `knockoff_py_vs_r_report_*.md` | 完整对比报告 |
| `knockoff_py_vs_r_merged_*.json` | 合并后的数据 |

---

## 评估指标

### 主要指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| **Power (检出率)** | TP / 真实信号数 | 100% |
| **FDR** | FP / 选中数 | ≤ 10% |
| **分离度** | 信号 W 均值 / 噪声 W 最大值 | 越高越好 |
| **时间** | 单次实验运行时间 | 越短越好 |

### 弱信号检测

特别关注特征 0-3 的检出率（β = 0.5, 1.0, 1.5, 2.0）

---

## 解读结果

### 典型输出

```
================================================================================
SUMMARY (Python)
================================================================================

NUMPY (50 experiments):
  Selected: 13.1 +/- 1.9
  Power:    100.0%
  FDR:      6.7%
  Time:     0.109s

CUPY (50 experiments):
  Selected: 13.2 +/- 1.7
  Power:    100.0%
  FDR:      7.9%
  Time:     0.126s

================================================================================
R results saved to: r_results.json

================================================================================
SUMMARY (R)
================================================================================

R-SDP (50 experiments):
  Selected: 12.8 +/- 1.5
  Power:    100.0%
  FDR:      5.9%
  Time:     1.234s

R-Equi (50 experiments):
  Selected: 13.5 +/- 1.8
  Power:    100.0%
  FDR:      7.2%
  Time:     0.876s
```

### 统计显著性

| p 值范围 | 标记 | 说明 |
|----------|------|------|
| p < 0.001 | *** | 极显著差异 |
| p < 0.01 | ** | 高度显著差异 |
| p < 0.05 | * | 显著差异 |
| p ≥ 0.05 | ns | 无显著差异 |

---

## 预计运行时间

| 阶段 | 预计时间 |
|------|----------|
| Python 50 次实验 | 10-30 分钟 |
| R 50 次实验 | 10-30 分钟 |
| 总计 | 20-60 分钟 |

---

## 常见问题

### Q: CuPy 运行时出现 CUDA error

**A:** 检查 CUDA 驱动：
```bash
nvidia-smi
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
```

### Q: R 脚本运行时报错

**A:** 确保 r-build 环境已安装所需包：
```bash
conda activate r-build
R -e "install.packages(c('knockoff', 'glmnet', 'jsonlite'))"
```

### Q: 如何修改实验次数

**A:** 编辑脚本顶部的 `N_EXPERIMENTS` 变量：
```python
N_EXPERIMENTS = 50  # 改为其他值
```

### Q: 如何修改信号强度

**A:** 编辑脚本顶部的 `BETA_SCALE` 变量：
```python
BETA_SCALE = 0.5  # 改为其他值（如 1.0 表示更强信号）
```

---

## 文件结构

```
statgpu/
├── dev/
│   ├── scripts/
│   │   ├── run_knockoff_full_remote_comparison.py  # 远程执行主脚本
│   │   ├── knockoff_50exp_comparison.py            # 本地对比脚本
│   │   └── merge_knockoff_r_comparison.py          # 结果合并脚本
│   └── docs/
│       ├── knockoff_remote_python_results.json     # Python 结果
│       ├── knockoff_remote_r_results.json          # R 结果
│       └── knockoff_py_vs_r_report_*.md            # 对比报告
└── statgpu/
    └── feature_selection/
        └── _knockoff_utils.py    # Knockoff 核心实现
```

---

## 参考文档

- [Knockoff 滤波器原理](https://arxiv.org/abs/1410.1671)
- [R knockoff 包文档](https://cran.r-project.org/package=knockoff)
- [统计gpu 文档](../README.md)

---

*最后更新：2026-04-19*
