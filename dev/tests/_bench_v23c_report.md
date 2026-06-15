# statgpu v23c 完整基准测试报告

> 2026-05-20 | 远程服务器: hz-4.matpool.com:28774 | Conda env: myconda

## 概览

| Section | 描述 | 通过率 | 状态 |
|---------|------|--------|------|
| A | 跨后端时间+精度 | 816/816 | ALL PASS |
| B | vs sklearn | 13/13 | ALL PASS |
| D | vs statsmodels | 68/68 | ALL PASS |
| E | 跨求解器一致性 | 146/146 | ALL PASS |
| **总计** | | **1043/1043** | **100%** |

Grade 分布: OK=155, ~=65, OK(obj)=7, MISMATCH=0

## Section A: 跨后端时间 (816 tests)

测试矩阵: 7 families x 10 penalties x 3 scales x 多求解器 x 3 backends (CPU/CuPy/Torch)

### 各规模平均耗时

| 规模 | 测试数 | CPU avg (ms) | CuPy avg (ms) | Torch avg (ms) | CuPy加速 | Torch加速 |
|------|--------|-------------|---------------|----------------|---------|----------|
| n=500, p=50 | 160 | 953.2 | 957.4 | 953.7 | 1.00x | 1.00x |
| n=2000, p=200 | 160 | 3994.8 | 15599.4 | 9107.6 | 0.26x | 0.44x |
| n=5000, p=500 | 88 | 2875.1 | 2167.5 | 1313.1 | 1.33x | 2.19x |

### 分析

- **n=500**: GPU overhead > 计算收益, CPU/GPU 基本持平
- **n=2000**: GPU overhead 显著, CPU 更快 (非平滑惩罚迭代次数多, GPU kernel launch 开销大)
- **n=5000**: GPU 优势明显, Torch 平均 2.19x 加速

### n=5000 各求解器 GPU 加速

| 求解器 | CuPy avg | Torch avg |
|--------|----------|-----------|
| fista | 1.52x | 2.56x |
| newton | 1.88x | 2.10x |
| irls | 2.31x | 2.40x |
| exact | 1.74x | 1.91x |

## Section B: Precision vs sklearn (13 tests, n=1000, p=50)

所有 13 个组合全部通过 (max|diff| < 1e-3 或 obj 更优)。

关键指标:
- squared_error+l1: max|diff| = 5.24e-06
- squared_error+l2: max|diff| = 9.77e-15
- logistic+l2: max|diff| = 1.36e-04 (lbfgs, 修复后)
- poisson+l2: max|diff| = 2.56e-04 (lbfgs, 修复后)
- gamma+l2: max|diff| = 6.09e-05 (lbfgs, 修复后)
- tweedie+l2: max|diff| = 5.80e-05 (lbfgs, 修复后)

## Section D: Precision vs statsmodels (68 tests, n=500, p=50)

所有 68 个组合全部通过。

关键指标:
- squared_error: max|diff| 普遍 < 1e-06
- logistic+l2: max|diff| = 5.51e-03 (OK(obj))
- logistic+none: max|diff| = 3.49e-04
- poisson+l2: max|diff| = 1.62e-03
- gamma+l2: max|diff| = 1.42e-03
- NB+l2: max|diff| = 8.03e-04

## Section E: 跨求解器一致性 (146 tests, CPU, n=2000, p=200)

所有 146 个组合全部通过。

目标函数值差异普遍在 1e-10 ~ 1e-13 级别, 表明所有求解器收敛到相同最优点。

## lbfgs 修复说明

### 问题

v22g 中 9 个 lbfgs+l2 组合显示 MISMATCH (max|diff| 最高 1.58e-01)。

### 根因

`lbfgs_solver` 的 fused GLM 路径中, 梯度只计算了 loss 部分, 遗漏了 penalty 梯度。
L-BFGS 收敛到 `loss_grad ≈ 0` (无正则化解) 而非正确的 `loss_grad + α·coef = 0`。

### 修复

在 `_solver.py` 的 `lbfgs_solver` 函数中, 在两处 `_fused_glm_value_and_gradient` 调用后
添加 `_smooth_penalty_gradient(penalty, coef)`:

1. 初始梯度计算
2. 迭代中梯度更新

### 修复效果

| 组合 | 修复前 max\|diff\| | 修复后 max\|diff\| |
|------|-------------------|-------------------|
| logistic+l2 vs sklearn | 1.13e-01 | 1.36e-04 |
| poisson+l2 vs sklearn | 2.16e-03 | 2.56e-04 |
| gamma+l2 vs sklearn | 1.27e-02 | 6.09e-05 |
| tweedie+l2 vs sklearn | 7.31e-03 | 5.80e-05 |
| logistic+l2 vs statsmodels | 2.75e-01 | 5.51e-03 |
| gamma+l2 vs statsmodels | 1.24e-02 | 1.42e-03 |
| NB+l2 vs statsmodels | 1.58e-01 | 8.03e-04 |
| tweedie+l2 vs statsmodels | 8.62e-03 | 1.31e-03 |
| NB+l2 跨求解器 | 5.32e-03 | 5.92e-08 |

## 测试环境

- GPU: NVIDIA (CUDA 11.7)
- CuPy: 13.6.0
- PyTorch: 2.0.0+cu117
- Python: 3.9 (conda myconda)
- 远程服务器: hz-4.matpool.com:28774

## 审计产物

- Section A 原始输出: `dev/tests/_bench_v22g_sectionA.txt`
- Section B/D/E 原始输出: `dev/tests/_bench_v23c_sectionBDE.txt`
- 基准测试脚本: `dev/tests/_bench_full_matrix.py`
