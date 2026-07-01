# statgpu vs External Frameworks: 3-Backend Precision + Performance

Date: 2026-06-24
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)

## Panel Data: statgpu vs linearmodels 6.1

### PanelOLS (Entity Fixed Effects) — 100K obs, 20 vars

| Backend | statgpu (s) | linearmodels (s) | Speedup | coef rel diff |
|---------|------------|-----------------|---------|---------------|
| numpy | 0.338 | 0.425 | 1.3x | 1.0e-15 |
| cupy | 0.037 | 0.425 | **11.6x** | 5.3e-16 |
| torch | 0.025 | 0.425 | **16.7x** | 5.0e-16 |

### RandomEffects — 100K obs, 20 vars

| Backend | statgpu (s) | linearmodels (s) | Speedup | coef rel diff |
|---------|------------|-----------------|---------|---------------|
| numpy | 1.013 | 0.496 | 0.5x | 1.7e-3 |
| cupy | 0.154 | 0.496 | **3.2x** | 1.7e-3 |
| torch | 0.147 | 0.496 | **3.4x** | 1.7e-3 |

**Precision:** PanelOLS 机器精度 (1e-15)。RE 微小差异 (1.7e-3) 因方差分量估计方法不同。

## GAM: statgpu vs pygam 0.10.1 (matched lam=0.6)

### 100K obs, 10 features, 20 splines

| Backend | statgpu (s) | pygam (s) | Speedup | pred rel diff |
|---------|------------|----------|---------|---------------|
| numpy | 1.882 | 5.705 | 3.0x | 2.50e-2 |
| cupy | 0.125 | 5.705 | **45.8x** | 2.50e-2 |
| torch | 0.111 | 5.705 | **51.3x** | 2.50e-2 |

### Auto GCV comparison

| Backend | statgpu lam | pygam lam | pred rel diff |
|---------|------------|----------|---------------|
| numpy | 0.313 | 3.981 | 2.33e-2 |
| torch | 0.313 | 3.981 | 2.33e-2 |

**精度分析：** 即使使用相同 lam，预测差异仍有 2.5%。原因：
1. **结点位置不同**：statgpu 用分位数结点，pygam 用均匀结点
2. **基函数归一化不同**：B-spline 基矩阵的缩放方式不同
3. **惩罚矩阵不同**：statgpu 用差分惩罚，pygam 用 P-spline 惩罚
4. **GCV 实现不同**：statgpu lam=0.31 vs pygam lam=3.98（差 10 倍）

两者都是正确的正则化估计，只是正则化路径不同。

## ANOVA: statgpu vs scipy 1.13.1

### f_oneway — 2M total observations

| Backend | statgpu (ms) | scipy (ms) | Speedup | F rel diff |
|---------|-------------|-----------|---------|------------|
| numpy | 17.24 | 17.33 | 1.0x | 5.1e-15 |
| cupy | 7.73 | 17.33 | **2.2x** | 5.5e-15 |
| torch | 18.98 | 17.33 | 0.9x | 5.3e-15 |

### f_twoway — 2M total observations

| Backend | statgpu (ms) | F_A |
|---------|-------------|-----|
| numpy | 37.78 | 35765.1 |
| cupy | 11.17 | 35765.1 |
| torch | 41.79 | 35765.1 |

**Precision:** F 统计量机器精度 (1e-15)。

## Summary: 3-Backend Performance vs External

| Module | External | numpy spd | cupy spd | torch spd | Precision |
|--------|----------|-----------|----------|-----------|-----------|
| PanelOLS | linearmodels | 1.3x | 11.6x | **16.7x** | 1e-15 |
| RandomEffects | linearmodels | 0.5x | 3.2x | 3.4x | 1.7e-3 |
| GAM | pygam | 3.0x | 45.8x | **51.3x** | 2.5% |
| f_oneway | scipy | 1.0x | 2.2x | 0.9x | 1e-15 |

**Key findings:**
1. **GAM torch 51.3x** — 最大加速，但精度差异 2.5%（算法不同）
2. **PanelOLS torch 16.7x** — 机器精度，完美匹配
3. **ANOVA cupy 2.2x** — 机器精度
4. **cupy 和 torch 性能接近**，torch 略快
5. **numpy 在小问题上最快**（无 GPU 开销）
