# 面板数据 (Panel Data) GPU 实现计划

**创建日期**: 2026-04-19  
**版本**: 0.1.0  
**状态**: 🔶 核心实现 (~45%) (2026-06-13, PR #57)

> 已实现: `PanelOLS` (单/双向固定效应), `RandomEffects` (Swamy-Arora), `PanelSummary`, `clustered_covariance`, `two_way_clustered_covariance`
> 缺失: PooledOLS, FamaMacBeth, BetweenOLS, FirstDifferenceOLS, IV 模型, HAC/Newey-West, Driscoll-Kraay

---

## 一、背景与目标

面板数据 (Panel Data) 同时包含截面维度 (cross-section) 和时间维度 (time-series)，在计量经济学、金融学、社会科学等领域有广泛应用。本计划旨在将主流的面板数据模型和检验方法用 GPU 加速实现，纳入 StatGPU 框架。

---

## 二、现有包功能调研

### 2.1 Python 生态系统

#### 2.1.1 linearmodels (推荐参考实现)

**包信息**: 
- 最新版本：7.0
- 文档：https://bashtage.github.io/linearmodels/
- 维护者：Kevin Sheppard (bashtage)

**核心功能**:

| 模型/功能 | 说明 | GPU 实现优先级 |
|-----------|------|---------------|
| `PanelOLS` | 固定效应模型 (单向/双向) | P0 |
| `RandomEffects` | 随机效应模型 | P0 |
| `FirstDifferenceOLS` | 一阶差分模型 | P1 |
| `BetweenOLS` | 组间回归模型 | P2 |
| `FamaMacBeth` | Fama-MacBeth 回归 | P1 |
| `IV2SLS` | 工具变量 2SLS | P2 |
| `GMM` | 广义矩估计 | P2 |

**协方差估计**:
- 稳健标准误 (HC0-HC3)
- 聚类标准误 (单维/多维)
- HAC (异方差自相关一致) 标准误
- Driscoll-Kraay 标准误

**假设检验**:
- F 检验 (模型整体显著性)
- Wald 检验 (线性约束)
- Breusch-Pagan LM 检验 (随机效应)
- Hausman 检验 (FE vs RE)

---

#### 2.1.2 statsmodels

**包信息**:
- 最新版本：0.14.6
- 文档：https://www.statsmodels.org/stable/

**面板相关功能**:

| 模块 | 功能 | 说明 |
|------|------|------|
| `MixedLM` | 线性混合效应模型 | 随机效应/多层模型 |
| `GLSAR` | GLS + 自相关误差 | AR(p) 误差结构 |
| `GEE` | 广义估计方程 | 相关数据结构 |
| `cov_type='hac-panel'` | 面板 HAC 协方差 | 异方差 + 自相关 |

**注意**: statsmodels 没有专门的面板数据模块，面板分析通常使用 `linearmodels` 或通过稳健协方差实现。

---

#### 2.1.3 scikit-learn

**间接支持**:
- `LinearRegression` + 手动固定效应编码
- `Ridge`/`Lasso` + 面板数据结构
- 无原生面板数据支持

---

### 2.2 R 语言生态系统

#### 2.2.1 plm (Panel Data Econometrics)

**包信息**:
- 最新版本：2.6-7 (2025 年 11 月)
- 文档：https://cran.r-project.org/package=plm
- GitHub: https://github.com/ycroissant/plm

**核心模型**:

| 函数 | 模型类型 | 说明 |
|------|----------|------|
| `plm()` | 基础面板模型 | within/fixed, random, pooling |
| `pgmm()` | GMM 面板模型 | Arellano-Bond, Blundell-Bond |
| `pggls()` | 广义可行 GLS | 组间异方差/自相关 |
| `pvcm()` | 变系数模型 | varying coefficients |

**假设检验函数**:

| 函数 | 检验类型 | 用途 |
|------|----------|------|
| `phtest()` | Hausman 检验 | FE vs RE 选择 |
| `pbgtest()` | Breusch-Godfrey | 序列相关检验 |
| `pwartest()` | Wooldridge AR(1) | 固定效应序列相关 |
| `pbsytest()` | Bera-Sosa-Yoon | 序列相关检验 |
| `pdwtest()` | Durbin-Watson | 序列相关检验 |
| `pbltest()` | Baltagi-Li | 随机效应序列相关 |
| `pwtest()` | Wooldridge | 不可观测效应检验 |
| `pwfdtest()` | Wooldridge FD | 固定效应检验 |
| `pcdtest()` | 截面相关检验 | 跨截面依赖 |
| `plmtest()` | LM 检验 | 个体/时间效应存在性 |
| `pFtest()` | F 检验 | 固定效应显著性 |

**稳健协方差**:
- `vcovHC()`: 异方差一致 (HC0-HC3)
- `vcovSC()`: 自相关 + 异方差一致
- `vcovDC()`: 双维聚类
- `vcovBK()`: Beck-Katz PCSE

---

#### 2.2.2 fixest (Fast Fixed-Effects)

**包信息**:
- 最新版本：0.14.0 (2026 年 3 月)
- 文档：https://lrberge.github.io/fixest/
- GitHub: https://github.com/lrberge/fixest

**核心特点**:
- **高性能**: C++ 后端，多重固定效应快速估计
- **灵活公式**: `feols(y ~ x1 | fe1 + fe2, data)`
- **丰富功能**:
  - `feols()`: OLS 固定效应
  - `feglm()`: GLM 固定效应 (Logit/Probit/Poisson)
  - `feNmls()`: 非线性最小二乘
  - `feIV()`: 工具变量估计

**推断功能**:
- 多重聚类标准误
- 自助法标准误
- 多重假设检验校正
- 事件研究法 (Event Study)
- 双重差分 (DID)

---

#### 2.2.3 lfe (Linear Fixed Effects)

**包信息**:
- 核心函数：`felm()`
- 特点：多重固定效应，替代 fixest

---

#### 2.2.4 其他重要包

| 包名 | 功能 | 说明 |
|------|------|------|
| `panelvar` | 面板 VAR | 动态面板向量自回归 |
| `panelr` | 面板数据处理 | `panel_data()` 结构 |
| `pder` | 面板计量经济学 | 配套教材的工具包 |
| `AER` | 应用计量 | 包含面板数据工具 |
| `lmtest` | 计量检验 | 与 plm 配合使用 |
| `sandwich` | 稳健协方差 | 各类 HC/HAC 估计 |

---

### 2.3 功能对比总览

| 功能类别 | Python (linearmodels) | R (plm/fixest) | GPU 实现可行性 |
|----------|----------------------|---------------|----------------|
| 固定效应模型 | ✅ PanelOLS | ✅ feols/plm | ✅ 高 (矩阵运算) |
| 随机效应模型 | ✅ RandomEffects | ✅ plm(random) | ✅ 中 (需要方差分量) |
| 一阶差分 | ✅ FirstDifference | ✅ diff/transform | ✅ 高 |
| 组间回归 | ✅ BetweenOLS | ✅ between | ✅ 高 |
| 工具变量 | ✅ IV2SLS | ✅ feIV/pgmm | ✅ 中 (2SLS 可并行) |
| GMM | ✅ GMM | ✅ pgmm | ✅ 中 (矩条件并行) |
| Fama-MacBeth | ✅ | ❌ | ✅ 高 (截面回归并行) |
| 稳健标准误 | ✅ 多种 | ✅ 多种 | ✅ 高 (面包公式) |
| 聚类标准误 | ✅ 单/多维 | ✅ 单/多维 | ✅ 高 |
| HAC 标准误 | ✅ | ✅ vcovSC | ⚠️ 中 (核函数计算) |
| Hausman 检验 | ✅ | ✅ phtest | ✅ 高 |
| 序列相关检验 | ⚠️ 部分 | ✅ 丰富 | ✅ 高 (辅助回归) |
| 截面相关检验 | ❌ | ✅ pcdtest | ✅ 高 (残差相关) |

---

## 三、GPU 加速机会分析

### 3.1 可并行化的核心计算

| 计算任务 | 并行策略 | 预期加速比 |
|----------|----------|------------|
| 组内变换 (Within Transformation) | 按个体/时间并行 | 5-10x |
| 固定效应投影 | 分块矩阵运算 | 10-20x |
| 大型矩阵求逆 | GPU 批处理 | 20-50x |
| Bootstrap 检验 | 完全并行 | 线性扩展 |
| 多重聚类标准误 | 按聚类组并行 | 5-10x |
| 序列相关检验 | 按个体并行 | 10-20x |
| Fama-MacBeth | 按时间截面并行 | 20-50x |
| GMM 矩条件 | 按样本/矩并行 | 10-30x |

### 3.2 内存考量

| 数据类型 | 内存占用 (N×T×K) | GPU 内存策略 |
|----------|------------------|-------------|
| 小面板 (N<100, T<20) | <1MB | 全部驻留显存 |
| 中面板 (N~1000, T~50) | ~100MB | 分块处理 |
| 大面板 (N>10000, T>100) | >1GB | 流式处理 + 压缩 |

---

## 四、实现计划

### 4.1 模块结构建议

```
statgpu/
├── panel/
│   ├── __init__.py
│   ├── _models/
│   │   ├── __init__.py
│   │   ├── _fixed_effects.py      # PanelOLS, 固定效应
│   │   ├── _random_effects.py     # RandomEffects
│   │   ├── _first_difference.py   # FirstDifferenceOLS
│   │   ├── _between.py            # BetweenOLS
│   │   ├── _fama_macbeth.py       # FamaMacBeth
│   │   └── _iv_gmm.py             # IV2SLS, GMM
│   ├── _covariance/
│   │   ├── __init__.py
│   │   ├── _robust.py             # HC0-HC3
│   │   ├── _clustered.py          # 聚类标准误
│   │   └── _hac.py                # HAC/Driscoll-Kraay
│   ├── _tests/
│   │   ├── __init__.py
│   │   ├── _hausman.py            # Hausman 检验
│   │   ├── _serial_corr.py        # 序列相关检验
│   │   ├── _heteroskedasticity.py # 异方差检验
│   │   └── _cross_section.py      # 截面相关检验
│   └── _utils/
│       ├── __init__.py
│       ├── _transformations.py    # Within/Between 变换
│       └── _panel_data.py         # 面板数据结构
```

---

### 4.2 优先级与时间表

#### 🔴 P0 - 核心模型 (4-6 周)

| 功能 | 预估工作量 | 依赖 |
|------|------------|------|
| `PanelOLS` (固定效应) | 2 周 | - |
| `RandomEffects` (随机效应) | 1 周 | PanelOLS |
| 稳健标准误 (HC0-HC3) | 1 周 | PanelOLS |
| 聚类标准误 (单维) | 1 周 | PanelOLS |
| `FixedEffectsCV` | 1 周 | PanelOLS |

**验收标准**:
- API 与 linearmodels/sklearn 兼容
- 支持 CuPy/Torch/Numpy 三后端
- 通过单元测试 (与 linearmodels 结果对比)
- 基准测试显示 5x+ 加速 (大样本)

---

#### 🟡 P1 - 扩展模型 (4-6 周)

| 功能 | 预估工作量 | 依赖 |
|------|------------|------|
| `FirstDifferenceOLS` | 1 周 | PanelOLS |
| `BetweenOLS` | 0.5 周 | PanelOLS |
| `FamaMacBeth` | 1 周 | PanelOLS |
| 聚类标准误 (多维) | 1 周 | P0 |
| HAC 标准误 | 1.5 周 | P0 |
| Hausman 检验 | 1 周 | P0 |
| 序列相关检验 (基础) | 1 周 | P0 |

---

#### 🟢 P2 - 高级功能 (6-8 周)

| 功能 | 预估工作量 | 依赖 |
|------|------------|------|
| `IV2SLS` (工具变量) | 2 周 | P1 |
| `GMM` (广义矩) | 2 周 | P1 |
| 截面相关检验 | 1 周 | P1 |
| 完整序列相关检验 | 1.5 周 | P1 |
| 异方差检验 | 1 周 | P1 |
| 自助法推断 | 1.5 周 | P1 |

---

### 4.3 测试策略

**单元测试**:
- 与 linearmodels 对比 (Python)
- 与 plm/fixest 对比 (R)
- 边缘情况：N=1, T=1, 不平衡面板

**基准测试**:
- 小/中/大面板性能对比
- CPU vs GPU 加速比
- 不同后端 (CuPy vs Torch) 对比

**数值精度**:
- 相对误差 < 1e-6 (与 CPU 参考实现)
- 标准误/检验统计量精度验证

---

### 4.4 数据接口设计

```python
from statgpu.panel import PanelOLS, RandomEffects
from statgpu.panel.covariance import clustered_cov
from statgpu.panel.tests import hausman_test

# 数据格式：支持 DataFrame 和数组
# 必需：entity_ids, time_ids, y, X

# 固定效应模型
mod = PanelOLS(entity_effects=True, time_effects=False)
results = mod.fit(y, X, entity_ids=firm_id, time_ids=year)

# 带聚类标准误
results = mod.fit(y, X, entity_ids=firm_id, time_ids=year,
                  cluster='entity')

# Hausman 检验
fe_mod = PanelOLS(entity_effects=True)
re_mod = RandomEffects()
stat, pval = hausman_test(y, X, entity_ids=firm_id)
```

---

## 五、技术挑战与解决方案

### 5.1 不平衡面板

**挑战**: 个体/时间维度缺失导致不规则数据结构

**解决方案**:
- 使用稀疏矩阵表示
- 分组索引加速查找
- 填充/掩码处理

---

### 5.2 大型固定效应投影

**挑战**: N 很大时虚拟变量矩阵过大

**解决方案**:
- 使用投影矩阵技巧 (避免显式构造)
- 利用分块对角结构
- 迭代求解 (共轭梯度)

---

### 5.3 多维聚类标准误

**挑战**: 多维权重矩阵计算复杂

**解决方案**:
- 按聚类组并行
- 使用 CGM (Cameron-Gelbach-Miller) 公式
- 预计算交叉项

---

### 5.4 自相关核函数 (HAC)

**挑战**: 核函数计算序列化

**解决方案**:
- 使用 Bartlett/Parzen/Newey-West 向量化
- 截断滞后并行计算
- 自适应滞后选择批处理

---

## 六、与现有 StatGPU 模块集成

### 6.1 依赖关系

| 面板模块 | 依赖的现有模块 |
|----------|----------------|
| 模型估计 | `linear_model` (Ridge/Lasso) |
| 假设检验 | `inference` (bootstrap, permutation) |
| 协方差估计 | `inference` (分布函数) |
| 数据变换 | (新增) |

### 6.2 后端复用

- CuPy: 矩阵运算、稀疏矩阵
- Torch: 自动微分 (可选)、批处理
- Numpy: CPU 参考实现

---

## 七、基准测试计划

### 7.1 测试场景

| 场景 | N | T | K | 说明 |
|------|---|---|---|------|
| 微观面板 | 1000 | 10 | 5 | 企业/家庭调查 |
| 中观面板 | 5000 | 20 | 10 | 行业/地区数据 |
| 宏观面板 | 100 | 50 | 15 | 国家/省份数据 |
| 大 N 小 T | 50000 | 5 | 8 | 横截面为主 |
| 小 N 大 T | 50 | 200 | 12 | 时间序列为主 |

### 7.2 对比基线

- Python: linearmodels 7.0
- R: plm 2.6-7, fixest 0.14.0
- CPU: statsmodels (参考)

### 7.3 性能目标

| 操作 | 目标加速比 (GPU vs CPU) |
|------|------------------------|
| 固定效应估计 | 10-20x |
| 稳健标准误 | 5-10x |
| 聚类标准误 | 10-15x |
| Hausman 检验 | 5-10x |
| 序列相关检验 | 10-20x |

---

## 八、参考文献与资源

### 8.1 计量经济学教材

1. **Baltagi, B. H.** (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
2. **Wooldridge, J. M.** (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
3. **Arellano, M.** (2003). *Panel Data Econometrics*. Oxford University Press.

### 8.2 软件文档

- **linearmodels**: https://bashtage.github.io/linearmodels/
- **plm**: https://cran.r-project.org/package=plm
- **fixest**: https://lrberge.github.io/fixest/
- **statsmodels**: https://www.statsmodels.org/stable/

### 8.3 关键论文

1. **Driscoll, J. C., & Kraay, A. C.** (1998). Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data. *Review of Economics and Statistics*, 80(4), 549-560.
2. **Cameron, A. C., Gelbach, J. B., & Miller, D. L.** (2011). Robust Inference with Multiway Clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.
3. **Arellano, M., & Bond, S.** (1991). Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations. *Review of Economic Studies*, 58(2), 277-297.

---

## 九、下一步行动

### 立即可开始

1. [ ] 阅读 linearmodels 的 PanelOLS 源码
2. [ ] 设计 `PanelData` 数据结构
3. [ ] 实现 Within 变换的 GPU 版本
4. [ ] 创建基础单元测试框架

### 本周目标

- [ ] 完成 `PanelOLS` 的骨架代码
- [ ] 实现固定效应投影的 GPU 核函数
- [ ] 编写 API 设计文档

### 本月目标

- [ ] `PanelOLS` 完整实现 (含三后端)
- [ ] 稳健标准误和聚类标准误
- [ ] 与 linearmodels 的基准对比测试

---

## 十、修订历史

| 日期 | 版本 | 修改内容 | 作者 |
|------|------|----------|------|
| 2026-04-19 | 0.1.0 | 初始版本 | - |

---

**备注**: 本计划基于 2025-2026 年 Python (linearmodels, statsmodels) 和 R (plm, fixest) 面板数据包的最新功能调研。实现细节可能根据实际开发进度调整。
