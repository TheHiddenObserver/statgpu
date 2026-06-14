---
name: ANOVA GPU 实现计划
description: Python 和 R 中方差分析 (ANOVA) 功能调研及 GPU 实现路线图
type: project
status: 🔶 基础实现 (~15%) (2026-06-13, PR #57)
---

# 方差分析 (ANOVA) GPU 实现计划

**状态**: 🔶 基础实现 (~15%)

> 已实现: `f_oneway` (单因素), `AnovaResult` — 支持 float32/float64, CPU/CuPy/Torch
> 缺失: 二因素/N因素 ANOVA, 重复测量, Welch ANOVA, 事后检验 (Tukey HSD), 效果量 (Cohen's f)

## 一、现有实现调研

### 1.1 Python 生态

#### SciPy (`scipy.stats`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 单因素 ANOVA | `f_oneway()` | 返回 F 值和 p 值 |
| Welch ANOVA | `f_oneway(..., equal_var=False)` | 不等方差假设 |
| 假设检验 | ✓ | 仅假设检验，无模型估计 |
| 效应量 | ✗ | 需手动计算 |
| 多因素 ANOVA | ✗ | 不支持 |
| 重复测量 | ✗ | 不支持 |

**特点**：
- 轻量级，仅基础假设检验
- 输入为原始数据数组
- 适用于快速单因素检验

#### Statsmodels (`statsmodels.stats.anova`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 一般线性模型 ANOVA | `anova_lm()` | 支持 OLS, WLS, GLS |
| 平方和类型 | Type I, II, III | `typ` 参数控制 |
| 重复测量 ANOVA | `AnovaRM` | 仅支持平衡设计 |
| 模型比较 | ✓ | 可比较嵌套模型 |
| 完整 ANOVA 表 | ✓ | F 值、p 值、自由度、平方和 |
| 多因素 ANOVA | ✓ | 通过公式接口 |
| 协方差分析 (ANCOVA) | ✓ | 通过回归框架 |

**特点**：
- 完整的统计推断框架
- 基于公式接口 (`formula`)
- 支持复杂实验设计
- 输出完整统计表格

#### Pingouin (`pingouin`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 单因素/多因素 ANOVA | `anova()` | 统一接口 |
| 重复测量 | `rm_anova()` | 支持被试内设计 |
| 混合设计 ANOVA | `anova()` | between/within 因子 |
| 效应量 | ✓ | η², ω², ηp² 自动计算 |
| 假设检验 | ✓ | 球形度、正态性等 |
| 事后检验 | ✓ | 集成 Tukey, Bonferroni |
| 贝叶斯 ANOVA | `bayesfactor_anova()` | 贝叶斯因子 |

**特点**：
- 最完整的 ANOVA 功能覆盖
- 自动效应量计算
- 完善的假设检验
- 用户友好输出

---

### 1.2 R 生态

#### Base R (`stats` 包)

| 功能 | 实现 | 说明 |
|------|------|------|
| 方差分析模型 | `aov()` | 拟合 ANOVA 模型 |
| ANOVA 表 | `anova()` | 生成标准 ANOVA 表 |
| 单因素/多因素 | ✓ | 通过公式指定 |
| 重复测量 | ✓ | `Error()` 项指定 |
| 平方和类型 | Type I (默认) | 顺序平方和 |
| 对比分析 | `contrasts()` | 自定义对比 |
| 事后检验 | `TukeyHSD()` | Tukey 诚实显著差异 |

**特点**：
- R 语言标准实现
- 与 `lm()` 共享底层架构
- 仅支持 Type I 平方和
- 需要完整数据 (无缺失值)

#### Car 包 (`car`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 高级 ANOVA 表 | `Anova()` | 支持多种模型类型 |
| 平方和类型 | Type I, II, III | 适用于不平衡设计 |
| 模型支持 | lm, glm, lmer 等 | 广泛兼容 |
| 类型 III 检验 | ✓ | 适合主效应 + 交互作用 |
| 多元线性假设 | `linearHypothesis()` | 一般线性假设检验 |
| 效应量 | `Anova(..., type=3)` + 手动计算 | |

**特点**：
- 处理不平衡设计的标准工具
- Type II/III 平方和实现
- 与多种模型对象兼容

#### Afex 包 (`afex`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 因子实验 ANOVA | `aov_ez()`, `aov_car()`, `aov_4()` | 便捷接口 |
| 被试间/被试内/混合 | ✓ | 自动识别设计 |
| 与 emmeans 集成 | ✓ | 自动传递结果 |
| 球形度校正 | ✓ | Greenhouse-Geisser, Huynh-Feldt |
| 混合模型支持 | ✓ | 通过 lme4 后端 |
| 效应量 | ✓ | 广义 eta 平方 |

**特点**：
- 心理学/神经科学领域标准
- 自动处理复杂设计
- 与事后检验无缝衔接

#### Emmeans 包 (`emmeans`)

| 功能 | 实现 | 说明 |
|------|------|------|
| 估计边际均值 | `emmeans()` | 最小二乘均值 |
| 成对比较 | `pairs()` | 事后两两比较 |
| 自定义对比 | `contrast()` | 任意线性对比 |
| 交互作用分析 | ✓ | 简单效应分析 |
| 趋势分析 | `emtrends()` | 斜率比较 |
| 多重比较校正 | ✓ | Tukey, Bonferroni, FDR 等 |
| 效应量 | ✓ | Cohen's d 等 |

**特点**：
- 事后检验的黄金标准
- 支持几乎所有模型类型
- 灵活的对比系统

---

### 1.3 功能对比总览

| 功能类别 | Python (SciPy) | Python (Statsmodels) | Python (Pingouin) | R (aov) | R (car::Anova) | R (afex+emmeans) |
|----------|----------------|----------------------|-------------------|---------|----------------|------------------|
| **假设检验** | | | | | | |
| 单因素 ANOVA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 多因素 ANOVA | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 重复测量 | ✗ | ✓ (AnovaRM) | ✓ | ✓ | ✓ | ✓ |
| 混合设计 | ✗ | 有限 | ✓ | ✓ | ✓ | ✓ |
| 协方差分析 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **模型估计** | | | | | | |
| 平方和计算 | 基础 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Type II/III | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 效应量 | ✗ | 手动 | ✓ | 手动 | 手动 | ✓ |
| **事后检验** | | | | | | |
| 成对比较 | ✗ | 有限 | ✓ | ✓ (TukeyHSD) | ✗ | ✓ |
| 自定义对比 | ✗ | ✗ | 有限 | 有限 | 有限 | ✓ |
| 多重校正 | ✗ | 有限 | ✓ | 有限 | ✗ | ✓ |

---

## 二、GPU 实现可行性分析

### 2.1 计算瓶颈分析

ANOVA 的主要计算步骤及 GPU 加速潜力：

| 计算步骤 | 计算复杂度 | GPU 加速潜力 | 说明 |
|----------|------------|--------------|------|
| 组内/组间平方和 | O(n) | 高 | 并行归约操作 |
| 均值计算 | O(n) | 高 | 并行归约 |
| F 统计量 | O(1) | 低 | 标量计算 |
| p 值计算 (F 分布) | O(1) | 中 | 特殊函数，可用 GPU 数学库 |
| 置换检验 (Permutation) | O(n × permutations) | 极高 | 完美并行 |
| 事后两两比较 | O(k² × n) | 高 | 多组间比较可并行 |
| 多元 ANOVA (MANOVA) | O(p³) | 高 | 矩阵运算 |

### 2.2 现有 GPU 实现参考

#### CUdanova (GitHub: jubbens/cudanova)
- 实现：**置换多元方差分析 (PERMANOVA)**
- 技术栈：CUDA, 多 GPU 支持
- 适用场景：高维数据，大量置换

#### CuPy 生态
- 与 NumPy/SciPy API 兼容
- 已有 GPU 版统计函数：`cupyx.scipy.stats`
- 可直接移植 SciPy 代码

---

## 三、GPU ANOVA 实现计划

### 3.1 实现目标

基于现有 `statgpu` 架构（支持 CuPy/PyTorch/Numpy 后端），实现：

1. **假设检验功能**
   - 单因素 ANOVA (One-way)
   - 多因素 ANOVA (Two-way, N-way)
   - 重复测量 ANOVA
   - 协方差分析 (ANCOVA)

2. **模型估计功能**
   - 平方和分解 (Type I, II, III)
   - 效应量计算 (η², ω², ηp²)
   - 完整 ANOVA 表生成

3. **事后检验功能**
   - 成对比较 (Tukey, Bonferroni, Sidak)
   - 自定义对比
   - 简单效应分析

### 3.2 阶段规划

#### 第一阶段：基础单因素 ANOVA (2-3 周)

**目标**：实现 `f_oneway` 的 GPU 加速版本

**任务**：
- [ ] 在 `statgpu/hypothesis/` 下创建 `_anova.py`
- [ ] 实现核心计算：
  - 组均值、总均值 (GPU 并行归约)
  - 组间平方和 (SSB)、组内平方和 (SSW)
  - F 统计量
  - p 值 (使用 `cupyx.scipy.special` 或 `torch.distributions`)
- [ ] 后端适配：
  - CuPy 后端 (`_cupy.py` 扩展)
  - PyTorch 后端 (`_torch.py` 扩展)
  - NumPy 后端 (参考实现)
- [ ] 与 SciPy `f_oneway` 输出对齐
- [ ] 单元测试验证数值一致性

**文件结构**：
```
statgpu/
├── hypothesis/
│   ├── __init__.py
│   └── _anova.py          # 新增：ANOVA 主实现
├── backends/
│   ├── _cupy.py           # 扩展：ANOVA 核函数
│   ├── _torch.py          # 扩展：ANOVA 核函数
│   └── _numpy.py          # 扩展：参考实现
```

---

#### 第二阶段：多因素 ANOVA 与平方和类型 (3-4 周)

**目标**：支持多因素设计和 Type I/II/III 平方和

**任务**：
- [ ] 实现一般线性模型 (GLM) 框架
- [ ] 设计矩阵构建 (GPU 加速)
- [ ] 平方和分解：
  - Type I (顺序)：sequential SS
  - Type II (调整)：adjusted for other main effects
  - Type III (边际)：adjusted for all effects
- [ ] 交互作用项处理
- [ ] 公式解析接口 (可选：依赖 `patsy`)
- [ ] 与 Statsmodels `anova_lm` 输出对齐

**关键算法**：
- Type I: 顺序拟合，逐个添加预测变量
- Type II/III: 使用对比矩阵和投影矩阵
- 核心：Hat 矩阵 H = X(X'X)⁻¹X' 的 GPU 实现

---

#### 第三阶段：重复测量与混合设计 (3-4 周)

**目标**：支持被试内设计和混合设计 ANOVA

**任务**：
- [ ] 被试内因子建模
- [ ] 误差项分解 (被试间 vs 被试内)
- [ ] 球形度假设检验 (Mauchly 检验)
- [ ] 自由度校正 (Greenhouse-Geisser, Huynh-Feldt)
- [ ] 与 Statsmodels `AnovaRM` 对齐
- [ ] 混合设计 (between-within) 支持

---

#### 第四阶段：事后检验与多重比较 (2-3 周)

**目标**：实现完整的事后检验套件

**任务**：
- [ ] 估计边际均值 (EMMs) 计算
- [ ] 成对比较：
  - Tukey HSD (学生化范围分布)
  - Bonferroni 校正
  - Sidak 校正
  - FDR (Benjamini-Hochberg)
- [ ] 自定义对比矩阵
- [ ] 简单效应分析 (交互作用分解)
- [ ] 效应量计算 (Cohen's d, Hedge's g)

---

#### 第五阶段：高级功能与优化 (2-3 周)

**目标**：完善功能并优化性能

**任务**：
- [ ] 置换检验 (Permutation ANOVA) - GPU 高度并行
- [ ] 多元方差分析 (MANOVA)
- [ ] 协方差分析 (ANCOVA)
- [ ] 贝叶斯因子 ANOVA (可选)
- [ ] 性能基准测试与优化
- [ ] 文档与示例

---

### 3.3 核心 API 设计

```python
from statgpu.hypothesis import anova

# 单因素 ANOVA (SciPy 兼容)
f_stat, p_val = anova.f_oneway(group1, group2, group3, backend='cupy')

# 多因素 ANOVA (Statsmodels 风格)
import statgpu as sgp
model = sgp.glm.OLS(y, X).fit()
anova_table = anova.anova_lm(model, typ=2)

# Pingouin 风格统一接口
anova_result = anova.anova(dv='score', between='condition', data=df, detailed=True)

# 事后检验
from statgpu.hypothesis import pairwise_tukey
tukey_result = pairwise_tukey(dv='score', group='condition', data=df)
```

---

### 3.4 技术依赖

| 依赖 | 用途 | 替代方案 |
|------|------|----------|
| `cupyx.scipy.special` | F 分布 CDF | 自定义 CUDA 核 |
| `torch.distributions.FisherSnedecor` | F 分布 p 值 | - |
| `patsy` | 公式解析 | 手动设计矩阵 |
| `scipy.stats.tukey_hsd` | 验证参考 | 手动实现 |

---

## 四、参考资源

### 4.1 文档与教程
- SciPy ANOVA: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
- Statsmodels ANOVA: https://www.statsmodels.org/dev/anova.html
- R aov: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/aov
- Car::Anova: https://www.rdocumentation.org/packages/car/versions/3.1-5/topics/Anova
- Emmeans: https://rvlenth.github.io/emmeans/
- Pingouin ANOVA: https://pingouin-stats.org/build/html/generated/pingouin.anova.html

### 4.2 GPU 实现参考
- CUdanova: https://github.com/jubbens/cudanova
- CuPy: https://cupy.dev/

### 4.3 统计参考
- Maxwell & Delaney (2004). Designing Experiments and Analyzing Data
- Field (2013). Discovering Statistics Using R

---

## 五、进度跟踪

| 阶段 | 预计开始 | 预计完成 | 实际完成 | 备注 |
|------|----------|----------|----------|------|
| 阶段 1: 单因素 ANOVA | - | - | - | |
| 阶段 2: 多因素 ANOVA | - | - | - | |
| 阶段 3: 重复测量 | - | - | - | |
| 阶段 4: 事后检验 | - | - | - | |
| 阶段 5: 高级功能 | - | - | - | |

---

*文档创建日期：2026-04-19*
*最后更新：2026-04-19*
