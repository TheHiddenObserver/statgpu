# 生存分析包功能调研与 GPU 实现计划

**创建日期**: 2026-04-19  
**作者**: TheHiddenObserver  
**状态**: 🔶 部分实现

> 已实现: `CoxPH` (Breslow/Efron, robust SE, cluster), `CoxPHCV` (骨架)
> 未实现: strata, frailty, time-varying covariates

---

## 一、现有包功能调研

### 1.1 Python 生态

#### 1.1.1 statsmodels

**官方文档**: https://www.statsmodels.org/dev/duration.html

| 功能模块 | 实现状态 | 说明 |
|----------|----------|------|
| `SurvfuncRight` | ✅ | 右删失数据的生存函数估计 (Kaplan-Meier) |
| Cox 比例风险模型 | ⚠️ 基础支持 | 通过 `PHReg` 类实现，功能较基础 |
| 参数化 AFT 模型 | ❌ | 未实现 |
| 竞争风险模型 | ❌ | 未实现 |
| 时依协变量 | ⚠️ 有限支持 | |
| 分层 Cox | ⚠️ 有限支持 | |
| 脆弱模型 (Frailty) | ❌ | 未实现 |

**特点**:
- 统计推断完整 (标准误、置信区间、假设检验)
- 与公式接口集成
- 性能中等，无 GPU 加速

---

#### 1.1.2 lifelines

**官方文档**: https://lifelines.readthedocs.io/  
**ResearchGate 论文**: https://www.researchgate.net/publication/334962719_lifelines_survival_analysis_in_Python

| 功能模块 | 实现状态 | 说明 |
|----------|----------|------|
| Kaplan-Meier 估计 | ✅ | `KaplanMeierFitter` |
| Nelson-Aalen 估计 | ✅ | `NelsonAalenFitter` |
| CoxPH | ✅ | `CoxPHFitter` |
| 参数化 AFT 模型 | ✅ | `WeibullAFTFitter`, `LogNormalAFTFitter`, `LogLogisticAFTFitter` |
| 加速失效时间模型 | ✅ | 多种分布支持 |
| 竞争风险模型 | ⚠️ | 基础支持 |
| 时依协变量 | ✅ | 支持 |
| 分层 Cox | ✅ | `strata` 参数 |
| 脆弱模型 (Frailty) | ⚠️ | 有限支持 |
| 惩罚 Cox | ✅ | L2 正则化 |

**特点**:
- 纯 Python 实现，API 友好
- 内置可视化功能
- 性能较慢，无 GPU 加速

---

#### 1.1.3 scikit-survival

**官方文档**: https://scikit-survival.readthedocs.io/  
**JMLR 论文**: https://www.jmlr.org/papers/volume21/20-729/20-729.pdf

| 功能模块 | 实现状态 | 说明 |
|----------|----------|------|
| CoxPH | ✅ | `CoxPHSurvivalAnalysis` |
| 随机生存森林 | ✅ | `RandomSurvivalForest` |
| 生存 SVM | ✅ | `FastSurvivalSVM` |
| 梯度提升生存 | ✅ | `GradientBoostingSurvivalAnalysis` |
| C-index 评估 | ✅ | 多种实现 |
| Integrated Brier Score | ✅ | |
| 参数化 AFT 模型 | ❌ | |
| 竞争风险模型 | ❌ | |

**特点**:
- 基于 scikit-learn API
- 机器学习方法丰富
- 无 GPU 加速 (依赖 scikit-learn CPU 后端)

---

#### 1.1.4 PySurvival

**官方文档**: https://square.github.io/pysurvival/

| 功能模块 | 实现状态 | 说明 |
|----------|----------|------|
| CoxPH | ✅ | `CoxPHModel`（半参数模型） |
| 随机生存森林 | ✅ | `RandomSurvivalForestModel` |
| MTLR | ✅ | 多任务逻辑回归生存模型 |
| SVM 生存模型 | ✅ | Kernel SVM 生存模型 |
| 参数化模型 | ✅ | Exponential / Weibull / Log-Logistic 等 |
| 推断统计量（SE/z/p） | ⚠️ 有限支持 | 更偏预测建模而非经典推断 |
| GPU 加速 | ❌ | 官方实现以 CPU 为主 |

**特点**:
- 模型覆盖面较广，包含传统与机器学习生存模型
- 对推断统计（类似 `coxph` 输出）支持较弱
- 生态活跃度较历史版本有所下降，编译安装兼容性需提前验证

---

### 1.2 R 语言生态

#### 1.2.1 survival 包

**CRAN**: https://cran.r-project.org/package=survival  
**PDF 文档**: https://cran.r-project.org/package=survival/survival.pdf

| 功能模块 | 函数 | 说明 |
|----------|------|------|
| Kaplan-Meier | `survfit()` | 金标准实现 |
| Cox 比例风险 | `coxph()` | 行业标杆，功能最全 |
| 参数化 AFT | `survreg()` | 多种分布支持 |
| 竞争风险 | `cuminc()` | 通过 `cmprsk` 扩展 |
| 时依协变量 | `coxph()` | 完整支持 (counting process 格式) |
| 分层 Cox | `coxph(strata())` | 完整支持 |
| 多状态模型 | `mstate` 扩展 | |
| 脆弱模型 | `coxph(frailty())` | 支持 |
| 惩罚 Cox | `coxph(penalty=)` | 支持 |

**`coxph()` 核心功能**:
- Ties 处理：Breslow / Efron / Exact
- 稳健协方差：Huber-White / Cluster-robust
- 时依权重
- 多状态建模基础

**`survfit()` 核心功能**:
- 从 Cox 模型预测生存曲线
- 多状态 Pr(state) 曲线
- 置信区间 (多种方法)

---

### 1.3 功能对比矩阵

| 功能 | statsmodels | lifelines | scikit-survival | PySurvival | R::survival | statgpu (当前) |
|------|-------------|-----------|-----------------|------------|-------------|----------------|
| Kaplan-Meier | ✅ | ✅ | ✅ | ✅ | ✅ (金标准) | ❌ |
| Nelson-Aalen | ❌ | ✅ | ❌ | ⚠️ | ✅ | ❌ |
| CoxPH (Breslow) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CoxPH (Efron) | ⚠️ | ✅ | ❌ | ⚠️ | ✅ | ✅ |
| CoxPH (Exact) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 参数化 AFT | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| 随机生存森林 | ❌ | ❌ | ✅ | ✅ | ✅ (randomForestSRC) | ❌ |
| 生存 SVM | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| 时依协变量 | ⚠️ | ✅ | ❌ | ❌ | ✅ | ❌ |
| 分层 Cox | ⚠️ | ✅ | ❌ | ❌ | ✅ | ❌ |
| 脆弱模型 | ❌ | ⚠️ | ❌ | ❌ | ✅ | ❌ |
| 竞争风险 | ❌ | ⚠️ | ❌ | ⚠️ | ✅ (cmprsk) | ❌ |
| 惩罚 Cox | ❌ | ✅ (L2) | ❌ | ✅ | ✅ | ✅ (L2) |
| 稳健协方差 | ✅ | ⚠️ | ❌ | ❌ | ✅ | ✅ (部分) |
| GPU 加速 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 二、GPU 加速生存分析研究现状

### 2.1 已有 GPU 实现

| 项目 | 链接 | 状态 |
|------|------|------|
| CUDA_Cox | https://github.com/Vafa-Andalibi/CUDA_Cox | 研究原型 |
| survivex | 论文：https://papers.ssrn.com/sol3/Delivery.cfm/24800526-6d5c-4f13-88a0-3e14bc4e701a | Python 库 |
| Massive Parallelization | https://www.tandfonline.com/doi/full/10.1080/10618600.2023.2213279 | JCGS 论文 |

**技术路线**:
- NVIDIA CUB 库用于并行归约操作
- CUDA kernels 计算偏似然和梯度
- 样本级并行 (100K+ 观测)

---

## 三、statgpu 生存分析模块现状

### 3.1 已实现功能

**文件结构**:
```
statgpu/survival/
├── __init__.py
├── _cox.py          # CoxPH 核心实现
├── _cox_cv.py       # 交叉验证 CoxPH
└── _cox_efron_cuda.py  # Efron ties CUDA 优化
```

**当前 `CoxPH` 支持**:
- ✅ Breslow / Efron ties 处理
- ✅ Newton-Raphson 优化
- ✅ 稳健协方差 (HC0/HC1/cluster)
- ✅ L2 正则化 (penalty 参数)
- ✅ C-index 计算
- ✅ 基线风险/累积风险估计
- ✅ `survfit()` 风格预测接口

**对标 R `survival::coxph()`**:
- 文档参见：`docs/models/coxph.md`
- 当前精度：与 R 一致 (见 benchmark 结果)

---

## 四、GPU 实现路线图

### 4.1 第一阶段：CoxPH 功能对齐 (P0 - 2026 Q2)

**目标**: 达到 R `survival::coxph()` 80% 核心功能

| 功能 | 优先级 | 预计工作量 | 依赖 |
|------|--------|-----------|------|
| Ties 处理 (Exact) | P0 | 2 周 | Efron 实现 |
| 时依协变量 | P0 | 3 周 | counting process 格式 |
| 分层 Cox | P0 | 2 周 | baseline 分层估计 |
| 稳健协方差 HC2/HC3/HAC | P1 | 1 周 | 已有 Linear/Ridge 基础 |
| 脆弱模型 (Frailty) | P1 | 3 周 | 随机效应估计 |

**关键设计决策**:
1. **时依协变量数据格式**: 采用 R 的 `(start, stop, event)` counting process 格式
2. **分层实现**: 每层独立 baseline hazard，共享 coef
3. **脆弱模型**: 使用 penalized partial likelihood 或 EM 算法

---

### 4.2 第二阶段：参数化 AFT 模型 (P1 - 2026 Q3)

**目标**: 实现 lifelines/R `survreg()` 核心 AFT 功能

| 模型 | 分布假设 | 优先级 |
|------|---------|--------|
| Weibull AFT | Weibull / Extreme Value | P0 |
| Log-Normal AFT | Normal (log-time) | P0 |
| Log-Logistic AFT | Logistic (log-time) | P1 |
| Generalized Gamma | 最通用 | P2 |

**GPU 加速点**:
- 对数似然计算 (样本级并行)
- Hessian 矩阵构建
- Newton-Raphson / EM 迭代

**API 设计**:
```python
from statgpu.survival import WeibullAFT, LogNormalAFT

model = WeibullAFT(device='cuda')
model.fit(X, time, event)
model.predict_survival(X_new, times)
```

---

### 4.3 第三阶段：高级生存模型 (P2 - 2026 Q4)

| 模型 | 参考实现 | GPU 潜力 |
|------|---------|---------|
| 随机生存森林 | scikit-survival / randomForestSRC | 高 (树并行) |
| 生存梯度提升 | XGBoost 风格 | 高 |
| 竞争风险 (Fine-Gray) | R::cmprsk | 中 |
| 多状态模型 | R::mstate | 中 |
| 联合模型 (纵向 + 生存) | JMbayes | 低 (MCMC) |

**推荐优先**:
1. **Fine-Gray 竞争风险**: 临床需求大，算法类似 CoxPH
2. **生存梯度提升**: 可复用 XGBoost/LightGBM 基础设施

---

### 4.4 第四阶段：生存分析工具链 (P3 - 2027 Q1)

| 工具 | 参考 | 说明 |
|------|------|------|
| 比例风险检验 | `cox.zph()` (R) | Schoenfeld 残差 |
| 模型诊断图 | survminer (R) | 集成 matplotlib |
| 样本量计算 | powerSurv (R) | 试验设计 |
| 时间依赖 ROC | timeROC (R) | 评估指标 |
| 校准曲线 | val.surv (rms) | 模型验证 |

---

## 五、技术架构建议

### 5.1 设备抽象层

沿用当前 `statgpu._config.Device` 枚举:
```python
class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
```

**策略**:
- 所有统计模型同时提供 CPU/GPU 实现
- CPU 路径用于验证和小数据
- GPU 路径用于大规模数据

---

### 5.2 与现有模块集成

| 现有模块 | 可复用组件 |
|---------|-----------|
| `statgpu._base.BaseEstimator` | 所有生存模型基类 |
| `statgpu._cv_base.CVEstimatorBase` | CoxPHCV 已使用 |
| `statgpu.linear_model._cox` | CoxPH 核心算法 |
| `statgpu.utils` | 数据验证/转换 |

---

### 5.3 外部验证框架

**对标目标** (按优先级):
1. **R::survival** (金标准)
2. **lifelines** (Python 最完整)
3. **statsmodels** (统计推断)
4. **scikit-survival / PySurvival** (机器学习与工程生态对齐)

**验证脚本模板**:
```python
# dev/benchmarks/compare_survival_R.py
# 1. 生成相同数据
# 2. 分别用 statgpu 和 R 拟合
# 3. 比较 coef/se/z/p/logLik/C-index
```

---

## 六、时间估算

| 阶段 | 时间范围 | 里程碑 |
|------|---------|--------|
| **Phase 1** | 2026 Q2 | CoxPH 功能对齐 R |
| **Phase 2** | 2026 Q3 | AFT 模型家族 |
| **Phase 3** | 2026 Q4 | 随机森林/梯度提升 |
| **Phase 4** | 2027 Q1 | 工具链完善 |

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 数值精度对齐困难 | 高 | 早期引入 R 对标测试 |
| GPU 内存不足 | 中 | 实现 stream/batch 处理 |
| 复杂模型收敛问题 | 中 | 提供多 solver 选项 |
| 文档/示例不足 | 低 | 跟随 R 生存分析教程 |

---

## 八、参考文献

1. Cox, D. R. (1972). Regression models and life-tables. *JRSS B*. https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
2. Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*. https://doi.org/10.2307/2529620
3. Efron, B. (1977). Efficiency of Cox's likelihood function. *JASA*. https://doi.org/10.1080/01621459.1977.10480613
4. Davidson-Pilon, C. et al. (2019). lifelines: survival analysis in Python. https://www.researchgate.net/publication/334962719_lifelines_survival_analysis_in_Python
5. Pölsterl, S. (2020). scikit-survival: A Library for Time-to-Event Analysis. *JMLR*. https://www.jmlr.org/papers/volume21/20-729/20-729.pdf
6. Therneau, T. M. (2024). survival package. https://cran.r-project.org/package=survival
7. PySurvival documentation. https://square.github.io/pysurvival/
8. Massive Parallelization of Massive Sample-Size Survival Analysis. *JCGS*. https://doi.org/10.1080/10618600.2023.2213279

---

## 九、下一步行动

1. **Phase 1 启动**: 实现 Ties (Exact) 和时依协变量
2. **建立 R 对标环境**: 配置 rpy2 或直接 Rscript 调用
3. **文档框架**: 创建 `docs/en/models/survival-aft.md` 等
4. **Cox entry+efron GPU 收尾（2026-04-22）**:
   - 保留并继续调优 `cupy_fused` 路径（`STATGPU_ENTRY_S2_FUSED_CUPY=1`）。
   - `torch.compile` 暂不作为默认路径：仅在 GPU Compute Capability >= 7.0（如 A30/RTX 4090）时启用。
   - 为 `torch.compile` 增加设备能力检测与自动回退（老卡如 P100 回退 eager，不中断训练/拟合）。

---

*本文档将随实现进度更新。详细任务追踪见 `TO_DO.md` 和 `PLAN_UNIFIED.md`。*
