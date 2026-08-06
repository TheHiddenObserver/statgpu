# 生存分析包功能调研与 GPU 实现计划

**创建日期**: 2026-04-19  
**作者**: TheHiddenObserver  
**更新日期**: 2026-07-12
**状态**: ✅ Phase 1 CoxPH 核心功能已完成；Phase 2+ 继续规划

> 已实现: `CoxPH`（Breslow/Efron/Exact ties、delayed entry、`(start, stop]`
> counting-process、strata、HC0/HC1/cluster robust SE、C-index、分层 baseline
> survival、无惩罚 AIC/BIC）、完整的 `CoxPHCV` L2 选择流程，以及 estimation-only 的
> `PenalizedCoxPHModel`（已验证 L1/L2/ElasticNet/SCAD/MCP）。
>
> 明确限制: Exact ties 目前仅支持 `cov_type="nonrobust"`；Efron/Exact 拟合后的
> baseline 使用常规 Breslow estimator；`PenalizedCoxPHModel` 不提供 SE/p-value/CI；
> GPU 是否更快取决于风险集结构和数据规模。
>
> 后续范围: Kaplan-Meier/Nelson-Aalen、Frailty、AFT、Fine-Gray、多状态模型和
> survival forest/boosting 尚未实现。

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
| CoxPH (Exact) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅（仅 nonrobust covariance） |
| 参数化 AFT | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| 随机生存森林 | ❌ | ❌ | ✅ | ✅ | ✅ (randomForestSRC) | ❌ |
| 生存 SVM | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| 时依协变量 | ⚠️ | ✅ | ❌ | ❌ | ✅ | ✅（`(start, stop]`） |
| 分层 Cox | ⚠️ | ✅ | ❌ | ❌ | ✅ | ✅（共享 coef、分层 baseline） |
| 脆弱模型 | ❌ | ⚠️ | ❌ | ❌ | ✅ | ❌ |
| 竞争风险 | ❌ | ⚠️ | ❌ | ⚠️ | ✅ (cmprsk) | ❌ |
| 惩罚 Cox | ❌ | ✅ (L2) | ❌ | ✅ | ✅ | ✅（L1/L2/EN/SCAD/MCP；estimation-only） |
| 稳健协方差 | ✅ | ⚠️ | ❌ | ❌ | ✅ | ✅（Breslow/Efron；Exact 暂不支持） |
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
├── _cox.py                    # CoxPH 公共 API、拟合、推断与预测
├── _cox_cv.py                 # CoxPHCV 选择、诊断与最终 refit
├── _risk_sets.py              # 三后端风险集与 tied-event 原语
├── _cox_counting.py           # counting-process 目标函数、solver、baseline
├── _cox_efron_cuda.py         # Efron CuPy 优化
├── _cox_efron_grad_hess_kernel.py
├── _cox_efron_triton.py
└── _cox_breslow_triton_kernel.py
```

**当前 `CoxPH` 支持**:
- ✅ Breslow / Efron / Exact ties 处理
- ✅ right-censored、delayed-entry 和 `(start, stop]` counting-process 输入
- ✅ `strata` 共享系数、独立风险集和 stratum-specific baseline
- ✅ `Surv(time, event)` 与 `Surv(start, stop, event)` formula 输入
- ✅ NumPy / CuPy / Torch-CUDA backend-native Newton 优化与风险集计算
- ✅ 稳健协方差 (HC0/HC1/cluster)
- ✅ L2 正则化 (penalty 参数)
- ✅ C-index 计算
- ✅ 基线风险/累积风险估计
- ✅ right-continuous `survfit()` 风格预测接口（分层模型预测时需提供 strata）

**当前 `CoxPHCV` 支持**:
- ✅ L2 penalty path 与 Breslow/Efron/Exact held-out partial likelihood
- ✅ delayed entry、start-stop、strata 与 subject-grouped folds
- ✅ 每个 candidate/fold 的 convergence、iterations 和 failure diagnostics
- ✅ 最优 penalty 的全数据 refit 与 `predict`/`score` 转发

**当前 `PenalizedCoxPHModel` 支持**:
- ✅ L1 / L2 / ElasticNet / SCAD / MCP 的 CPU 目标值与 KKT 验证
- ✅ CuPy / Torch-CUDA parity gates
- ✅ 无截距（Cox partial likelihood 中 intercept 不可识别）
- ⚠️ 仅估计；`compute_inference=True` 显式抛出 `NotImplementedError`
- ⚠️ 目前只处理 `(time, event)` 与 Breslow/Efron，不等同于 `CoxPH` Phase-1 全功能

**对标 R `survival::coxph()`**:
- 文档参见：`docs/en/models/coxph.md`
- Breslow/Efron delayed-entry/strata 在 CPU 测试中对标 statsmodels；Exact 使用
  brute-force tied-partition 测试，三后端结果由 parity gates 验证。
- 完整的 R `survival::coxph()` Exact/counting-process 对标仍是后续验证项，不能据此
  声称已经覆盖 R 的全部推断语义。

### 3.2 2026-07-12 GPU benchmark 快照

最终 quick/full artifact 来自 NVIDIA RTX 5880 Ada、float64、1 次 warm-up 和 2 次
计时重复。下列比值为 `NumPy fit time / GPU fit time`，大于 1 才表示 GPU 更快；
fit timing 包含优化、推断和 baseline 估计，数据传输单独计时。

| 场景 | CuPy / NumPy | Torch / NumPy | 结论 |
|------|--------------|---------------|------|
| quick delayed entry (700×8) | 0.647x | 0.959x | 两个 GPU backend 均较慢 |
| full delayed entry (2,500×16) | 1.044x | 1.374x | 两个 GPU backend 均加速，Torch 更明显 |
| full stratified start-stop (2,400×16, 4 strata) | 0.241x | 0.411x | 均慢于 NumPy |
| full standard heavy ties (20,000×32, Efron) | 0.850x | 0.436x | 均慢于 NumPy |
| full Exact ties (120×4) | 0.069x | 0.095x | 动态规划正确性优先，均慢于 NumPy |

Artifact: `results/survival_completion_2026-07-12.json` 与
`results/survival_completion_full_2026-07-12.json`。这些结果不构成普遍 speedup
承诺；在 crossover 规模得到系统估计前，`device="cpu"` 仍是小型/复杂风险集场景的
合理选择。相同 artifact 中，NumPy/CuPy/Torch-CUDA 的 CV 均选择同一 penalty，
最终 refit 的 coef/bse backend 差异小于 `1e-16`。

---

## 四、GPU 实现路线图

### 4.1 第一阶段：CoxPH 核心功能对齐（P0，2026-07-12 已完成）

**目标**: 完成 Exact、counting-process、strata 与可审计 CV 的共享三后端实现。

| 功能 | 优先级 | 状态 | 当前边界 |
|------|--------|------|----------|
| Ties 处理 (Exact) | P0 | ✅ 完成 | nonrobust covariance；baseline 使用 Breslow convention |
| 时依协变量 | P0 | ✅ 完成 | `(start, stop]` counting-process，支持 subject ID |
| 分层 Cox | P0 | ✅ 完成 | 共享 coef，独立风险集和 baseline |
| CoxPHCV | P0 | ✅ 完成 | L2 path、Exact/Breslow/Efron、subject-grouped folds |
| 稳健协方差 HC2/HC3/HAC | P1 | ⏳ 后续 | 当前为 nonrobust/HC0/HC1/cluster |
| 脆弱模型 (Frailty) | P1 | ⏳ 后续 | 需要随机效应估计与新的推断合同 |

**关键设计决策**:
1. **时依协变量数据格式**: 采用 R 的 `(start, stop, event)` counting process 格式
2. **分层实现**: 每层独立 baseline hazard，共享 coef
3. **风险集实现**: NumPy/CuPy/Torch 共用相同语义，区间采用左开右闭 `(start, stop]`
4. **Exact 实现**: elementary-symmetric dynamic programming；优先正确性和数值稳定性
5. **脆弱模型**: 保留为后续独立 phase，候选方案为 penalized partial likelihood 或 EM

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
| **Phase 1** | 2026-07-12 完成 | Exact/start-stop/strata/CV 核心能力 |
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

1. **Phase 1 性能后续**: 针对 standard heavy ties、stratified start-stop 和 Exact
   风险集优化；在得到多规模 crossover 曲线前不做普遍 GPU speedup 承诺。
2. **独立 R 对标**: 通过 `Rscript` 对比 Exact、counting-process、strata、robust
   covariance 与 baseline prediction；当前 artifact 未调用 R。
3. **推断增强**: 评估 HC2/HC3/HAC、Exact robust covariance，以及 penalized Cox
   的 debiased/post-selection inference；在方法学验证前维持 estimation-only。
4. **Phase 2 AFT**: Weibull、Log-Normal、Log-Logistic；先定义三后端 likelihood、
   censoring 和 inference contract，再实现 GPU kernel。
5. **Phase 3 高级模型**: Frailty、Fine-Gray、多状态、survival forest/boosting 均
   保留为后续独立里程碑，不纳入本次 Phase 1 完成声明。

---

*本文档将随实现进度更新。详细任务追踪见 `TO_DO.md` 和 `PLAN_UNIFIED.md`。*
