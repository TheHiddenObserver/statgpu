# statgpu TO DO

## 当前缺失清单（与 sklearn/statsmodels/R 的功能可比性差距）

### 通用层面的缺失（所有方法大概率都适用）
- `robust covariance / cluster-robust / HAC` 等稳健标准误：statsmodels/R 有多种协方差估计；statgpu 目前大多是按经典公式做（例如线性模型按 `(X'X)^{-1}` 推断）。
- `sparse input` 支持不足：sklearn 对稀疏矩阵（CSR/CSC）支持很完整；这里多是 dense 数组路径。
- “模型选择/调参工具”缺失：没有内置 `path / cv / grid-search / warm_start`（sklearn 的 `LassoCV/RidgeCV/LassoLarsIC`、R 的 `glmnet`/`caret` 这一类能力基本不在项目里）。
- `normalize/standardize/center` 这类数据预处理选项缺失：sklearn 往往提供标准化/预处理开关；statgpu 主要依赖你手动准备数据。
- KKT/Stopping 语义仍不完全等价：你虽然已经可以做“同等价 stopping（KKT）”的对比，但 sklearn/statsmodels 内部停止准则/尺度不完全可控，因此“参数 tol 一样就数值意义等价”很难完全保证。
- GPU 侧的某些指标是简化/近似：例如 Cox 的 C-index GPU 计算不是严格精确的 pairwise 比较，而是简化近似。

### `LinearRegression`（对比 sklearn/statsmodels/R 的缺失）
- 只实现了 OLS 的基础形式，没有 GLS/加权最小二乘更完整的分支（虽有 `sample_weight`，但不等价于 statsmodels 的一整套协方差建模）。
- 标准误/统计检验以经典形式为主，缺少 statsmodels 常见的稳健协方差选项与诊断表（例如异方差稳健、聚类稳健等）。
- statsmodels/R 的公式接口（`formula`、因子水平、交互项、缺失值处理）功能缺失。

### `Ridge`（缺失点）
- solver/求解器策略缺失：sklearn ridge 有多种 solver（例如 `svd/cholesky/sag/sparse_cg` 等）与数值稳定性选项；statgpu 主要是闭式/Cholesky 路径为主。
- warm start、Ridge 路径（不同 alpha 的连续解）缺失：sklearn 有 `RidgeClassifierCV` 等体系，R 里常见 `glmnet`/`rrugf` 这类 path 工具。
- 推断统计的协方差/自由度处理功能相对简化：相对 statsmodels 的可选项更少。

### `Lasso`（缺失点最明显的一个）
- `ElasticNet`/`l1_ratio`：目前是纯 L1（lasso），没有直接的 elastic-net 组合接口（R 的 `glmnet`/sklearn 的 `ElasticNet`/`LassoLars` 体系更完整）。
- `positive`/系数符号约束、特征选择策略（如 sklearn 的不同算法路径）缺失。
- warm_start 与“正则化路径”（从大 alpha 到小 alpha 的连续路径）缺失。
- 推断统计的统计学完整性有限：目前的 SE/p-value 主要沿用“线性回归经典推断”的结构（本质上忽略了 L1 造成的非光滑/选择效应）。这和“真正严谨的 Lasso 推断（需要 KKT/选择调整或专门推断方法）”不是同一层级。
- CPU/GPU 的优化器对齐程度有限：你现在已经通过 KKT 等价 stopping 做到了可比较，但要做到“参数 tol 完全等价且目标尺度等价”仍需要更多校准/映射。

### `LogisticRegression`（对比 sklearn/statsmodels/R 的缺失）
- 只支持二分类（binomial）为主；缺少多分类（multinomial/softmax）能力。
- 只支持 L2（通过 `C`）的正则形式；缺少 L1（`penalty='l1'`）或 elastic-net。
- 训练优化器只有 IRLS 这一路径（sklearn 还有 liblinear/liblbfgs 等多实现、数值稳定/速度选项更多）。
- 稳健标准误/聚类稳健、以及更多诊断输出缺失。

### `CoxPH`（对比 R `coxph` 的缺失点较多）
- 没有实现 strata（分层基线风险）、frailty（随机效应/混合效应）、time-varying covariates 更完整框架。
- 缺少 robust/cluster 方差估计、sandwich 这类 Cox 常见增强项（R `coxph` 支持非常多选项）。
- GPU 端的 `C-index`：实现里有“简化近似”的特征（不是严格 pairwise exact），因此如果把 Cox 当做严格性能/评估基准，需要注意这一点。
- Penalized Cox（L1/L2 正则化的 Cox）、分组惩罚等能力缺失。

---

## TODO 任务（建议按优先级）

### 最近已完成（2026-04）
- `Lasso` 推断方法语义化重命名：
  - `cpu_ols_inference`（旧名兼容：`naive_ols`）
  - `gpu_ols_inference`（旧名兼容：`gpu_naive_ols`）
- `Lasso` 的 `gpu_ols_inference` 路径改为尽量在 GPU 上完成 `SE/t/p/CI` 计算（减少 `scipy.stats` 依赖和大块 CPU 传输）。
- 新增显存管理开关：`gpu_memory_cleanup`，已覆盖 `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`。
- 新增 benchmark：
  - `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
  - `examples/benchmark_gpu_memory_cleanup.py`
- 修复 `LogisticRegression.fit()` 在 CUDA 输入 `cupy.ndarray` 时的隐式 `np.asarray` 转换报错。

### P0：统计推断严谨性（Inference Rigor）
- 为线性模型实现稳健协方差（`robust covariance / cluster-robust / HAC` 等），并对齐 statsmodels 的可选项与数值定义。
- 为 Lasso 引入更接近真实 *post-selection inference* 的推断方案（例如 de-biased lasso / selective inference），并与当前 bootstrap/naive 形成对照基线。
- 在 GPU 路径上实现 Lasso 更严谨的推断（至少把 bootstrap 这类需要多次 refit 的方案尽量 GPU 化或减少大规模 CPU 交换），同时补齐 benchmark 的 `accuracy + runtime + breakdown`（数据构造/fit/inference 拆分、warmup/repeats）。
- 提供所有模型推断输出的一致性与可用性：`SE/t/z/p/CI`、自由度、`AIC/BIC/LLF` 等关键统计在不同 `device`/`compute_inference` 组合下行为一致。
- 为 CoxPH 实现 robust/cluster（sandwich）方差估计，并尽量减少 CUDA 下的必要 host/device 传输。
- 给 CoxPH 的 `C-index` 同时提供“严格 pairwise 计算”和“近似计算”的可控开关，并在文档/benchmark 中明确近似误差边界。

### P1：API parity / 功能补齐
- 为 Lasso 增加 `ElasticNet`（`l1_ratio`）与 `positive` 约束，并补齐 warm_start 与正则化路径（alpha path）。
- 为 Ridge 增加 warm_start / Ridge paths，以及推断统计与 `summary()` 输出，尽量对齐 LinearRegression/Lasso 的统计输出口径。
- 扩展 LogisticRegression 到 multinomial/softmax、多分类；支持 `L1` 或 elastic-net 正则，并补齐稳健标准误与更多诊断输出。
- 扩展 CoxPH 到 strata、frailty、time-varying covariates，并补齐 penalized Cox（L1/L2）与相关诊断。
- 增加稀疏输入（CSR/CSC）支持，并确保稀疏/密集解的一致性或给出可解释差异。

### P2：模型选择/调参与数据预处理
- 增加 `path / cv / grid-search / warm_start` 等模型选择工具，并统一 `stopping/KKT` 的标定脚本与输出。
- 增加数据预处理开关（`center/standardize/normalize` 等），并把这些选项纳入可复现实验流程，降低用户手工差异对对比的影响。

### P3：GPU/CPU 对齐与基准框架（Bench Framework）
- 给所有模型的 benchmark 补齐“数据构造 vs fit vs inference”拆分时间统计，并默认包含 GPU warmup/repeats。
- 统一等价 stopping 口径（KKT violation），并提供可复现的标定脚本，保证 tol/cnvrg_tol 与目标 KKT 阈值之间的映射有明确定义。
- 统一 result diff 指标体系（例如 `L_inf`、`L2_rel`、以及推断统计 `bse/t/p/CI` 的数值差异），形成可对外的报告模板。
- 为 `gpu_memory_cleanup` 增加模型级 benchmark 报告模板：默认记录 `fit_ms`、`pool_used_fit`、`pool_total_fit`、`pool_used_reset`、`pool_total_reset`。

