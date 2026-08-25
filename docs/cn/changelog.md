# Changelog

> 语言：中文<br>
> 最后更新：2026-08-18<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 2026-08-09 — Panel Stage C 协方差补齐（PR #126）

Stage C 完成 Panel Tier-1 的协方差与推断能力，同时保持 estimator coefficient 与标准化 fit-statistic 定义不变。`robust` 继续表示既有 HC1；`hc0`、`hc2`、`hc3` 按各 estimator 的实际 transformed fit space 计算。`RandomEffects` 在 quasi-demeaned GLS score 上支持 robust/HC、clustered 与 Driscoll-Kraay covariance，并在 Swamy-Arora between auxiliary regression 没有正 residual degrees of freedom 时 fail closed；one-/two-way cluster 支持显式 `group_debias=True`；如果任一 clustering dimension 少于两个不同 group，clustered inference 现在会 fail closed，而不会返回退化的 single-cluster sandwich。`PooledOLS(cov_type="hac")` 继续表示 legacy row-order Bartlett/Newey-West 路径。

本次还系统强化了 two-way fixed-effect 的收敛与 prediction。Panel formula 中 pipe 明确命名的 metadata 现在作为权威来源：与显式 entity/time IDs 冲突时会 fail closed，缺失的 pipe 命名列不能再由无关显式 IDs 替代；RandomEffects 的第二个 pipe time 变量也只在 Driscoll-Kraay covariance 实际使用时才允许，同时 fixed-effect magic tokens 会被拒绝，而不会被重新解释为 grouping metadata。entity/time projection metadata 只在迭代前 factorize 一次，并在所选 backend 上复用；收敛判据直接检查两个 effect 维度的 residual group mean，对数值上被固定效应完全吸收的方向使用 scale-aware roundoff floor，同时公开 fail-closed 的 `demean_max_iter`/`demean_tol` 控制以处理弱连通 panel。unbalanced panel 的 two-way fixed effects 改为联合恢复；若已知 entity/time label 分属 disconnected incidence graph 的不同 component，则该预测不可识别并明确报错。formula prediction 在 Patsy 会删除 input row 或 formula transformation 生成非有限 design value 时会 fail closed；prediction 也不再把任意少一列的矩阵猜成“省略 intercept”，只有与拟合设计一致时才按原位置和原数值恢复显式 non-unit constant。已知 fixed-effect label 的预测现在会恢复 centered level grand mean，使 `PanelOLS.predict()` 返回完整的 fixed-effect level projection；formula 临时启用的 effect 不再泄漏到后续 refit，超过两个 fixed-effect 变量的 formula 会 fail closed，而无 FE 的 `PanelOLS` formula 会保留 Patsy/R 默认 intercept（`0 +` / `-1` 继续表示显式 no-intercept）；no-intercept 的 `rsquared_within` 现在使用标准 uncentered total sum of squares。其余强化还包括 rank-deficient coefficient identifiability，并让 classical Hausman 在任一 fitted coefficient vector 非唯一时 fail closed；此外还包括 `FirstDifferenceOLS` duplicate/time 语义、HC2/HC3 leverage 稳定性、metadata alignment、CuPy scatter-add、RandomEffects formula intercept/name 行为以及 quadratic-spectral weight。外部定义固定对齐 `statsmodels==0.14.6`、`linearmodels==7.0`、R `plm==2.6-7` 与 R `sandwich==3.1-3`。

最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 只在存在溢出风险时使用 reduction-length scaling，以避免全局 magnitude normalization 额外造成的信息损失，但不宣称可以恢复任意病态 cancellation remainder；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration。shared panel inference 不再施加绝对 variance floor：exact-zero variance 下，零 coefficient 的 statistic 为 0，非零 coefficient 为带符号无穷。Classical model F、pooling F 与 Breusch-Pagan LM 使用 overflow-safe centering 和 subnormal-safe backend normalization。residual-based covariance 现在把 tiny-design/projection scale 的恢复推迟到 cancellation 之后：projection coordinate 只在 projection×residual 可能溢出时缩放，cluster 与 DK score 先 grouping 再做 selective Gram scaling，而且 residual vector 不会被全局 magnitude normalization，因此巨大 cancellation 旁边仍可表示的小 component 可以保留，同时已经安全的 subnormal-design 精度不被额外 normalization 破坏。two-way clustering 按 partition 等价而不是任意 code 编号识别 nested dimension，并在恢复尺度前代数消去相同 marginal/intersection component。range-aware symmetrization/inclusion-exclusion 与 HAC/DK 的 pre-Gram、full-lag accumulator 共同避免可恢复的中间溢出。maintained physical Stage-C runner 已把 diagnostic-scale、zero-variance、pre-Gram、tiny-design、mixed-range、nested-partition、covariance extreme-scale 与 lag-accumulation 分支加入 CuPy 与 Torch CUDA 验证。


物理验证按照 exact-source evidence chain 记录。历史 Stage-C 与 Fama-MacBeth artifact 只继续对各自原始 numerical head 有效。此前接受的 P100 source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` 对当前 PR branch 已属于历史证据，因为后续 review-fix loop 修改了有效的 Fama-MacBeth 与 shared panel least-squares 路径；在提升 merge readiness 之前必须重新完成 exact-head CuPy/Torch CUDA acceptance。Tesla P100 上，broad Stage-C runner 在 CuPy 与 Torch 上各自通过 35/35 estimator/covariance case 与 12/12 public primitive；专用 HAC chronology runner 也通过 ordered-categorical/numeric 等价、lexical negative control 和 shared backend-native Student-t inference。Fama-MacBeth 现在对 exact-size GPU batch 使用保守的 Gram-spectrum certificate：只有明显 well-conditioned 的 period 才允许使用 batched Gram solve，任何 uncertified period 仍由原有 SVD rank policy 负责。accepted scaling artifact 中，CuPy/Torch 的 GPU-over-NumPy median-time ratio 分别为：micro（64×128×4）**0.549/0.343**、medium（128×1,024×8）**0.204/0.168**、large（128×4,096×16）**0.114/0.109**，对应约 1.82×/2.92×、4.91×/5.97× 和 8.75×/9.16× speedup。所有 measured GPU scale 都是一个 `gram-certified` batch、一次 control synchronization、零 SVD fallback；该 resident-array timing protocol 不包含 host-to-device input transfer，因此这些结果属于特定 workload/hardware 的物理证据，不是普遍 GPU 性能保证。focused Fama-MacBeth gate 同时验证 chronology/formula/rank/inference、backend-native public array、backend-native distribution inference 与 prediction/device provenance。最终四个 physical runner 的 artifact 均保存在 `results/pr126_p100_fama_fix/`，并指向同一个 numerical source。

### 验证（2026-08-22）

针对 PR 分支的最新一轮 review-fix 循环继续加固数值与设备路径，并在 exact head `5068da3f` 上重跑完整物理矩阵：

- **双向聚类协方差性能**：精确的逐行 dyadic two-sum fallback（普通均衡面板在约 6.5k 行以上必然触发，10k 行时每次 CuPy fit 约 1000 秒）现在由 residual-acceptance 检查门控——普通设计停留在向量化 Gram 路径，只有真正可恢复的 cancellation residual 才回退到精确行级展开。Tesla P100 上 `pooled_cluster_two_way` 的 10k 行 CuPy fit 从 **约 1018 秒降到约 1.3 秒**（Torch 约 0.2 秒；100k 行约 0.4 秒），`benchmark_panel_stage_c_covariance.py` 的 60 行矩阵约 40 秒完成（此前超时）。
- **数值加固**：CuPy `maximum.at`/`cupyx.scatter_max` 对 1e7..1e308 量级的 float64 返回 `inf`（CuPy 13.6 实测），组内 min/max scatter 改为顺序 host scatter；Torch CUDA SVD 改用精确 `gesvd` driver（默认 `gesvdj` 会在结构零位置泄漏约 1e-16，被巨大响应放大）；失败的 panel fit 保留实际执行后端 provenance；Student-t(1) 的 p-value 改用良态的 `2 atan(1/x)/pi` 形式，极端统计量（如 |t|=1e154）保留可表示尾部（此前 subtractive survival 在约 1e15 即坍缩为 0）；formula side-array 对齐对超长输入 fail closed。
- **CuPy 设备亲和性**：后端可用性探测不再切换当前 CUDA device，panel 分配（scatter 目标、dummy 矩阵、行权重、SVD 单位阵）绑定到参考 device；新增物理 device-affinity gate（`validate_panel_cupy_device_affinity_gpu.py`）覆盖 CuPy 与 Torch CUDA。
- 全部 12 个 physical runner 在 exact head 的 Tesla P100（CuPy 13.6.0 / Torch 2.0.0+cu117）上通过：Stage-C correctness（每后端 35 case + 12 primitive）、focused Fama-MacBeth oracle + certified-Gram provenance、HAC chronology、极端 t(2) 尾部、device affinity、Fama-MacBeth scaling、RHS cancellation、rank precedence、intercept cancellation。产物：`results/pr126_perf_fix_528d967e/`、`results/pr126_review_fix_da3604ee/`。

## 2026-08-08

### PR #122 — Panel Tier-1 diagnostics Stage B

- 新增公开的结构化 `PanelTestResult`、`PanelFitStatistics` 以及维护中 panel estimator 的标准化 `fit_statistics_`。新的 fit statistics 包含 parameter-based within/between/overall R²、显式定义的 adjusted R²，以及在存在 residual-OLS 拟合空间时的 classical homoskedastic model F。
- Stage-A 的 coefficient inference 与 legacy R²/df 行为保持不变。特别是 `PanelOLS` 继续公开历史 residual df 和 BSE/t/p/CI；Stage-B diagnostics 使用单独的标准 fixed-effect nuisance-rank df，经典 Hausman 只读取按该标准 denominator 重标度的小型 diagnostic covariance，不修改公共 inference。
- 新增 fixed-effects classical pooling F、one-way entity error-components Breusch-Pagan LM（包含 Baltagi-Li unbalanced-panel 公式）以及 classical one-way entity FE-vs-RE Hausman。计量上不适用的情况返回结构化 reason；Hausman covariance difference 若为奇异 PSD，则使用明确记录的 generalized-inverse/rank extension；若实质 indefinite，则直接报告不可用。
- `PooledOLS.fit()` 与 `FamaMacBeth.fit()` 的可选 `entity_ids` 只用于 Stage-B within/between fit statistics 和 panel BP-LM。Pooled HAC 稳定排序现在让 entity diagnostic metadata 与 X/y 使用完全相同的 permutation；formula missing-row filtering 也会在形成 diagnostics 前对齐 observation-level side arrays。
- 增加 analytic/fitted regression、维护中的 Python 3.9 + Torch 2.0 CPU parity，以及可执行的 `linearmodels==7.0` definition-alignment job。FirstDifference 的外部比较只在两边 transformed sample 定义一致的 panel 上执行；Stage B 不会为了 external gate 静默改变 Stage-A 对内部缺期采用 adjacent-observed-row differencing 的既有契约。
- 新增 `dev/benchmarks/validate_panel_stage_b_gpu.py` 作为 exact-head physical correctness/provenance gate。此前在数值实现 `a57efcea29b0e87ecb89865c5a6902d5773812c6` 上接受的 P100 artifact 继续作为不可变的历史证据保留：CuPy 与 Torch 各自通过全部 17 个 estimator case，requested/executed backend 一致且无 fallback；focused disconnected two-way FE artifact 也把 df=1 inference boundary 验证到机器精度。该运行中每个 backend 的 4 个 Hausman parameterization 都是正确的结构化 `applicable=false` case，因此它们验证了 applicability/reason parity，但没有在物理 GPU 上执行 applicable Hausman 的 statistic/p-value/df 路径。
- 重新打开的 physical gate 已在精确 clean measurement head `2701aa9feb3796c33c94e6480fcb78c80c6a809c` 上闭合：Tesla P100 的 CuPy 与 Torch 各自通过全部 17 个 estimator case 和 5 个 Hausman diagnostic，requested/executed backend 一致且没有 CPU fallback。新增的 48-observation nonzero-effect fixture 在两个 backend 上均为 `applicable=true`、df=1；Hausman statistic 相对 NumPy 的最大差异不超过 `1.10e-13`，p-value 不超过 `2.19e-14`。新的 44-row canonical validation source 保留该分支的 statistic/pvalue/df；旧的 42-row a57efcea source 继续作为历史审计证据保留。本次证据不包含 timing 或 speedup 声明。

关联：Issue #93 与 pull request #122。

### PR #121 — CuPy inverse-quantile LUT 正确性修复

- 修正 CuPy `betaincinv()` 与 `gammaincinv()` 的 LUT cache tuple 顺序。LUT builder 原本已经按 `(x_grid, y_grid)` 存储，但缓存读取时反向解包，导致 inverse lookup 在错误坐标轴上搜索，并可能把 quantile 推到 clipped boundary。
- 该问题由 Panel Stage A 物理 GPU 验证暴露：Tesla P100 上 CuPy `t.isf(0.025, 45)` 曾得到接近 0 的 critical value，从而产生 zero-width confidence interval。修正后的两行数值实现返回 `2.014103388876289`，与 SciPy reference 的绝对差为 `4.04e-09`。
- 增加 maintained regression coverage，覆盖 raw inverse-beta/inverse-gamma cache reuse、公开 CuPy Student-t/Beta/F/Gamma/chi-square PPF/ISF 与 round trip、df=1/10/45/60/80 的 Student-t LUT/native-fallback 边界、module-level distribution proxy、legacy inverse-quantile alias，以及 representative Panel inference consumer。
- 在精确数值实现 head `f768b312d05f47debdb8fa13ae4da09b27d00239` 上完成 expanded physical validation：Tesla P100、Python 3.9.16、CuPy 13.6.0、clean working tree。Student-t 的最大 PPF/ISF 绝对误差为 `4.04e-09`，Beta 为 `5.49e-13`/`1.68e-13`，F 为 `4.51e-12`，Gamma 为 `1.45e-08`，chi-square 为 `2.90e-08`，均明显小于 maintained inverse-quantile accuracy contract。
- reconstructed Panel CI 与实际 shared Panel inference consumer 分别在 `3.42e-10` 与 `1.96e-10` 的最大绝对误差内匹配 reference；此前退化为 zero-width 的区间现已恢复为非退化区间，并与 Torch/reference 结果一致。

关联：Issue #120 与 pull request #121。

## 2026-08-07

### PR #119 — Panel Tier-1 共享框架 Stage A

- 为 Issue #93 增加内部 `BasePanelModel`、`PanelIndexInfo`、`PanelTestResult` 与 `PanelFitStatistics` 基础层。Stage A 只建立共享生命周期和 panel 结构契约；Hausman、pooling F、Breusch-Pagan LM 以及扩展 fit statistics 仍属于 Stage B。
- 将 panel estimator 中已经存在的 residual-based OLS covariance 分派集中到共享 registry，同时保持各模型原有的 nonrobust scaling、HC1 correction、one-/two-way cluster、HAC、rank/df 约定和 unsupported-name 行为。此阶段不新增 HC0/HC2/HC3 或 Driscoll-Kraay。
- 在统计上合理的边界内，将 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 与 `FamaMacBeth` 迁移到共享生命周期 helper；fixed-effect recovery/prediction、Swamy-Arora variance component 与 quasi-demeaning、Fama-MacBeth beta-series covariance 继续保持模型专用实现。
- 保持现有 formula、缺失行对齐、intercept/effect token、prediction output、summary schema/打印行为、balanced/unbalanced 语义、residual-df 定义以及显式 device 不允许静默 fallback 的契约。
- 在任何 panel source 重构之前先提交并通过 pre-refactor golden suite，并在重构后持续作为回归 gate；专用 Python 3.9 + Torch 2.0 CPU CI 现在也执行共享 panel metadata/covariance/inference 测试，避免 optional Torch 缺失时静默跳过。

Stage B diagnostics 与 Stage C covariance 扩展继续由 Issue #93 跟踪；Stage A 不会把这些尚未实现的能力写成公开支持。

### PR #116 — Torch LogisticRegressionCV strict-CUDA 修复

- 修复 maintained Torch strict-CUDA `LogisticRegressionCV` 在 batched GPU IRLS 路径中的 mixed-precision 失败。CV 现在按当前 working dtype 分配参数与 ridge diagonal，并在 validation scoring 前保持 coefficient/intercept path 为 backend-native。
- 增加 float32/float64 CV、weighted/unweighted、intercept/no-intercept 以及完整 selector consumer 的 regression coverage；新增 Python 3.9 + Torch 2.0 CPU CI gate，避免 optional Torch 缺失时相关回归测试被静默跳过。
- 在精确数值实现 head `e6e4846b06604ed53e65fc9afd9054bd5777098f` 上完成物理 GPU 验证：Tesla P100-SXM2-16GB、Python 3.9.16、PyTorch 2.0.0+cu117 / CUDA 11.7、CuPy 13.6.0。四个 focused Torch CUDA case 均与 CPU reference 选择相同的 `C=0.2`；最大 mean-loss 差异小于 `6.2e-8`，float64 路径达到机器精度一致。
- canonical 六类 CV rerun 中，statgpu 的 18 个 NumPy/CuPy/Torch backend row 全部成功，failed candidate/fold 均为 0，最终 refit 全部收敛。`LogisticRegressionCV` 在 NumPy、CuPy、Torch 和 sklearn 上均选择 `C=0.1`；Torch 与 NumPy 的 validation-loss 差异小于 `4.7e-8`。
- 历史 pre-fix P100 failure source 保持不可变并继续注册；post-fix exact-head source 单独从 `results/pr116_p100/cv_benchmark_pr116_p100.json` 注册，`focused_validation.json` 作为 validation-only evidence 保留，不作为 dashboard timing source。

关联：Issue #112 与 pull request #116。

## 0.2.4 — 2026-08-06

### Logistic 回归与 GLM 正确性

- 修正任意受支持 link 下 Binomial IRLS 的 Fisher 权重、工作响应、线搜索目标、后端原生 warm start 与二次惩罚校验。
- 强化直接 `LogisticRegression` 的响应与控制参数校验、事务式重拟合、收敛状态、整数硬预测、单列响应处理和有限阈值契约。
- NumPy、CuPy 与 Torch 的拟合后 logistic likelihood 统一使用数值稳定的 `LogisticLoss`；likelihood、AIC、BIC、伪 R² 与收敛状态不再依赖协方差推断是否开启。
- 单一类别目标仍可计算 confusion-matrix 与硬分类指标；ROC-AUC 和 average precision 继续保留明确的类别支持要求。
- CuPy/Torch 的解析权重保持设备原生，并修正加权 IRLS 曲率、likelihood、dispersion 与 sandwich inference 语义。
- 统一 GLM 在拟合、线搜索、诊断量和协方差中的 analytic-weight 语义；对权重整体缩放不会改变估计量或报告结果。
- 为 scalar GLM 增加后端原生的响应域、实数性、有限值、形状和长度校验，覆盖 penalized 与 CV 入口。
- formula sample weight 仅在 Patsy 完成缺失行筛选后对齐，并修正 Gaussian GLM FISTA 的加权中心化。

### 交叉验证、推断与 estimator 契约

- 使 `RidgeCV`、`ElasticNetCV` 与 `LogisticRegressionCV` 具备失败安全语义：每次拟合前清除旧状态，只有最终全数据重拟合成功后才发布所选参数。
- 保留显式 Torch/CuPy 请求，并将 `device="auto"` 的最终重拟合固定到 CV 阶段选定的后端。
- Logistic 与 Elastic Net 默认正则化网格现在纳入解析权重，并满足整数权重的行复制等价性。
- Penalized CV 保留声明的验证损失与解析权重；编程、shape、CUDA OOM 和 device 错误不再被转换为 candidate `NaN` 或无关的 MSE fallback。
- 完成独立 `ElasticNet` 与 `ElasticNetCV` 最终重拟合的 NumPy、CuPy、Torch 推断契约；各 fold 模型仍只用于估计。
- 修正 ElasticNet/Ridge 缩放说明：在共享平均损失约定下，`ElasticNet(alpha, l1_ratio=0)` 与 `Ridge(alpha)` 一致。
- 使公共有限值 guard、clone、sklearn tags、嵌套 `set_params` 与 fitted-state 失效处理具有事务性，并兼容旧版 scikit-learn clone identity 检查。

### Solver 与后端安全性

- 修正 solver matrix：Newton、L-BFGS 与 L-BFGS-B 会拒绝不支持的非光滑惩罚，不再只优化目标函数中的光滑部分。
- 删除错误的 Euclidean-prox Newton 快捷路径。光滑 L2/无惩罚目标继续使用 Newton；非光滑 proximal-Newton 请求会显式转到 backend-native FISTA，直到实现 Hessian-metric proximal solver。
- 将 Armijo、线性方程、CV grid 与 inference fallback 收窄到明确的数值域或秩失败；CUDA OOM、device、index、契约和其他 runtime failure 原样抛出。
- FISTA、Newton 系列、L-BFGS 系列与 ADMM 的 warm start 统一跟随预处理设计矩阵的 backend、device 和 dtype。
- 补全 ADMM 的合法 Cholesky fallback，并强化 L-BFGS-B 的可行方向、后端原生 bounds 与 NaN-bound 校验。
- 增加集中且可观测的 Torch compile policy：未设置、`auto` 与 `disable` 默认 eager；`default` 和 `reduce-overhead` 仅作为显式 opt-in。只有已知 CUDA Graph 输出生命周期错误会触发永久 eager fallback。
- 通过惰性导出 `CoxPartialLikelihoodLoss` 移除 `statgpu.glm_core` 与 Cox loss 的包初始化循环；全新解释器不再依赖特定导入顺序。

### 文档与发布准备

- 使中英文 LogisticRegression、ElasticNet、cross-validation、solver algorithm 与 solver/penalty 文档与当前实现保持一致。
- 删除无法由当前 exact-head 环境支持的通用 GPU 加速比、后端阈值与统一系数误差声明；性能建议改为针对实际 workload 做 benchmark。
- 明确 maintained pytest coverage 与手工物理 GPU diagnostics 的 ownership 边界。
- 将包版本更新为 `0.2.4`，并新增 `.github/releases/v0.2.4.md` 作为 GitHub Release 的权威正文。

### 验证

- PR #87 最终 implementation head 通过完整 CPU suite：2239 passed、719 skipped；同时通过 static/documentation contracts、Python 3.9–3.12 regression、scikit-learn 1.2.2/1.3.2/latest compatibility 与 release-package validation。
- 未改变的数值实现已通过物理 NVIDIA GPU 验证：RTX 4090 + PyTorch 2.8.0+cu128 的选定 compile/CUDA Graph matrix 为 9/9，并通过 runtime assertions；Tesla P100 + CuPy 13.6.0 也通过对应 runtime assertions。
- 当前 focused release PR 只修改版本元数据与发布文档；创建 `v0.2.4` tag 前，必须确保 exact release-head 的 hosted gates 全部通过。

关联：Issue #45、Issue #81、Issue #82、Issue #83，以及 pull request #87。

## 0.2.3 — 2026-08-04

### 生存分析

- 完成 CoxPH Phase 1：支持 Breslow、Efron 与 Exact ties，delayed entry、
  `(start, stop]` counting-process 数据、共享系数的分层模型、subject identifier，
  以及 `Surv(start, stop, event)` 公式输入。
- 为 NumPy、CuPy 与 Torch-CUDA 增加共享的 Cox risk-set objective、gradient、
  information matrix 与 baseline estimation primitive；Exact tied-event partition
  使用 backend-native dynamic programming。
- `CoxPHCV` 的 held-out partial likelihood 现支持全部 tie method、delayed entry、
  start-stop row、strata 与按 subject 分组的 fold。
- 强化 Cox inference、centered risk-set 数值计算、log-domain baseline prediction、
  公式 NA 对齐、奇异 information 检查、CV cache identity、fold eligibility、
  selected-penalty 全数据 refit 与失败 fit 的状态清理。
- 强化 L1、L2、Elastic Net、SCAD 与 MCP penalized Cox estimation；移除不可识别
  intercept，修正 Cox-specific warm start，并使 Torch Efron 的 value、gradient
  与 Hessian 路径保持原生实现。

### 交叉验证与分组惩罚

- 请求 CoxPHCV two-stage 或 successive-halving 时，统一执行一次显式 exhaustive
  full-precision candidate pass，在保持确定性选择语义的同时避免重复完整 grid fit。
- 一次性 `CoxPHCV.cv_splits` iterator 可在重复 fit、scikit-learn clone、参数重建
  与 pickle 中复用。
- 公开 Group Lasso 与 Adaptive Group Lasso 在支持的 backend 上统一采用 generic
  loss-gradient 与 exact group-proximal 路径。

### 验证与打包

- Hosted workflow #960 已在最终审查 head
  `f05a44ad363b46612e956e137e2f00d040765acb` 上通过：文档、static、完整 CPU
  与 Python 3.9–3.12 regression job 均通过；完整 CPU suite 为 1881 passed、
  662 skipped。
- 最终 exact-head 物理 GPU promotion artifact 已作为
  [schema-3 evidence](https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4)
  持久发布。它记录 134/134 项检查通过、child 与 nested return code 均为 0、
  gate-failure 数组为空、运行前后源码状态干净，SHA-256 为
  `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`。
- 包版本更新为 `0.2.3`。新增 release-package validation：检查版本一致性，构建
  pure-Python wheel 与 sdist，执行 `twine check`，核验 artifact 内容，并在干净
  环境中分别 smoke-install 两种发行包。

## 更早的历史记录

截至 2026-08-03 的详细条目保留在
[归档 changelog](changelog-history-through-2026-08-03.markdown)。