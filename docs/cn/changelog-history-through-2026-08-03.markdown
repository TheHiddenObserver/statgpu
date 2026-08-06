# Changelog

> 语言：中文<br>
> 最后更新：2026-08-03<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 2026-08

### 修复（2026-08-03）— PR #80 CV device-sizing matrix 后续

- 标量响应 `PenalizedGLM_CV.fit()` 新增 list 与一次性 generator 的端到端覆盖，
  证明 auto-device sizing 接收 materialize 后的实际 fold 数。Penalized Cox 的通用
  fallback 工作量只统计可评估 fold，同时保留 skipped-fold 原因与完整有限证据选择。
- 中英文 device 表现在区分经验 `n * p`/feature 规则与通用聚合工作量 fallback。
  精确源码 schema-19 P100 证据绑定提交
  `0bc131767bef1eeec45805073431e666f690b78c`：CuPy 与 Torch 各通过 14/14 个
  structured case 及 553 项定向测试；44/44 个 Git-blob hash 全部匹配，
  `source_clean=true` 且 `gate_failures=[]`。
- 架构章节现已分别展示标量响应与惩罚 Cox 的执行顺序，包括一次性 fold
  materialization、可评估 fold 的设备工作量估算，以及 Cox automatic grid 在选定
  backend 上的构造位置。
- 标量响应 CV 现在会在设备路由前校验完整用户 alpha 网格：非法标量值伴随 warning
  被过滤，空网格或过滤后为空会重新生成默认网格，shape/type 错误会在 candidate 与
  重拟合前失败。Ridge 文档现在区分 CPU-only 精确特征分解的 CV/重拟合计算与选定的
  预测 backend 契约。精确源码 schema-20 P100 证据绑定提交
  `a7053af2cb628880708cf2e4bfab121b1354725a`：CuPy 与 Torch 各通过 14/14 个
  structured case 及 581 项定向测试；44/44 个 Git-blob hash 全部匹配，
  `source_clean=true` 且 `gate_failures=[]`。
- 混合 Python/object 网格现在会在 dtype promotion 前拒绝布尔值与字符串/bytes，避免
  它们变成候选 alpha。标量端到端覆盖现已包括 L1、L2、ElasticNet、SCAD、MCP、
  Adaptive L1、Group Lasso、Group SCAD 与 Group MCP；精确源码物理 runner 升级到
  schema 21。
- 首轮 schema-21 P100 验证发现 Torch Group Lasso 在 CUDA block-coordinate solve
  内把 group metadata 建在 CPU。现在 group index、flat index 与 group-size weight
  都会在 candidate fit 前通过共享 backend array helper，按 design matrix 的设备一次性
  归一化。
- 最终精确源码 schema-21 证据绑定提交
  `5bb55ede04eecb5ab7689a400e864996fb514240`：CuPy 与 Torch 各通过 14/14 个
  structured case 及全部九类标量 penalty，630 项定向测试通过，记录的 45 个
  Git-blob hash 全部匹配，`source_clean=true` 且 `gate_failures=[]`。

### 修复（2026-08-02）— PR #80 惩罚 Cox CV 与后端后续修复

- `PenalizedGLM_CV(loss="cox_ph")` 现在会保留 `(time, event)` target，在
  NumPy/CuPy/Torch 上为 L1/L2/ElasticNet/SCAD/MCP 提供 strict CV，使用 held-out
  Cox partial likelihood 评分，要求完整且有限的 fold 证据，并以无截距
  `PenalizedCoxPHModel` 完成重拟合。所有候选无效时会事务性失败，不再选择第一个 alpha。
- `CoxPH(device="auto")` 会固定拟合后端用于预测与评分；`CompositePenalty`
  会保留 sklearn <=1.2 要求的构造器参数对象身份；`CoxPHCV` 会在任何 CV 工作前拒绝
  side-array shape 错误。精确源码 schema-16 P100 证据绑定提交
  `d688f760d8a0678c3c52c657a50178dad1b5ab3d`：CuPy 与 Torch 均通过 14/14 个 case，
  516 项定向测试通过，43 个源码 hash 全部匹配，`source_clean=true` 且 `gate_failures=[]`。
- Penalized-Cox 自定义 fold 现在复用 cast 前的严格索引校验，并支持一般的非空、
  互不重叠 split，包括前向与 repeated 设计。ElasticNet 自动网格按 `l1_ratio`
  使用零模型 KKT 缩放；纯 L2 明确记录 heuristic，无 penalty 别名作为不可调能力
  被拒绝，`device="auto"` 会先探测 CUDA backend 是否实际可用再回退 CPU。精确源码
  schema-17 P100 证据绑定提交 `f9e974b33c080c36a1a0cf1ca3508baca09f4939`：
  CuPy/Torch 均通过 14/14 个 case 与 541 项定向测试；44 个 Git-blob hash 全部匹配，
  `source_clean=true` 且 `gate_failures=[]`。
- Auto-device 工作量现在使用规范化后的实际 custom-fold 数，不再使用 constructor 的
  `cv` 值。Cox capability 文本收窄到 L1/L2/ElasticNet/SCAD/MCP，generic alpha-grid
  guide 也明确 Cox 的 hard-failure 语义。精确源码 schema-18 P100 证据绑定提交
  `a2d6a97d092d51a506421b67eea90fa71b5f8ac4`：CuPy/Torch 均通过 14/14 个 case
  与 544 项定向测试；44 个 Git-blob hash 全部匹配，`source_clean=true` 且
  `gate_failures=[]`。

### 修复（2026-08-01）— PR #80 无事件 stratum 预测后续

- `predict_survival()` 现在把已拟合但没有观察 failure 的 stratum 空 baseline 视为合法：
  累计 baseline hazard 恒为零，生存率精确为 1；存储的 time/hazard shape 不匹配仍失败。
- NumPy/CuPy/Torch 测试覆盖显式与自动 times、混合有事件/无事件预测行以及 `CoxPHCV`
  委托。物理 runner 升级为 schema 15 并加入机器可读专用 case；backend-import 检查名称
  也缩窄到实际审计的 dispatch 范围，避免声称覆盖整个 model layer。精确源码 commit
  `0d33a4fa64e7bf023407c4f691d008995ae67493` 的 P100 验证通过 CuPy 与 Torch 各
  13/13 个 case 及 475 项定向测试；39 个 Git-blob hash 全部匹配，
  `source_clean=true` 且 `gate_failures=[]`。

## 2026-07

### 修复（2026-07-29）— PR #80 prepared capability 后续审查

- `CoxPH.set_params()` 现在只校验 choice 与数值参数，不再改写公开表示；构造器、
  `set_params()` 与 `fit()` 因此遵守同一套 clone-stable 参数契约，计算仍通过不可变
  的私有 fit snapshot 使用规范化值。
- 普通 `CoxPHCV` fold 现在使用显式的 CV-owned trusted prepared capability。这些
  backend 数组在结构上仍然可变，但在完整 penalty path 生命周期内由当前 CV orchestration
  私有持有，因此每个候选可直接复用
  failure-group 元数据，不再执行 O(np) 的居中排序内容扫描，也不再临时构建设计矩阵；
  调用者持有的低层 prepared state 仍执行严格内容校验。
- canonical public solver path 现在传递带类型的
  `_PreparedCountingProcessInputs` 或 `_PreparedOrdinaryRightCensoredState`。
  active path 不再依赖原先三个相互约束的 flag；低层 prepared 元数据本身即可选择
  ordinary fast path，同时继续兼容显式请求 fast path 的直接调用。
- HC0、HC1 与 cluster 推断现在会拒绝少于两个独立单元的输入；HC1 还要求
  `n_units > n_features`，再应用精确的
  `n_units / (n_units - n_features)` 修正。稳健协方差对角线采用尺度感知的
  负值检查，退化 sandwich meat 不再生成零标准误和虚假的极端显著性。
- 协方差 benchmark 不再把 statsmodels 的模型协方差错误标记为 HC1；R 可用时
  会实际执行 `survival::coxph`，并在 JSON 中记录独立单元数、修正公式与明确的
  unsupported 原因；PHReg 若返回非有限系数推断，也会被标记为 unsupported。
- 秩亏 HC0/cluster 协方差不再进入无门禁的全参数求解。有效的边际稳健推断会保留，
  joint Wald test 则通过显式 availability/failure metadata 与 summary 输出标记；
  summary 也会区分 robust Wald 和经典 likelihood-ratio/score test。外部协方差向量
  现在必须具有精确长度且全部有限；R 与 statsmodels 显式使用对齐的 Newton
  `max_iter`/`tol`，JSON 同步记录 solver contract。精确源码 schema-11 在
  Tesla P100 上通过 CuPy/Torch 各 11/11 个 case 与 353 个定向测试；对齐后的
  Breslow/Efron R HC1 和 cluster 结果约在 `1e-16` 量级一致。
- Covariance 验证现在区分正定、PSD 但秩亏以及实质性非 PSD 三种谱状态：第一种支持
  完整推断，第二种保留有效边际结果并关闭 joint Wald，第三种令 strict inference
  事务性失败。Cox 从 inference package 复用该谱/Wald policy。外部 benchmark 的
  unsupported 行现在统一写入 `covariance_contract="unsupported"`，并单独记录
  请求的 contract 与失败原因。精确源码 schema-12 在 Tesla P100 上通过 CuPy/Torch
  各 11/11 个 case 与 358 个定向测试；记录的 32 个 Git-blob hash 全部匹配，且
  `gate_failures=[]`。
- 显式 stratified 拟合即使训练数据只有一个 stratum，生存预测也必须提供训练时已知
  的标签，`CoxPHCV` 委托路径遵守相同契约。`termination_reason_` 继续表示解释后的
  三类结果，新增 `optimization_stop_reason_` 公开 `max_iter` 等底层 solver 原始退出
  原因。EN/CN 模型页已用固定 source commit 的 schema-12 证据表替换过期的
  schema-6 pending 声明与多轮 review 时间线，并明确 artifact 的适用范围。精确源码
  schema-13 在 P100 上通过 CuPy/Torch 各 11/11 个 case 与 432 个定向测试；记录的
  34 个 Git-blob hash 全部匹配，且 `gate_failures=[]`。
- 正 L2 惩罚的 nonrobust Cox 推断现在使用固定惩罚强度的频率学派 estimating-equation
  协方差 `A^-1 J A^-1`，不再把 penalized curvature inverse 当作抽样协方差发布。
  provenance 明确记录推断目标、fixed-penalty 条件以及未校正 CV 选择；经典
  LR/score/AIC/BIC 仍保持关闭。`score()` 与 `predict_survival()` 现在复用同一套
  strata shape/已知标签编码，并在各 backend 上返回一致的公开错误。schema-14
  物理 GPU runner 已覆盖这两类契约。
- PR79 canonical Cox validator 现在与固定 penalty 的频率学派协方差
  `A^-1 J A^-1` 以及公开 delayed-entry 边界 `start < failure_time <= stop`
  一致。独立解析回归可区分该协方差与旧 curvature inverse，并覆盖一行恰好在
  failure time 进入的边界。精确源码 commit
  `0e48291de3c78dcfa6063e11947c43274e70c6c9` 的 schema-14 验证已在 Tesla P100
  通过 CuPy 与 Torch 各 12/12 个 case 及 468 项定向测试；39 个 Git-blob hash
  全部匹配，`source_clean=true` 且 `gate_failures=[]`。
- EN/CN CoxPH 模型页现在明确记录 objective、estimating equation、总 likelihood
  尺度的 penalty 口径与固定 penalty 推断限制，并提供可运行的 NumPy、CuPy CUDA、
  Torch CUDA 拟合和 CV 示例、R 外部证据及常见失败 FAQ。英文日期与损坏的参考文献
  页码分隔符也已和当前内容同步。
- 前一版 prepared-capability schema-9 精确 clean source commit 已通过 Paramiko
  在远程 `myconda` 的 Tesla P100
  上刷新。CuPy 与 Torch 各通过 10/10 structured cases，其中包括 fold strict-content
  重复扫描次数为零和公开 setter 表示稳定；物理 GPU targeted matrix 通过 321 项测试，
  记录的 29 个 Git-blob hash 全部匹配，且 `gate_failures=[]`。证据提交随后通过全部
  7 个 hosted docs、static、full-CPU 与 Python 3.9–3.12 jobs。
- 精确源码 commit `4570b9dca4cb771edfb1c29efb564c0e5340227f` 的 schema-10
  验证已在 Tesla P100 通过：CuPy 与 Torch 各通过 11/11 structured cases，
  targeted matrix 通过 343 项测试，31 个 Git-blob hash 全部匹配，
  `source_clean=true` 且 `gate_failures=[]`。在 `n=3000`、`p=10` 下，R
  `survival::coxph` 的 HC1 与 cluster 系数、标准误和 p-value 与 StatGPU 的差异
  约为 `1.4e-16`；statsmodels HC1 及其动态返回非有限值的 cluster 推断均在严格
  JSON 中明确标记为 unsupported。证据提交
  `8cb02c0e782b8719f86efea172059f5e801ab685` 随后在 Actions run
  `30451833466` 中通过全部 7 个 hosted jobs。

### 修复（2026-07-29）— PR #80 schema-7 物理 GPU 复验

- 精确源码的 schema-7 复验在 Tesla P100 上通过 282 项定向测试以及全部 18 个
  CuPy/Torch case gate。machine-readable JSON 记录 clean source commit、29 个源码
  hash，并直接验证 prepared-state 内容错配会被拒绝，以及 packed GPU target 的
  完整 host transfer provenance 会被如实报告。

### 修复（2026-07-29）— PR #80 schema-8 边界修复

- Penalized 与 canonical Cox 现在共享同一套 backend-neutral 预测矩阵规范：多特征
  模型的一维输入表示一条完整观测，单特征模型的一维输入表示多条观测；错误特征数和
  高维输入会在 backend matmul 前统一拒绝。低层 right-censored fast path 会拒绝
  非零 start 或多个 strata，避免 objective 与 baseline 使用不同的 risk-set 语义。
  `CoxPH` 和 `CoxPHCV` 拟合时改用不可变的私有 active controls，不再改写公开构造
  参数。schema-8 精确源码复验在 Tesla P100 上通过 318 项定向测试以及全部 20 个
  CuPy/Torch case gate，记录的 29 个源码 hash 均与 clean source commit 一致。

### 修复（2026-07-29）— PR #80 复审补充

- 复用的 right-censored loss state 现在会在底层求解前，于当前 backend
  上核对 `X`、time 和 event 的实际内容；同 shape 的其他数据或 prepare
  后的原地修改不再可能把旧 objective 与新 baseline 混用。`CoxPHCV`
  解包 CuPy/Torch packed target 时保留原生切片，因此完整 host transfer
  会如实进入 CV provenance。Cox 构造参数延迟到 fit 时规范化，penalized
  prediction/score 则统一复用 `BackendBase` 与共享的布尔、实数校验器。

### 修复（2026-07-29）— PR #80 最终后续审查

- 普通 GPU Breslow/Efron 拟合现在会如实报告完整排序 time/event 的
  device-to-host 传输。`CoxPHCV` 在一次完整 selector 调用中为每个 fold 只构造
  一次排序设计、失败组、event index 与 Efron fraction，并由全部 staged penalty
  pass 复用；该复用受显式 workspace 门禁约束，超限时会按 stage 重建，而不会
  保留无界的多 fold GPU cache。所有情况都不再为每个候选重复传输 target 和构造
  元数据。delayed-entry、strata 与 subject 拟合也会披露需要保留的
  完整 side-vector 传输；不含 side array 的路径不再复制虚构的全零 start 向量。
  未使用的 cluster/评分 unique labels 也不再物化到 host。
- `CoxPH`、`CoxPHCV` 与 `PenalizedCoxPHModel` 现在共享严格的 hazard-ratio
  数值契约：若有限 log-risk 的指数超出 float64 可表示的有限正数范围，则抛出
  `FloatingPointError`，不再返回无穷、零或使用 estimator 特有的隐式截断；原始
  log-risk 仍可通过 `predict_risk_score()` 获取。普通非分层生存预测会保留拟合时
  的 centered log-baseline，不再回退到直接计算 `exp(X @ coef)`。
- CV cache 诊断通过 `selection_cache_hit`、`selection_origin_device`、
  `requested_fit_device` 以及本次调用的准备/传输计数区分 selection 来源与当前调用。
  preparation 总数与实际向量复制总数分别记录。规范 Cox 每次公开 fit 只 reset 一次，公开 `CoxFitNumericalError` 同时从
  `statgpu` 和 `statgpu.survival` 导出。

### 修复（2026-07-27）— PR #80 后续审查

- 所有公开 `CoxPH.fit()` 现在统一使用稳定的 shared risk-set objective。普通
  nonrobust Breslow/Efron 使用有界 suffix-moment 快速路径，在保持近线性行扩展的同时，
  可稳定处理 `[-1000, 0, 1000]` 配合非零初始系数的有限输入；start-stop、strata、
  robust 与 Exact 场景继续使用对应的 backend-native shared kernel。
- 显式 Breslow `(n, p, p)` Hessian 工作区受
  `STATGPU_BRESLOW_HESSIAN_MAX_BYTES` 控制（默认 512 MiB）；CPU 超限时使用
  incremental grouped moment，CuPy 使用有界 grouped GEMM。CUDA OOM/runtime
  错误不再被 fused kernel 吞掉或误报为 information singular，least-squares 只针对
  已识别的 singular/ill-conditioned 线性求解失败。
- `CoxPH`、`CoxPHCV`、公开 `score()` 与 held-out partial likelihood 均在实数转换前
  拒绝 complex 输入。score test 通过 `score_test_available_` 与
  `score_test_failure_reason_` 暴露可用状态；device 错误原样传播，null information
  奇异则明确记录。ordinary 与 counting-process concordance 在无可比较 pair 时统一返回 `0.5`。

- `CoxPH(gpu_memory_cleanup=True)` 现在会在每个公开预测和评分调用结束后执行两类
  allocator 清理钩子，异常退出也不例外。`CoxPHCV` 在外层公开边界统一负责清理，并在
  内部最终 estimator 上关闭清理，因此每次 CV 预测或评分只执行一轮 allocator 清理与同步。
  摘要会输出真实的矩阵或 formula 接口，以及
  counting-process、strata、subject、cluster 与 ties 元数据，不再伪造 R 调用。规范
  Cox 路径与 `CoxPHCV` 最终重拟合现在统一发布 `ParameterInferenceResult`，并同步
  parameter、z、p-value 和置信区间字段。
- `CoxPHCV` 现在会对 CV 选择与最终重拟合的完整流程报告 full-host-transfer
  provenance，并分别暴露 CV/refit 字段。专用的 candidate numerical exception 使 CV
  可以排除非有限 penalty，但不会吞掉 input、CUDA、allocator 或编程错误。strata
  在每个 fold 中只 factorize 并通过 shared backend 准备一次；不计算 inference/C-index
  的 candidate fit 不再传入 cluster 或 subject label。规范 `CoxPH` 通过单一 reset
  contract 初始化状态，历史 risk-set cache 仅保留在测试 adapter 中。
- low-level concordance 会在转换前验证 `subject_id` 是否为有限、严格整数且在 int64
  范围内。survival risk-set 规范化复用了共享 backend 的数组、标量、zeros、eye 与
  integer-code helper；公开 fit 边界逻辑直接定义在 estimator 上，不再通过 import-time
  adapter 安装。
- 规范 `CoxPH` estimator 与公开 dispatch 继续位于 `_cox.py`，且不再继承或导入历史
  mixin。各 backend 的 information inversion 已作为无状态 helper 移入
  `_cox_inference.py`；不活跃的 CPU、CuPy 与 Torch 参考 kernel 仅通过
  `_cox_legacy.py` 中的显式组合 adapter 用于测试，公开 survival 导入不会再加载可选的
  legacy 探测逻辑。
- `CoxPHCV` 的 NumPy、CuPy 与 Torch held-out Breslow、Efron、Exact likelihood 现统一
  经过 shared counting-process objective。稳定的 NumPy log-likelihood-only 专用路径
  位于 risk-set 实现中，既保留原 suffix 路径性能，也避免 CV 模块重复维护统计定义。
  formula side array 统一使用一个保留 backend 的对齐 helper，CV prediction 文档也明确
  返回 NumPy、CuPy 或 Torch 原生数组。

### 验证（2026-07-27）— PR #80 后续审查

- exact clean commit 的 P100 产物在 `n=4096`、`p=12` 下记录了同步的
  NumPy/CuPy/Torch 中位时间：continuous Breslow 为 0.1003/0.0367/0.0373 秒，
  continuous Efron 为 0.2316/0.0501/0.0488 秒，heavy-ties Breslow 为
  0.0234/0.0184/0.0187 秒，heavy-ties Efron 为 0.1858/0.0214/0.0198 秒。
  所有重复均收敛且有限，六个 extreme-predictor 后端/ties 组合也全部有限：
  `results/benchmark_frontend_sources/coxph_stability_resource_pr80_20260727.json`。
- Penalized Cox SCAD/MCP 现在每次拟合只预处理、排序和传输一次 survival 分组元数据；
  FISTA-LLA 使用只计算梯度的热路径，按周期合并有限性与收敛状态传输，并在 allocator
  清理前释放 loss 持有的训练数组。
- trusted gradient 现使用有界行分块内的 scaled direct first moment，既保持最大
  predictor 离开后续风险集时的 denominator 稳定性，也避免在约 `1e15` 的正负矩之间
  发生灾难性消减。该路径明确保留 predictor-range scalar check，不再宣称 zero-sync；
  每个行块最多 65,536 行、两百万个 moment 元素，因此已移除的 signed-log scan 不会
  再产生随完整 `n` 增长的临时工作区。
- FISTA-LLA 会计入包含最终收敛更新在内的每次 proximal update，并准确记录各 alpha
  的累计迭代数。GPU event 校验只传输一个含两个 boolean 的状态向量，不再复制完整
  packed target；Torch 2.0 转换前会先规范化合法的 host `uint64` strata。
  NumPy、CuPy、Torch 的 `X`、time、event、start、stop 与 coefficient 复数输入均在
  转为实数之前明确拒绝。
- machine-readable 的物理 P100 产物记录其精确 clean source commit、Cox/FISTA/fit
  源码哈希、24 组同步次数与 gradient 对齐、48 组 SCAD/MCP
  coefficient/objective/KKT/finite-state 结果、6 组同步性能结果及 2 组物理 GPU 工作区测量：
  `results/benchmark_frontend_sources/penalized_cox_trusted_gradient_pr80_20260727.json`。
  产物明确标注 fresh-process cold-start 未测量，同时分别记录 warm process 中的首次
  fit 与紧接着的 steady-state fit，并说明未计入的初始化或编译成本。

### 优化（2026-07-27）— PR #80 后续审查

- 普通 right-censored Exact ties 在所有 strata 上使用一次分段前缀 DP。带 delayed
  entry 且 strata 数量至少为 8 的 GPU 工作负载可使用受内存门禁保护的全局 batch；
  较小场景使用有界的逐-stratum batch。

### 修复（2026-07-27）— PR #80 后续审查

- strata 在转为整数前会拒绝小数、非有限值和超出 int64 范围的标签，包括过大的
  unsigned 标签；可由 int64 表示的 `uint64` 标签在 NumPy、CuPy、Torch 中均会接受。
  `STATGPU_TORCH_EXACT_SCAN_STRATEGY` 支持 `auto`、`native` 和 `channelwise`；
  保守的 `auto` 只在已有实测证据的 Torch 2.0 + Pascal/P100 组合启用分通道扫描。

- 公开 Cox fit 边界会保留 packed CuPy/Torch target，重新校验可变的 device 与
  boolean control，在 cast 前拒绝 complex prediction 输入，并在 refit 失败后事务性
  清理状态。`inference_mode="approx"` 现明确记录为统一精确推断路径的
  compatibility-only alias；公开 estimator 的 strata 文档也与实际支持的可 factorize
  host 标签保持一致。

### 优化（2026-07-27）— PR #80 后续审查

- `STATGPU_COX_GROUP_MAX_BYTES` 现在会在分配前约束 Breslow/Efron delayed-entry
  failure-group 工作区。若单个 risk set 已超过上限，则使用数值稳定的 backend-native
  row-streaming moment fallback，不再因最小 dense batch size 为 1 而产生无界工作区。
  最终 schema-v3 exact-source P100 复验通过 121 项定向测试。在 `n=4096`、`p=128`
  和 8 MiB 上限下，旧估算为 1,056,768 bytes 并选择 dense，修正后估算为
  9,445,376 bytes 并选择 streaming；CuPy 与 Torch 均实际记录到 streaming
  路径，且与 NumPy 的最大差异为 `4.441e-15`。Concordance 现在对 event 与 sample
  两个轴同时分块，严格保证每块不超过两百万个 pair；物理 GPU 的 `n=2,000,001`
  边界场景使用了两个 sample tile，普通、counting-process 与 penalized 的全删失评分
  均返回 `0.5`：
  `results/benchmark_frontend_sources/coxph_concordance_boundary_pr80_20260728.json`。
- ordinary concordance 现在会在当前 backend 上累积全部 tile 计数，并仅在循环结束后
  批量传输一次标量，不再为每个 tile 触发三次 host synchronization。

### 验证（2026-07-27）— PR #80 后续审查

- 维护的 delayed-entry + 3-strata P100 基准在 10,240 行时测得
  NumPy/CuPy/Torch 中位时间 136.02/36.50/21.95 秒，即 GPU 相对 NumPy 提速
  3.73 倍/6.20 倍；该产物与新增的 strata-count 产物均为零 gate failure。
- 同一 P100 的 `n=4096`、`p=12`、64 个 time bin 场景在排除一次 warmup 后，
  direct-moment SCAD 的 NumPy/CuPy/Torch 中位时间为 0.08350/0.03148/0.02137 秒，
  MCP 为 0.08469/0.03100/0.02133 秒。CuPy/Torch 对 SCAD 的提速为 2.65/3.91 倍，
  对 MCP 为 2.73/3.97 倍；产物明确将其标为同步 warm timing，而不是 fresh-process latency。
- 刷新的 schema-v4 exact-source completion 产物在 Tesla P100 上使用 CuPy 13.6.0 与
  Torch 2.0.0+cu117，通过 159 项定向测试。它验证了公开清理的正常和异常路径、
  `CoxPHCV` 外层单一清理 ownership、真实 summary、共享无状态 inference result、整数
  subject code、ordinary concordance 单次标量传输、直接 backend 复用、不存在
  import-time method replacement，以及私有 legacy 组合隔离；同时记录
  `source_clean=true`、21 个经 Git blob 校验的源码哈希和
  `gate_failures=[]`：
  `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`。

### 优化（2026-07-26）— PR #80 分层 Exact 组合路径

- 多 strata 的 Exact 拟合此前无法进入两条单 strata 快速路径，而会退回到按
  `stratum × failure time` 执行的 Python/设备循环。新路径利用分层部分似然的
  可加性，对每个 stratum 复用 nested right-censored 或有界 batched
  counting-process objective；NumPy 也可在内存门禁内使用 batched Exact 处理
  delayed-entry 工作负载。
- 在 Tesla P100-SXM2-16GB 上（`p=4`、三个 strata、完整拟合及推断），
  `n=160` 时 R/NumPy/CuPy/Torch 中位时间为
  0.0180/0.0143/0.1742/0.0747 秒，`n=15,360` 时为
  0.258/0.2263/0.2181/0.1341 秒，`n=61,440` 时为
  1.118/0.9874/0.2285/0.1384 秒。两个 GPU 后端均在实测 `n=15,360`
  超过 R；`n=61,440` 时 CuPy 与 Torch 分别比 R 快 4.89 倍和 8.08 倍。
  显式 GPU 在小型分层拟合中仍受 kernel launch 开销限制。
- R 4.4.1/survival 3.8.9 对齐为零 gate failure；系数、Exact 部分对数似然与
  协方差的最大差异分别为 `5.84e-10`、`8.15e-10`、`4.45e-12`。
- 可复用 benchmark 为 `dev/benchmarks/benchmark_exact_ties_scaling.py`
  的 `--scaling-scenario strata`；可审计产物为
  `results/benchmark_frontend_sources/coxph_exact_strata_pr80_20260726.json`。

### 优化（2026-07-26）— PR #80 Torch Exact 通道扫描

- 在 Tesla P100、PyTorch 2.0.0+cu117 上 profiling nested Exact 后发现：一维 CUDA
  前缀和很快，但对 4 或 16 个尾部矩通道执行长轴 `cumsum(dim=0)` 会主导 Torch
  用时。
- 对至少 2,048 行且尾部通道不超过 64 的 Torch CUDA 输入，Exact 现在把各通道
  转为连续布局，执行高效的一维扫描，再在设备上拼回结果。
  `STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` 与
  `STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` 可配置门禁；小样本、宽张量和 CPU
  仍使用原生扫描。
- 额外通道扫描工作区已计入现有 512 MiB nested Exact 内存决策。若基础 DP 可容纳
  而额外扫描工作区不足，则 nested 算法继续使用原生 Torch 扫描。
- 在同步 bounded-tie 工作负载（`p=4`、最大 tie size 为 8、完整拟合及推断）中，
  `n=15,360` 的 R/NumPy/CuPy/Torch 中位时间为
  0.295/0.273/0.0949/0.0558 秒，`n=61,440` 为
  1.323/1.465/0.1114/0.0662 秒，`n=122,880` 为
  2.691/3.043/0.1430/0.1000 秒。最大规模下 Torch 比优化前快 30.32 倍、比 R
  快 26.92 倍、比 NumPy 快 30.44 倍、比 CuPy 快 1.43 倍。
- R 4.4.1/survival 3.8.9 对齐为零 gate failure；最大系数、Exact 部分对数似然和
  协方差差异为 `1.30e-09`、`5.12e-09`、`5.01e-12`。本地 13 文件矩阵通过
  **297 项**、97 项可选依赖 skip；真实 P100 矩阵通过 **392 项**、2 项预期 skip。
- 可复用入口为 `dev/benchmarks/benchmark_exact_ties_scaling.py`，输出
  `results/exact_ties_scaling.json`；最终 artifact hash 记录于
  `dev/reviews/pr80_review_fix.md`。

### 优化（2026-07-26）— PR #80 普通右删失 Exact 完整拟合

- 大样本分阶段 profiling 表明，Exact likelihood 前缀已不再是完整拟合瓶颈；
  Breslow baseline 推断仍会对普通右删失数据执行 `失败组 × 样本` 风险掩码扫描。
- 将该常用路径改为每个 stratum 内按 stop time 降序的一次 log-risk 前缀：
  NumPy 使用 `logaddexp.accumulate`，Torch 使用 `logcumsumexp`，CuPy 在保守
  predictor-range 门禁内使用平移后的累积和。delayed entry 与极端 CuPy
  predictor 保留数值稳定的后端原生 fallback。
- 在 `n=61,440`，NumPy/CuPy/Torch 的 baseline 阶段从
  6.847/5.988/3.328 秒降至 0.0202/0.00701/0.00265 秒。最终本地受影响矩阵为
  **226 passed、37 skipped、0 failed**；13 文件真实 P100 完整矩阵为
  **388 passed、2 个预期 skip、0 failed**。
- 在同步 P100 bounded-tie 工作负载（`p=4`、最大 tie size 为 8、完整拟合及推断）
  中，`n=15,360` 的 R/NumPy/CuPy/Torch 中位时间为
  0.305/0.282/0.0971/0.361 秒，`n=61,440` 为
  1.293/1.469/0.113/1.510 秒，`n=122,880` 为
  2.589/3.023/0.1518/3.031 秒。最大规模下 CuPy 比 R 快 17.05 倍、比 NumPy
  快 19.91 倍；小规模 `n=1920` GPU 拟合仍受 kernel launch 限制。
- R 4.4.1/survival 3.8.9 对齐仍为零 gate failure；综合场景相对 R 的最大系数、
  exact partial log-likelihood 与 model covariance 差异为
  `1.30e-09`、`5.46e-12`、`5.01e-12`。
- 可复用验证入口为 `dev/benchmarks/benchmark_exact_ties_scaling.py`，
  输出 `results/exact_ties_scaling.json`。


### 改进（2026-07-25）— v0.2.2 发布准备

- **版本与打包**：
  - 将 `pyproject.toml` 和 `statgpu/__init__.py` 的版本从 0.2.1 更新为 0.2.2；
  - 保留 tag 触发的 PyPI workflow 和 `STATGPU_NO_EXT=1` 构建策略，生成通用
    `py3-none-any` wheel 与 source distribution；
  - 继续以 Python 3.9 至 3.12 作为维护中的 CI 版本矩阵。
- **纳入的维护范围**：
  - 包含下方条目与可审计产物所记录的 PR #79 正确性、后端契约、推断和验证工作；
  - 包含 PR #84 对发布入口 README、文档门户、方法清单、中英文模型/后端指南和
    确定性文档契约的更新。
- **发布文件**：
  - `pyproject.toml`
  - `statgpu/__init__.py`
  - `CHANGELOG.md`
  - `docs/en/changelog.md`
  - `docs/cn/changelog.md`

### 验证（2026-07-25）— v0.2.2 发布候选

- 两处版本声明均为 0.2.2；实时 PyPI 元数据显示最新版本仍为 0.2.1，远端仓库中
  不存在 `v0.2.2` 标签。
- 文档链接检查与维护中文档契约检查全部通过，共覆盖 122 个维护中文档文件。
- 完整 CPU-only suite 结果为 **1051 passed、257 skipped、0 failed**。
- `STATGPU_NO_EXT=1` 成功生成 `statgpu-0.2.2-py3-none-any.whl` 和
  `statgpu-0.2.2.tar.gz`，两个制品均通过 `twine check`。
- 已审计 wheel/sdist 元数据、归档路径与内容，未发现本地配置、凭据、缓存或无关结果包。
- wheel 与 sdist 均在全新环境中从已安装的 `site-packages` 导入 statgpu 0.2.2，
  并通过 CPU `LinearRegression` smoke test。

### 新增与修复（2026-07-25）— PR #80 Cox Phase-1 完成

- 将原本基于 0.2.1 的 PR #80 head 与 0.2.2 发布树对齐，同时保留 0.2.2
  版本及 PR #79 的 inference/KKT 契约。
- 增加共享计数过程风险集引擎，覆盖 Breslow、Efron、Exact ties、delayed entry、
  `(start, stop]` 时变行、strata、惩罚、robust/cluster 推断与 subject-aware
  concordance。
- 扩展 `CoxPHCV` 的 start/strata/subject 传递、subject-preserving folds、Exact
  held-out likelihood、后端一致的最终 refit 与 inference-mode provenance。
- 修复最终 KKT 收敛、open-left `start < event_time` 边界、baseline hazard 构造、
  后端原生预测/评分，以及 GPU benchmark 同步计时和源码版本记录。
- 将 CuPy/Torch 的密集 Efron 累积矩与 log-likelihood 子步骤向量化；对于单个
  stratum 的普通 right-censored Exact 拟合，NumPy/CuPy/Torch 现在跨嵌套风险集复用
  elementary-symmetric 前缀 DP，并用按事件时间排序的分段前缀和移除
  `失败组 × 样本` 密集掩码。delayed entry、多个 strata、score residuals、
  工作区超限和保守数值范围门禁继续使用后端原生的 normalized batch/逐组 fallback。
  两个 Exact 工作区上限默认均为 512 MiB，并在密集分配前完成检查。
- 复用默认零初值的 null objective、不需要 score residuals 时已接受的 final
  objective，以及求解器已计算的 null score/information，避免 Exact 拟合与 score
  test 中的重复求值。
- 2026-07-25 的本地 NumPy quick gate 已通过全部可执行 correctness、inference、
  CV、schema 与外部对齐检查。随后通过 Paramiko 在远程 Tesla P100 的 `myconda`
  环境中验证准确的 reviewed source，发现并修复 Torch prediction、
  scikit-learn 1.2.2 clone 与测试边界问题。最终真实 GPU 矩阵为 **384 passed、
  2 个预期 skip、0 failed**；NumPy、CuPy、Torch 的 quick/full benchmark schema
  均通过且没有 gate failure。
- 同步后的 full benchmark 中，heavy ties 中位拟合时间为 NumPy 0.477 秒、CuPy
  0.179 秒、Torch 0.212 秒，较早的 Efron 优化仍使 CuPy/Torch 提速 8.36 倍和
  24.31 倍。最终 nested-Exact benchmark 在同一 Tesla P100 上（`p=4`、最大 tie
  size 为 8、完整拟合并计算推断）测得 `n=960` 的 R/NumPy/CuPy/Torch 时间为
  0.029/0.0253/0.1686/0.0941 秒，`n=1920` 时为
  0.047/0.0585/0.2690/0.1590 秒。在 `n=1920`，StatGPU 三条路径相对 reviewed
  pre-prefix NumPy/CuPy/Torch 实现分别提速约 928 倍/41.0 倍/41.6 倍，且未使用
  隐式 CPU fallback。可复用脚本为 `dev/benchmarks/benchmark_exact_ties_scaling.py`。
- 将该基准扩展为可选的 R 4.4.1/survival 3.8.9
  `coxph(ties="exact")` 外部对齐。right-censored、delayed-entry、strata 及组合场景
  在三个 StatGPU 后端上均通过系数、exact log-likelihood、协方差和收敛门禁；相对
  R 的最大差异分别为 `1.30e-09`、`4.55e-13`、`5.01e-12`。bounded
  right-censored `n=1920` 场景的 R/NumPy/CuPy/Torch 时间为
  0.047/0.0585/0.2690/0.1590 秒；另一个 delayed-entry `n=160` 场景则为
  57.079/0.167/0.544/0.353 秒，表明 Exact 性能强烈依赖风险集形状。

### 验证（2026-07-24）— PR #79 exact-head 最终闭环

最终 review 的生产代码 head 为
`c85750d63d4e6dbc9d988847566c20f5fa862e91`。

- exact-head GitHub Actions Tests run #545 通过；
- Python 3.9、3.10、3.11、3.12 regression job 全部通过；
- 完整 CPU suite 为 **1074 passed、275 skipped、0 failed**；
- clean-head canonical smoke pipeline 通过，`canonical_eligible=True`，verdict 为 `PASS`；
- 维护中的 Tesla P100 suite 执行 **33 个检查全部通过**，另有两个预期 skip；
- CoxPH、Linear 与 Panel 的维护路径均满足 PR79 验收合同。

另外执行的六个旧 GPU 诊断脚本没有纳入维护 pytest Gate。其转换、替换或移除由
[Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83) 跟踪。

### 修复（2026-07-24）— 最终公开合同与文档同步

- 修正 CoxPH delayed-entry 支持矩阵：robust/cluster covariance 在
  `compute_inference=True` 时显式报错；同一拟合在 `compute_inference=False` 时允许仅估计，
  推断字段保持未设置。
- 明确 `CoxPHCV` 在最终 refit 时执行相同 inference guard。
- 文档化 PooledOLS 后端保持预测、稳定 HAC `time_index` 排序和有效秩 residual degrees of freedom。
- 明确秩亏 PooledOLS：fitted value、prediction、RSS、rank 与拟合空间检查仍有效；
  系数级推断由于不唯一识别而标记为 `NOT_COMPARABLE`。
- 同步 README、中英文 CoxPH/Panel 模型页、双语 release summary 与 PR79 审计报告。
- 删除陈旧的硬编码 final accuracy artifact。只有在 exact target SHA 上重新执行完整 raw campaign，
  并通过当前 aggregator 与 renderer 后，才可以重新提交 full canonical report。

### 修复（2026-07-23）— PR #79 完整 review 闭环

- 统一 CPU/CuPy/Torch CoxPH 的最终 KKT、line search、终止状态和公共结果字段；
- 新增默认 strict、显式 opt-in 的 approx 稳健推断与 provenance 字段；
- Cox 预测与评分保持后端原生，baseline hazard 使用向量化风险集，移除受影响的 Torch Hessian materialization，
  并避免 nonrobust GPU 推断无条件复制完整训练数据；
- 强化 PR79 diagnostics 与 canonical report：missing、failed、duplicate、non-finite、dirty、wrong-SHA 证据全部 fail closed；
- 新增行为回归并同步中英文 Cox 支持矩阵。

### 验证历史（2026-07-21）

较早的 Tesla P100 完整 campaign 在代码 head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad` 上通过：

- Gate A：160 passed、0 failed、2 个预期 skip；
- Gate B：1100 passed、0 failed、124 skipped、1 个 strict XFAIL；
- Gate C：10/10 metamorphic 检查通过；
- Gate D：审计路径未发生完整设计矩阵 GPU-to-CPU 传输；
- Gate E：CuPy 与 Torch 各重复 15 次，未发现显存泄漏；
- Gate F：记录三个规模下的同步 Tesla P100 性能基线；
- Gate G：Ridge/scikit-learn 与线性回归/statsmodels 对齐通过。

后续在 `786af9e2eb4742a56e5203b4380b03aec63a3ac8` 上进行的 exact-head campaign
又通过了 17/17 个 focused physical-GPU 检查。这些历史 SHA 仍是可审计证据，
但上方 2026-07-24 条目才是最终 PR head 闭环。

### 性能基线 — Tesla P100

以下为特定硬件下的回归基线，不构成跨硬件性能保证。

| 数据形状 | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

环境：Tesla P100-SXM2-16GB、Python 3.9、CuPy 13.6.0、PyTorch 2.0.0+cu117。

### 已知非阻塞后续工作

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81)：共享的后端原生 NaN/Inf 验证；
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82)：为 scikit-learn <=1.2 clone identity 重构公开构造器；
- [Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83)：转换或移除未纳入维护测试树的旧 GPU 诊断脚本。

## 历史变更记录

截至 2026-07-14 的详细记录保留在
[归档 changelog](changelog-history-through-2026-07-14.md)。
