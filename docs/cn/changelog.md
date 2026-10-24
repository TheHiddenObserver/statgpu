# Changelog

> 语言：中文<br>
> 最后更新：2026-10-24<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 未发布 — 工作区修复

### Benchmark 资格完善（PR #258）

- CV 干净同步公开拟合计时与 selector/refit 插桩诊断分离，保留目标函数、折分、参数网格及数值阈值。
- 干净计时拒绝活动中的 Torch 原生 profiler，并检查 CV/最终 refit 的显式收敛状态，包括 Cox RuntimeWarning 对应的不收敛。
- Gaussian ingress 观察移到计时外；workflow 增加正确性预检并明确 host/mixed 计时口径。
- 扩展 learning 算法阶段证据并分离 helper 追踪与计时；未闭合或混合计算不标记为全原生。
- 增加七套统一资格与扫描入口，保留精确节点清单、独立线程 profile、有限规模及不可改写的原始证据。GPU 独占未验证时仅为诊断，不产生加速比或 dashboard 资格。
- 新增 `run_full_benchmark_matrix.py`，统一先规划、后显式执行的注册/独立 lane；支持多 seed/shape profile、受支持的命名配置与确定性分批。逐 case 的资源/oracle 边界会拒绝或明确排除不支持的点，不静默缩小输入。
- 覆盖核算不再只看七套注册 suite：区分 275 个注册身份、默认 21 次调用/1692 行代表性矩阵、独立 runner 文件以及有限配置矩阵。保留 blocked、不适用和未执行项，不能把进程成功退出解释为全覆盖。
- 独立登记公开 KDE bootstrap、hybrid phase、Cox/panel 专项、外部对照、compile 与 memory 等计时协议。wrapper/子程序及重叠 scale 点不是额外独立证据；legacy 或不安全入口不会自动执行。
- 明确 solver/family/weights/sparse/precision、规模/oracle、统计校准和硬件方面的剩余缺口。输入 dtype、CUDA 输出归属和 host/hybrid 工作流不等于原生计算精度或全算法 CUDA 执行。参见[资格指南](guides/benchmark-qualification.md)及 `dev/benchmarks/BENCHMARK_QUALIFICATION_GAPS.md`。
- 不修改生产算法或 Logistic 默认值。计划、源码盘点及合成 harness 验证都不是物理 GPU 结果；历史运行不能证明改过的源码或扩大的矩阵。新源码仍需物理 GPU 验收。

### PR #257 — 完整扫描正确性与证据修复

- 在 benchmark 的独立 oracle/报告边界显式导出原生拟合数组，包括 Fama–MacBeth 系数；不改变原生拟合及数值容差。P100 复验失败保留为历史，需按修复后的新源码重跑。

- 隔离注入式 Torch CPU 测试与机器上的 CUDA 可用性；公开 GAM CPU 路由、数值容差及物理 GPU 测试不变。首轮 P100 失败保留为历史，仍需按新源码复验。
- 恢复 GPU 平滑二次目标的 FISTA 停止检查，不增加迭代预算或放宽收敛容差。
- squared-error L2 PenalizedGLM_CV 最终重拟合保留在请求的数值后端。
- Torch F 生存函数对分子自由度为 2 使用稳定的原生精确表达式，其他自由度行为不变。
- Nyström 节点分解与 GAM 节点选择改用所选后端；KernelPCA 分解失败不再静默转到 CPU 重试，公开主机报告边界仍明确保留。
- 加强逐算法数值操作证据和失败诊断。本地 CPU 测试不能替代待运行的 P100 复验，原扫描证据及仅诊断性能资格保持不变。

这些源码变更不包含在已发布的 0.3.0 wheel 或 sdist 中。下方历史发布准备记录保留
原有措辞，包版本没有调整；此前的 P100 验收不能证明本次数值变更已通过验证。

### 原生 KDE bootstrap（2026-10-22）

- CuPy bootstrap 改用原生 RNG、重拟合及分位数计算；支持 Torch bootstrap 和
  Torch 加权 KDE。共享的 Torch 权重与查询形状修复也适用于核回归，但不包含
  `bandwidth_per_feature`。
- 公开区间结果仍为 NumPy 数组。原生字符串选择器每次重复重新拟合，NumPy 一维
  高斯快速路径仍固定初始因子。原生种子机制会改变随机序列及 Generator 状态消耗。
- 可选的 `workspace_bytes` 约束管理缓冲区的估计预算，不是峰值内存硬上限。
  详见[指南](guides/native-kde-bootstrap.md)。
- 高斯对数密度归约排除零概率质量观测。原生选择器使用稳定排序和固定 PCA 载荷
  符号，NumPy 选择器算法不变。由于加权分位数重采样对反向不保持不变，即使 PCA
  主轴非退化，加权多元因子也可能明显不同。最大特征值重复或近似重复还会引入
  方向歧义并发出 `RuntimeWarning`，公开区间调用会汇总内部拟合的警告。
  原生投影诊断数组保留在后端设备上，选择器 `to_dict()` 仍为浅层转换。
- 历史实验候选的 GPU 结果不能用于证明这次公开实现的正确性或性能。

### 修复（2026-10-20）

- **原生采样（#173）**：F 采样将公开参数 `dfn`/`dfd` 转换为 NumPy 的
  `dfnum`/`dfden`，不再因关键字不匹配抛出 `TypeError`。全部 15 种原生采样器
  应用显式输出 `dtype`；默认 `None` 仍返回 float64，离散分布也不例外。
  NumPy/SciPy CPU 随机生成后再转换到所选后端的方式不变，不代表原生 GPU RNG。
  回归覆盖：`dev/tests/test_distribution_sampling_regressions.py`。
- **逆高斯和负二项稳健协方差**：HC0/HC1 的 sandwich bread 改用观测损失
  Hessian，而非期望 Fisher scoring 曲率。惩罚曲率、分析权重得分 meat、HC1
  行数修正与固定/已选调参条件不变。优化器 Hessian 和 nonrobust 协方差仍采用
  Fisher；独立的 oracle 路径原本已使用观测 bread。普通拟合、受支持的光滑
  M-estimation 和 CV 最终重拟合受影响，因此拟合系数不变时稳健标准误也可能改变。
  导数解释限于均值未裁剪的范围，没有新增裁剪求导规则。
  回归覆盖：`dev/tests/test_glm_observed_bread.py`。
- **周期三次样条（#216）**：以解析单侧端点函数值、一阶和二阶导数替代范围外
  有限差分。确定为秩三的投影返回 `n_knots + 1` 个 float64 列，秩无法确定时明确
  失败。常数函数仍在基空间内，但没有专门的截距列。使用旧基的模型须重新拟合；
  没有新增可复用训练投影或外推 API。自然样条的数值限制保持不变。
  回归覆盖：`dev/tests/test_cyclic_spline_periodic_constraints.py`。
- **UMAP NumPy 2 分派（#191）**：近邻搜索和负采样 RNG 根据已解析的后端身份
  分派，不再根据数组是否有 `device` 属性判断。NumPy CPU NNDescent 不再转入
  Torch 路径。exact/auto 近邻选择、主机 SciPy 图组装、谱初始化和 CPU 一维布局
  的限制不变。回归覆盖：`dev/tests/test_unsupervised_umap.py`。

### 验证（2026-10-20）

上述定向回归和 benchmark smoke 属于本地正确性检查，不是物理 CUDA 或性能验收。
最终数量及源码身份以本轮验证产物为准；不可用的 GPU 路径仍待验证，不能沿用之前的
发布证据推定通过。

## 0.3.0 — 即将发布

本版本汇总自 0.2.5 以来已合并的变更。此前标为“未发布”或计划纳入 0.2.6 的条目统一归入 0.3.0，不表示曾发布独立的 0.2.6。安装与升级概览见[版本发布说明](../../.github/releases/v0.3.0.md)。

### 主要变更与升级说明

- **官方网站上线**：[statgpu.github.io](https://statgpu.github.io/) 提供中英文文档的在线入口。
- Gaussian 线性模型推断及受支持的稀疏 Gaussian 残差自助法，现在在拟合使用的 NumPy、CuPy 或 Torch 后端执行数值计算。最终报告数组仍可能采用 NumPy；模型能够在 GPU 上拟合，并不表示所有推断方法都支持 GPU。
- 惩罚模型直接拟合使用 `solver`，`LassoCV` 的交叉验证选择阶段使用 `cv_solver`。若希望保持直接拟合行为，删除已弃用的 `cpu_solver` 即可，不要机械地把其值复制给 `solver`，否则可能改用另一种算法。推断方法采用 `inference_method="post_selection_ols"` 替代已弃用的 `cpu_ols`/`gpu_ols`，硬件由 `device` 选择。参见[求解器迁移指南](guides/penalized-solver-api-migration.md)与[推断方法](guides/inference-modes.md)。
- **推断结果变化**：自动逐节点 Lasso 调参改为由标准化设计决定，不再依赖响应尺度，因此已有去偏区间和 P 值可能改变。需要固定辅助惩罚时，以关键字传入正数 `nodewise_alpha`；它不用于选择预测模型的惩罚强度。
- **更严格的错误处理**：非高斯 L1/ElasticNet 系数推断以及不兼容的 Quantile 求解器、惩罚组合现在明确报错，不再返回不完整结果或悄悄替换算法。SCAD/MCP oracle 推断仍仅支持 CPU，并以已经选定的变量集合为条件；oracle 推断失败会清除已拟合状态。受支持的光滑非高斯 L2/无惩罚 M-estimation 和交叉验证最终重拟合推断均不校正调参选择的不确定性。
- Oracle 收敛检查仍因分布族而异：只有 Tweedie 与负二项重拟合新增相对一阶得分检查。Poisson 等其他分布族保留子模型原有停止准则，包括相应的绝对容差。小尺度数据可能需要更严格的 `tol` 或调整尺度；范围检查不是收敛证明，相对得分检查也不提供系数或协方差误差界。参见 [oracle 数值限制](guides/penalized-glm-inference.md#无惩罚重拟合与分布族设置)。
- RidgeCV 按自定义训练子集计算，包含带权和反向索引的数据折。它仍使用有限精度的原始 Gram 矩；提升已形成矩的精度无法恢复已经丢失的信息。遇到相应错误时可关闭混合精度或调整输入尺度，不能仅因系数有限就认定结果准确。
- 聚类自助法按完整组抽样，每次样本行数可以不同。缺失聚类标签会报错，`"nan"` 等字面字符串 ID 仍有效。普通公式预测仍可能删除缺失行，但 LinearRegression 评分现在会拒绝由此产生的形状不匹配。
- 新教程同时说明现有修复与限制。UMAP 邻域与初始化限制、KernelPCA 重新拟合失败后的状态问题、KernelPCA/Nystroem 的 Torch 设备放置限制以及 knockoff 并列统计量问题，并未由文档更新修复。使用相关流程前请查阅[模型指南](models/README.md)。

### 发布准备的验证范围（2026-10-19）

PR #249 在历史源码 `0969d9da21b293ac569905bc95bd1fbdcb51f13c` 上记录了 **4,229 项 CPU 通过**、**233 项 Torch CPU 通过**、**824 项 CUDA 通过**（**CuPy 412 项 + Torch CUDA 412 项**），以及 **6 项原生 CUDA 探测通过**。收集到 **5,288 项测试**，其中含 **2 项可选双 GPU 用例**。实际运行使用单张 Tesla P100；测试收集不等于执行，这些结果不能证明双 GPU 验收通过。合并后的基线 `5b9d42476f7b921918b7202142f7d39ffd10cdd0` 对应同一 Git 树 `6065449b`，其 3 个托管工作流、14 项检查已通过。以上属于历史基线结果，不是整个新版本的认证，也不是性能提速结论。

发布前仍须按照[发布流程](../../RELEASING.md)，对最终发布源码完成所需 CPU、Python 3.9–3.12、文档与静态检查、打包、wheel/sdist 干净安装、Ubuntu/Windows/macOS wheel 冒烟测试及相应物理 GPU 验证。下文各功能的历史记录仅适用于其中注明的源码、验证程序和硬件。准备这些说明不等于创建标签或向 PyPI 发布。

### 其他文档与开发工具变更（PR #165、#170、#136、#141）

- PR #165 增加了 [float32 L-BFGS 数值说明](guides/lbfgs-float32-precision-contract.md)与诊断测试。原生 float32 优化可能因后端舍入差异产生不同迭代轨迹；应同时比较目标值与平稳性，需要更严格的系数一致性时使用同后端 float64 对照。这次变更没有替换 L-BFGS 生产算法。
- PR #170 改善中文模型与无监督学习说明；PR #168 随后整理了中英文对应的教程和参考导航。
- PR #136、#141 更新贡献者流程与项目工具，没有新增公开数值方法。

以下保留详细的 API、统计含义与数值行为变更。

### 数值行为修复（PR #249，2026-10-17 ~ 2026-10-19）

#### 修复（2026-10-17 ~ 2026-10-19）

- **RidgeCV 自定义训练行（#243）**：
  - 在添加正则项之前，拒绝已存储 Gram 对角元精确为零、对应交叉乘积分量却非零的不一致状态。CuPy float32 检查改为在原设备上读取同宽整数视图并屏蔽符号位，修正浮点转换导致的不一致漏检和合法输入误拒。正负零仍视为零，非零次正规存储值以及合法零特征、零响应场景得以保留；其他后端与精度沿用原有判定路径。这不重建因舍入、抵消或冲零而丢失的因子信息；发生该精度错误时，可关闭混合精度或缩放输入。
  - 原次正规因子混合精度样本保留为矩已丢失时的明确精度失败回归，并增加 float64 解析成功对照。提升精度后的系数评分成功回归改用正规 Gram 元素，仍要求超过 float32 范围的系数，并同时检查分数分离和 alpha 选择。这与此前收窄的原始矩契约一致，不宣称原丢失因子场景已能准确求解。
  - 修正困难原生 Gram 求解在不抛出线性代数错误时，仍可能返回有限但非最小范数秩一解的问题。奇异备用求解前先检查数值秩；完全独立的坐标块保留可辨识的小方向，很小的系统采用受限的向上缩放，耦合尺度下无法一致判定数值秩时则明确报错。普通批量路径和原始加权目标保持不变，不恢复此前撤回的因子重建保证。
  - 对 float32 次正规数存储范围内的有效惩罚，先检查再转换精度；当其影响求解时，在同一设备上使用 float64 运算，避免 CuPy 将次正规数冲为零而错误拒绝。超出所配置精度的正有限舍入范围时仍报错。空特征最终重拟合不再调用不适用的零维 CuPy 三角求解，保留原有拟合与推断生命周期。
  - 保留原先物理失败输入，并如上明确修正次正规因子测试契约。最终重拟合的近零系数检查改用独立高精度目标与根据工作精度推导的矩形成、求解误差界，不再要求抵消结果逐位为零；其余严格交叉验证得分、所选 alpha 和预测断言保持原容差。CPU/Torch CPU 运行不代表物理 CUDA 通过。
  - 显式提供 alpha 网格时，加权零特征设计恢复仅截距或零预测模型的合法交叉验证行为。空斜率谱跳过不适用的归约，仍保留折评分、输入校验和最终重拟合。自动生成 alpha 网格仍要求至少一列斜率特征。
  - 撤回开发期间额外提出的因子 SVD、完全重复因子压缩、补偿投影及极端或次正规因子恢复承诺。CPU 加权折保留原始加权目标；数值困难的 GPU 折在同一设备上逐个 alpha 求解原生 Gram 系统，并按数值秩处理奇异系统；已形成的矩会提升为 float64，但不重建原始因子。既有可计算的原始 Gram 场景仍有回归覆盖，包括秩一或重复设计，以及原始权重使加权矩可表示的次正规尺度样本。原生正规方程的条件数和表示范围限制仍然存在，系数有限本身不证明准确。详见 [RidgeCV 数值范围](reference/linear-model-api.md#custom-ridgecv-training-subsets)。
  - 加权验证先计算可表示的有限残差，再施加平方根权重，以保留残差相减时的有效消减。加权验证 MSE 非有限时抛出 `FloatingPointError`，并提示调整尺度。RidgeCV 还会在发布任何交叉验证或重拟合状态之前，拒绝最终系数或截距非有限的结果；直接 Ridge 的行为不变。
  - 原无权重折统计量优化即使收到较小的训练子集，也可能把验证集补集当作训练行。现在无论是否提供样本权重，都严格使用指定的训练与验证行，支持空窗、隔离期以及未覆盖全部观测的验证集。
  - 加权折保留相对观测值的中心化及正子集权重校验。原始加权 Gram 矩阵和交叉乘积保留原加权目标；统一缩放权重后的数值不变性受它们在求解精度下的表示范围和精度限制，包括次正规数精度损失。加权 GPU 矩先以原生 float64 形成再转换精度；加权中心化与加权验证始终使用 float64，常规混合精度求解使用 float32 Gram 矩阵。有效折规模缩放后的正则项在该转换之前形成。GPU 特征分解不施加固定的正特征值下限。有效正则项无法表示为有限正数，或者矩、正则化系统或系数非有限时，抛出 `FloatingPointError`；这不能检测所有有限但不准确的结果。
  - 反向切片的 NumPy 训练或验证索引此前保留了 Torch 不接受的负步长，尤其会在检查原生 Torch 样本权重时失败。现在完成索引校验后，仅在必要时复制数组以调整内存布局，包括反向切片的单元素数组，同时保留指定行顺序和调用方输入。回归测试以独立闭式解核对完整划分、训练子集、部分验证覆盖及单元素划分的交叉验证评分与全数据重拟合，覆盖两种截距设置及无权重、单位权重、非均匀权重。另有 Torch CPU 转换测试检查索引转换；物理 CUDA 测试仍依赖相应硬件。
  - 索引必须是一维、非空、范围内的整数；每个子集内不得重复，训练与验证之间不得重叠。即使不足四行或只有一个 alpha 候选，显式划分也会计算评分；仍要求 `cv>=2`，但实际数据折以显式划分为准。每个训练子集和验证子集的权重总量都须为正，单个观测的权重仍可为零。只有各训练集确为相应验证集补集，且各验证集恰好划分全部行时，才复用全数据统计量减去验证集统计量的计算方式。所选模型仍在传给 `fit` 的全部行上重拟合。
  - 实现与回归覆盖：`statgpu/linear_model/cv/_ridge_cv.py`、`dev/tests/test_issue243_ridge_custom_folds.py`，以及 `dev/tests/test_pr168_linear_cycle7.py` 中独立的解析折损失对照。
- **SCAD/MCP 无惩罚 oracle 重拟合（#227）**：
  - 保留负二项原行序的严格收敛回归，并增加统计目标相同、容差不变的行重排成功对照。一个 Python 3.11 / SciPy 1.10 / NumPy 1.24 的 BLAS 环境在子模型得分达到容差前便受损失分辨率限制；测试保留既有的明确拒绝与完整状态清理，并增加确定性的首次子模型警告对照。这项测试兼容修正不改变共享求解器，也不宣称收敛已不受行序影响。
  - 非高斯 oracle 重拟合现在先检查各分布族的预测值、均值和协方差未被裁剪的范围，再返回不确定性；Tweedie 和负二项重拟合还会检查相对一阶得分。这会拒绝被裁剪的 Tweedie/Gamma/逆高斯重拟合、Poisson 均值下限造成的失真、负二项提前停止或模型协方差曲率下限造成的失真，以及 logistic 协方差曲率下限造成的失真。检查只使用正权重行，保留符合条件的精确端点。检查失败会抛出 `RuntimeError` 并清除拟合和报告状态。较小 `tol` 可能有助于未被裁剪、仅提前停止的重拟合，但不能消除裁剪；只需要点估计时可设置 `compute_inference=False`。仅负二项可按得分尺度推导绝对容差并重试一次，最终相对阈值不变，最多执行两次各自受 `max_iter` 限制的子模型拟合；失败仍清除状态。高斯重拟合与共享求解器保持不变。参见[各分布族的范围和失败处理说明](guides/penalized-glm-inference.md#无惩罚重拟合与分布族设置)。
  - 修正保留的物理 GPU RidgeCV 混合精度回归：raw float32 Gram 乘积或惩罚参数无法表示时，要求明确报错，与此前已说明的恢复范围收窄保持一致。保留 float64 和普通尺度的成功对照，并新增原生 Torch CPU 选择器检查；本次修正没有移除或跳过物理 GPU 用例。
  - Gamma/Tweedie 的非稳健 oracle 离散尺度现在按实际分布族方差计算 Pearson 残差，不再给分母施加固定下限。因此，可表示的小响应尺度不会再产生虚假偏窄的区间。既有分析权重和残差自由度约定保留；无法表示的尺度仍会明确报错。
  - 修正活跃集及仅截距 oracle 重拟合中，正标准误极小时的 z 统计量及正态参考 p 值。现在用估计值除以实际正标准误，保持估计值和标准误同步缩放时的检验结果；标准误恰好为零时仍保留原有约定。
  - 非稳健 oracle 推断若估计离散尺度为零，但任一正权重行上的响应与计算得到的拟合均值不同，现在会抛出 `RuntimeError`，零设计行也包含在内。离散尺度为正时，所有活跃参数的模型方差都必须为正；协方差非对角条目仍可为零。计算残差真正为零的拟合仍可具有零估计离散尺度；logistic、Poisson 和负二项重拟合仍使用固定离散尺度 1。这些数值失败检查不构成一般的协方差条件数保证。
  - 稳健推断若把权重直接除以最大权重，可能把正的小权重舍入为零，即使该观测的大残差对协方差有重要贡献。现在统一缩放权重时保留正权重及实质得分贡献。选择共同尺度时，保守界限同时检查下溢与溢出风险。保守缩放区间为空时，若实际聚合乘积为有限值，所需的非零得分平方均值仍处于正规浮点范围，且被忽略微小项的界限低于通常的聚合舍入误差，仍可使用未缩放权重。若工作精度下既无法验证稳定缩放，也无法验证受限的原始权重计算，或稳健得分、曲率或活跃参数协方差非有限，则抛出 `RuntimeError`，不发布无效不确定性；此时仍可能存在有限的精确协方差。尺度不变性仍受加权乘积的可表示性与精度限制，包括次正规数本身的精度限制；真正为零的得分、个别为零的方差及未入选参数的 `NaN` 不确定性仍被允许。若活跃参数的 HC0/HC1 协方差整体为零，但有效行上的得分因子证明得分矩阵非零，现在会抛出 `RuntimeError`。零权重和零设计行不会触发该检查；它不保证检测每个局部下溢的协方差条目，也不改变非稳健协方差行为。子模型重拟合使用的权重不变。
  - 对需要估计未知离散尺度的 `nonrobust` oracle（高斯、两种 Gamma 链接、逆高斯及 Tweedie），若正权重观测数不大于入选系数数（含有效截距），现在抛出 `ValueError`。这避免饱和样本或补入零权重行后报告零标准误、单点区间。残差自由度计数、Poisson/负二项的固定尺度协方差，以及既有 HC0/HC1 行为保留；Gamma/Tweedie 的 Pearson 尺度修正见上文。
  - 原构造过程会遗漏分布族参数，并可能保留默认正则化或选择非预期设备。非高斯 oracle 现在明确使用 `C=0`，保留已拟合的 Gamma 链接、负二项离散参数及 Tweedie 幂参数，传递收敛控制，并显式在 CPU 上重拟合：logistic/Poisson 使用 IRLS，其他受支持的非高斯分布族使用 Newton。
  - 预测与评分也使用拟合时的负二项/Tweedie 分布族参数，正确处理具体估计器的默认参数及显式覆盖；Gamma 预测遵循拟合时的逆链接或 log 链接。预测仍使用惩罚系数。
  - adaptive-L1 初始化使用相同的负二项/Tweedie 分布族参数，包括具体估计器的默认参数与显式覆盖。负二项和逆高斯 oracle 的 HC0/HC1 协方差现在用观测 Hessian 构造三明治协方差的两侧矩阵；非稳健协方差仍采用 Fisher 信息，HC1 仍按原始观测数进行有限样本校正，其中包括零权重行。Oracle 的 HC0/HC1 协方差现在会在计算得分外积前统一缩放权重，避免权重的共同尺度导致下溢或溢出；重拟合权重与非稳健协方差的既有约定不变。
  - Oracle 计算过程中发生错误时，通用估计器和具体估计器都会清除拟合、预测和推断报告状态，包括此前成功拟合留下的状态；重新使用前须成功拟合。子模型未收敛时，在返回 oracle 系数表之前抛出 `RuntimeError`。对于有效响应全零的 Poisson/负二项/Tweedie，以及有效响应只有一个类别的 logistic，会先检查无惩罚目标是否存在有限解；自由截距可持续改善目标时，抛出 `ValueError`，不再发布虚假的有限不确定性。零权重行不会改变这一边界。无截距时检查入选设计中的方向，保留具有有限目标的合法正负混合设计；边界设计若在数值上无法可靠判定，则明确抛出 `RuntimeError`。这不是一般的分离检测器。没有入选斜率但包含截距时，仅在目标存在有限解时重新拟合仅含截距的模型；无截距的空逆链接 Gamma 模型则被拒绝。预测仍使用惩罚系数，未选中系数的不确定性仍为 `NaN`，普通正态参考区间不校正选择不确定性。Oracle 推断仍仅支持 CPU。
  - 实现与回归覆盖：`statgpu/linear_model/penalized/_inference_mixin.py`、具体负二项/Tweedie 估计器的分布族参数解析，以及 `dev/tests/test_issue227_oracle_refit_contract.py`。
- **整群重采样（#199）**：
  - 调整嵌套标签回归对照以兼容不同 NumPy 转换行为：独立计入估计器原有有限值检查，并严格要求仅增加一次快照转换。缺失标签拒绝、零回调、dtype、分组和样本断言以及生产分组逻辑保持不变；故障注入对照可识别缺失或多余的快照转换。
  - 结构化掩码数组的群组标签现在按字段归约掩码，支持嵌套记录和子数组字段。全为假的掩码保留原先有效的分组；字段实际被掩蔽时，抛出既有的缺失标签 `ValueError`，不再触发结构化类型的归约错误。
  - 嵌套数组转换器按自然推断类型返回的非标量结果，现在会在与字符串标签合并前接受缺失检查，避免数值 NaN 被字符串化；同一转换结果复用于分组。主机数组中实际被掩蔽的标签也会报错；全为假的掩码及字面缺失样式字符串仍有效。若自定义转换本身已经返回字符串，仍无法恢复丢失的缺失状态。
  - 保留序列包装对象在最外层已有的后端传输转换，群组分组继续使用 `.get()` 或 `.cpu()` 转换后的标签。预校验不会改用序列迭代替代这些钩子，也不会额外执行它们；嵌套普通序列仍检查缺失值。
  - 标签验证不再对嵌套的自定义数组对象额外执行 object 类型转换，而是保留其正常转换行为；数组协议显式声明为 `None` 时，原有转换错误也不会被绕过。转换结果为 object 数组时仍检查缺失标签；若自定义转换已经把缺失值替换成字符串，仍适用文档中说明的限制。
  - 混合字符串序列此前可能在检查前把真正的 NaN 转成字符串 `"nan"`。模块函数与估计器辅助方法现在都会在转换前检查主机标签，覆盖 object 数组，以及 list、tuple、deque、UserList、普通长度与索引协议等受支持主机序列中的 `None`、NaN、NaT 及 pandas 缺失标量。缺失标签检查和分组共用保留的同一组主机序列标签，消除校验新增的原始序列读取，同时保留估计器原有的有限值检查。预校验不消费迭代器，也不额外调用自定义数组或设备转换协议，包括通过动态属性提供的数组转换。真正的字符串标签 `"nan"`、`"None"`、`"NaT"` 仍然有效；原生数值后端的分组方式和带种子的重采样不变。回归测试覆盖两个入口、保留类型的 object 数组、合法字符串对照、已接受的序列形式、调用方 dtype/形状保留，以及不变的自定义转换调用次数。
  - 原先按累计行数停止抽样的规则可能截断群组。现在每次按原始群组数有放回地抽样，并拼接完整群组，保持不同数组之间的行对应关系。不等大小群组产生不同的重采样行数，回调逐次接收样本，不作批量试调用；等大小群组的稠密抽样与固定种子行为保持不变。
  - 不等大小群组使用 `force_vectorized=True` 会报错，唯一例外是为恰好一个一维数组设置 `statistic_hint="mean"`。缺失群组标签抛出 `ValueError`；分层或组内置换标签的独立限制仍然保留。元数据增加 `resampling_unit`、`n_clusters`、`clusters_per_resample` 和 `variable_sample_size`。
  - 实现与回归覆盖：`statgpu/inference/_resampling.py`、`dev/tests/test_issue199_whole_cluster_bootstrap.py`，以及可执行的中英文重采样示例。
- **LinearRegression 评分形状（#178）**：
  - 原先单列响应与一维预测相减时，可能静默广播成残差矩阵。现在单目标评分等价地接受 `(n,)` 或 `(n,1)`；真正的多目标响应必须与预测形状完全一致。行数或目标维度不匹配时，在所选后端计算残差之前抛出 `ValueError`。
  - 公式预测仍会删除存在缺失值的查询行；评分遇到缩短的预测与完整响应时会报错，不再广播。常数目标仍返回 0，多目标评分仍为各目标 R² 的均值。
  - 实现与回归覆盖：`statgpu/linear_model/wrappers/_linear.py`、`dev/tests/test_issue178_linear_score_shape.py`、单列响应学习示例，以及 `dev/tests/test_pr168_linear_cycle8.py` 中的公式评分回归。

#### 验证（2026-10-17）

- 补充针对性的数值与文档回归，不放宽独立对照或数值容差。仅把已修复的 oracle、RidgeCV 训练子集及公式评分案例从预期失败改为正常测试；其他预测与数值限制仍单独保留。物理 CUDA 检查仅在硬件可用时运行；CPU 执行不代表物理 GPU 验证或性能结果。

### 多重检验的 Torch 设备处理（PR #171）

#### 修复（2026-10-15）

- **多重检验保留 Torch 输入设备**：
  - 模块级 `adjust_pvalues`、`multipletests` 和 `combine_pvalues` 此前选择 Torch 时，没有将数组分配绑定到输入张量的具体设备。因此 CUDA 可用时，转换可能移动 CPU 输入，也可能将秩、校正因子或权重分配到其他 CUDA 设备。
  - 现在无论使用 `backend="auto"` 还是显式 `"torch"`，Torch `pvalues` 张量都会保留其 CPU 或确切的 CUDA 设备编号。Float32 输入在同一设备上升为 float64；临时数组、传入或默认权重以及输出均位于该设备。权重不参与后端或设备选择。
  - 非 Torch P 值显式指定 `backend="torch"` 时，CUDA 可用则使用当前具体 CUDA 设备，否则使用 CPU。MPS 等不支持的 Torch P 值设备会引发 `ValueError`，因为这些函数仅支持 CPU 与 CUDA 上的 float64 计算，不会静默改用其他设备。
  - 模块级函数仍不受 `set_device` 影响。估计器辅助方法选择 Torch 时，仍保留独立的 CUDA 要求和输入转换；共享后端工厂的行为不变。NaN `alpha`、权重总和溢出及极端概率尾部的已有数值警告仍然适用。
  - 同步更新中英文多重检验 API、设备指南与估计器辅助方法参考，并在 `dev/tests/test_multiple_testing_device_ownership.py` 中新增设备归属回归测试，其中物理 CUDA 检查仅在相应硬件可用时运行。本次本地更新不声称已完成物理 GPU 验证。

### 入门指南与 API 参考（PR #168）

#### 改进（2026-10-10）

- 在每次公开调用内复用协方差有限值检查已经规范化的工作输入。受支持的 Torch 输入走 NumPy 路径时，`fit`、`score`、`predict` 和 `mahalanobis` 只执行一次 Torch 到 NumPy 的规范化，包括预测中的嵌套调用；原生 Torch FP8 调用也复用在原设备上完成的升精度结果。所有传入的数值参数仍接受检查，包括统计计算中忽略的 `y`，估计器不会缓存输入。数值结果、后端选择、输入归属及异常行为保持不变。`dev/tests/test_covariance_normalization_reuse.py` 检查转换次数与这些保留契约；转换次数不代表实测运行时间加速。

#### 修复（2026-10-10）

- **协方差 Torch 转换与 FP8 验证**：全部七个协方差估计器在公共有限值检查之前规范化受支持的实数 Torch 输入。NumPy 路径先传至 CPU，再转为 `float64`，保留 `bfloat16`、`float8_e5m2`、惰性负值视图及 MPS 到 CPU 的转换。原生 Torch 的 `float8_e5m2` 检查先在原设备上转换临时数组，不再依赖原始 FP8 类型的有限值检查算子。此前 Torch 2.5.1 CUDA 会在进入协方差转换函数之前拒绝这些 FP8 输入。NaN、正无穷与负无穷仍会报错；拟合后端、原始输入与底层存储、梯度、显式设备不可用时报错以及已有 CuPy/pandas 处理保持不变。测试位于 `dev/tests/test_covariance_torch_inputs.py` 与 `dev/tests/test_covariance_fp8_prevalidation.py`，包含模拟缺失算子的执行顺序回归。CPU 结果不能代替真实 CUDA 或 MPS 验收，仍须在相应硬件上重跑。其他 float8 格式、复数、稀疏及量化输入的支持范围不变。

- 恢复全部七个协方差估计器对数值型 pandas `DataFrame` 的拟合、评分、预测与 Mahalanobis 距离支持，以及单特征 `Series` 的拟合支持。此前设备转换误将 pandas 需要键参数的 `get` 方法当作 CuPy 传输方法。现在先规范化普通类数组输入，真正的 CuPy 数组和 Torch 张量仍按已有设备策略处理。回归测试对照 NumPy 结果、检查非法输入，并执行真实 Torch CPU 查询；物理 CUDA 测试仅在硬件可用时运行。

#### 修复（2026-10-09）

- `f_oneway`、`cohens_f` 和 `f_twoway` 现在拒绝观测值中的 NaN、正无穷和负无穷，与其他维护中的 ANOVA 方法保持一致。有限常数组和残差自由度不足时的已有结果定义不变。
- 协方差估计器现在遵守显式指定的计算设备，即使输入为 Torch 张量或 CuPy 数组：`cpu` 使用 NumPy，`cuda` 要求 CuPy CUDA，`torch` 要求 Torch CUDA；显式请求的后端不可用时会报错。只有估计器与全局设备策略均为 `auto` 时，才保留输入所属后端的既有行为。评分、Mahalanobis 距离及交叉验证使用拟合时的计算后端。
- 修复中英文 GLM 公共参数表的 `C` 行，完整保留渲染后的惩罚归一化公式、`C=0` 行为与求解器例外说明。

#### 修复（2026-10-06）

- 修复原生 Torch 自动生成 model-X knockoff 的设备与种子处理：局部随机数生成器和随机矩阵现在使用输入张量的确切设备；即使 CUDA 可用，CPU 输入也在 CPU 上构造，且 `random_state` 生效、构造不消耗 Torch 全局随机数状态。沿用 `None` 转为零种子的规则时，多次 Torch 构造会重复同一随机矩阵；如需不同且可复现的逐次抽样，应显式提供整数种子。这不保证跨后端或跨设备逐位一致，也不修复独立的 Lasso 缓存和设备分派限制。

#### 改进（2026-10-09）

- 将模型与面板方法的验证记录、基准看板维护步骤移到开发者文档，用户页保留统计假设、数值警告与对照设置；明确 RidgeCV 自定义训练行问题属于已知实现缺陷，并保留外部交叉验证的绕行方法。把 PR78 历史看板计划移到开发者计划目录，用生成数据链接代替容易过时的当前数量，并按实际工作负载与已有来源限定历史速度结果。

- 将模型教程整理为配有说明的短步骤：导入与输入含义、数据准备、拟合、预测及结果解释。后续的推断、调参和 GPU 示例在适用时明确复用前文数据，避免重复整套数据生成或 CPU 拟合。中英文保持相同的学习顺序，并保留各模型的公式、假设与使用限制。
- 为原先依赖未定义变量的示例补齐数据准备。示例测试按照读者实际阅读的顺序执行代码块和明确声明的前置步骤，同时保留预测、推断和交叉验证结果的检查。

#### 改进（2026-10-06）

- UMAP 缺失邻边的预期失败只识别特定有限图；异常图会正常报错，真正修复 UMAP 后会触发 strict XPASS。

- 说明普通 LinearRegression/GLM 公式预测静默删除缺失行的限制，以及后续广播可能产生错误但有限的 LinearRegression 评分。新增完整查询与保留索引的检查步骤，并区分已经拒绝预测缺失行的惩罚模型。PR #249 已修复错误的 LinearRegression 评分广播；普通公式预测仍可能删除缺失查询行，预测前仍应检查查询数据完整性。

- 将 Poisson 与通用交叉验证示例改为可独立运行、固定随机种子的 CPU 流程，包含独立测试集、保持时间顺序且留有间隔的数据折，以及期望计数、倍数效应和不确定性的解释。补全类型化 GLM 的构造参数参考和 CV 运行时帮助，明确模型族专属参数与实际公开方法。
- 修正 UMAP 参考交叉熵的求和范围，排除自身配对；补全 TSNE 条件概率、联合概率和可实现困惑度的定义，并说明 UMAP 吸引曲线拟合中的主机计算。即使 TSNE 返回有限的嵌入，也不代表它实现了一个本来不可能达到的困惑度。
- 从通用参考直接链接完整的核方法和样条 API。通过独立数值特征与变异检查收紧已知缺陷测试，避免把无关 NaN、有限但错误的结果或运行时异常误标为预期失败。

- 补全 Ridge、SCAD 和 MCP 的学习流程，提供可独立运行的 CPU 示例，明确不受惩罚的截距，并完善模型方法、公式、权重参考与运行时帮助。说明原生 Torch model-X 的随机种子语义、采样器选项的实际分发范围，以及 KernelPCA 重新拟合失败后混合新旧状态的风险；其中自定义 RidgeCV 训练子集缺陷已由 PR #249 在本版本修复；RidgeCV 仍受上文所述的数值范围限制，KernelPCA 重新拟合失败后的状态问题仍然存在。
- 扩展线性模型、推断、特征选择、生存分析、平滑及十二类无监督估计器的中英文解释、数学定义与可运行示例。[完整 API 索引](README.md#完整-api-参考)汇集构造参数、方法与拟合结果参考。
- 澄清推断目标、选择和调参后的条件推断、权重、重采样假设与可用的结果报告方式；同步说明 Ridge、SCAD、MCP 与 Poisson 的实际默认行为和支持的接口，并区分 Cox 模型族中求和与按行数平均的部分似然。
- 说明失败重拟合、非有限输出、数据几何与设备放置的当前限制及检查方法。[坐标缩放说明](models/nonparametric/nonparametric.md)和 [NMF 指南](unsupervised/nmf.md)解释了极小计量单位可能改变当前结果的问题，给出一致缩放、复用变换及还原预测或密度的做法。[Ridge 指南](models/regression/ridge.md)另提供训练数据确定的中心化步骤，以减轻大坐标偏移造成的数值问题。
- 澄清采用通用设备分派的估计器在 `device="auto"` 下会继承全局设备策略，并补齐公开构造函数帮助。保留的基准报告按其原始源码与硬件标为历史测量。
- 除文档、帮助文字与回归测试外，本 PR 还修复了原生 Torch model-X 的随机矩阵设备选择和局部随机数生成器使用、ANOVA 对非有限输入的拒绝行为，以及协方差模型的设备选择。其他单独跟踪的数值与状态限制仍未修复，UMAP 生产代码也没有改动。CPU 示例与分派检查不能作为物理 GPU 执行或性能证据。

### 逻辑回归分类器求解器选择（PR #172，2026-10-03）

#### 新增

- `LogisticRegression` 与 `LogisticRegressionCV` 现在接受 `solver="auto"`、`"irls"` 和 `"lbfgs"`；`auto` 保持原有 IRLS 行为。这是即将发布的 0.3.0 新增接口，0.2.5 尚不提供。
- L-BFGS 保留带权 Bernoulli 损失之和与 `||coef||**2/(2*C)` 的目标函数，截距不受惩罚，直接拟合的 `C=0` 仍表示无正则化。实现时将整个目标函数除以样本数或权重总和，因此不会改变 `C` 的含义。
- NumPy、CuPy、Torch 均在所选后端拟合。GPU 交叉验证的 `auto`/`irls` 保留批处理 IRLS；显式 L-BFGS 在所选后端以 float64 逐项拟合候选，并在最终重拟合中使用同一求解器。`gpu_cv_mixed_precision` 仅影响批处理 IRLS。
- 新增已拟合属性 `solver_`、汇总中的请求/实际求解器、事务式校验与分求解器的收敛状态。原有协方差推断保持不变，不校正收缩偏差或交叉验证选参的不确定性。
- 实现文件：`statgpu/linear_model/wrappers/_logistic.py`、新增 `_logistic_solver.py`、`statgpu/linear_model/cv/_logistic_cv.py`，以及 `statgpu/solvers/_lbfgs.py` 中按需启用的梯度收敛判断。回归测试：`dev/tests/test_logistic_solver_contract.py`。这些测试定义与 CPU/Torch-CPU 运行本身不能证明真实 CUDA 执行；最终发布验证范围见本版本开头的说明。

```python
from statgpu.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], solver="lbfgs", device="cpu")
model.fit(X, y)
print(model.C_, model.solver_)
```

### Quantile 求解器与推断更新（PR #166）

#### 修复

- **公开 Quantile 求解器边界**：直接拟合、交叉验证和公开底层接口现在采用一致的响应/设计矩阵形状、停止条件、延续路径、样本权重和求解器兼容性规则。普通 FISTA，以及底层普通 L-BFGS 在未传权重或均匀权重下的历史兼容行为继续保留；FISTA-BB、共享 ADMM、Newton、Proximal Newton 和 L-BFGS-B 会在数值迭代前明确拒绝 Quantile。历史 `quantile_cd_solver` 会忽略 `sample_weight`，且不能可靠表示不受惩罚的截距，因此仅保留名称的导入兼容性，调用时直接报错；标量 SCAD/MCP 使用 Proximal IRLS-CD。
- **直接拟合与交叉验证的输入对齐**：Quantile 的响应变量校验保持在当前数值后端上执行，并统一覆盖通用/类型化直接拟合与 `PenalizedGLM_CV`；公开底层 Quantile 路径也会在开始数值计算前拒绝含 NaN/Inf 的 `X/y`。非法的自定义交叉验证折和权重总质量为 0 的折，会在自动构造 `alpha` 网格之前报错；延续路径尺度、自适应/分组惩罚的归属关系与实际拟合目标保持一致。预测和评分会拒绝非法形状，不再允许 NumPy 广播产生外形正常但语义错误的结果。
- **独立 `QuantileRegression` 的推断正确性**：进行残差重抽样前，先按目标分位数的经验分位点对拟合残差做中心化，使重抽样误差分布的经验 τ 分位数为 0；随后每个重拟合样本都使用后端原生的批量 Quantile IRLS/MM，求解调用者请求的 Quantile 目标。多特征、非中位数的 bootstrap 重拟合目标已与独立 Quantile 线性规划解对齐，替代此前可能在 pinball 最优点之上提前停止的批量次梯度/FISTA 路径。真正非均匀解析权重下的独立模型推断会明确报错，因为当前尚未实现相应的加权核方法/bootstrap 推断；均匀权重继续对应等价的未加权推断目标。这里的 bootstrap 是基于可交换中心化残差的 i.i.d. 残差 bootstrap，不将其描述为对一般异方差稳健的 wild/multiplier bootstrap。
- **独立模型的推断生命周期与参数校验**：bootstrap 至少需要 2 次重抽样；非法的推断方法、核函数、带宽、停止条件、布尔参数或分位数设置会在进入后端数值计算前报错。Hall-Sheather、Bofinger 和 Chamberlain 规则在 `q ± h` 离开 `(0,1)`、最终带宽不是有限正数，或零点残差密度估计不是有限正数时，都会在发布协方差前报错。`score()` 按文档返回负 pinball loss，并支持可选解析权重。失败拟合不会保留半成品推断结果；`gpu_memory_cleanup=True` 也会正确识别执行记录中的 CuPy 后端，并覆盖成功与失败路径。
- **运行时帮助与 API 文档**：Quantile 次梯度的运行时帮助与实际导数保持一致；通用带惩罚模型的运行时帮助明确列出公开的 Quantile 损失接口。求解器封装在重新加载和不同导入顺序下保持安全，同时保留历史公开模块的导入身份。
- **光滑 Quantile FISTA 的后端一致性**：NumPy、CuPy 与 Torch 的 L2/无惩罚 Quantile 回溯路径采用与 CPU 一致的逐迭代“已接受点 + 目标稳定性”收敛语义。GPU 路径把已接受候选点的目标值、系数变化和 L2 跟踪量合并到该次 Armijo 检查原本就需要的同步中，避免额外的收敛检查或惩罚跟踪主机同步。
- **异步稀疏 Quantile FISTA**：异步非光滑 Quantile 交叉验证继续采用延迟检查，将 Nesterov 动量上限设为 `0.5`；只有在预热阶段之后连续两次目标检查都没有达到容差尺度的改善时，才把固定步长减半并重启动量。自适应控制复用已有的延迟同步，并把 L1 惩罚跟踪合并到同一次检查中，因此不会增加主机同步。GPU 上的 Quantile 稀疏交叉验证会对每个折中的 L1/ElasticNet/Adaptive-L1 候选模型以及最终全数据重拟合统一使用维护中的异步/自适应 FISTA；此前这些内部 `model.fit()` 调用可能回落到通用 pinball 回溯，并在非光滑折点处触发 Armijo 失败。普通直接 Quantile FISTA 的公开行为不变。
- **独立线性规划验证基准**：物理验收中的异步加权 L1 用例改用 SciPy/HiGHS 的线性规划最优值作为独立参考。在这组确定性数据上，CPU 通用 FISTA 的诊断目标值比真实凸最优值高约 `8.939e-4`，因此不再把“接近 CPU FISTA”作为正确性标准；原有 `5e-4` 阈值没有放宽，而是直接针对线性规划最优目标。schema v23 还把 L1 交叉验证的逐折/逐 `alpha` 验证得分、最终选择的 `alpha` 和最终全数据惩罚目标全部改为使用独立 HiGHS 线性规划参考。固定用例上 CPU 交叉验证得分相对线性规划的最大偏差达到 `3.1335e-3`，因此 CPU 结果只保留作诊断参考；GPU 得分和最终重拟合目标的容差均未放宽。
- **底层标量 Quantile SCAD/MCP 的精度修正（Issue #169）**：公开底层 `fista_lla_path` 路径的标量 Quantile SCAD/MCP 现在委托给与直接拟合、交叉验证相同的专用 Proximal IRLS-LLA 引擎。此前的融合式固定步 FISTA 内层求解会在非光滑 pinball 代理目标的凸加权 L1 最优点之上停滞，却仍然报告收敛；分组惩罚以及带热启动或返回路径的底层调用继续使用融合式 FISTA-LLA 引擎。标量物理门禁 schema v3 用非平凡收敛用例取代了零对零的步长探针，并以独立 SciPy/HiGHS 线性规划不动点作为验收参考（实测差距约 `4.6e-11`，容差 `1e-8`），同时保留专门的预算耗尽告警用例。
- **底层 Quantile 的校验与告警边界**：自适应/分组自适应权重以及 `AdaptiveL1Penalty` 的控制参数现在会拒绝复数输入，而不是静默截断；直接调用 `QuantileLoss.irls()` 会拒绝零特征设计；单步标记延续路径现在求解请求的目标 `alpha`；零预算的融合 LLA 回退不再抛出 `UnboundLocalError`；重新发出的 Quantile 收敛告警会指向调用者。Torch 核方法的 sandwich 推断让协方差权重跟随工作张量 dtype，不再受默认 float32 影响，跨后端带宽一致性恢复到紧容差。

#### 验证

- 新增有针对性的回归测试，覆盖 Python 版本下的测试收集安全性、公开求解器别名与重新加载幂等性、底层形状/路径/权重拒绝、直接拟合与交叉验证的响应校验、预测/评分的广播保护、Torch 响应后端保持、独立模型的失败事务/参数控制/内存清理、核方法带宽定义域失败，以及一个 `tau=0.2` 的集成批量 bootstrap 检查，用于区分调用者请求的分位数与错误的互补分位数方向。
- 扩展 CUDA 回归覆盖，加入非均匀加权 L1 FISTA、独立模型 bootstrap、自动 Group SCAD/MCP 直接拟合/交叉验证、加权标量底层 Quantile LLA 对独立线性规划不动点的精度与预算耗尽诊断、平坦 IRLS 行为，以及 Group LLA 导数在 CuPy/Torch 上的设备驻留检查。

### Quantile 求解器来源对齐（PR #164 / Issue #163）

#### 修复

- Penalized Quantile 的求解器身份现在与实际执行算法一致。L2/无惩罚的 `solver="auto"` 解析为普通 Quantile IRLS，L1/ElasticNet 继续使用维护中的 FISTA 系列，SCAD/MCP 解析为专用 Proximal IRLS-CD。`proximal_irls_cd` 仍然只是内部记录解析结果与实际执行的标签，不成为公开可显式请求的 `solver=` 关键字。
- 不兼容的显式 Quantile 求解器请求现在会在进入数值调度或 CV 网格计算之前直接报错，不再静默执行另一种算法。直接拟合、CV 候选与折、以及选定的全数据重拟合的请求/解析/实际执行求解器身份因而保持一致。
- 非中位数 Quantile CV 评分会保留调用者配置的分位数水平，包括使用解析验证权重时；类型化 `PenalizedQuantileRegression(quantile=q)` 也会在克隆安全的构造、自适应 L1 初始化与 `score()` 中保留同一个 `q`。
- SCAD/MCP Quantile 的截距现在作为 pinball 目标中不受惩罚的坐标直接优化。若 LLA 替代目标完全变平，则通过维护中的完整 `QuantileLoss.irls()` 内核闭合，而不是继续使用对角近似。
- Torch 上的 Quantile 执行现在会让 L2 惩罚对角项、IRLS 热启动、Proximal IRLS-CD 的 epsilon/threshold/tolerance 标量以及回退权重始终跟随当前张量的 dtype/device；热启动使用 Torch 原生克隆，而不是 NumPy/CuPy 的 `.copy()` 路径。

#### 验证

- 精确干净的数值源码 `2de971402004efc703dc98f510942db3980988e4` 已在 Tesla P100-SXM2-16GB 上通过冻结的 schema-v1 物理验收，环境为 CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2、Python 3.9.16。CuPy/Torch CUDA 的直接拟合/交叉验证共 **12/12** 用例全部通过，并记录具体 `cuda:0` 设备来源与预期的 IRLS/FISTA/`proximal_irls_cd` 求解器身份。
- 冻结容差没有放宽：L2 系数/截距最大误差为 `6.938893903907228e-15`（阈值 `2e-5`），L2 CV 得分最大误差为 `7.965850201685498e-15`（阈值 `2e-5`），SCAD 惩罚目标最大误差为 `1.4085439563257807e-06`（阈值 `2e-4`）。
- 规范的确切源码 artifact 为 `dev/reviews/pr164_quantile_solver_provenance_gpu.json`。commit `230eaebbd4dfaf090e84011fe0eb190339389411` 直接位于已验证的数值源码之上，并且只新增该 artifact。这份记录仅说明 PR #164 在所标注源码上的验收；后续版本仍须单独验证，不能据此认定变更后的数值代码或验证程序已通过。

### Quantile IRLS 惩罚契约修复（PR #162 / Issue #161）

#### 修复

- 底层 `QuantileLoss.irls()` 现在只接受无惩罚或 L2。ElasticNet、L1、SCAD/MCP、group/adaptive penalty 以及未知 penalty 对象会在数值迭代前直接报错，不再允许 IRLS 只处理声明目标中的光滑 L2 部分而忽略非光滑项。
- 公开估计器契约保持不变：显式 Quantile `solver="irls"` 仍然只属于 L2/无惩罚维护路径；非光滑惩罚继续通过 FISTA 系列或 Quantile 专用的非凸求解算法处理。
- IRLS 文档字符串与可执行的低层契约现在一致；不支持的直接底层调用会得到明确错误，而不是返回看似合理但只优化了部分惩罚的结果。

#### 验证

- 增加针对性回归测试，覆盖直接拟合下 ElasticNet/L1/SCAD/未知 penalty 在线性求解前被拒绝、无惩罚/L2 路径保留、解析权重整体正比例缩放不变性、既有估计器级失败边界，以及 Torch CPU 上的 L2 后端保持。
- 本修复删除的是不受支持的路径，并未改变维护中的 None/L2 IRLS 数值算法，因此本身不新增物理 CUDA 验收要求。

### GLM 显式 Newton/L-BFGS 的解析权重支持（PR #151 / Issue #150）

#### 变更

- 普通 `GeneralizedLinearModel` 在受支持的 GLM 分布族和链接函数上，显式 `solver="newton"` 与 `solver="lbfgs"` 现在可以接受真正的非均匀解析 `sample_weight`，不会因为权重存在而静默替换成 IRLS/FISTA，也不会改变显式的 NumPy/CuPy/Torch 执行请求。
- 维护中的 GLM 光滑求解器在完整求解过程中统一使用同一个归一化解析权重目标。review/fix 使近似均匀权重分类在整体正比例缩放与排列下保持不变，并在执行 dtype 转换前先归一化，整数设计仍保留分数权重，float32 原始求和溢出不会在稳定归一化之前误拒绝，并保留普通拟合后诊断/推断实际求解时使用的求解器准备好的权重身份。在 Newton/L-BFGS 的 M-estimation 路径上，同一权重身份会在协方差计算前等价缩放为均值为 1 的表示，因此整体正比例缩放解析权重不会在数值求解容差之外改变系数及 nonrobust/HC0/HC1 推断，同时不会静默重定义无关的 IRLS/FISTA 加权语义。`ParameterInferenceResult` 在构造方遗漏标准字段时也会公开实际的 M-estimation `cov_type`。L-BFGS 保留可见的线搜索失败，只在“所要求的目标下降”和“实际参数位移”都已低于维护中的数值分辨率时允许受限的浮点 Armijo 接受。
- `GammaRegression(link="inverse_power")` 的显式 Newton/L-BFGS 现在使用由 Gamma 损失函数拥有的光滑训练域契约，而不再对无截距情况作一刀切拒绝。被定义域截断的极小 L-BFGS 步长本身不构成收敛；若定义域内的拟牛顿 Armijo 搜索耗尽，会用重新计算定义域上界的最速下降方向再尝试一次，若梯度尚未收敛且恢复仍失败则明确报错。可行的无截距设计可以正常拟合；零权重观测不约束域可行性；真正不可行、数值上无法认证或被定义域边界钉住的拟合会显式失败。
- 同一 inverse-Gamma 域契约已经闭合到带惩罚 L2 Newton/L-BFGS 和 `PenalizedGLM_CV`：框架热启动只有在域内时才复用；CV 评分使用声明的 `inverse_power` 目标，不再错误使用 log 链接评估器；最终全数据重拟合保留链接。`PenalizedGammaRegression` 同时保持历史 `loss_kwargs["link"]` 优先级，并通过 sklearn clone/get_params 契约。
- 既有 `solver="auto"`、IRLS/FISTA、显式 smooth-solver `C`、Ordered GLM、standalone LogisticRegression、非 inverse Gamma 链接，以及成功拟合后的求解器/后端/设备来源语义保持不变，除非上文明确说明属于本次修复范围。
- 普通显式 Newton/L-BFGS 重拟合现在对校验、优化、推断与来源信息发布实施原子事务。失败的重拟合会完整恢复上一次成功的估计器状态，而不会暴露“旧系数 + 新尝试元数据”的混合对象；第一次拟合失败仍保持未拟合状态。此前成功执行的求解器/后端/设备来源会保留，但失败的尝试不会被发布成新的执行证据。

#### 验证

- 历史 hosted/physical evidence 继续只对各自精确源码有效。数值/validator 源码 `9eb39cee2c0e691160f528ab687b7379d26f3e42` 曾通过全部 7 个 hosted PR workflow 和 Tesla P100 schema-v5 验收；该源码的完整 CPU 测试套件为 **3395 passed / 831 skipped / 0 failed**。后续新一轮 review 已修改生产推断与结果来源信息，因此这些 hosted 结果和 schema v5 都不能作为后续源码的验收。
- Tesla P100 schema-v3 artifact 对应 `c6781cb6a2e1fe500f325e832d23cdc80a99b564`，schema-v4 inverse-Gamma artifact 对应 `9ef5b34ffc8abf133bc6262e98a855d5efe37d3c`，schema-v5 artifact 对应 `9eb39cee2c0e691160f528ab687b7379d26f3e42`；三者现在都只作为各自源码的不可变**历史确切源码证据**保留。对应文件为 `dev/reviews/pr151_glm_weighted_explicit_solvers_gpu.json`、`dev/reviews/pr151_inverse_gamma_domain_gpu_v4.json` 与 `dev/reviews/pr151_final_gpu_v5.json`。
- schema v6 是不可变的**历史失败 validator**，不再是当前验收门禁。其 float32 溢出推断用例除了预期的 float32 解析权重外，还意外把 `X/y` 一起转成 float32，从而额外引入了无关的 float32 设计 CuPy L-BFGS 跨后端一致性要求。这个独立精度问题由 Issue #160 跟踪；PR151 不会为了让该错误用例通过而放宽冻结容差或扩大通用 L-BFGS 的修改范围。
- 修正后的 schema v7 记录了 PR #151 当时通过的物理 CUDA 验收。精确干净的数值/validator 源码 `0302262ef8242b31f2f17b7835b6c08aeda89904` 已在 Tesla P100-SXM2-16GB（CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2、Python 3.9.16）上通过完整 v3→v4→v5→v7 链，记录 `status: success` 与 `source_clean: true`。当前规范 artifact 为 `dev/reviews/pr151_final_gpu_v7.json`；较早通过的 `e1cbf3756d269a6073461de8331da11c6235fb4e` schema-v7 artifact 作为历史确切源码证据保留在 `dev/reviews/pr151_final_gpu_v7_e1cbf375.json`。与旧运行的数值结果保持一致，最大 review 收尾误差仍为 `9.719913152128612e-09`，位置仍是同一个 penalized/Newton/Torch p 值比较。commit `29b8e8056fb9bdfb34e2b76982d433c0be3d844f` 直接位于已验证数值源码之上并记录最终证据；当时随后的文档更新没有改变已验证实现。PR #151 已合并；PR #165 随后说明了 Issue #160 调查的 float32 精度行为，没有替换 L-BFGS 生产算法。这些历史产物不构成最终 0.3.0 源码的验收。

### 后端原生 Gaussian residual bootstrap（PR #147 / Issue #145）

#### 变更

- 将既有 unweighted Gaussian `residual_bootstrap` 从仅 NumPy 执行扩展到拟合记录在案的 NumPy/CuPy/Torch 后端与具体设备；三个后端共享同一个确定性的后端无关残差索引调度，后端/设备来源一旦漂移即直接报错，不回退 CPU。
- 子样本重拟合保留已拟合的惩罚家族、调参、截距、求解器/停止条件、Lipschitz 与 SCAD/MCP LLA 控制，并保持子样本推断关闭。`PenalizedGLM_CV` 仍只在选定的全数据最终重拟合上执行一次 bootstrap，并明确报告对 CV 选中惩罚的条件化、未校正选择不确定性。
- 加权、robust/HC、HAC/block、非 Gaussian 与 Cox bootstrap 语义仍不支持并直接报错。大规模设计/响应/残差/bootstrap 响应数组与子样本优化保持在记录的数值后端/设备上；最终报告仍采用 NumPy 边界，仅允许小型控制/参数快照跨越该边界。

#### 验证

- 增加确定性的 hosted 覆盖，覆盖 NumPy 路径保留、L1/ElasticNet/SCAD/MCP、formula 与公开包装器消费方、精确设备子样本上下文、失败事务、安装器幂等性、CV 仅最终重拟合语义与不支持的行。
- `dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py` schema v1 是冻结的确切源码 CuPy/Torch CUDA 验收门禁。PR #147 已于 2026-09-12 合并。这里说明的是该功能的验证程序，不表示最终 0.3.0 发布源码已经完成物理 GPU 验证。

### Penalized GLM 推断契约修复（PR #142）

#### 变更

- 通用与类型化 penalized-GLM 估计器统一以 `inference_method="auto"` 作为公开对账边界；专用稀疏 Gaussian 包装器保留既有显式默认。成功拟合会区分请求/解析/报告的方法，并记录推断目标与调参/选择条件。
- 受支持的光滑非 Gaussian L2 / 无惩罚推断解析为固定惩罚的 `m_estimation`，当前支持 nonrobust/HC0/HC1 协方差；非 Gaussian L1/ElasticNet 系数推断改为直接报错，不再发布历史上只含 L2 curvature 的全向量部分 sandwich。
- 残差 `bootstrap` 限定为 `cov_type="nonrobust"` 的无权重稀疏 Gaussian 残差自助法。PR #147 已将其扩展到拟合记录的 NumPy、CuPy 或 Torch 后端；重拟合保留原惩罚、调参及截距设置，至少需要 2 次抽样，不会静默改用 CPU。带权、robust/HC/HAC、非 Gaussian、Cox 及 Gaussian L2 残差自助法仍不支持。
- 非 Gaussian M-estimation 以拟合记录在案的 NumPy/CuPy/Torch 后端与具体设备为准，并覆盖 Torch↔CuPy 异构输入容器对齐。维护中的 Newton 求解器现在支持真正的非均匀解析权重，并让目标值、梯度、Hessian 与 Armijo 试探使用同一个归一化加权目标，因此加权光滑 L2/无惩罚的公开 `solver="auto"` 不再需要 PR #142 临时的 FISTA 覆盖。直接拟合与 `PenalizedGLM_CV` 的选择/最终重拟合重新服从规范调度；适用的 logistic/Poisson L2 行执行后端原生 Newton，同时公开请求仍保持 `auto`，PR #151 随后使近似均匀权重判定对整体正比例缩放与排列保持不变，详见上文权重说明。
- `PenalizedGLM_CV` 新增推断控制，并且系数推断只在选定的全数据最终重拟合上执行一次；结果条件于 CV 选中的惩罚，并明确报告 `penalty_selection_adjusted_=False`。

#### 验证

- 增加针对性契约、formula、clone/兼容性、失败事务、无惩罚、加权 CV、安装器幂等性、加权 Newton 目标/兼容性与跨后端一致性回归，并同步中英文模型/CV/推断文档。
- `dev/benchmarks/validate_penalized_glm_inference_gpu.py` schema v4 是维护中的物理 CUDA 验收门禁。精确干净的数值源码 `db448d718f523eacf97bcb3c419e376c9812362d` 已在 Tesla P100（CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2）通过：Logistic 无权重保留显式 FISTA 覆盖，Logistic 加权与 Poisson 加权/无权重按规范 `solver="auto"` → Newton 执行；CuPy、Torch、Torch→CuPy 与 CuPy→Torch 四条路径均通过未放宽的系数/截距（`2e-6`）与推断（`1e-5`）阈值，并记录具体 `cuda:0` 来源信息。保留的 artifact 为 `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json`。后续仅修正 changelog/证据散文的仅文档提交，经用户明确批准可复用这一不可变的数值源码 artifact；任何数值、validator、求解器、后端、推断或容差变化都会重新打开物理验收。

### 逐节点 Lasso 推断调参（PR #139）

#### 变更

- 为稀疏 Gaussian 去偏推断公开 `nodewise_alpha=None`，适用于 `PenalizedGeneralizedLinearModel`、`PenalizedLinearRegression`、`Lasso`、`ElasticNet`、`LassoCV` 与 `ElasticNetCV`。显式值须为有限正实数；`PenalizedGLM_CV` 仍不支持非默认的此项设置。
- 默认逐节点惩罚不再依赖响应尺度，而由标准化后的中心化/加权设计决定：`sqrt(2 * log(max(p, 2)) / n_nodewise)`。非均匀分析权重下采用 Kish 型有效样本量；`p=1` 的解析情形无需逐节点惩罚。
- 逐节点精度矩阵在标准化坐标下求解并检查 KKT 条件，再按 `D^{-1} Theta_Z D^{-1}` 变回原坐标。已拟合属性记录请求值和实际调参值，不同逐节点调参不会误用旧的精度矩阵缓存。
- 去偏推断目标、协方差与参考分布含义、预测系数以及变量选择后的 OLS 行为保持不变；不新增旧响应尺度规则的公共模式。迁移说明见[逐节点调参指南](guides/nodewise-alpha-migration.md)。

#### 验证

- 固定设计迁移模拟比较旧响应相关规则、新自动规则与代表性显式规则，设置为 `n=320`、`p=18`、120 次重复和两种响应噪声尺度；默认规则的逐节点惩罚不随噪声尺度改变，旧规则相差 8 倍。该模拟中的覆盖误差未劣化，平均区间宽度改善，不代表所有数据下的普遍保证。
- Hosted 测试覆盖调参输入、KKT 检查、缓存隔离、公共 API/clone/refit、CV 最终重拟合推断、公式接口、GPU 不可用时的行为及已提交模拟产物。
- `dev/benchmarks/validate_nodewise_alpha_gpu.py` 用于真实 CUDA 上的 CuPy/Torch 验证，包括自动/显式调参、NumPy 对齐、响应尺度不变性、权重缩放与跨容器输入。列出验证器不表示本条记录新增了物理运行结果。

### Post-selection OLS 推断 API 清理（PR #138 / Issue #137）

#### 变更

- 为稀疏 Gaussian `Lasso`、`ElasticNet` 以及公开通用 `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")` 接口增加与硬件无关的规范 `inference_method="post_selection_ols"`。旧 `cpu_ols` / `gpu_ols` 作为一个兼容周期的 `FutureWarning` 别名保留；`LassoCV` 在 CV 兼容边界继续接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼法。
- “统计方法是什么”与“在哪个硬件执行”严格正交。显式 `device="cpu"`、`"cuda"` 或 `"torch"` 即使面对异构输入容器仍具有权威性；只有真正的 AUTO 策略才允许保留原生 CuPy/Torch-CUDA 输入。LassoCV 现在让 CV 与选中 alpha 的最终重拟合固定在同一解析后端，并把 CuPy 响应/权重对齐到设计矩阵的具体 CUDA 序号。
- `post_selection_ols` 在拟合记录在案的 NumPy/CuPy/Torch 后端上执行无惩罚活跃集 OLS/WLS 重拟合，同时保留惩罚 `coef_` 用于预测。nonrobust 继续使用 Student-t 与历史非活跃坐标占位。秩亏的活跃设计使用有效秩计算残差自由度，并通过设计级 Moore-Penrose/SVD 完成系数重拟合与协方差 bread；robust/HAC 以及空活跃集无截距情形保留调用者请求的协方差/参考分布族。
- 活跃集重拟合的诊断状态与惩罚拟合的 R-squared/F/对数似然/AIC/BIC 归属分离。`summary()` 分开报告惩罚拟合与选择后残差自由度；formula 路径保持 categorical/缺失行/样本权重对齐；失败的重拟合直接报错，不保留上一轮成功拟合或当前半成品推断状态。
- 统一稀疏 Gaussian 解析权重语义：NumPy/CuPy/Torch 直接拟合与加权 LassoCV 都先在原始观测上按权重中心化，再使用等价的 `sqrt(w * n / sum(w))` 行变换。默认 CV alpha 网格、折目标、加权验证 MSE 与最终重拟合使用同一约定；所有权重为同一正常数时精确等价于无权重 CV。加权非 Gaussian 稀疏 GLM 继续保留各自损失专属、感知样本权重的目标。
- NumPy/CuPy/Torch 的 `debiased` 统一到同一个中心化平均损失工作问题，使省略权重、全 1 权重与全局等比例缩放解析权重的结果一致。含截距的 simultaneous max-|Z| 推断现在让原始坐标系下的截距影响真正进入 bootstrap 最大值；成功重拟合会先清除过期的 simultaneous/精度状态再发布新结果。
- 字符串与公开 `Penalty` 对象形式共享同一个稀疏 Gaussian 迁移与 AUTO 路由契约。clone/get-params/set-params、警告调用点、LassoCV 最终重拟合归属、后端/设备来源、formula 路由与失败事务都有维护中的回归覆盖。

#### 验证

- hosted 验证覆盖 Python 3.9/3.12、Torch 2.0 CPU、完整 CPU 测试套件、scikit-learn 1.2.2/1.3.2/current 维护兼容性、static/ruff、文档、发布包与 benchmark 前端契约。
- `dev/benchmarks/validate_post_selection_ols_gpu.py` 是最终物理 CUDA 验收门禁，目前为 **schema v7 / 22 个用例**：保留原始 4 个直接 Lasso 用例，并增加 18 个 CuPy/Torch 收尾用例，覆盖 ElasticNet/通用稀疏 Gaussian、加权/无权重 debiased、真实加权多 alpha LassoCV 选择+最终重拟合、秩亏 SVD 重拟合、Penalty 对象 AUTO 路由、空活跃集 HC3 与含截距 simultaneous max-|Z|。既有 post-selection 数值容差没有放宽。
- PR #138 已合并。此前 Tesla P100 产物仅证明各自标注源码上的运行，不能作为最终 0.3.0 源码的验收；托管检查不能替代物理 GPU 运行，本条也不作 GPU 性能声明。

### Penalized solver API 清理（PR #135）

#### 变更

- 公开直接拟合的惩罚估计器统一以与后端无关的 `solver` 作为直接拟合的权威算法选择器。旧 `cpu_solver` 暂时保留，计划在未来破坏性版本中移除，调用者显式使用时产生 `FutureWarning`，但不会被静默映射成 `solver`，从而保持当前统一引擎的实际数值行为。
- `LassoCV` 将 `solver`（最终全数据重拟合）与 `cv_solver`（CV 折/路径）分开；`cv_solver="auto"` 在 CPU 上解析为坐标下降，在 CUDA/Torch 上解析为 FISTA，拟合后的 `cv_solver_` 记录实际执行算法。
- 已弃用的 `LassoCV(cpu_solver=...)` 保留历史阶段语义：CPU 上继续作为旧 CV 求解器别名；CUDA/Torch 上会 warning，但保持非权威，因此不会替换维护中的 GPU FISTA 路径。

#### 兼容性

- 省略直接拟合的 `cpu_solver` 与框架内部重建不会产生弃用噪声，包括 scikit-learn 1.2 的 `get_params() -> constructor` clone 路径和较新版本的 `__sklearn_clone__` 路径。
- 显式 `set_params(cpu_solver=...)` 以及旧式 `LassoCV(..., cpu_solver=...).fit(...)` 的 warning 会指向调用者，而不是 statgpu 内部重建/校验调用栈。
- 迁移指南明确区分“删除已经非权威的 direct `cpu_solver` 以保持当前实际行为”和“把旧算法意图显式搬到 `solver`、主动改变实际 solver”两种操作。

#### 验证

- 增加针对性求解器/弃用回归覆盖，覆盖直接拟合求解器权威性、参数省略与显式旧值、sklearn clone/重建、内部 helper 警告抑制、`set_params`、LassoCV CPU/GPU 别名、`cv_solver_` 与警告调用点。
- 维护兼容性 workflow 会在 scikit-learn 1.2.2、1.3.2 与 current 上运行该针对性测试集；该功能的历史结果属于 PR #135，最终 0.3.0 源码的验证须单独完成。

### Gaussian 后端原生推断（PR #129 / Issue #127）

#### 变更

- 维护中的 Gaussian 线性模型现在把协方差、标准误、统计量、p 值与置信区间的**数值计算**保留在实际执行的 NumPy/CuPy/Torch 后端；数值推断完成后，既有报告属性/结果仍可生成最终 NumPy 快照。
- Normal/Student-t 推断统一路由到维护中的参考分布层，并覆盖稳定的 df=1/df=2 极端尾部；若实际执行后端来源缺失或非法，则直接报错，而不是静默选择 NumPy。
- Ridge/L2 推断保持既有平均损失约定与正规方程中的 `n_eff * alpha` 映射，包括加权拟合与 `RidgeCV` 最终重拟合推断。

#### 验证

- 增加公开 `LinearRegression`、formula、加权/robust、秩亏、多目标、float32、statsmodels 对齐、无主机传输、非 L2 委托以及 Ridge/RidgeCV 回归覆盖，并增加针对性的 hosted CI workflow。
- 增加维护中的确切提交物理 CUDA validator，覆盖 CuPy/Torch 的干净工作树证明、请求/实际执行后端与具体设备来源、协方差/BSE/统计量/p 值/置信区间误差、加权/秩亏/多目标/小自由度，以及 `RidgeCV` 最终重拟合推断。
- [最终验收记录](https://github.com/TheHiddenObserver/statgpu/pull/129#issuecomment-5479959733)报告：确切数值源码 [`8d9f422e2fcf648e412838747b9ab5881b3fb7ef`](https://github.com/TheHiddenObserver/statgpu/commit/8d9f422e2fcf648e412838747b9ab5881b3fb7ef) 的七项 hosted workflow 与最终审查通过；两个物理 CUDA 验证器均在干净检出的 Tesla P100-SXM2-16GB `cuda:0` 上退出 0。后端原生矩阵使用 schema 5、CuPy 13.6.0 和 Torch 2.0.0+cu117；CuPy 秩恢复验证器为 schema 1，5/5 通过。
- 该记录列出的历史产物为 `results/pr129_gaussian_gpu_8d9f422e/gaussian_backend_native_8d9f422e.json` 和 `results/pr129_gaussian_gpu_8d9f422e/gaussian_cupy_rank_recovery_8d9f422e.json`。这些结果仅对应上述源码，不是对后续版本的重新验证；本条不作 GPU 加速声明。
- PR #129 后于 2026-08-31 合并为 [`b17c8a6c161065bfbe9efa9f60d0fad69a464773`](https://github.com/TheHiddenObserver/statgpu/commit/b17c8a6c161065bfbe9efa9f60d0fad69a464773)。较早的待验收状态记录属于当时检查点，不能作为最终状态。

## 0.2.5 — 2026-08-26（已发布）

### 新增

- **Panel Tier-1 Stage C 协方差与推断**：面向变换类 panel estimators 的 HC0/HC2/HC3 与 legacy HC1（`robust`）协方差；带可选 `group_debias=True` 的一路/两路聚类协方差；Bartlett、Parzen、Quadratic-Spectral 核的 Driscoll-Kraay 协方差；`RandomEffects` 在 quasi-demeaned GLS 分数上的 robust/HC 推断；`PooledOLS` 的 legacy row-order HAC（支持有序 categorical 时间顺序）。
- **诊断**：classical Hausman FE-vs-RE、pooling F、Breusch-Pagan LM、within/between/overall/adjusted R-squared 与 model F——在 NumPy/CuPy/Torch 上对极端 float64 量级 overflow-safe。
- **事务式 panel fits**：行保持的 formula prediction 与 fail-closed refit 语义。

### 修复

- CuPy `maximum.at`/`scatter_max` 对约 1e7..1e308 量级的 float64 返回 `inf`；组内 min/max scatter 现按量级门控（`<= 1e6` 走原生 GPU scatter），两条路径均精确。
- Torch CUDA SVD 要求精确的 `gesvd` driver，不可用时 fail closed；默认 `gesvdj` driver 会在结构零位置泄漏 ~1e-16，被巨大响应放大成错误系数。
- panel coefficient-resolution certificate 改为确定性误差界（不再依赖 LAPACK 版本相关的 SVD 舍入）；无法解析的近共线满秩设计 fail closed，单列 FE 吸收设计与秩亏设计报告实际秩。
- formula side-array 对齐仅接受原始 formula-data 行数或保留行数两种长度，其余 fail closed。
- 失败的 panel fit 保留实际执行后端 provenance 并清空 fitted/inference 状态。

### 优化

- Tesla P100 上两路聚类协方差 10k 行每次 CuPy fit 从约 1000 秒降至约 1.3 秒（过度宽泛的行级展开 fallback 改为 residual-acceptance 检查，普通均衡面板停留在向量化 Gram 路径）。
- Fama-MacBeth resident-array scaling（P100，本发布产物）：CuPy/Torch 的 GPU-over-NumPy median-time ratio 为 micro **1.314/0.706**、medium **0.174/0.126**、large **0.092/0.084**——Torch 在每个规模都快于 NumPy（1.4×/7.9×/11.9×），CuPy 自 medium 起 crossover，micro 启动开销段略慢于 NumPy，全部测量均为单次 `gram-certified` batch、零 SVD fallback。

### 验证

- 已在经验证的数值源 `697de113` 上完成 exact-source Tesla P100 验收：12 个 physical runner 全部通过。产物：`results/pr126_release_697de113/`。发布 head 本身是 PR #128 合入 `master` 的 merge commit；`697de113` 是这些 artifact 所验证的不可变数值源。
- TestPyPI 彩排：纯 Python wheel 在干净环境中从 `test.pypi.org` 安装并通过导入与 CPU fit/predict smoke test。

## 2026-08-09 — Panel Stage C 协方差补齐（PR #126）

Stage C 完成 Panel Tier-1 的协方差与推断能力，同时保持 estimator coefficient 与标准化 fit-statistic 定义不变。`robust` 继续表示既有 HC1；`hc0`、`hc2`、`hc3` 按各 estimator 的实际 transformed fit space 计算。`RandomEffects` 在 quasi-demeaned GLS score 上支持 robust/HC、clustered 与 Driscoll-Kraay covariance，并在 Swamy-Arora between auxiliary regression 没有正 residual degrees of freedom 时 fail closed；one-/two-way cluster 支持显式 `group_debias=True`；如果任一 clustering dimension 少于两个不同 group，clustered inference 现在会 fail closed，而不会返回退化的 single-cluster sandwich。`PooledOLS(cov_type="hac")` 继续表示 legacy row-order Bartlett/Newey-West 路径。

本次还系统强化了 two-way fixed-effect 的收敛与 prediction。Panel formula 中 pipe 明确命名的 metadata 现在作为权威来源：与显式 entity/time IDs 冲突时会 fail closed，缺失的 pipe 命名列不能再由无关显式 IDs 替代；RandomEffects 的第二个 pipe time 变量也只在 Driscoll-Kraay covariance 实际使用时才允许，同时 fixed-effect magic tokens 会被拒绝，而不会被重新解释为 grouping metadata。entity/time projection metadata 只在迭代前 factorize 一次，并在所选 backend 上复用；收敛判据直接检查两个 effect 维度的 residual group mean，对数值上被固定效应完全吸收的方向使用 scale-aware roundoff floor，同时公开 fail-closed 的 `demean_max_iter`/`demean_tol` 控制以处理弱连通 panel。unbalanced panel 的 two-way fixed effects 改为联合恢复；若已知 entity/time label 分属 disconnected incidence graph 的不同 component，则该预测不可识别并明确报错。formula prediction 在 Patsy 会删除 input row 或 formula transformation 生成非有限 design value 时会 fail closed；prediction 也不再把任意少一列的矩阵猜成“省略 intercept”，只有与拟合设计一致时才按原位置和原数值恢复显式 non-unit constant。已知 fixed-effect label 的预测现在会恢复 centered level grand mean，使 `PanelOLS.predict()` 返回完整的 fixed-effect level projection；formula 临时启用的 effect 不再泄漏到后续 refit，超过两个 fixed-effect 变量的 formula 会 fail closed，而无 FE 的 `PanelOLS` formula 会保留 Patsy/R 默认 intercept（`0 +` / `-1` 继续表示显式 no-intercept）；no-intercept 的 `rsquared_within` 现在使用标准 uncentered total sum of squares。其余强化还包括 rank-deficient coefficient identifiability，并让 classical Hausman 在任一 fitted coefficient vector 非唯一时 fail closed；此外还包括 `FirstDifferenceOLS` duplicate/time 语义、HC2/HC3 leverage 稳定性、metadata alignment、CuPy scatter-add、RandomEffects formula intercept/name 行为以及 quadratic-spectral weight。外部定义固定对齐 `statsmodels==0.14.6`、`linearmodels==7.0`、R `plm==2.6-7` 与 R `sandwich==3.1-3`。

最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 只在存在溢出风险时使用 reduction-length scaling，以避免全局 magnitude normalization 额外造成的信息损失，但不宣称可以恢复任意病态 cancellation remainder；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration。shared panel inference 不再施加绝对 variance floor：exact-zero variance 下，零 coefficient 的 statistic 为 0，非零 coefficient 为带符号无穷。Classical model F、pooling F 与 Breusch-Pagan LM 使用 overflow-safe centering和 subnormal-safe backend normalization。residual-based covariance 现在把 tiny-design/projection scale 的恢复推迟到 cancellation 之后：projection coordinate 只在 projection×residual 可能溢出时缩放，cluster 与 DK score 先 grouping 再做 selective Gram scaling，而且 residual vector 不会被全局 magnitude normalization，因此巨大 cancellation 旁边仍可表示的小 component 可以保留，同时已经安全的 subnormal-design 精度不被额外 normalization 破坏。two-way clustering 按 partition 等价而不是任意 code 编号识别 nested dimension，并在恢复尺度前代数消去相同 marginal/intersection component。range-aware symmetrization/inclusion-exclusion 与 HAC/DK 的 pre-Gram、full-lag accumulator 共同避免可恢复的中间溢出。maintained physical Stage-C runner 已把 diagnostic-scale、zero-variance、pre-Gram、tiny-design、mixed-range、nested-partition、covariance extreme-scale 与 lag-accumulation 分支加入 CuPy 与 Torch CUDA 验证。

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
