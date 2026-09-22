# Changelog

> 语言：中文<br>
> 最后更新：2026-09-20<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 未发布 — Quantile 求解器与推断更新（PR #166，目标 0.2.6）

### 修复

- **公开 Quantile 求解器边界**：直接拟合、交叉验证和公开底层接口现在采用一致的响应/设计矩阵形状、停止条件、延续路径、样本权重和求解器兼容性规则。普通 FISTA，以及底层普通 L-BFGS 在未传权重或均匀权重下的历史兼容行为继续保留；FISTA-BB、共享 ADMM、Newton、Proximal Newton 和 L-BFGS-B 会在数值迭代前明确拒绝 Quantile。历史 `quantile_cd_solver` 会忽略 `sample_weight`，且不能可靠表示不受惩罚的截距，因此仅保留名称的导入兼容性，调用时直接报错；标量 SCAD/MCP 使用 Proximal IRLS-CD。
- **直接拟合与交叉验证的输入对齐**：Quantile 的响应变量校验保持在当前数值后端上执行，并统一覆盖通用/类型化直接拟合与 `PenalizedGLM_CV`；公开底层 Quantile 路径也会在开始数值计算前拒绝含 NaN/Inf 的 `X/y`。非法的自定义交叉验证折和权重总质量为 0 的折，会在自动构造 `alpha` 网格之前报错；延续路径尺度、自适应/分组惩罚的归属关系与实际拟合目标保持一致。预测和评分会拒绝非法形状，不再允许 NumPy 广播产生外形正常但语义错误的结果。
- **独立 `QuantileRegression` 的推断正确性**：进行残差重抽样前，先按目标分位数的经验分位点对拟合残差做中心化，使重抽样误差分布的经验 τ 分位数为 0；随后每个重拟合样本都使用后端原生的批量 Quantile IRLS/MM，求解调用者请求的 Quantile 目标。多特征、非中位数的 bootstrap 重拟合目标已与独立 Quantile 线性规划解对齐，替代此前可能在 pinball 最优点之上提前停止的批量次梯度/FISTA 路径。真正非均匀解析权重下的独立模型推断会明确报错，因为当前尚未实现相应的加权核方法/bootstrap 推断；均匀权重继续对应等价的未加权推断目标。这里的 bootstrap 是基于可交换中心化残差的 i.i.d. 残差 bootstrap，不将其描述为对一般异方差稳健的 wild/multiplier bootstrap。
- **独立模型的推断生命周期与参数校验**：bootstrap 至少需要 2 次重抽样；非法的推断方法、核函数、带宽、停止条件、布尔参数或分位数设置会在进入后端数值计算前报错。Hall-Sheather、Bofinger 和 Chamberlain 规则在 `q ± h` 离开 `(0,1)`、最终带宽不是有限正数，或零点残差密度估计不是有限正数时，都会在发布协方差前报错。`score()` 按文档返回负 pinball loss，并支持可选解析权重。失败拟合不会保留半成品推断结果；`gpu_memory_cleanup=True` 也会正确识别执行记录中的 CuPy 后端，并覆盖成功与失败路径。
- **运行时帮助与 API 文档**：Quantile 次梯度的运行时帮助与实际导数保持一致；通用带惩罚模型的运行时帮助明确列出公开的 Quantile 损失接口。求解器封装在重新加载和不同导入顺序下保持安全，同时保留历史公开模块的导入身份。
- **光滑 Quantile FISTA 的后端一致性**：NumPy、CuPy 与 Torch 的 L2/无惩罚 Quantile 回溯路径采用与 CPU 一致的逐迭代“已接受点 + 目标稳定性”收敛语义。GPU 路径把已接受候选点的目标值、系数变化和 L2 跟踪量合并到该次 Armijo 检查原本就需要的同步中，避免额外的收敛检查或惩罚跟踪主机同步。
- **异步稀疏 Quantile FISTA**：异步非光滑 Quantile 交叉验证继续采用延迟检查，将 Nesterov 动量上限设为 `0.5`；只有在预热阶段之后连续两次目标检查都没有达到容差尺度的改善时，才把固定步长减半并重启动量。自适应控制复用已有的延迟同步，并把 L1 惩罚跟踪合并到同一次检查中，因此不会增加主机同步。GPU 上的 Quantile 稀疏交叉验证会对每个折中的 L1/ElasticNet/Adaptive-L1 候选模型以及最终全数据重拟合统一使用维护中的异步/自适应 FISTA；此前这些内部 `model.fit()` 调用可能回落到通用 pinball 回溯，并在非光滑折点处触发 Armijo 失败。普通直接 Quantile FISTA 的公开行为不变。
- **独立线性规划验证基准**：物理验收中的异步加权 L1 用例改用 SciPy/HiGHS 的线性规划最优值作为独立参考。在这组确定性数据上，CPU 通用 FISTA 的诊断目标值比真实凸最优值高约 `8.939e-4`，因此不再把“接近 CPU FISTA”作为正确性标准；原有 `5e-4` 阈值没有放宽，而是直接针对线性规划最优目标。schema v23 还把 L1 交叉验证的逐折/逐 `alpha` 验证得分、最终选择的 `alpha` 和最终全数据惩罚目标全部改为使用独立 HiGHS 线性规划参考。固定用例上 CPU 交叉验证得分相对线性规划的最大偏差达到 `3.1335e-3`，因此 CPU 结果只保留作诊断参考；GPU 得分和最终重拟合目标的容差均未放宽。

### 验证

- 新增有针对性的回归测试，覆盖 Python 版本下的测试收集安全性、公开求解器别名与重新加载幂等性、底层形状/路径/权重拒绝、直接拟合与交叉验证的响应校验、预测/评分的广播保护、Torch 响应后端保持、独立模型的失败事务/参数控制/内存清理、核方法带宽定义域失败，以及一个 `tau=0.2` 的集成批量 bootstrap 检查，用于区分调用者请求的分位数与错误的互补分位数方向。
- 扩展 CUDA 回归覆盖，加入非均匀加权 L1 FISTA、独立模型 bootstrap、自动 Group SCAD/MCP 直接拟合/交叉验证、加权标量底层 FISTA-LLA 刷新、平坦 IRLS 行为，以及 Group LLA 导数在 CuPy/Torch 上的设备驻留检查。

## 未发布 — Quantile 求解器来源对齐（PR #164 / Issue #163，目标 0.2.6）

### 修复

- Penalized Quantile 的求解器身份现在与实际执行算法一致。L2/无惩罚的 `solver="auto"` 解析为普通 Quantile IRLS，L1/ElasticNet 继续使用维护中的 FISTA 系列，SCAD/MCP 解析为专用 Proximal IRLS-CD。`proximal_irls_cd` 仍然只是内部记录解析结果与实际执行的标签，不成为公开可显式请求的 `solver=` 关键字。
- 不兼容的显式 Quantile 求解器请求现在会在进入数值调度或 CV 网格计算之前直接报错，不再静默执行另一种算法。直接拟合、CV 候选与折、以及选定的全数据重拟合的请求/解析/实际执行求解器身份因而保持一致。
- 非中位数 Quantile CV 评分会保留调用者配置的分位数水平，包括使用解析验证权重时；类型化 `PenalizedQuantileRegression(quantile=q)` 也会在克隆安全的构造、自适应 L1 初始化与 `score()` 中保留同一个 `q`。
- SCAD/MCP Quantile 的截距现在作为 pinball 目标中不受惩罚的坐标直接优化。若 LLA 替代目标完全变平，则通过维护中的完整 `QuantileLoss.irls()` 内核闭合，而不是继续使用对角近似。
- Torch 上的 Quantile 执行现在会让 L2 惩罚对角项、IRLS 热启动、Proximal IRLS-CD 的 epsilon/threshold/tolerance 标量以及回退权重始终跟随当前 tensor 的 dtype/device；热启动使用 Torch 原生克隆，而不是 NumPy/CuPy 的 `.copy()` 路径。

### 验证

- 精确干净的数值源码 `2de971402004efc703dc98f510942db3980988e4` 已在 Tesla P100-SXM2-16GB 上通过冻结的 schema-v1 物理验收，环境为 CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2、Python 3.9.16。CuPy/Torch CUDA 的直接拟合/交叉验证共 **12/12** 用例全部通过，并记录具体 `cuda:0` 设备来源与预期的 IRLS/FISTA/`proximal_irls_cd` 求解器身份。
- 冻结容差没有放宽：L2 系数/截距最大误差为 `6.938893903907228e-15`（阈值 `2e-5`），L2 CV 得分最大误差为 `7.965850201685498e-15`（阈值 `2e-5`），SCAD 惩罚目标最大误差为 `1.4085439563257807e-06`（阈值 `2e-4`）。
- 规范的确切源码 artifact 为 `dev/reviews/pr164_quantile_solver_provenance_gpu.json`。commit `230eaebbd4dfaf090e84011fe0eb190339389411` 直接位于已验证的数值源码之上，并且只新增该 artifact。后续 release/changelog 收尾严格属于仅文档变更，并显式复用这一不可变的数值源码验收；任何数值、validator、求解器、后端或容差变化都会重新要求物理复跑。

## 未发布 — Quantile IRLS 惩罚契约修复（PR #162 / Issue #161，目标 0.2.6）

### 修复

- 底层 `QuantileLoss.irls()` 现在只接受无惩罚或 L2。ElasticNet、L1、SCAD/MCP、group/adaptive penalty 以及未知 penalty object 会在数值迭代前直接报错，不再允许 IRLS 只处理声明目标中的光滑 L2 部分而忽略非光滑项。
- 公开估计器契约保持不变：显式 Quantile `solver="irls"` 仍然只属于 L2/无惩罚维护路径；非光滑惩罚继续通过 FISTA 系列或 Quantile 专用的非凸求解算法处理。
- IRLS 文档字符串与可执行的低层契约现在一致；不支持的直接底层调用会得到明确错误，而不是返回看似合理但只优化了部分惩罚的结果。

### 验证

- 增加针对性回归测试，覆盖直接拟合下 ElasticNet/L1/SCAD/未知 penalty 在线性求解前被拒绝、无惩罚/L2 路径保留、解析权重整体正比例缩放不变性、既有估计器级失败边界，以及 Torch CPU 上的 L2 后端保持。
- 本修复删除的是不受支持的路径，并未改变维护中的 None/L2 IRLS 数值算法，因此本身不新增物理 CUDA 验收要求。

## 未发布 — GLM 显式 Newton/L-BFGS 的解析权重支持（PR #151 / Issue #150，目标 0.2.6）

### 变更

- 普通 `GeneralizedLinearModel` 在受支持的 GLM 分布族和链接函数上，显式 `solver="newton"` 与 `solver="lbfgs"` 现在可以接受真正的非均匀解析 `sample_weight`，不会因为权重存在而静默替换成 IRLS/FISTA，也不会改变显式的 NumPy/CuPy/Torch 执行请求。
- 维护中的 GLM 光滑求解器在完整求解过程中统一使用同一个归一化解析权重目标。review/fix 使近似均匀权重分类在整体正比例缩放与排列下保持不变，并在执行 dtype 转换前先归一化，整数设计仍保留分数权重，float32 原始求和溢出不会在稳定归一化之前误拒绝，并保留普通拟合后诊断/推断实际求解时使用的求解器准备好的权重身份。在 Newton/L-BFGS 的 M-estimation 路径上，同一权重身份会在协方差计算前等价缩放为均值为 1 的表示，因此整体正比例缩放解析权重不会在数值求解容差之外改变系数及 nonrobust/HC0/HC1 推断，同时不会静默重定义无关的 IRLS/FISTA 加权语义。`ParameterInferenceResult` 在构造方遗漏标准字段时也会公开实际的 M-estimation `cov_type`。L-BFGS 保留可见的线搜索失败，只在“所要求的目标下降”和“实际参数位移”都已低于维护中的数值分辨率时允许受限的浮点 Armijo 接受。
- `GammaRegression(link="inverse_power")` 的显式 Newton/L-BFGS 现在使用由 Gamma 损失函数拥有的光滑训练域契约，而不再对无截距情况作一刀切拒绝。被定义域截断的极小 L-BFGS 步长本身不构成收敛；若定义域内的拟牛顿 Armijo 搜索耗尽，会用重新计算定义域上界的最速下降方向再尝试一次，若梯度尚未收敛且恢复仍失败则明确报错。可行的无截距设计可以正常拟合；零权重观测不约束域可行性；真正不可行、数值上无法认证或被定义域边界钉住的拟合会显式失败。
- 同一 inverse-Gamma 域契约已经闭合到带惩罚 L2 Newton/L-BFGS 和 `PenalizedGLM_CV`：框架热启动只有在域内时才复用；CV 评分使用声明的 `inverse_power` 目标，不再错误使用 log 链接评估器；最终全数据重拟合保留链接。`PenalizedGammaRegression` 同时保持历史 `loss_kwargs["link"]` 优先级，并通过 sklearn clone/get_params 契约。
- 既有 `solver="auto"`、IRLS/FISTA、显式 smooth-solver `C`、Ordered GLM、standalone LogisticRegression、非 inverse Gamma 链接，以及成功拟合后的求解器/后端/设备来源语义保持不变，除非上文明确说明属于本次修复范围。
- 普通显式 Newton/L-BFGS 重拟合现在对校验、优化、推断与来源信息发布实施原子事务。失败的重拟合会完整恢复上一次成功的估计器状态，而不会暴露“旧系数 + 新尝试元数据”的混合对象；第一次拟合失败仍保持未拟合状态。此前成功执行的求解器/后端/设备来源会保留，但失败的尝试不会被发布成新的执行证据。

### 验证

- 历史 hosted/physical evidence 继续只对各自精确源码有效。数值/validator 源码 `9eb39cee2c0e691160f528ab687b7379d26f3e42` 曾通过全部 7 个 hosted PR workflow 和 Tesla P100 schema-v5 验收；该源码的完整 CPU 测试套件为 **3395 passed / 831 skipped / 0 failed**。后续新一轮 review 已修改生产推断与结果来源信息，因此这些 hosted 结果和 schema v5 都不能作为后续源码的验收。
- Tesla P100 schema-v3 artifact 对应 `c6781cb6a2e1fe500f325e832d23cdc80a99b564`，schema-v4 inverse-Gamma artifact 对应 `9ef5b34ffc8abf133bc6262e98a855d5efe37d3c`，schema-v5 artifact 对应 `9eb39cee2c0e691160f528ab687b7379d26f3e42`；三者现在都只作为各自源码的不可变**历史确切源码证据**保留。对应文件为 `dev/reviews/pr151_glm_weighted_explicit_solvers_gpu.json`、`dev/reviews/pr151_inverse_gamma_domain_gpu_v4.json` 与 `dev/reviews/pr151_final_gpu_v5.json`。
- schema v6 是不可变的**历史失败 validator**，不再是当前验收门禁。其 float32 溢出推断夹具除了预期的 float32 解析权重外，还意外把 `X/y` 一起转成 float32，从而额外引入了无关的 float32 设计 CuPy L-BFGS 跨后端一致性要求。这个独立精度问题由 Issue #160 跟踪；PR151 不会为了让该错误夹具通过而放宽冻结容差或扩大 generic L-BFGS 的修改范围。
- 修正后的 schema v7 已成为最终接受的物理 CUDA 验收。精确干净的数值/validator 源码 `0302262ef8242b31f2f17b7835b6c08aeda89904` 已在 Tesla P100-SXM2-16GB（CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2、Python 3.9.16）上通过完整 v3→v4→v5→v7 链，记录 `status: success` 与 `source_clean: true`。当前规范 artifact 为 `dev/reviews/pr151_final_gpu_v7.json`；较早通过的 `e1cbf3756d269a6073461de8331da11c6235fb4e` schema-v7 artifact 作为历史确切源码证据保留在 `dev/reviews/pr151_final_gpu_v7_e1cbf375.json`。与旧运行的数值结果保持一致，最大 review 收尾误差仍为 `9.719913152128612e-09`，位置仍是同一个 penalized/Newton/Torch p 值比较。commit `29b8e8056fb9bdfb34e2b76982d433c0be3d844f` 直接位于已验证数值源码之上并记录最终证据；之后的 changelog 收尾仅属于仅文档变更，不重新打开物理验收。PR #151 现已 **merge-ready**；Issue #160 继续作为非阻塞后续事项，Issue #152 保持已完成。

## 未发布 — 后端原生 Gaussian residual bootstrap（PR #147 / Issue #145，目标 0.2.6）

### 变更

- 将既有 unweighted Gaussian `residual_bootstrap` 从仅 NumPy 执行扩展到拟合记录在案的 NumPy/CuPy/Torch 后端与具体设备；三个后端共享同一个确定性的后端无关残差索引调度，后端/设备来源一旦漂移即直接报错，不回退 CPU。
- 子样本重拟合保留已拟合的惩罚家族、调参、截距、求解器/停止条件、Lipschitz 与 SCAD/MCP LLA 控制，并保持子样本推断关闭。`PenalizedGLM_CV` 仍只在选定的全数据最终重拟合上执行一次 bootstrap，并明确报告对 CV 选中惩罚的条件化、未校正选择不确定性。
- 加权、robust/HC、HAC/block、非 Gaussian 与 Cox bootstrap 语义仍不支持并直接报错。大规模设计/响应/残差/bootstrap 响应数组与子样本优化保持在记录的数值后端/设备上；最终报告仍采用 NumPy 边界，仅允许小型控制/参数快照跨越该边界。

### 验证

- 增加确定性的 hosted 覆盖，覆盖 NumPy 路径保留、L1/ElasticNet/SCAD/MCP、formula 与公开包装器消费方、精确设备子样本上下文、失败事务、安装器幂等性、CV 仅最终重拟合语义与不支持的行。
- `dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py` schema v1 是冻结的确切源码 CuPy/Torch CUDA 验收门禁。PR #147 转 Ready 或合并前仍必须完成物理 CUDA 验收；hosted CI 不能替代该门禁。

## 未发布 — Penalized GLM 推断契约修复（PR #142，目标 0.2.6）

### 变更

- 通用与类型化 penalized-GLM 估计器统一以 `inference_method="auto"` 作为公开对账边界；专用稀疏 Gaussian 包装器保留既有显式默认。成功拟合会区分请求/解析/报告的方法，并记录推断目标与调参/选择条件。
- 受支持的光滑非 Gaussian L2 / 无惩罚推断解析为固定惩罚的 `m_estimation`，当前支持 nonrobust/HC0/HC1 协方差；非 Gaussian L1/ElasticNet 系数推断改为直接报错，不再发布历史上只含 L2 curvature 的全向量部分 sandwich。
- 残差 `bootstrap` 明确限定为 `cov_type="nonrobust"` 的无权重 CPU Gaussian 残差 bootstrap；重拟合保留真实惩罚/调参/截距契约，至少需要 2 次重抽样，且 CuPy/Torch 已执行拟合不会静默切到 CPU 做重抽样。
- 非 Gaussian M-estimation 以拟合记录在案的 NumPy/CuPy/Torch 后端与具体设备为准，并覆盖 Torch↔CuPy 异构输入容器对齐。维护中的 Newton 求解器现在支持真正的非均匀解析权重，并让目标值、梯度、Hessian 与 Armijo 试探使用同一个归一化加权目标，因此加权光滑 L2/无惩罚的公开 `solver="auto"` 不再需要 PR #142 临时的 FISTA 覆盖。直接拟合与 `PenalizedGLM_CV` 的选择/最终重拟合重新服从规范调度；适用的 logistic/Poisson L2 行执行后端原生 Newton，同时公开请求仍保持 `auto`，并保留历史浮点均匀权重 `allclose` 语义。
- `PenalizedGLM_CV` 新增推断控制，并且系数推断只在选定的全数据最终重拟合上执行一次；结果条件于 CV 选中的惩罚，并明确报告 `penalty_selection_adjusted_=False`。

### 验证

- 增加针对性契约、formula、clone/兼容性、失败事务、无惩罚、加权 CV、安装器幂等性、加权 Newton 目标/兼容性与跨后端一致性回归，并同步中英文模型/CV/推断文档。
- `dev/benchmarks/validate_penalized_glm_inference_gpu.py` schema v4 是维护中的物理 CUDA 验收门禁。精确干净的数值源码 `db448d718f523eacf97bcb3c419e376c9812362d` 已在 Tesla P100（CuPy 13.6.0、Torch 2.0.0+cu117、NumPy 1.24.2）通过：Logistic 无权重保留显式 FISTA 覆盖，Logistic 加权与 Poisson 加权/无权重按规范 `solver="auto"` → Newton 执行；CuPy、Torch、Torch→CuPy 与 CuPy→Torch 四条路径均通过未放宽的系数/截距（`2e-6`）与推断（`1e-5`）阈值，并记录具体 `cuda:0` 来源信息。保留的 artifact 为 `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json`。后续仅修正 changelog/证据散文的仅文档提交，经用户明确批准可复用这一不可变的数值源码 artifact；任何数值、validator、求解器、后端、推断或容差变化都会重新打开物理验收。

## 未发布 — Post-selection OLS 推断 API 清理（PR #138 / Issue #137）

### 变更

- 为稀疏 Gaussian `Lasso`、`ElasticNet` 以及公开 generic `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")` surface 增加与硬件无关的规范 `inference_method="post_selection_ols"`。旧 `cpu_ols` / `gpu_ols` 作为一个兼容周期的 `FutureWarning` 别名保留；`LassoCV` 在 CV 兼容边界继续接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼法。
- “统计方法是什么”与“在哪个硬件执行”严格正交。显式 `device="cpu"`、`"cuda"` 或 `"torch"` 即使面对异构输入容器仍具有权威性；只有真正的 AUTO 策略才允许保留原生 CuPy/Torch-CUDA 输入。LassoCV 现在让 CV 与选中 alpha 的最终重拟合固定在同一解析后端，并把 CuPy 响应/权重对齐到设计矩阵的具体 CUDA 序号。
- `post_selection_ols` 在拟合记录在案的 NumPy/CuPy/Torch 后端上执行无惩罚活跃集 OLS/WLS 重拟合，同时保留惩罚 `coef_` 用于预测。nonrobust 继续使用 Student-t 与历史非活跃坐标占位。秩亏的活跃设计使用有效秩计算残差自由度，并通过设计级 Moore-Penrose/SVD 完成系数重拟合与协方差 bread；robust/HAC 以及空活跃集无截距情形保留调用者请求的协方差/参考分布族。
- 活跃集重拟合的诊断状态与惩罚拟合的 R-squared/F/对数似然/AIC/BIC 归属分离。`summary()` 分开报告惩罚拟合与选择后残差自由度；formula 路径保持 categorical/缺失行/样本权重对齐；失败的重拟合直接报错，不保留上一轮成功拟合或当前半成品推断状态。
- 统一稀疏 Gaussian 解析权重语义：NumPy/CuPy/Torch 直接拟合与加权 LassoCV 都先在原始观测上按权重中心化，再使用等价的 `sqrt(w * n / sum(w))` 行变换。默认 CV alpha 网格、折目标、加权验证 MSE 与最终重拟合使用同一约定；所有权重为同一正常数时精确等价于无权重 CV。加权非 Gaussian 稀疏 GLM 继续保留各自损失专属、感知样本权重的目标。
- NumPy/CuPy/Torch 的 `debiased` 统一到同一个中心化平均损失工作问题，使省略权重、全 1 权重与全局等比例缩放解析权重的结果一致。含截距的 simultaneous max-|Z| 推断现在让原始坐标系下的截距影响真正进入 bootstrap 最大值；成功重拟合会先清除过期的 simultaneous/精度状态再发布新结果。
- 字符串与公开 `Penalty` 对象形式共享同一个稀疏 Gaussian 迁移与 AUTO 路由契约。clone/get-params/set-params、warning 调用点、LassoCV 最终重拟合归属、后端/设备来源、formula 路由与失败事务都有维护中的回归覆盖。

### 验证

- hosted 验证覆盖 Python 3.9/3.12、Torch 2.0 CPU、完整 CPU 测试套件、scikit-learn 1.2.2/1.3.2/current 维护兼容性、static/ruff、文档、发布包与 benchmark 前端契约。
- `dev/benchmarks/validate_post_selection_ols_gpu.py` 是最终物理 CUDA 验收门禁，目前为 **schema v7 / 22 个用例**：保留原始 4 个直接 Lasso 用例，并增加 18 个 CuPy/Torch 收尾用例，覆盖 ElasticNet/generic 稀疏 Gaussian、加权/无权重 debiased、真实加权多 alpha LassoCV 选择+最终重拟合、秩亏 SVD 重拟合、Penalty 对象 AUTO 路由、空活跃集 HC3 与含截距 simultaneous max-|Z|。既有 post-selection 数值容差没有放宽。
- 之前 Tesla P100 artifact 只继续作为各自历史确切提交的不可变证据。后续 review/fix 循环已修改有效的生产数值路径，因此当前源码的物理验收仍然 **等待最终确切干净提交上的 schema-v7 22/22 CuPy/Torch CUDA 复跑**。hosted 检查不替代该门禁，本变更也不做 GPU 性能声明。

## 未发布 — Penalized solver API 清理（PR #135）

### 变更

- 公开直接拟合的惩罚估计器统一以与后端无关的 `solver` 作为直接拟合的权威算法选择器。旧 `cpu_solver` 暂时保留一个兼容周期，调用者显式使用时产生 `FutureWarning`，但不会被静默映射成 `solver`，从而保持当前统一引擎的实际数值行为。
- `LassoCV` 将 `solver`（最终全数据重拟合）与 `cv_solver`（CV 折/路径）分开；`cv_solver="auto"` 在 CPU 上解析为坐标下降，在 CUDA/Torch 上解析为 FISTA，拟合后的 `cv_solver_` 记录实际执行算法。
- 已弃用的 `LassoCV(cpu_solver=...)` 保留历史阶段语义：CPU 上继续作为旧 CV 求解器别名；CUDA/Torch 上会 warning，但保持非权威，因此不会替换维护中的 GPU FISTA 路径。

### 兼容性

- 省略直接拟合的 `cpu_solver` 与框架内部重建不会产生弃用噪声，包括 scikit-learn 1.2 的 `get_params() -> constructor` clone 路径和较新版本的 `__sklearn_clone__` 路径。
- 显式 `set_params(cpu_solver=...)` 以及旧式 `LassoCV(..., cpu_solver=...).fit(...)` 的 warning 会指向调用者，而不是 statgpu 内部重建/校验调用栈。
- 迁移指南明确区分“删除已经非权威的 direct `cpu_solver` 以保持当前实际行为”和“把旧算法意图显式搬到 `solver`、主动改变实际 solver”两种操作。

### 验证

- 增加针对性求解器/弃用回归覆盖，覆盖直接拟合求解器权威性、参数省略与显式旧值、sklearn clone/重建、内部 helper warning 抑制、`set_params`、LassoCV CPU/GPU 别名、`cv_solver_` 与 warning 调用点。
- 维护兼容性 workflow 会在 scikit-learn 1.2.2、1.3.2 与 current 上运行该针对性测试集；最终确切提交的 hosted 结果在 PR #135 的最终源码提交完成 CI 后记录。

## 未发布 — Gaussian 后端原生推断（PR #129 / Issue #127）

### 变更

- 维护中的 Gaussian 线性模型现在把协方差、标准误、统计量、p 值与置信区间的**数值计算**保留在实际执行的 NumPy/CuPy/Torch 后端；数值推断完成后，既有报告属性/结果仍可生成最终 NumPy 快照。
- Normal/Student-t 推断统一路由到维护中的参考分布层，并覆盖稳定的 df=1/df=2 极端尾部；若实际执行后端来源缺失或非法，则直接报错，而不是静默选择 NumPy。
- Ridge/L2 推断保持既有平均损失约定与正规方程中的 `n_eff * alpha` 映射，包括加权拟合与 `RidgeCV` 最终重拟合推断。

### 验证

- 增加公开 `LinearRegression`、formula、加权/robust、秩亏、多目标、float32、statsmodels 对齐、无主机传输、非 L2 委托以及 Ridge/RidgeCV 回归覆盖，并增加针对性的 hosted CI workflow。
- 增加维护中的确切提交物理 CUDA validator，覆盖 CuPy/Torch 的干净工作树证明、请求/实际执行后端与具体设备来源、协方差/BSE/统计量/p 值/置信区间误差、加权/秩亏/多目标/小自由度，以及 `RidgeCV` 最终重拟合推断。
- 已审查的实现提交上的 hosted 门禁与新一轮完整 diff review 已全部通过；最终验收仍要求在最终源码 SHA 上执行确切干净提交的 CuPy/Torch CUDA 验证。PR #129 当前保持 open/unmerged，#127 尚不能标记为 `COMPLETE`。本变更不做 GPU 加速声明。

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