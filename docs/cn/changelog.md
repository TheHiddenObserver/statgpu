# Changelog

- 将 AUTO 模式的 RidgeCV、ElasticNetCV 与 LogisticRegressionCV 最终重拟合固定到 CV 选参时使用的后端，避免选参后在 Torch 与 CuPy 之间静默漂移。

- 在公开 RidgeCV、ElasticNetCV 与 LogisticRegressionCV 调度中保留 `device='auto'`，使 GPU 常驻输入继续使用其原有后端；LogisticRegressionCV 现可在不完整复制到 CPU 的情况下验证 0/1 响应。

- Logistic 与 ElasticNet CV 的默认正则化网格现在纳入解析权重并满足整数权重的行复制等价性；CV 的 GPU 数组设备检查不再掩盖运行时错误。

- 专用 Ridge、ElasticNet 与 Logistic CV 现在严格保留显式 Torch/CuPy 后端选择，统一规范化 Device 枚举，并在生成网格或提前返回前验证解析权重。

- 修正 NumPy、CuPy 与 Torch 下解析权重 LogisticRegression 的 IRLS：权重仅进入 WLS 曲率而不进入工作响应分母，且加权似然与推断保持同一目标；同时收窄 penalized-CV alpha 网格与 CuPy 精确 Ridge 的降级范围，使编程错误、CUDA OOM 与设备错误继续抛出。

- 完成惩罚 CV 降级边界加固：可选 Lipschitz 提示统一识别 NumPy/CuPy/Torch 的秩失败，而 alpha 网格估计不再隐藏内存或 GPU 基础设施错误。

- 保持惩罚 CV 的声明验证目标：非 Gaussian 损失不再静默退化为 MSE，平方损失应急路径保留验证权重，GPU 基础设施错误会穿透多层 CV 降级并原样抛出。

- 收窄 GPU 线性代数降级条件：仅真实的秩亏/非正定失败可转用最小二乘、伪逆、ridge 或零块恢复；CUDA OOM、设备、索引与实现错误将原样抛出。

> 语言：中文<br>
> 最后更新：2026-08-05<br>
> 页面定位：变更记录<br>
> 切换：[English](../en/changelog.md)

## 未发布 — PyTorch、输入校验与 sklearn 兼容性维护

### 运行时安全

- Armijo 回溯不再把通用 `out of range` 错误当作可恢复数值 trial，因此 index/device 编程错误会原样抛出。
- proximal-Newton 现在会对明确的数值域 ValueError trial 执行回溯，同时保留无关的契约与 runtime failure。
- shared backend 线性方程求解现在仅对明确的秩失败使用 least-squares 降级，并保留 CUDA OOM/device RuntimeError。
- shared NumPy constructor 现在与 CuPy/Torch 一样跟随浮点 reference dtype；整数 reference 仍采用 float64 数值默认值。
- FISTA 系列 warm start 现在跟随预处理设计矩阵；smooth proximal-Newton 权重会在 loss 计算前转换到当前 backend/device/dtype。
- Newton 系列 Armijo 回溯现在仅忽略明确的数值域 trial failure，并保留 CUDA OOM/device/runtime 基础设施错误。
- solver sample-weight 校验现在会保留 CUDA OOM/device 等 backend RuntimeError，不再将其掩盖为普通输入 ValueError。
- 可执行 solver matrix 现在将 Elastic Net 视为非光滑惩罚，并通过 FISTA 而不是仅支持光滑目标的 solver 验证其精度。
- Newton、L-BFGS 与 L-BFGS-B 现在会对 Elastic Net 和其他非光滑惩罚显式失败，不再只优化其中的光滑部分。
- Newton 系列、L-BFGS 系列与 ADMM 的 warm start 现在统一跟随预处理设计矩阵的 backend、device 与 dtype，不再保留调用方原始数组的位置。
- 删除会重复计入光滑惩罚、从而优化错误目标的 Euclidean-prox Newton 快捷路径；光滑目标保留 Newton，非光滑目标在 Hessian-metric proximal 求解器完成前显式使用 FISTA。
- 补全 ADMM 的 Cholesky 降级初始化，并强化 L-BFGS-B 的可行方向与 NaN bounds 校验。
- 相邻的 Newton、proximal-Newton、ADMM、FISTA-BB、L-BFGS 与 L-BFGS-B 路径现在会在曲率计算前校验权重，仅对真正的奇异系统降级，保持 proximal Newton 与 CuPy bounds 的 dtype/device，并采用正确的梯度平方 Armijo 斜率。
- direct solver 与 penalized-CV 的 sample-weight 检查现在保持在所选 backend，并在 weighted Lipschitz 运算前执行；权重总和溢出会被拒绝，HC1 analytic-weight inference 对全局权重缩放保持不变。
- statgpu 内部迭代式 Torch kernel 统一通过集中式 compile policy。
  默认不再使用会启用 CUDA Graph 的 `reduce-overhead`；用户仍可通过
  `STATGPU_TORCH_COMPILE_MODE` 显式选择 `default`、
  `reduce-overhead` 或完全 eager。遇到已知 CUDA Graph 输出生命周期错误时，
  对应 callable 会永久回退 eager；其他运行时错误不会被吞掉。
- 维护矩阵覆盖的公共 estimator 数值入口采用 NumPy、CuPy 或 Torch 原生
  reduction 检查 NaN/Inf，不把完整 GPU 数组搬回 CPU；矩阵覆盖
  fit/predict/transform、inverse-transform、scoring、初始化数组和 panel ID，
  同时保留 formula 路径对缺失行的专属语义。
- formula sample weight 在 Patsy 确定保留行之后才进行对齐，并检查一维形状、
  finite、非负性与正权重和；Torch/CuPy 的对齐及 inference 权重保持在设备端。
- Gaussian GLM 的 FISTA 路径改用加权的特征均值与响应均值 profile intercept；
  在零惩罚时与闭式 weighted least squares 一致，不再优化错误的未加权中心化目标。
- GLM 的 sample weight 统一采用 analytic-weight 语义，覆盖 IRLS ridge scaling、
  line search、归一化 pseudo-loglikelihood、AIC/BIC、dispersion 与 sandwich inference；
  对全部权重作统一倍数缩放不会改变估计量或报告的诊断量。
- 所有支持的 GLM family（包括 penalized 与 CV estimator）都在 solver 或 fold
  dispatch 之前执行 backend-native response-domain validation；scalar GLM response
  支持非空实数的一维或单列输入，并在 solver/fold dispatch 前拒绝非实数、多列或长度不匹配；
  design matrix 与 analytic sample weight 也在 model、formula、CV 和 direct IRLS
  路径中共享 backend-native 的实数、finite、shape 与 length 契约；active IRLS/FISTA 编译
  统一走 centralized compile policy，且不再把无关的
  线性代数、显存或 device 错误伪装成 fallback。

### Estimator 与测试契约

- 构造函数原始参数与运行时标准化属性分开保存，使旧版 scikit-learn 的
  constructor identity clone 检查也能通过。
- `.gitignore` 不再隐藏应维护的 `test_*.py`；手工 GPU 诊断脚本使用独立目录
  和明确的 ownership policy。

关联：Issue #45、Issue #81、Issue #82、Issue #83。
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
