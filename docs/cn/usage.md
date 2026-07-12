# statgpu 文档入口（中文）

> 语言: 中文
>
> 最后更新: 2026-07-12
>
> 页面定位: 中文文档入口
>
> 切换: [English](../en/usage.md)

中文入口，详细内容按“快速开始 / 核心指南 / 方法文档 / 基准脚本”拆分到
`docs/cn/`；英文对应页面位于 `docs/en/`。

## 1) 快速开始

- [快速上手](getting-started/quickstart.md)
- [设备与显存管理](guides/device-and-memory.md)
- [推断配置（Lasso）](guides/inference-modes.md)
- [Distribution API 使用指南（原生 GPU + 显式 Fallback）](guides/distribution-api.md)
- [多重检验：P值校正与合并（BH/BY/Holm/Bonferroni/Hochberg + Fisher/Cauchy/Stouffer）](guides/multiple-testing-combine-pvalues.md)
- [变更记录](changelog.md)

安装提示：
- GPU 环境请按 CUDA 主版本选择 CuPy wheel：
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) 方法文档（按模块扩展）

总览索引：
- [模型总览](models/README.md)
- [GeneralizedLinearModel 与 Penalized GLM](models/generalized-linear-model.md)
- [PoissonRegression](models/poisson-regression.md)
- [Knockoff 特征选择](models/knockoff.md)
- [有序广义线性模型 (Logit/Probit)](models/ordered.md)
- [非参数方法](models/nonparametric.md)

### 线性模型 `statgpu.linear_model`
- [LinearRegression](models/linear-regression.md)
- [GeneralizedLinearModel 与 Penalized GLM](models/generalized-linear-model.md)
- [PoissonRegression](models/poisson-regression.md)
- [Ridge](models/ridge.md)
- [Lasso](models/lasso.md)
- [ElasticNet](models/elastic-net.md)
- [LogisticRegression](models/logistic-regression.md)

### 生存分析 `statgpu.survival`
- [CoxPH、CoxPHCV 与 PenalizedCoxPHModel](models/coxph.md)

当前已实现方法：
- `LinearRegression`
- `GeneralizedLinearModel`
- `PoissonRegression`
- `PenalizedLinearRegression`
- `PenalizedLogisticRegression`
- `PenalizedPoissonRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`
- `LassoCV`
- `LogisticRegression`
- `CoxPH` ✅（NumPy/CuPy/Torch）
  - `ties=breslow/efron/exact`
  - 支持 delayed entry、`(start, stop]`、`strata`、重复行 `subject_id`
  - `cov_type=nonrobust/hc0/hc1/cluster` 在三后端可用；Exact 当前仅 `nonrobust`
  - 支持 C-index、统一 Breslow baseline、分层生存预测及无惩罚拟合的 AIC/BIC
- `CoxPHCV` ✅（NumPy/CuPy/Torch）
  - L2 penalty 网格的 held-out 部分似然搜索 + 全量重拟合
  - 支持 Breslow/Efron/Exact、start-stop、strata 和按 `subject_id` 分组折叠
- `PenalizedCoxPHModel` ✅（NumPy/CuPy/Torch）
  - L1/L2/Elastic Net/SCAD/MCP；SCAD/MCP 使用 FISTA-LLA
  - 无截距、仅估计；不提供惩罚 Cox 推断或基线风险
- `OrderedLogitRegression` / `OrderedProbitRegression` ✅ (三后端)
  - 有序响应模型（累积 logit/probit 链接函数）
  - 跨后端精度修复 (2026-04-26)：coef 最大差异 < 1e-2

当前导出的 CV 类：
- `RidgeCV` ✅ (完整实现，支持 GPU 加速交叉验证)
- `LogisticRegressionCV` ✅ (完整实现，支持 GPU 加速交叉验证)
- `CoxPHCV` ✅（L2 CV、三后端、计数过程/Exact 轴）

当前已实现特征选择：
- `knockoff_filter`
- `fixed_x_knockoff_filter`
- `model_x_knockoff_filter`
- `KnockoffSelector`
- `FixedXKnockoffSelector`

推断能力摘要：
- `LinearRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac`（CPU+GPU）
- `Ridge`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac`（CPU+GPU）
- `Lasso`: `inference_method=cpu_ols_inference/gpu_ols_inference/bootstrap`
- `LogisticRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac`（CPU+GPU）
- `CoxPH`: `cov_type=nonrobust/hc0/hc1/cluster`（NumPy/CuPy/Torch）；Exact 仅 `nonrobust`
- 多重比较工具：`statgpu.adjust_pvalues` / `statgpu.multipletests`（`bh/by/holm/bonferroni/hochberg`）
- 全局 p 值合并：`statgpu.combine_pvalues`（`fisher/cauchy/stouffer`）
- 有序响应模型：`OrderedLogitRegression` / `OrderedProbitRegression`（CPU/CuPy/Torch）
- 统一重采样引擎：`statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) 基准与验证

- [基准脚本索引](guides/benchmarks.md)

当前重点脚本：
- `dev/benchmarks/_bench_inference_timing.py`（多重检验计时, p=100-10k）
- `dev/benchmarks/_bench_inference_timing_large.py`（多重检验计时, p=50k-1M）
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`
- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`

最新生存分析产物：

- [`results/survival_completion_2026-07-12.json`](../../results/survival_completion_2026-07-12.json)（quick）
- [`results/survival_completion_full_2026-07-12.json`](../../results/survival_completion_full_2026-07-12.json)（full）

两份产物使用 NVIDIA RTX 5880 Ada Generation、float64，fit 计时包括优化、推断和
baseline，数据传输单独计时。full delayed-entry 的 CuPy/Torch 相对 NumPy 为
1.044×/1.374×；full stratified start-stop 为 0.241×/0.411×，Exact 与普通重 ties
也慢于 CPU。quick delayed-entry 为 0.647×/0.959×。因此 GPU 收益取决于规模与风险集
结构，当前结果没有建立通用 crossover 阈值。

两份产物还验证了 stratified start-stop + subject-grouped `CoxPHCV`：NumPy、CuPy、
Torch 选择同一 penalty，最终 refit 系数和标准误的最大后端差异小于 $10^{-16}$。

最新非参数产物：
- 公平核对齐运行 `20260415_103036`（对角核设置下与 statsmodels 达到机器精度对齐）
- local-linear 优化运行 `20260415_120903`（多维 local-linear：CPU 约 4.8-5.4x，GPU 约 115-116x）

最新三方协方差产物：
- `results/remote_covariance_full_compare_2026-04-10.json`（`statsmodels` / `statgpu CPU` / `statgpu GPU`，`hc2/hc3/hac`）

建议给协作者跑的大规模计时命令：

```bash
python dev/benchmarks/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_all_large_results.json
```

## 4) 协作建议

- 跑性能对比时，优先使用 `dev/benchmarks/benchmark_all_methods_large_scale.py`
- 报告结果时至少包含：设备信息、数据规模、`repeats/warmup`、是否包含 inference
- 若新增功能，请同步更新：
  - `docs/cn/models/*.md` 与对应英文页
  - `docs/cn/guides/benchmarks.md` 与对应英文页（如新增脚本）
  - `docs/cn/changelog.md` 与 `docs/en/changelog.md`
