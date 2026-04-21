# statgpu 文档入口（中文）

> 语言: 中文  
> 最后更新: 2026-04-18  
> 页面定位: 中文文档入口  
> 切换: [English](USAGE.md)

语言切换：
- English: [USAGE.md](USAGE.md)

`USAGE_CN.md` 为中文入口，详细内容按“快速开始 / 核心指南 / 方法文档 / 基准脚本”拆分到 `docs/`。

## 1) 快速开始

- [快速上手](docs/getting-started/quickstart.md)
- [设备与显存管理](docs/guides/device-and-memory.md)
- [推断配置（Lasso）](docs/guides/inference-modes.md)
- [Distribution API 使用指南（原生 GPU + 显式 Fallback）](docs/guides/distribution-api.md)
- [全局 p 值合并（Fisher/Cauchy/ACAT）](docs/guides/multiple-testing-combine-pvalues.md)
- [变更记录](docs/changelog.md)

安装提示：
- GPU 环境请按 CUDA 主版本选择 CuPy wheel：
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) 方法文档（按模块扩展）

总览索引：
- [模型总览](docs/models/README.md)
- [Knockoff 特征选择](docs/models/knockoff.md)
- [非参数方法](docs/models/nonparametric.md)

### 线性模型 `statgpu.linear_model`
- [LinearRegression](docs/models/linear-regression.md)
- [Ridge](docs/models/ridge.md)
- [Lasso](docs/models/lasso.md)
- [LogisticRegression](docs/models/logistic-regression.md)

### 生存分析 `statgpu.survival`
- [CoxPH](docs/models/coxph.md)

当前已实现方法：
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoCV`
- `LogisticRegression`
- `CoxPH` ✅ (Torch backend)
  - `cov_type=nonrobust/hc0/hc1/cluster` (cluster 为 CPU 路径)
  - `ties=breslow/efron` (Efron 带数值稳定性 clipping 保护)
  - 支持 C-index、baseline hazard、AIC/BIC
  - **性能**: Torch GPU 在 n=5000, p=20 规模下实现 15.44x 加速 (vs statsmodels)
  - 详见 `results/coxph_benchmark_report_2026-04-20.md` 综合性能对比报告

当前导出的 CV 类：
- `RidgeCV` ✅ (完整实现，支持 GPU 加速交叉验证)
- `LogisticRegressionCV` ✅ (完整实现，支持 GPU 加速交叉验证)
- `CoxPHCV` (骨架，待实现完整 CV 训练/搜索逻辑)

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
- 多重比较工具：`statgpu.adjust_pvalues` / `statgpu.multipletests`（`bh/by/holm/bonferroni`）
- 全局 p 值合并：`statgpu.combine_pvalues`（`fisher/cauchy/acat`）
- 统一重采样引擎：`statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) 基准与验证

- [基准脚本索引](docs/benchmarks.md)

当前重点脚本：
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`
- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`

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
  - `docs/models/*.md`
  - `docs/benchmarks.md`（如新增脚本）
  - `docs/changelog.md`
