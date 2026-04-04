# statgpu 文档入口（中文）

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 中文文档入口  
> 切换: [English](USAGE.md)

语言切换：
- English: [USAGE.md](USAGE.md)

`USAGE_CN.md` 为中文入口，详细内容按“快速开始 / 核心指南 / 方法文档 / 基准脚本”拆分到 `docs/`。

## 1) 快速开始

- [快速上手](docs/getting-started/quickstart.md)
- [设备与显存管理](docs/guides/device-and-memory.md)
- [推断配置（Lasso）](docs/guides/inference-modes.md)
- [变更记录](docs/changelog.md)

安装提示：
- GPU 环境请按 CUDA 主版本选择 CuPy wheel：
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) 方法文档（按模块扩展）

总览索引：
- [Models Overview](docs/models/README.md)

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
- `LogisticRegression`
- `CoxPH`

推断能力摘要：
- `LinearRegression`: `cov_type=nonrobust/hc0/hc1`（CPU+GPU）
- `Lasso`: `inference_method=cpu_ols_inference/gpu_ols_inference/bootstrap`
- `LogisticRegression`: `cov_type=nonrobust/hc0/hc1`（CPU+GPU）

## 3) 基准与验证

- [基准脚本索引](docs/benchmarks.md)

当前重点脚本：
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`

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
