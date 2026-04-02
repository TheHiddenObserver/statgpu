# statgpu 文档入口

`USAGE.md` 作为统一入口，详细内容按“快速开始 / 核心指南 / 方法文档 / 基准脚本”拆分到 `docs/`。

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
- `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
- `examples/benchmark_gpu_memory_cleanup.py`
- `examples/benchmark_all_methods_large_scale.py`

建议给协作者跑的大规模计时命令：

```bash
python examples/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out examples/bench_all_large_results.json
```

## 4) 面向未来的方法扩展规范

当新增统计方法时，建议按以下结构新增文档，避免全部放进 `USAGE.md`：

- `docs/models/<method-name>.md`：方法专页（参数、示例、输出、限制）
- `docs/guides/<topic>.md`：跨模型主题（设备、推断、调优、对比口径）
- `docs/benchmarks.md`：基准脚本入口与结果解读口径
- `docs/changelog.md`：按日期记录能力增量、行为变化和兼容性说明

## 5) 协作建议（给合作者）

- 跑性能对比时，优先使用 `examples/benchmark_all_methods_large_scale.py`
- 报告结果时至少包含：设备信息、数据规模、`repeats/warmup`、是否包含 inference
- 若新增功能，请同步更新：
  - `docs/models/*.md`
  - `docs/benchmarks.md`（如新增脚本）
  - `docs/changelog.md`

## 6) 文档阅读顺序（建议）

1. `docs/getting-started/quickstart.md`（快速跑通）
2. `docs/guides/device-and-memory.md`（设备与显存策略）
3. `docs/models/README.md` + 对应模型页（参数与输出）
4. `docs/benchmarks.md`（性能测试与命令模板）
5. `docs/changelog.md`（近期能力变更）

## 7) 排查入口（常见）

- CUDA 不可用：先看 `docs/getting-started/quickstart.md` 的设备控制示例
- 推断统计差异：看 `docs/guides/inference-modes.md` 与模型页 `cov_type/inference_method`
- 显存占用偏高：看 `docs/guides/device-and-memory.md` 的 `gpu_memory_cleanup`
