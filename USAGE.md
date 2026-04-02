# statgpu 文档入口

`USAGE.md` 作为统一入口，不再堆叠所有方法细节。  
详细内容按“快速开始 / 核心指南 / 方法文档 / 基准脚本”拆分到 `docs/`。

## 1) 快速开始

- [快速上手](docs/getting-started/quickstart.md)
- [设备与显存管理](docs/guides/device-and-memory.md)
- [推断配置（Lasso）](docs/guides/inference-modes.md)

## 2) 方法文档（按模块扩展）

### 线性模型 `statgpu.linear_model`
- [LinearRegression](docs/models/linear-regression.md)
- [Ridge](docs/models/ridge.md)
- [Lasso](docs/models/lasso.md)
- [LogisticRegression](docs/models/logistic-regression.md)

### 生存分析 `statgpu.survival`
- [CoxPH](docs/models/coxph.md)

## 3) 基准与验证

- [基准脚本索引](docs/benchmarks.md)

当前重点脚本：
- `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
- `examples/benchmark_gpu_memory_cleanup.py`

## 4) 面向未来的方法扩展规范

当新增统计方法时，建议按以下结构新增文档，避免全部放进 `USAGE.md`：

- `docs/models/<method-name>.md`：方法专页（参数、示例、输出、限制）
- `docs/guides/<topic>.md`：跨模型主题（设备、推断、调优、对比口径）
- `docs/benchmarks.md`：基准脚本入口与结果解读口径

`USAGE.md` 只保留导航与统一索引。
