# 基准脚本索引

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 基准脚本索引  
> 切换: [English](en/benchmarks.md)

语言切换：[English](en/benchmarks.md)

## 推断相关

- `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
  - 对比 `cpu_ols_inference` vs `gpu_ols_inference`
  - 输出时间与 `coef/bse/t/p/conf_int` 差异

## 显存管理

- `examples/benchmark_gpu_memory_cleanup.py`
  - 对比 `gpu_memory_cleanup=False/True`
  - 输出 `fit_ms` 与 CuPy memory pool 指标

## 训练性能 / 停止准则

- `examples/benchmark_lasso_cpu_gpu_tol.py`
- `examples/compare_lasso_kkt_stopping.py`

## 全方法大数据量耗时对比

- `examples/benchmark_all_methods_large_scale.py`
  - 覆盖：`LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - 支持 CPU/GPU 双设备、warmup/repeats、可选 inference 计时
  - 关键点：数据构造与 host->device 迁移在计时外，默认只统计 `fit()`

推荐运行命令：

```bash
python examples/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --include-external \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out examples/bench_all_large_results.json
```

若要把推断统计的计算时间也纳入计时，追加：

```bash
--compute-inference
```

## 外部框架对标（数值 + 时间）

- `examples/benchmark_external_frameworks.py`
  - 优先对标：`statsmodels`、`sklearn`
  - 可选对标：`R`（若系统有 `Rscript` 与对应包）
  - 输出：`fit_ms` + 系数/推断统计差异（可输出 JSON）

推荐运行命令（statsmodels + sklearn）：

```bash
python examples/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow \
  --skip-r
```

推荐运行命令（含 R）：

```bash
python examples/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow
```

对标门禁建议（务必统一口径）：
- 显式固定同一特征集合（避免 `y ~ .` 误包含目标/辅助列）
- 显式固定 `ties`（如 `cox-ties=breslow` 或 `efron`）
- 显式记录正则参数与迭代阈值（`alpha/C/max_iter/tol`）

## Cox 协方差专项基准

- `examples/benchmark_cox_cluster.py`
  - 对比 `CoxPH cov_type=nonrobust/hc1/cluster` 的时间与数值差异
  - 覆盖 `statgpu CPU/GPU` 与 `statsmodels.PHReg`（可用时）
