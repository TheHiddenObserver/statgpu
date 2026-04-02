# 基准脚本索引

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
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out examples/bench_all_large_results.json
```
