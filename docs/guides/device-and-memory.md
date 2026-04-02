# 设备与显存管理

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 指南文档  
> 切换: [English](../en/guides/device-and-memory.md)

语言切换：[English](../en/guides/device-and-memory.md)

## 设备选择

每个模型都支持 `device`：
- `device="cpu"`
- `device="cuda"`
- `device="auto"`（默认）

## 显存管理开关：`gpu_memory_cleanup`

以下模型支持：
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LogisticRegression`
- `CoxPH`

参数行为：
- `gpu_memory_cleanup=False`（默认）  
  使用 CuPy memory pool 缓存已分配显存，重复 `fit` 通常更快。
- `gpu_memory_cleanup=True`  
  每次 `fit` 后尝试释放 pool block，降低常驻显存占用。

示例：

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
model.fit(X, y)
```

## 何时开启

- 显存紧张、多模型并行：建议开
- 单模型连续训练追求吞吐：建议关

## 基准脚本

参见 `examples/benchmark_gpu_memory_cleanup.py`，可对比：
- `fit_ms`
- `pool_used_fit`
- `pool_total_fit`
- `pool_used_reset`
- `pool_total_reset`
