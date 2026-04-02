# 快速上手

## 安装

```bash
cd statgpu
pip install -e .
```

## 最小示例

```python
import numpy as np
from statgpu.linear_model import LinearRegression

X = np.random.randn(1000, 20)
y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)

model = LinearRegression(device="cuda")
model.fit(X, y)
print(model.score(X, y))
```

## 常用设备控制

```python
import statgpu as sg

sg.set_device("auto")   # 默认: 有 CUDA 就用 GPU
sg.set_device("cuda")   # 强制 GPU
sg.set_device("cpu")    # 强制 CPU
print(sg.cuda_available())
```

更多内容见：
- [设备与显存管理](../guides/device-and-memory.md)
- [推断配置（Lasso）](../guides/inference-modes.md)
