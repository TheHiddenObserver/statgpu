# statgpu 使用指南

## 安装

```bash
cd statgpu
pip install -e .
```

## 快速开始

### 1. 线性回归

```python
from statgpu.linear_model import LinearRegression
import numpy as np

# 生成数据
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 10

# 拟合模型
model = LinearRegression(device='cuda')  # 自动使用 GPU
model.fit(X, y)

# 查看完整统计输出
model.summary()

# 预测
y_pred = model.predict(X)
```

### 2. Ridge 回归

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device='cuda')
model.fit(X, y)
model.summary()
```

### 3. Lasso 回归

```python
from statgpu.linear_model import Lasso

model = Lasso(alpha=0.1, device='cuda')
model.fit(X, y)
model.summary()
```

### 4. Logistic 回归

```python
from statgpu.linear_model import LogisticRegression

# 二分类问题
model = LogisticRegression(device='cuda')
model.fit(X, y_binary)
model.summary()

# 预测概率
proba = model.predict_proba(X)
```

### 5. Cox 生存分析

```python
from statgpu.survival import CoxPH

# time: 生存时间
# event: 事件指示 (1=事件发生, 0=删失)
model = CoxPH(device='cuda')
model.fit(X, time, event)
model.summary()
```

## 设备控制

```python
import statgpu as sg

# 全局设置
sg.set_device('cuda')   # 强制使用 GPU
sg.set_device('cpu')    # 强制使用 CPU
sg.set_device('auto')   # 自动选择 (默认)

# 检查 GPU 是否可用
print(sg.cuda_available())

# 单个模型设置
model = LinearRegression(device='cuda')  # 仅此模型使用 GPU
```

## 输出说明

所有模型都提供 R 风格的完整统计输出：

- **coef**: 系数估计值
- **std err**: 标准误
- **t/z**: t 统计量或 z 统计量
- **P>|t/z|**: p 值
- **[0.025, 0.975]**: 95% 置信区间
- **R-squared**: R 方 (回归模型)
- **AIC/BIC**: 信息准则
- **Log-Likelihood**: 对数似然

## GPU 加速

大数据集时 GPU 显著加速：

| 数据规模 | CPU | GPU | 加速比 |
|---------|-----|-----|--------|
| 100K × 200 | 238ms | 64ms | 3.7x |
| 500K × 200 | 3.3s | 0.5s | 6.6x |
| 1M × 500 | 19s | 2.4s | 7.9x |

## 验证

所有模型结果已与以下工具验证一致：
- ✅ sklearn
- ✅ statsmodels
- ✅ R (lm, glm, coxph)
