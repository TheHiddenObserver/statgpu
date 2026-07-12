# 回归诊断

> 语言：中文  
> 最后更新：2026-07-12  
> 切换：[English](../../en/guides/regression-diagnostics.md)

`RegressionDiagnostics(model)` 从兼容回归模型读取拟合设计、响应、残差与尺度，
提供原始/标准化/内部及外部 studentized residual、杠杆值、Cook 距离和 VIF。
秩亏设计使用 pseudoinverse 计算 hat diagonal，外部 studentization 使用删除单个
观测后的残差方差。

```python
from statgpu import LinearRegression, RegressionDiagnostics

model = LinearRegression().fit(X, y)
diag = RegressionDiagnostics(model)
print(diag.leverage)
print(diag.externally_studentized_residuals)
print(diag.cooks_distance)
print(diag.vif())
```

诊断属于报告侧 CPU 工具：拟合数组只复制一次到 NumPy，以调用 SciPy 分布检验并
生成可读摘要。这是显式边界，不是训练路径的静默回退。参考测试与
`statsmodels.OLSInfluence` 对齐。
