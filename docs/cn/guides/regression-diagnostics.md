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

`RegressionDiagnostics` 在 CPU 上运行，即使模型原先在 GPU 上拟合也是如此。
创建诊断对象会把所需的已拟合数组转换为 NumPy，因此应预留 CPU 内存和数据传输开销。
这一操作不会重新拟合模型。
