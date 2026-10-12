# 回归诊断

> 语言：中文  
> 最后更新：2026-07-12  
> 切换：[English](../../en/guides/regression-diagnostics.md)

拟合回归模型后，可以检查残差、具有较高杠杆值或影响力的观测，以及特征之间的共线性。
`RegressionDiagnostics(model)` 从兼容模型的已拟合结果计算这些诊断量，
提供原始残差、标准化残差、内部及外部学生化残差、杠杆值、Cook 距离和 VIF。

准备好设计矩阵 `X` 与响应 `y` 后，先拟合线性回归，再读取诊断结果：

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

诊断计算使用模型保存的设计矩阵、响应、残差与尺度。秩亏设计通过伪逆计算帽子矩阵的对角线（杠杆值）；
外部学生化残差使用删除单个观测后的残差方差。
