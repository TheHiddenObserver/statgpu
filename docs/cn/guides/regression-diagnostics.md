# 回归诊断

> 语言：中文  
> 最后更新：2026-07-12  
> 切换：[English](../../en/guides/regression-diagnostics.md)

拟合回归模型后，可以检查残差、具有较高杠杆值或影响力的观测，以及特征之间的共线性。
`RegressionDiagnostics(model)` 从兼容模型的已拟合结果计算这些诊断量，
提供原始残差、标准化残差、内部及外部学生化残差、杠杆值、Cook 距离和 VIF。

杠杆值较高，表示这一行的预测变量组合与大多数观测差异较大；它提示该观测有较大的潜在影响，但不等于该行数据有错。Cook 距离综合残差大小和杠杆值，衡量单个观测对拟合的影响。VIF（方差膨胀因子）检查每个特征能在多大程度上被其他特征线性解释，帮助发现系数难以分别估计的共线性问题。

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

`leverage`、`externally_studentized_residuals` 和 `cooks_distance` 中索引为 `i` 的值都对应拟合数据的第 `i` 行；`vif()` 按非截距设计列的顺序返回每列的结果。例如，数组 `X` 有 100 行、两列非常数特征时，上述三个逐行诊断数组各有 100 个值，VIF 有两个值，自动加入的截距不占 VIF 条目。公式拟合后，应对应公式处理后保留的观测行和展开的设计列，它们可能不同于原始数据框。

可以先找出 Cook 距离较大的行，回到原始数据检查录入、测量和特殊情况，并比较保留与去掉这些行后的拟合作为敏感性检查。诊断量用来确定值得调查的观测，不是自动删样本规则。

应结合样本量和模型假设理解这些结果，不要把某个阈值当成普遍适用的删样本规则。

`RegressionDiagnostics` 在 CPU 上运行，即使模型原先在 GPU 上拟合也是如此。
创建诊断对象会把所需的已拟合数组转换为 NumPy，因此应预留 CPU 内存和数据传输开销。
这一操作不会重新拟合模型。

诊断计算使用模型保存的设计矩阵、响应、残差与尺度。秩亏设计通过伪逆计算帽子矩阵的对角线（杠杆值）；
外部学生化残差使用删除单个观测后的残差方差。
