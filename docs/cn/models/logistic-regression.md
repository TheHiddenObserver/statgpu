# LogisticRegression

> 语言: 中文  
> 最后更新: 2026-10-03
> 页面定位: 模型文档  
> 切换: [English](../../en/models/logistic-regression.md)

语言切换：[English](../../en/models/logistic-regression.md)

## 概览（Overview）

`LogisticRegression` 提供二分类 Logit 的 IRLS 或 L-BFGS 拟合、稳健协方差和分类评估，支持 CPU/GPU。`LogisticRegressionCV` 通过交叉验证选择 L2 正则化参数 `C`，再使用全部训练数据重拟合。当前接口支持二分类及可选的 L2 正则化，不支持多分类、L1 或 elastic-net。

## 路径（Path）

- `statgpu.linear_model.LogisticRegression`
- `statgpu.linear_model.LogisticRegressionCV`

## 目标函数（Objective Function）

当 `C` 为正数时，两种求解器最小化相同的求和形式目标函数，并支持样本权重：

$$
\min_{b,\beta}\; -\sum_i w_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{\|\beta\|_2^2}{2C},
\qquad p_i=\sigma(b+x_i^\top\beta).
$$

`sample_weight` 指定 \(w_i\)，省略时 \(w_i=1\)。截距 \(b\) 不受惩罚；`fit_intercept=False` 时不拟合截距。正数 `C` 越大，正则化越弱。保留的特殊值 `C=0` 表示不使用惩罚，**不表示**无限强的正则化。负数及非有限的 `C` 均无效。

L-BFGS 在优化时将**整个目标函数（包括惩罚项）**除以样本数或样本权重总和。这不会改变最优解或 `C` 的含义，但会影响梯度停止准则的数值尺度。与其他库比较时，应对齐损失的求和或平均约定、惩罚系数、样本权重、截距处理以及收敛设置。

## 求解器选择与收敛

- `solver="auto"` 解析为 `"irls"`，保持默认拟合行为。
- `solver="irls"` 使用迭代加权最小二乘法，`tol` 用于判断参数步长范数。
- `solver="lbfgs"` 使用有限内存 BFGS，`tol` 用于判断缩放后目标函数的梯度范数；参数步长很小本身不能判定为收敛。
- `max_iter` 限制所选求解器的迭代次数。请检查 `converged_` 和 `n_iter_`；未达到停止准则时会发出 `ConvergenceWarning`。
- `get_params()["solver"]` 保留用户指定值，拟合后的 `solver_` 记录实际解析值。`summary()` 显示两者及收敛状态，但要求 `compute_inference=True`。

当 `C` 为正数时，系数的得分方程为 \(\sum_i w_i x_i(y_i-p_i)-\beta/C=0\)，截距对应的方程没有惩罚项。IRLS 和 L-BFGS 以同一解为目标，但迭代次数及相同 `tol` 下的精度可能不同。适当缩放条件较差的特征有助于两种求解器收敛。L-BFGS 无需在每次优化迭代中求解 Hessian 线性系统，但开启推断仍需要计算协方差。

### 后端与交叉验证兼容性

下表所有组合均支持二元响应、可选截距、样本权重和 L2 正则化。直接拟合还支持无惩罚的 `C=0`；CV 搜索使用正数 `C` 候选值。

| 指定求解器 | 实际求解器 | NumPy CPU 拟合 / CV | CuPy CUDA 拟合 / CV | Torch CUDA 拟合 / CV |
|---|---|---|---|---|
| `"auto"` | `"irls"` | IRLS / 依次进行 IRLS 拟合 | IRLS / 批处理 IRLS | IRLS / 批处理 IRLS |
| `"irls"` | `"irls"` | IRLS / 依次进行 IRLS 拟合 | IRLS / 批处理 IRLS | IRLS / 批处理 IRLS |
| `"lbfgs"` | `"lbfgs"` | L-BFGS / 依次进行 L-BFGS 拟合 | L-BFGS / 依次进行 GPU L-BFGS 拟合 | L-BFGS / 依次进行 GPU L-BFGS 拟合 |

`device="cpu"`、`"cuda"` 和 `"torch"` 分别选择 NumPy、CuPy CUDA 和 Torch CUDA。显式指定的 GPU 后端不可用或拟合失败时会报错，不会静默回退到 CPU。只有 `device="auto"` 允许自动选择设备。另见[求解器与惩罚项兼容性](../guides/solver-penalty-matrix.md)。

CV 中的 `auto`/`irls` 保留 GPU 批处理方式。L-BFGS 在所选后端上依次拟合候选项，不使用批处理 IRLS 求解器。`gpu_cv_mixed_precision` 仅影响 GPU 批处理 IRLS；L-BFGS 使用 float64。选出 `C_` 后，使用同一指定求解器在全部数据上重拟合，`solver_` 记录实际解析值。不支持的求解器名称会引发 `ValueError`。

## 协方差与推断（Covariance/Inference）

推断默认使用大样本正态近似（z 统计口径），支持：

- `cov_type="nonrobust"`：经典信息矩阵协方差
- `cov_type="hc0"`：White/sandwich 稳健协方差
- `cov_type="hc1"`：HC0 加自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于杠杆值修正
- `cov_type="hc3"`：更保守的刀切法风格修正
- `cov_type="hac"`：Newey-West（Bartlett 核）自相关稳健协方差
- `hac_maxlags`：仅对 `cov_type="hac"` 生效

`_bse`、`_zvalues`、`_pvalues` 和 `_conf_int` 要求 `compute_inference=True`。似然、AIC、BIC、伪 R² 与 `converged_` 在 `compute_inference=False` 时仍可用；协方差相关字段不可用。

两种求解器在拟合得到的估计值处使用相同的协方差计算。`C` 为正数时，信息矩阵或夹心估计的外层矩阵包含岭惩罚的曲率。这些区间与检验不会自动修正收缩偏差，也不考虑通过 CV 选择 `C` 所带来的额外不确定性。切换求解器不会改变这一统计解释。CV 的推断结果通过最终 `estimator_` 获取。

## 参数（Parameters）

`LogisticRegression` 的完整构造参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合不受惩罚的截距 |
| `C` | `1.0` | 正数表示 L2 强度的倒数；`0` 关闭正则化 |
| `max_iter` | `100` | 所选求解器的最大迭代次数 |
| `tol` | `1e-4` | 正数停止阈值，具体含义随求解器变化，见上文 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | CPU 作业数设置；不会并行执行 Logit 求解或 CV 候选拟合 |
| `compute_inference` | `True` | 是否计算推断统计量 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `gpu_memory_cleanup` | `False` | 拟合后尽可能清理 CuPy 内存池 |
| `hac_maxlags` | `None` | HAC 最大滞后阶，默认按 Newey-West 风格经验规则选择 |
| `solver` | `"auto"` | `auto` / `irls` / `lbfgs`；`auto` 解析为 `irls` |

`LogisticRegressionCV` 的完整构造参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `Cs` | `None` | 正数 `C` 候选值；`None` 根据数据自动生成网格 |
| `n_Cs` | `100` | 自动生成的候选值数量 |
| `C_min_ratio` | `1e-3` | 自动网格中最小与最大 `C` 的比值 |
| `cv` | `5` | 未提供 `cv_splits` 时的折数 |
| `cv_splits` | `None` | 显式的 `(训练索引, 验证索引)` 对 |
| `fit_intercept` | `True` | 是否拟合不受惩罚的截距 |
| `max_iter` | `100` | 候选拟合与最终重拟合的迭代上限 |
| `tol` | `1e-4` | 所选求解器的停止阈值 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | CPU 作业数设置；不会并行执行候选拟合 |
| `compute_inference` | `True` | 是否为最终重拟合计算推断统计量 |
| `cov_type` | `"nonrobust"` | 最终重拟合的协方差类型，取值见上文 |
| `gpu_memory_cleanup` | `False` | 尽可能清理 CuPy 内存池 |
| `random_state` | `None` | 自动生成 CV 划分时的随机种子 |
| `gpu_cv_mixed_precision` | `True` | GPU 批处理 IRLS 的混合精度设置；L-BFGS 不使用此设置 |
| `solver` | `"auto"` | 候选拟合与最终重拟合所用的求解器 |

两种估计器均支持 `fit(X, y, sample_weight=None)`。`y` 为 0/1 二元响应；提供的样本权重必须有限、非负且总和为正。`LogisticRegressionCV` 不提供 `hac_maxlags` 参数，其最终重拟合的 HAC 使用默认滞后阶规则。

## 拟合与交叉验证示例

```python
import numpy as np
from statgpu.linear_model import LogisticRegression, LogisticRegressionCV

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
probability = 1 / (1 + np.exp(-(0.3 + X @ [0.8, -0.5, 0.2, 0.0])))
y = rng.binomial(1, probability)
weights = rng.uniform(0.5, 1.5, size=len(y))

model = LogisticRegression(
    solver="lbfgs", device="cpu", C=1.0,
    max_iter=300, tol=1e-7, cov_type="hc1",
).fit(X, y, sample_weight=weights)
print(model.get_params()["solver"], model.solver_, model.converged_)
print(model.predict_proba(X[:3]))

cv_model = LogisticRegressionCV(
    Cs=[0.1, 1.0, 10.0], cv=3, random_state=42,
    solver="lbfgs", device="cpu", max_iter=300, tol=1e-7,
    compute_inference=False,
).fit(X, y, sample_weight=weights)
print(cv_model.C_, cv_model.solver_, cv_model.estimator_.converged_)
print(cv_model.mean_loss_)
```

将 `solver` 改为 `"auto"` 或 `"irls"`，即可在相同数据上比较 IRLS。安装了受支持的 GPU 后端后，可将 `device` 改为 `"cuda"` 或 `"torch"`，其余示例及求解器选项同样适用。CV 选择平均验证集对数损失最小的候选项。`mean_loss_` 保存各候选值的平均损失，`cv_results_["loss_path"]` 保存逐折损失，`best_score_` 为所选平均损失的负数。

## 严格与近似模式的差别

当前接口未提供独立的 `strict/approx` 推断开关。稳健协方差类型（`hc*`/`hac`）的选择主要体现统计假设与计算成本之间的权衡。两种求解器都不会额外修正收缩偏差或模型选择后的推断。

## 输出（Outputs）

- `fit(X, y, sample_weight=None) -> self`
- 系数与拟合状态：`coef_`、`intercept_`、`n_iter_`、`solver_`、`converged_`
- 预测：`predict_proba(X)`、`predict(X)`、`predict_with_threshold(X, threshold)`
- 推断属性：`_bse`、`_zvalues`、`_pvalues`、`_conf_int`；使用 `summary()` 打印汇总
- 拟合与信息准则：`loglikelihood`、`loglikelihood_null`、`pseudo_rsquared`、`aic`、`bic`
- 分类指标：`accuracy`、`precision`、`recall`、`f1`、`auc`、`average_precision`
- CV：`C_`、`Cs_`、`mean_loss_`、`cv_results_`、`best_score_`、`cv_selected_device_`、`solver_` 及最终 `estimator_`；最终收敛状态为 `estimator_.converged_`

分类评估 API：
- `confusion_matrix`、`classification_table`
- `roc_curve`、`roc_auc_score`
- `precision_recall_curve`、`average_precision_score`
- `evaluate_classification`
- `statgpu.evaluation.evaluate_binary_classification`
- `plot_roc_curve`、`plot_precision_recall_curve`（依赖 `matplotlib`）

## 常见问题（FAQ）

- **如何拟合无正则 Logit MLE？**  
  使用 `C=0`；较大的正数 `C`（例如 `1e10`）则近似无正则拟合。发生完全分离时，有限的无正则 MLE 可能不存在。
- **为什么输出 z 统计量而不是 t？**  
  Logit 推断通常采用大样本正态近似。
- **评估方法是否直接在 GPU 上计算？**  
  核心预测与指标计算使用所选的受支持后端；绘图时转为 NumPy 数据。

## 外部验证（External Validation）

与 `statsmodels.Logit` 的对齐测试位于：

- `dev/tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`

`HC2/HC3/HAC` 三方产物见：
- `results/remote_covariance_full_compare_2026-04-10.json`

## 参考（References）

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
