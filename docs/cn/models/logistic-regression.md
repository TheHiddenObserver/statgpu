# LogisticRegression

> 语言: 中文  
> 最后更新: 2026-05-20  
> 页面定位: 模型文档  
> 切换: [English](../en/models/logistic-regression.md)

语言切换：[English](../en/models/logistic-regression.md)

## 概览（Overview）

`LogisticRegression` 提供二分类 Logit（IRLS + L2）训练与推断，支持 CPU/GPU。当前范围聚焦二分类；多分类、L1、elastic-net 暂未纳入主路径（见 `TO_DO.md`）。

## 路径（Path）

`statgpu.linear_model.LogisticRegression`

## 目标函数（Objective Function）

最小化二分类负对数似然与 L2 正则项（`C` 为正则强度倒数，`C` 越大正则越弱）。当 `C` 很大（如 `1e10`）时可近似无正则 MLE。

## 估计方程（Estimating Equation）

采用 IRLS/Newton/L-BFGS 求解，受 `max_iter` 与 `tol` 控制；`fit_intercept=True` 时联合估计截距。v23c (2026-05) 起，`solver="lbfgs"` 在各后端上正确支持 L2 惩罚。

## 协方差与推断（Covariance/Inference）

推断默认使用大样本正态近似（z 统计口径），支持：

- `cov_type="nonrobust"`：经典信息矩阵协方差
- `cov_type="hc0"`：White/sandwich 稳健
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `C` | `1.0` | 正则强度倒数（越大正则越弱） |
| `max_iter` | `100` | IRLS 最大迭代数 |
| `tol` | `1e-4` | 收敛阈值 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `True` | 是否计算推断统计 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.linear_model import LogisticRegression

# CPU
m_cpu = LogisticRegression(
    device="cpu",
    C=1e10,
    cov_type="hc1",
    compute_inference=True,
)
m_cpu.fit(X, y_binary)

# GPU
m_gpu = LogisticRegression(
    device="cuda",
    C=1e10,
    cov_type="hac",
    hac_maxlags=4,
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X_gpu, y_gpu)
```

## strict/approx 差异（strict/approx difference）

当前接口未暴露独立 `strict/approx` 开关。默认路径用于高一致性推断；GPU 与 CPU 在极小数值误差范围内可能存在差异。

## 输出（Outputs）

- `fit(X, y) -> self`
- 预测：`predict_proba(X)`、`predict(X)`、`predict_with_threshold(X, threshold)`
- 推断属性：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合与信息准则：`loglikelihood`, `loglikelihood_null`, `pseudo_rsquared`, `aic`, `bic`
- 常用属性：`coef_`, `intercept_`, `n_iter_`
- 汇总：`summary()`

分类评估 API：
- `confusion_matrix`、`classification_table`
- `roc_curve`、`roc_auc_score`
- `precision_recall_curve`、`average_precision_score`
- `evaluate_classification`
- `statgpu.evaluation.evaluate_binary_classification`
- `plot_roc_curve`、`plot_precision_recall_curve`（依赖 `matplotlib`）

## 常见问题（FAQ）

- **如何近似无正则 Logit MLE？**  
  使用较大 `C`（例如 `1e10`）。
- **为什么输出 z 统计量而不是 t？**  
  Logit 推断通常采用大样本正态近似。

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
