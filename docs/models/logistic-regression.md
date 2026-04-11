# LogisticRegression

> 语言: 中文  
> 最后更新: 2026-04-10  
> 页面定位: 模型文档  
> 切换: [English](../en/models/logistic-regression.md)

语言切换：[English](../en/models/logistic-regression.md)

路径：`statgpu.linear_model.LogisticRegression`

## 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `C` | `1.0` | 正则强度倒数（越大正则越弱） |
| `max_iter` | `100` | IRLS 最大迭代数 |
| `tol` | `1e-4` | 收敛阈值 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | 是否计算推断统计 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶；为空时使用 Newey-West 风格默认规则 |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `fit_intercept`
- `C`
- `max_iter`
- `tol`
- `device`
- `compute_inference`
- `cov_type`：`nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
- `hac_maxlags`：仅 `cov_type="hac"` 生效；未指定时按样本规模自动推断
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=True)
m.fit(X, y_binary)
proba = m.predict_proba(X)
```

## 稳健协方差（HC0/HC1/HC2/HC3/HAC）

`LogisticRegression` 支持稳健协方差：

- `cov_type="nonrobust"`：经典信息矩阵协方差
- `cov_type="hc0"`：White/sandwich 稳健协方差
- `cov_type="hc1"`：在 `hc0` 基础上做自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 的稳健修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差，可通过 `hac_maxlags` 控制滞后阶

### CPU 示例

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(
    device="cpu",
    C=1e10,              # 近似无正则 MLE
    cov_type="hc1",
    compute_inference=True,
)
m.fit(X, y_binary)
print(m._bse, m._pvalues, m._conf_int)
```

### GPU 示例（纯 GPU 推断路径）

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(
    device="cuda",
    C=1e10,
  cov_type="hac",
  hac_maxlags=4,
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m.fit(X_gpu, y_gpu)
print(m._bse, m._pvalues, m._conf_int)
```

## 外部框架一致性

项目已补充与 `statsmodels.Logit` 的一致性测试（`HC0/HC1`，CPU+GPU）：

- `dev/tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`

`HC2/HC3/HAC` 的三方（`statsmodels` / `statgpu CPU` / `statgpu GPU`）统一产物见：

- `results/remote_covariance_full_compare_2026-04-10.json`

## 输出统计

- 系数：`intercept_`, `coef_`, `n_iter_`
- 推断：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合质量：`loglikelihood`, `loglikelihood_null`, `pseudo_rsquared`, `aic`, `bic`
- 分类指标：`accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`
- 汇总：`summary()`

## 分类评估 API

- `predict_with_threshold(X, threshold=0.5)`：自定义阈值预测标签
- `confusion_matrix(X, y, threshold=0.5)`：返回 `[[TN, FP], [FN, TP]]`
- `classification_table(X, y, threshold=0.5)`：返回准确率/精确率/召回率/F1/特异度等字典
- `roc_curve(X, y)`：返回 `fpr, tpr, thresholds`
- `roc_auc_score(X, y)`：返回 ROC AUC
- `precision_recall_curve(X, y)`：返回 `precision, recall, thresholds`
- `average_precision_score(X, y)`：返回 Average Precision（AP）
- `evaluate_classification(X, y, threshold=0.5, include_curves=True)`：一次性批量返回 confusion/table/ROC/PR/AUC/AP（单次概率计算）
- `statgpu.evaluation.evaluate_binary_classification(y_true, y_score, threshold=0.5, include_curves=True, backend='auto')`：对外部概率输入进行一次性批量评估（无需绑定模型对象）
- `plot_roc_curve(X, y, ax=None, label=None)`：绘制 ROC 曲线，返回 matplotlib 轴对象
- `plot_precision_recall_curve(X, y, ax=None, label=None)`：绘制 PR 曲线，返回 matplotlib 轴对象

外部概率评估示例：

```python
from statgpu import evaluate_binary_classification

# y_true: 0/1 标签；y_score: 正类概率
out = evaluate_binary_classification(
  y_true,
  y_score,
  threshold=0.5,
  include_curves=True,
  backend="auto",  # auto / numpy / cupy / torch
)

print(out["classification_table"])
print(out["roc_auc"], out["average_precision"])
```

当 `device="cuda"` 时，`confusion_matrix` / `classification_table` / `roc_curve` /
`roc_auc_score` / `precision_recall_curve` / `average_precision_score` 默认走 GPU
计算路径，不进行 GPU 到 CPU 的数组转换（绘图接口会在绘图前转换为 NumPy）。

> 绘图接口依赖 `matplotlib`（可选依赖）。

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict_proba(X)`：返回两列概率（0 类和 1 类）
- `predict(X)`：返回二分类标签
- `predict_with_threshold(X, threshold)`：按自定义阈值返回标签
- 常用属性：`coef_`, `intercept_`, `n_iter_`, `aic`, `bic`

## 常见问题

- **Q: 想近似无正则 Logit MLE 怎么做？**  
  A: 可用较大 `C`（如 `1e10`）作为近似。
- **Q: 为什么是 z 统计量不是 t？**  
  A: Logit 推断通常采用大样本正态近似（z 统计口径）。

## 当前范围

- 目前聚焦二分类 Logit（IRLS + L2 路径）。
- 多分类 / L1 / elastic-net 仍在待办中（见 `TO_DO.md`）。
