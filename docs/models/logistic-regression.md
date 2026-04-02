# LogisticRegression

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
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `fit_intercept`
- `C`
- `max_iter`
- `tol`
- `device`
- `compute_inference`
- `cov_type`：`nonrobust` / `hc0` / `hc1`
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=True)
m.fit(X, y_binary)
proba = m.predict_proba(X)
```

## 稳健协方差（HC0/HC1）

`LogisticRegression` 支持稳健协方差：

- `cov_type="nonrobust"`：经典信息矩阵协方差
- `cov_type="hc0"`：White/sandwich 稳健协方差
- `cov_type="hc1"`：在 `hc0` 基础上做自由度修正 `n/(n-k)`

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
    cov_type="hc1",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m.fit(X_gpu, y_gpu)
print(m._bse, m._pvalues, m._conf_int)
```

## 外部框架一致性

项目已补充与 `statsmodels.Logit` 的一致性测试（`HC0/HC1`，CPU+GPU）：

- `tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`

## 输出统计

- 系数：`intercept_`, `coef_`, `n_iter_`
- 推断：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合质量：`loglikelihood`, `loglikelihood_null`, `pseudo_rsquared`, `aic`, `bic`
- 分类指标：`accuracy`, `precision`, `recall`, `f1`
- 汇总：`summary()`

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict_proba(X)`：返回两列概率（0 类和 1 类）
- `predict(X)`：返回二分类标签
- 常用属性：`coef_`, `intercept_`, `n_iter_`, `aic`, `bic`

## 常见问题

- **Q: 想近似无正则 Logit MLE 怎么做？**  
  A: 可用较大 `C`（如 `1e10`）作为近似。
- **Q: 为什么是 z 统计量不是 t？**  
  A: Logit 推断通常采用大样本正态近似（z 统计口径）。

## 当前范围

- 目前聚焦二分类 Logit（IRLS + L2 路径）。
- 多分类 / L1 / elastic-net 仍在待办中（见 `TO_DO.md`）。
