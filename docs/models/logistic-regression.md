# LogisticRegression

路径：`statgpu.linear_model.LogisticRegression`

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
