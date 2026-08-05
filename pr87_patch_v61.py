from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:180]!r}"
        )
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_section(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"{path}: section markers are not unique")
    lo = text.index(start)
    hi = text.index(end, lo)
    p.write_text(text[:lo] + replacement + text[hi:], encoding="utf-8")


# Standalone ElasticNet must expose the inference contract already implemented
# by PenalizedLinearRegression.
wrapper = "statgpu/linear_model/wrappers/_elasticnet.py"
replace_once(
    wrapper,
    '''        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
    ):
''',
    '''        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ):
''',
)
replace_once(
    wrapper,
    '''            gpu_memory_cleanup=gpu_memory_cleanup,
            stopping=stopping,
        )
''',
    '''            gpu_memory_cleanup=gpu_memory_cleanup,
            stopping=stopping,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
        )
''',
)

# The CV-selected final model must honor the public compute_inference flag.
cv_path = "statgpu/linear_model/cv/_elasticnet_cv.py"
replace_once(
    cv_path,
    '''            fit_intercept=self._fit_intercept,
            device=refit_device,
        )
''',
    '''            fit_intercept=self._fit_intercept,
            device=refit_device,
            n_jobs=self.n_jobs,
            compute_inference=self._compute_inference_enabled,
            inference_method="debiased",
        )
''',
)

# Tests: actual CPU inference, CV propagation, constructor contract, and
# physical GPU matrix entries (skipped on hosted CPU runners).
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_elasticnet_wrapper_cpu_debiased_inference_contract" in test_text:
    raise RuntimeError("v61 tests already present")
test_text += r'''


def _elasticnet_inference_fixture(seed=20260805):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, 3))
    beta = np.array([1.4, -0.9, 0.65])
    y = 0.35 + X @ beta + rng.normal(scale=0.35, size=X.shape[0])
    return X.astype(np.float64), y.astype(np.float64)


def _assert_elasticnet_inference_contract(model, n_features=3):
    n_params = n_features + 1
    assert model._compute_inference_enabled is True
    assert model._inference_result is not None
    assert model._inference_result.method == "debiased"
    assert np.asarray(model._params).shape == (n_params,)
    assert np.asarray(model._bse).shape == (n_params,)
    assert np.asarray(model._pvalues).shape == (n_params,)
    assert np.asarray(model._conf_int).shape == (n_params, 2)
    assert np.all(np.isfinite(np.asarray(model._params)))
    assert np.all(np.isfinite(np.asarray(model._bse)))
    assert np.all(np.isfinite(np.asarray(model._pvalues)))
    assert np.all(np.isfinite(np.asarray(model._conf_int)))
    assert model.summary() is not None


def test_elasticnet_wrapper_cpu_debiased_inference_contract():
    from statgpu.linear_model import ElasticNet

    X, y = _elasticnet_inference_fixture()
    model = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=2000,
        tol=1e-7,
        device="cpu",
        compute_inference=True,
        inference_method="debiased",
    ).fit(X, y)

    _assert_elasticnet_inference_contract(model)
    params = model.get_params(deep=False)
    assert params["compute_inference"] is True
    assert params["inference_method"] == "debiased"
    assert params["cov_type"] == "nonrobust"
    assert params["hac_maxlags"] is None


def test_elasticnet_cv_compute_inference_runs_on_final_refit():
    from statgpu.linear_model import ElasticNetCV

    X, y = _elasticnet_inference_fixture(seed=20260806)
    model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=1500,
        tol=1e-7,
        device="cpu",
        compute_inference=True,
        random_state=17,
    ).fit(X, y)

    assert model.estimator_ is not None
    _assert_elasticnet_inference_contract(model.estimator_)
    assert model.summary() is not None


def test_elasticnet_cv_passes_inference_flag_to_final_model(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module

    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.02])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *args, **kwargs: (0.02, 0.5, details),
    )
    observed = []

    class FakeElasticNet:
        def __init__(self, *args, **kwargs):
            observed.append(kwargs)
            self.coef_ = np.zeros(3)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return np.zeros(int(X.shape[0]))

    monkeypatch.setattr(module, "ElasticNet", FakeElasticNet)
    X, y = _elasticnet_inference_fixture(seed=20260807)
    model = module.ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        device="cpu",
        compute_inference=True,
        n_jobs=3,
    ).fit(X, y)

    assert len(observed) == 1
    assert observed[0]["compute_inference"] is True
    assert observed[0]["inference_method"] == "debiased"
    assert observed[0]["n_jobs"] == 3
    assert model.estimator_ is not None


def test_torch_cuda_elasticnet_inference_contract():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import ElasticNet, ElasticNetCV

    X_np, y_np = _elasticnet_inference_fixture(seed=20260808)
    X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")

    direct = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=1200,
        tol=1e-6,
        device="torch",
        compute_inference=True,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(direct)

    cv_model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=800,
        tol=1e-6,
        device="torch",
        compute_inference=True,
        random_state=19,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(cv_model.estimator_)


def _require_physical_cupy_v61():
    cp = pytest.importorskip("cupy")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"requires a physical CuPy CUDA backend: {exc}")
    if count < 1:
        pytest.skip("requires a physical CuPy CUDA backend")
    return cp


def test_cupy_elasticnet_inference_contract():
    cp = _require_physical_cupy_v61()
    from statgpu.linear_model import ElasticNet, ElasticNetCV

    X_np, y_np = _elasticnet_inference_fixture(seed=20260809)
    X = cp.asarray(X_np)
    y = cp.asarray(y_np)

    direct = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=1200,
        tol=1e-6,
        device="cuda",
        compute_inference=True,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(direct)

    cv_model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=800,
        tol=1e-6,
        device="cuda",
        compute_inference=True,
        random_state=23,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(cv_model.estimator_)
'''
test_path.write_text(test_text, encoding="utf-8")

# Model documentation: replace stale planned/unsupported claims.
en_section = '''## Covariance/Inference

`ElasticNet` is estimation-only by default. Set `compute_inference=True` to run
post-fit inference through the shared penalized-linear inference engine. The
default `inference_method="debiased"` uses nodewise Lasso to construct a
bias-corrected estimator, standard errors, z statistics, p-values, and 95%
confidence intervals. `summary()` is available after inference succeeds.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `compute_inference` | `False` | Enable post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where the selected inference method supports HAC |

Debiased inference is implemented for NumPy, CuPy, and Torch fitting paths. CPU
validation is part of the hosted test suite; physical CUDA validation for CuPy
and Torch remains a required remote gate for each exact release candidate.
Post-selection OLS is a heuristic and does not provide valid selective-inference
coverage. Inference is conditional on the selected regularization parameters
and does not alter the fitted penalized coefficients.

For `ElasticNetCV`, `compute_inference=True` applies inference only to the final
full-data refit after alpha and `l1_ratio` have been selected. Fold models remain
estimation-only.

'''
replace_section(
    "docs/en/models/elastic-net.md",
    "## Covariance/Inference\n",
    "## strict/approx difference\n",
    en_section,
)

cn_section = '''## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，将通过共享的
penalized-linear 推断引擎执行拟合后推断。默认
`inference_method="debiased"` 使用 nodewise Lasso 构造偏误校正估计量、标准误、
z 统计量、p 值与 95% 置信区间；推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

NumPy、CuPy 与 Torch 拟合路径均已实现 debiased 推断。CPU 验证属于托管测试套件；
每个精确发布候选仍必须通过 CuPy 与 Torch 的物理 CUDA 远程验证。
Post-selection OLS 只是启发式方法，不保证有效的选择后覆盖率。推断以已选定的正则化
参数为条件，不会改变原 penalized coefficient。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio`
选定后的全数据最终重拟合；各折模型仍仅用于估计和评分。

'''
replace_section(
    "docs/cn/models/elastic-net.md",
    "## 协方差/推断\n",
    "## strict/approx 区别\n",
    cn_section,
)

# Update parameter tables in model docs.
replace_once(
    "docs/en/models/elastic-net.md",
    '''| `gpu_memory_cleanup` | `False` | Clean GPU memory after fit (CuPy only) |
''',
    '''| `gpu_memory_cleanup` | `False` | Clean GPU memory after fit (CuPy only) |
| `compute_inference` | `False` | Compute post-fit coefficient inference |
| `inference_method` | `"debiased"` | Debiased, post-selection OLS, or bootstrap inference |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where supported |
''',
)
replace_once(
    "docs/cn/models/elastic-net.md",
    '''| `gpu_memory_cleanup` | `False` | 拟合后清理 GPU 内存（仅 CuPy） |
''',
    '''| `gpu_memory_cleanup` | `False` | 拟合后清理 GPU 内存（仅 CuPy） |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | Debiased、post-selection OLS 或 bootstrap 推断 |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |
''',
)

# CV guide must describe the public flag it already exposes.
replace_once(
    "docs/en/guides/cross-validation.md",
    '''| `n_alphas` | int | `100` | Number of alphas. |

#### PenalizedGLM_CV-Specific
''',
    '''| `n_alphas` | int | `100` | Number of alphas. |
| `compute_inference` | bool | `False` | Run debiased inference on the final full-data ElasticNet refit. |

Fold fits remain estimation-only; inference is computed only after the selected
`alpha` and `l1_ratio` are refit on all observations.

#### PenalizedGLM_CV-Specific
''',
)
replace_once(
    "docs/cn/guides/cross-validation.md",
    '''| `n_alphas` | int | `100` | Alpha 数量。 |

### PenalizedGLM_CV 专用
''',
    '''| `n_alphas` | int | `100` | Alpha 数量。 |
| `compute_inference` | bool | `False` | 对最终全数据 ElasticNet 重拟合执行 debiased 推断。 |

各折拟合仍仅用于估计；只有在所选 `alpha` 与 `l1_ratio` 使用全部观测重拟合后才计算推断。

### PenalizedGLM_CV 专用
''',
)

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Completed the public ElasticNet inference contract: the standalone wrapper now exposes and forwards inference options, and ElasticNetCV honors `compute_inference=True` on its final full-data refit with NumPy/CuPy/Torch matrix tests.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Completed the public ElasticNet inference contract: the standalone wrapper now exposes and forwards inference options, and ElasticNetCV honors `compute_inference=True` on its final full-data refit with NumPy/CuPy/Torch matrix tests.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 完成公开 ElasticNet 推断契约：独立 wrapper 现暴露并透传推断选项，ElasticNetCV 的最终全数据重拟合会真实执行 `compute_inference=True`，并补充 NumPy/CuPy/Torch 矩阵测试。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
