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


# Shared mapping from the CV-selected backend to the final refit device.
device_path = "statgpu/linear_model/cv/_device.py"
replace_once(
    device_path,
    '''def validate_cv_sample_weight(sample_weight, n_samples):
    """Validate analytic CV weights before any grid or degenerate return."""
    if sample_weight is None:
        return None
    return validate_glm_sample_weight(sample_weight, n_samples)
''',
    '''def cv_refit_device(device, backend_name):
    """Return the final-fit device matching the backend used for CV.

    Explicit requests remain unchanged.  AUTO is pinned to the backend chosen
    from the design matrix so parameter selection and the final refit cannot
    silently use different GPU libraries.
    """
    device_name = normalize_cv_device(device)
    if device_name != Device.AUTO.value:
        return Device(device_name)
    mapping = {
        "numpy": Device.CPU,
        "cupy": Device.CUDA,
        "torch": Device.TORCH,
    }
    try:
        return mapping[str(backend_name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CV backend {backend_name!r}; expected numpy, cupy, or torch"
        ) from exc


def validate_cv_sample_weight(sample_weight, n_samples):
    """Validate analytic CV weights before any grid or degenerate return."""
    if sample_weight is None:
        return None
    return validate_glm_sample_weight(sample_weight, n_samples)
''',
)

# RidgeCV: resolve once for both selection and final refit.
ridge_path = "statgpu/linear_model/cv/_ridge_cv.py"
replace_once(
    ridge_path,
    "from ._device import resolve_cv_backend, validate_cv_sample_weight\n",
    "from ._device import (\n"
    "    cv_refit_device,\n"
    "    resolve_cv_backend,\n"
    "    validate_cv_sample_weight,\n"
    ")\n",
)
replace_once(
    ridge_path,
    '''        device_name = self._device

        # Run CV to select alpha
''',
    '''        device_name = self._device
        _, cv_backend_name, _, _, _, _ = resolve_cv_backend(device_name, X)
        refit_device = cv_refit_device(device_name, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Run CV to select alpha
''',
)
replace_once(
    ridge_path,
    "            device=self._device,\n",
    "            device=refit_device,\n",
)
replace_once(
    ridge_path,
    "        self.estimator_ = None\n\n    def fit(self, X, y, sample_weight=None):\n",
    "        self.estimator_ = None\n"
    "        self.cv_selected_device_ = None\n\n"
    "    def fit(self, X, y, sample_weight=None):\n",
)

# ElasticNetCV.
elastic_path = "statgpu/linear_model/cv/_elasticnet_cv.py"
replace_once(
    elastic_path,
    "from ._device import resolve_cv_backend, validate_cv_sample_weight\n",
    "from ._device import (\n"
    "    cv_refit_device,\n"
    "    resolve_cv_backend,\n"
    "    validate_cv_sample_weight,\n"
    ")\n",
)
replace_once(
    elastic_path,
    '''        device_request = self._device

        # Normalize l1_ratio to list
''',
    '''        device_request = self._device
        _, cv_backend_name, _, _, _, _ = resolve_cv_backend(device_request, X)
        refit_device = cv_refit_device(device_request, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Normalize l1_ratio to list
''',
)
replace_once(
    elastic_path,
    "            device=self._device,\n",
    "            device=refit_device,\n",
)
replace_once(
    elastic_path,
    "        self.estimator_ = None\n\n    def _fit_cv(self, X, y, sample_weight=None):\n",
    "        self.estimator_ = None\n"
    "        self.cv_selected_device_ = None\n\n"
    "    def _fit_cv(self, X, y, sample_weight=None):\n",
)

# LogisticRegressionCV.
logistic_path = "statgpu/linear_model/cv/_logistic_cv.py"
replace_once(
    logistic_path,
    "from ._device import resolve_cv_backend, validate_cv_sample_weight\n",
    "from ._device import (\n"
    "    cv_refit_device,\n"
    "    resolve_cv_backend,\n"
    "    validate_cv_sample_weight,\n"
    ")\n",
)
replace_once(
    logistic_path,
    '''        # Keep AUTO unresolved until resolve_cv_backend can inspect X.
        device_name = self._device

        # Run CV to select C
''',
    '''        # Keep AUTO unresolved until resolve_cv_backend can inspect X.
        device_name = self._device
        _, cv_backend_name, _, _, _, _ = resolve_cv_backend(device_name, X)
        refit_device = cv_refit_device(device_name, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Run CV to select C
''',
)
replace_once(
    logistic_path,
    "            device=self._device,\n",
    "            device=refit_device,\n",
)
replace_once(
    logistic_path,
    "        self.estimator_ = None\n\n    def fit(self, X, y, sample_weight=None):\n",
    "        self.estimator_ = None\n"
    "        self.cv_selected_device_ = None\n\n"
    "    def fit(self, X, y, sample_weight=None):\n",
)

# Tests.
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_cv_refit_device_pins_auto_to_selected_backend" in test_text:
    raise RuntimeError("v58 tests already present")
test_text += r'''


def test_cv_refit_device_pins_auto_to_selected_backend():
    from statgpu._config import Device
    from statgpu.linear_model.cv._device import cv_refit_device

    assert cv_refit_device("auto", "numpy") == Device.CPU
    assert cv_refit_device("auto", "cupy") == Device.CUDA
    assert cv_refit_device("auto", "torch") == Device.TORCH
    assert cv_refit_device("cpu", "torch") == Device.CPU
    assert cv_refit_device("cuda", "torch") == Device.CUDA
    with pytest.raises(ValueError, match="Unknown CV backend"):
        cv_refit_device("auto", "mystery")


@pytest.mark.parametrize(
    "module_name,class_name,selector_name,model_name,y,details",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            "Ridge",
            np.arange(6, dtype=np.float64),
            {
                "alpha": 0.5,
                "alphas": np.array([0.5]),
                "mse_path": np.array([[1.0]]),
                "mean_mse": np.array([1.0]),
            },
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            "LogisticRegression",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            {
                "C": 1.0,
                "Cs": np.array([1.0]),
                "loss_path": np.array([[0.5]]),
                "mean_loss": np.array([0.5]),
            },
        ),
    ],
)
def test_public_cv_auto_refit_uses_cv_selected_backend(
    monkeypatch,
    module_name,
    class_name,
    selector_name,
    model_name,
    y,
    details,
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    backend = object()
    monkeypatch.setattr(
        module,
        "resolve_cv_backend",
        lambda device, X: ("auto", "torch", backend, True, False, True),
    )
    monkeypatch.setattr(module, selector_name, lambda *args, **kwargs: details)
    observed = []

    class FakeModel:
        def __init__(self, *args, device=None, **kwargs):
            observed.append(device)
            self.coef_ = np.zeros(2)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

    monkeypatch.setattr(module, model_name, FakeModel)
    estimator = getattr(module, class_name)(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    estimator.fit(X, y)

    assert observed == [Device.TORCH]
    assert estimator.cv_selected_device_ == Device.TORCH


def test_elasticnet_cv_auto_refit_uses_cv_selected_backend(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module
    from statgpu._config import Device

    backend = object()
    monkeypatch.setattr(
        module,
        "resolve_cv_backend",
        lambda device, X: ("auto", "cupy", backend, True, True, False),
    )
    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.5])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *args, **kwargs: (0.5, 0.5, details),
    )
    observed = []

    class FakeElasticNet:
        def __init__(self, *args, device=None, **kwargs):
            observed.append(device)
            self.coef_ = np.zeros(2)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

    monkeypatch.setattr(module, "ElasticNet", FakeElasticNet)
    estimator = module.ElasticNetCV(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    estimator.fit(X, np.arange(6, dtype=np.float64))

    assert observed == [Device.CUDA]
    assert estimator.cv_selected_device_ == Device.CUDA
'''
test_path.write_text(test_text, encoding="utf-8")

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Pinned AUTO-mode RidgeCV, ElasticNetCV, and LogisticRegressionCV final refits to the backend selected during CV, preventing silent Torch/CuPy backend drift after parameter selection.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Pinned AUTO-mode RidgeCV, ElasticNetCV, and LogisticRegressionCV final refits to the backend selected during CV, preventing silent Torch/CuPy backend drift after parameter selection.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 将 AUTO 模式的 RidgeCV、ElasticNetCV 与 LogisticRegressionCV 最终重拟合固定到 CV 选参时使用的后端，避免选参后在 Torch 与 CuPy 之间静默漂移。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
