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


# Preserve the public AUTO request until the dedicated CV router sees X.
replace_once(
    "statgpu/linear_model/cv/_ridge_cv.py",
    "        device_name = self._get_compute_device().value\n",
    "        device_name = self._device\n",
)

replace_once(
    "statgpu/linear_model/cv/_elasticnet_cv.py",
    "        compute_device = self._get_compute_device()\n",
    "        device_request = self._device\n",
)
replace_once(
    "statgpu/linear_model/cv/_elasticnet_cv.py",
    "            device=compute_device,\n",
    "            device=device_request,\n",
)

logistic_path = "statgpu/linear_model/cv/_logistic_cv.py"
insert_after = '''from ._device import resolve_cv_backend, validate_cv_sample_weight
'''
helper = '''from ._device import resolve_cv_backend, validate_cv_sample_weight


def _validate_binary_cv_response(y):
    """Validate a strict 0/1 response without copying GPU arrays to NumPy."""
    from statgpu.glm_core._logistic import LogisticLoss

    values = LogisticLoss().validate_response(y)
    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        valid = torch.all((values == 0) | (values == 1))
    elif module.startswith("cupy"):
        import cupy as cp

        valid = cp.all((values == 0) | (values == 1))
    else:
        valid = np.all((values == 0) | (values == 1))
    if not bool(valid.item() if hasattr(valid, "item") else valid):
        raise ValueError("LogisticRegressionCV requires binary y (0 or 1)")
    return values
'''
replace_once(logistic_path, insert_after, helper)

replace_once(
    logistic_path,
    '''        # Validate y is binary
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        unique_y = np.unique(y_arr)
        if not np.all(np.isin(unique_y, [0.0, 1.0])):
            raise ValueError(
                f"LogisticRegressionCV requires binary y (0 or 1), "
                f"got unique values: {unique_y[:10]}"
            )

        device_name = self._get_compute_device().value
''',
    '''        # Preserve response residency; only a scalar validity decision syncs.
        _validate_binary_cv_response(y)

        # Keep AUTO unresolved until resolve_cv_backend can inspect X.
        device_name = self._device
''',
)

# Tests.
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_public_dedicated_cv_preserves_auto_device_request" in test_text:
    raise RuntimeError("v57 tests already present")
test_text += r'''


@pytest.mark.parametrize(
    "module_name,class_name,selector_name,y",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            np.arange(6, dtype=np.float64),
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            "_select_elasticnet_params_cv",
            np.arange(6, dtype=np.float64),
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        ),
    ],
)
def test_public_dedicated_cv_preserves_auto_device_request(
    monkeypatch, module_name, class_name, selector_name, y
):
    import importlib
    from statgpu.linear_model.cv._device import normalize_cv_device

    module = importlib.import_module(module_name)
    observed = []

    def probe(*args, **kwargs):
        observed.append(kwargs["device"])
        raise RuntimeError("device request probe")

    monkeypatch.setattr(module, selector_name, probe)
    estimator = getattr(module, class_name)(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="device request probe"):
        estimator.fit(X, y)

    assert len(observed) == 1
    assert normalize_cv_device(observed[0]) == "auto"


def test_logistic_cv_binary_validation_preserves_torch_response():
    torch = pytest.importorskip("torch")
    from statgpu.linear_model.cv._logistic_cv import _validate_binary_cv_response

    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    assert _validate_binary_cv_response(y) is y
    with pytest.raises(ValueError, match="binary y"):
        _validate_binary_cv_response(
            torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        )


def test_auto_cv_router_prefers_gpu_resident_input_backend(monkeypatch):
    import statgpu.linear_model.cv._device as device_mod

    class FakeTorchCudaArray:
        __module__ = "torch"
        device = "cuda:0"

    calls = []

    class FakeBackend:
        pass

    def backend_probe(*, backend, device):
        calls.append((backend, device))
        return FakeBackend()

    monkeypatch.setattr(device_mod, "get_backend", backend_probe)
    resolved = device_mod.resolve_cv_backend("auto", FakeTorchCudaArray())
    assert resolved[0] == "auto"
    assert resolved[1] == "torch"
    assert resolved[3] is True
    assert calls == [("torch", "cuda")]
'''
test_path.write_text(test_text, encoding="utf-8")

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Preserved `device='auto'` through public RidgeCV, ElasticNetCV, and LogisticRegressionCV dispatch so GPU-resident inputs retain their owning backend; LogisticRegressionCV now validates 0/1 responses without a full GPU-to-CPU copy.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Preserved `device='auto'` through public RidgeCV, ElasticNetCV, and LogisticRegressionCV dispatch so GPU-resident inputs retain their owning backend; LogisticRegressionCV now validates 0/1 responses without a full GPU-to-CPU copy.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 在公开 RidgeCV、ElasticNetCV 与 LogisticRegressionCV 调度中保留 `device='auto'`，使 GPU 常驻输入继续使用其原有后端；LogisticRegressionCV 现可在不完整复制到 CPU 的情况下验证 0/1 响应。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
