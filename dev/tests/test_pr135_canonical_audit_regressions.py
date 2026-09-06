import warnings

import numpy as np
import pytest

import statgpu.linear_model.wrappers._lasso as lasso_impl
from statgpu._config import Device, set_device
from statgpu.linear_model import Lasso, LassoCV


def _regression_data(seed=808, n=36, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.6, 0.25, 0.0])
    y = X @ beta + rng.normal(scale=0.15, size=n)
    return X, y


@pytest.mark.parametrize(
    ("resolved_device", "backend_name"),
    [(Device.CUDA, "cupy"), (Device.TORCH, "torch")],
)
def test_lassocv_auto_constructor_preserves_globally_resolved_backend(
    monkeypatch, resolved_device, backend_name
):
    X, y = _regression_data()
    sample_weight = np.ones(X.shape[0], dtype=np.float64)
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="auto",
        compute_inference=False,
        random_state=13,
    )
    assert model._device is Device.AUTO

    # Exercise the real public global-device resolution while mocking only the
    # physical array conversion so the regression stays CPU-hosted.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        set_device(resolved_device.value)
    converted = {}

    def fake_to_array(value, device=None, backend=None):
        marker = object()
        converted[id(value)] = (marker, device, backend)
        return marker

    monkeypatch.setattr(model, "_to_array", fake_to_array)

    def synthetic_selection(X_cv, y_cv, **kwargs):
        assert X_cv is converted[id(X)][0]
        assert y_cv is converted[id(y)][0]
        assert kwargs["sample_weight"] is converted[id(sample_weight)][0]
        assert kwargs["device"] == resolved_device.value
        assert kwargs["cpu_solver"] == "fista"
        return {
            "alpha": 0.03,
            "alphas": np.asarray([0.03, 0.08], dtype=np.float64),
            "mse_path": np.asarray(
                [[0.30, 0.31, 0.29], [0.42, 0.40, 0.41]], dtype=np.float64
            ),
            "mean_mse": np.asarray([0.30, 0.41], dtype=np.float64),
        }

    def successful_refit(self, *args, **kwargs):
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(lasso_impl, "_select_lasso_alpha_cv", synthetic_selection)
    monkeypatch.setattr(Lasso, "fit", successful_refit)

    try:
        assert model._get_compute_device() is resolved_device
        model.fit(X, y, sample_weight=sample_weight)

        assert model.cv_solver_ == "fista"
        for original in (X, y, sample_weight):
            _, converted_device, converted_backend = converted[id(original)]
            assert converted_device is resolved_device
            assert converted_backend == backend_name
    finally:
        set_device("auto")


@pytest.mark.parametrize("method_alias", ["glmnet_cv", "glmnet.cv"])
def test_lassocv_glmnet_alias_rejects_explicit_fista_on_cpu(method_alias):
    X, y = _regression_data()
    model = LassoCV(
        alphas=[0.03, 0.08],
        cv=3,
        device="cpu",
        method=method_alias,
        cv_solver="fista",
        compute_inference=False,
    )

    # Public raw constructor identity is retained for sklearn compatibility,
    # while the runtime normalized value drives solver semantics.
    assert model.method == method_alias
    assert model._method == "glmnet"
    with pytest.raises(ValueError, match="method='glmnet'"):
        model.fit(X, y)


@pytest.mark.parametrize("method_alias", ["glmnet_cv", "glmnet.cv"])
def test_lassocv_glmnet_alias_reports_coordinate_descent_for_auto(method_alias):
    model = LassoCV(device="cpu", method=method_alias, cv_solver="auto")
    assert model._resolve_cv_solver("cpu") == "coordinate_descent"


def test_lasso_cv_selector_validates_torch_inputs_on_torch_backend(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class FakeBackend:
        def asarray(self, value, dtype=None):
            return value

    def fake_get_backend(*, backend="auto", device="auto"):
        calls.append((backend, device))
        return FakeBackend()

    monkeypatch.setattr(lasso_impl, "get_backend", fake_get_backend)
    details = lasso_impl._select_lasso_alpha_cv(
        torch.ones((3, 2)),
        torch.ones(3),
        sample_weight=torch.ones(3),
        alphas=[0.1],
        cv_folds=2,
        device="torch",
        return_details=True,
    )

    assert details["alpha"] == pytest.approx(0.1)
    assert calls == [("torch", "cuda")]


def test_lasso_bilingual_docs_cover_complete_constructor_inventory():
    import inspect
    from pathlib import Path

    public_params = [
        name for name in inspect.signature(Lasso.__init__).parameters if name != "self"
    ]
    for path in (
        Path("docs/en/models/lasso.md"),
        Path("docs/cn/models/lasso.md"),
    ):
        text = path.read_text(encoding="utf-8")
        missing = [name for name in public_params if f"`{name}`" not in text]
        assert missing == [], f"{path} missing public Lasso parameters: {missing}"


def test_lassocv_bilingual_cache_docs_match_current_key_contract():
    from pathlib import Path

    stale_fragments = (
        "_LASSO_CV_ALPHA_CACHE_MAXSIZE = 16",
        "first 5 indices per fold",
        "每 fold 前 5 个 index",
        "sample_weight_shape",
        "data_digest -- from `_hash_data`",
        "data_digest — 来自 `_hash_data`",
        "memory address changes",
        "内存地址变化",
    )
    required_fragments = (
        "_array_identity_token",
        "_make_lasso_cv_auto_cache_key",
        "gpu_cv_mixed_precision",
        "STATGPU_LASSO_CV_CACHE_SIZE",
        "64",
    )
    for path in (
        Path("docs/en/guides/cross-validation.md"),
        Path("docs/cn/guides/cross-validation.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert all(fragment not in text for fragment in stale_fragments)
        assert all(fragment in text for fragment in required_fragments)
