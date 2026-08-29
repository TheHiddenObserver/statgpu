from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


# 1) Shared concrete-device CuPy transfer primitive.
replace_once(
    "statgpu/backends/_utils.py",
    '''    raise ValueError(f"Unsupported backend: {backend_name}")\n\n\ndef _to_numpy(x):\n''',
    '''    raise ValueError(f"Unsupported backend: {backend_name}")\n\n\ndef _cupy_asarray_on_device(value, target_device: int, dtype=None):\n    """Return a CuPy array on one concrete CUDA device.\n\n    CuPy may preserve an existing native array when ``asarray`` can avoid a\n    copy.  When the source lives on another GPU, explicitly copy it while the\n    target device is current so downstream arithmetic cannot mix devices.\n    """\n    import cupy as cp\n\n    target_device = int(target_device)\n    with cp.cuda.Device(target_device):\n        if (\n            isinstance(value, cp.ndarray)\n            and int(value.device.id) != target_device\n        ):\n            value = cp.copy(value)\n        return cp.asarray(value, dtype=dtype)\n\n\ndef _to_numpy(x):\n''',
)

# 2) Shared Gaussian inference uses the concrete-device transfer primitive.
replace_once(
    "statgpu/linear_model/_gaussian_inference.py",
    '''from statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n''',
    '''from statgpu.backends._array_ops import _linalg_exception_is_rank_failure\nfrom statgpu.backends._utils import _cupy_asarray_on_device\n''',
)
replace_once(
    "statgpu/linear_model/_gaussian_inference.py",
    '''        if target_device is None:\n            target_device = int(cp.cuda.runtime.getDevice())\n        with cp.cuda.Device(target_device):\n            return cp.asarray(value, dtype=target_dtype or cp.float64)\n''',
    '''        if target_device is None:\n            target_device = int(cp.cuda.runtime.getDevice())\n        return _cupy_asarray_on_device(\n            value, target_device, dtype=target_dtype or cp.float64\n        )\n''',
)

# 3) PGLM fit freezes native fitted parameters through Gaussian inference and
# aligns side arrays to the actual X device.
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''from statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n''',
    '''from statgpu.backends._array_ops import _linalg_exception_is_rank_failure\nfrom statgpu.backends._utils import _cupy_asarray_on_device\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        self._inference_precomputed = False\n        self._precomputed_gaussian_state = None\n        self._clear_inference_state()\n''',
    '''        self._inference_precomputed = False\n        self._precomputed_gaussian_state = None\n        self._native_fit_coef = None\n        self._native_fit_intercept = None\n        self._clear_inference_state()\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''            with cp.cuda.Device(cupy_device_id):\n                y_arr = cp.asarray(y_arr)\n                if _sw_arr is not None:\n                    _sw_arr = cp.asarray(_sw_arr)\n''',
    '''            y_arr = _cupy_asarray_on_device(y_arr, cupy_device_id)\n            if _sw_arr is not None:\n                _sw_arr = _cupy_asarray_on_device(_sw_arr, cupy_device_id)\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)\n        self._fitted = True\n''',
    '''        self._compute_post_fit_gaussian_inference(X, y, sample_weight=_sw_arr)\n        # Numerical inference may allocate after the solver's own cleanup.\n        # Honor gpu_memory_cleanup at the true end of the fit transaction.\n        if backend_name == "cupy":\n            self._cleanup_cuda_memory()\n        elif backend_name == "torch":\n            self._cleanup_torch_memory()\n        self._fitted = True\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        params_np = _to_numpy(params)\n        self.n_iter_ = n_iter\n        if self._effective_intercept:\n            self.coef_ = params_np[:p]\n            self.intercept_ = float(params_np[p])\n            self._params = np.concatenate([[self.intercept_], self.coef_])\n        else:\n            self.coef_ = params_np.copy()\n            self.intercept_ = 0.0\n            self._params = self.coef_.copy()\n        self._df_resid = self._nobs - (\n            X_arr.shape[1] + (1 if self._effective_intercept else 0)\n        )\n''',
    '''        defer_gaussian_reporting = (\n            backend_name in ("cupy", "torch")\n            and self._compute_inference_enabled\n            and self.loss == "squared_error"\n            and str(getattr(self._penalty, "name", self.penalty)).lower() == "l2"\n        )\n        self.n_iter_ = n_iter\n        if defer_gaussian_reporting:\n            if self._effective_intercept:\n                self._native_fit_coef = params[:p]\n                self._native_fit_intercept = params[p]\n            else:\n                self._native_fit_coef = params\n                self._native_fit_intercept = None\n            self.coef_ = None\n            self.intercept_ = None\n            self._params = None\n        else:\n            params_np = _to_numpy(params)\n            if self._effective_intercept:\n                self.coef_ = params_np[:p]\n                self.intercept_ = float(params_np[p])\n                self._params = np.concatenate([[self.intercept_], self.coef_])\n            else:\n                self.coef_ = params_np.copy()\n                self.intercept_ = 0.0\n                self._params = self.coef_.copy()\n        self._df_resid = self._nobs - (\n            X_arr.shape[1] + (1 if self._effective_intercept else 0)\n        )\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        params_np = _to_numpy(params)\n        self.n_iter_ = n_iter\n        if self._effective_intercept:\n            self.intercept_ = float(params_np[0])\n            self.coef_ = params_np[1:]\n            self._params = np.concatenate([[self.intercept_], self.coef_])\n        else:\n            self.intercept_ = 0.0\n            self.coef_ = params_np.copy()\n            self._params = self.coef_.copy()\n        self._df_resid = self._nobs - (\n            X_arr.shape[1] + (1 if self._effective_intercept else 0)\n        )\n''',
    '''        defer_gaussian_reporting = (\n            backend_name in ("cupy", "torch")\n            and self._compute_inference_enabled\n            and self.loss == "squared_error"\n            and str(getattr(self._penalty, "name", self.penalty)).lower() == "l2"\n        )\n        self.n_iter_ = n_iter\n        if defer_gaussian_reporting:\n            if self._effective_intercept:\n                self._native_fit_intercept = params[0]\n                self._native_fit_coef = params[1:]\n            else:\n                self._native_fit_intercept = None\n                self._native_fit_coef = params\n            self.coef_ = None\n            self.intercept_ = None\n            self._params = None\n        else:\n            params_np = _to_numpy(params)\n            if self._effective_intercept:\n                self.intercept_ = float(params_np[0])\n                self.coef_ = params_np[1:]\n                self._params = np.concatenate([[self.intercept_], self.coef_])\n            else:\n                self.intercept_ = 0.0\n                self.coef_ = params_np.copy()\n                self._params = self.coef_.copy()\n        self._df_resid = self._nobs - (\n            X_arr.shape[1] + (1 if self._effective_intercept else 0)\n        )\n''',
)

# 4) The post-fit router consumes native GPU parameters and creates the public
# NumPy coefficient/intercept snapshot only after numerical inference.
replace_once(
    "statgpu/linear_model/penalized/_base.py",
    '''        self._inference_precomputed = False\n        self._precomputed_gaussian_state = None\n        # Simultaneous inference state\n''',
    '''        self._inference_precomputed = False\n        self._precomputed_gaussian_state = None\n        self._native_fit_coef = None\n        self._native_fit_intercept = None\n        # Simultaneous inference state\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_base.py",
    '''        def _run_gaussian_inference_on_fit_device():\n            state = build_gaussian_fit_state(\n                X,\n                y,\n                self.coef_,\n                self.intercept_,\n                self._effective_intercept,\n''',
    '''        if backend_name in ("cupy", "torch"):\n            coef_for_inference = getattr(self, "_native_fit_coef", None)\n            intercept_for_inference = getattr(self, "_native_fit_intercept", None)\n            if coef_for_inference is None:\n                raise RuntimeError(\n                    "Gaussian GPU inference requires native fitted parameters; "\n                    "refusing a reporting-array host round trip."\n                )\n            if self._effective_intercept and intercept_for_inference is None:\n                raise RuntimeError(\n                    "Gaussian GPU inference is missing the native fitted intercept."\n                )\n        else:\n            coef_for_inference = self.coef_\n            intercept_for_inference = self.intercept_\n\n        def _run_gaussian_inference_on_fit_device():\n            state = build_gaussian_fit_state(\n                X,\n                y,\n                coef_for_inference,\n                intercept_for_inference if self._effective_intercept else 0.0,\n                self._effective_intercept,\n''',
)
replace_once(
    "statgpu/linear_model/penalized/_base.py",
    '''        # Convert only after covariance, distribution, p-value, and CI work.\n        self._apply_gaussian_reporting_state(state)\n        if result is None:\n''',
    '''        # Convert only after covariance, distribution, p-value, and CI work.\n        self._apply_gaussian_reporting_state(state)\n        if self._effective_intercept:\n            self.intercept_ = float(self._params[0])\n            self.coef_ = self._params[1:].copy()\n        else:\n            self.intercept_ = 0.0\n            self.coef_ = self._params.copy()\n        self._native_fit_coef = None\n        self._native_fit_intercept = None\n        if result is None:\n''',
)

# 5) Update hosted fixtures to provide the native fitted state now required by
# the fail-closed GPU router.
replace_once(
    "dev/tests/test_gaussian_inference_public_consumers.py",
    '''    if backend_name == "torch":\n        model._selected_backend_device = "cpu"\n    model.coef_ = coef.copy()\n    model.intercept_ = float(intercept)\n''',
    '''    if backend_name == "torch":\n        torch = pytest.importorskip("torch")\n        model._selected_backend_device = "cpu"\n        model._native_fit_coef = torch.as_tensor(coef, dtype=torch.float64)\n        model._native_fit_intercept = torch.tensor(intercept, dtype=torch.float64)\n    model.coef_ = coef.copy()\n    model.intercept_ = float(intercept)\n''',
)
replace_once(
    "dev/tests/test_gaussian_inference_no_host_transfer.py",
    '''    model._selected_backend_name = "torch"\n    model._selected_backend_device = "cpu"\n    model.coef_ = coef.copy()\n    model.intercept_ = float(intercept)\n\n    model._compute_post_fit_gaussian_inference(X, y)\n''',
    '''    model._selected_backend_name = "torch"\n    model._selected_backend_device = "cpu"\n    model._native_fit_coef = torch.as_tensor(coef, dtype=torch.float64)\n    model._native_fit_intercept = torch.tensor(intercept, dtype=torch.float64)\n    # Reporting fields are intentionally absent: the GPU router must consume\n    # the native fit state and only populate them after numerical inference.\n    model.coef_ = None\n    model.intercept_ = None\n\n    model._compute_post_fit_gaussian_inference(X, y)\n''',
)

# Add focused behavioral tests for cross-device copying, delayed parameter
# reporting, and cleanup ordering.
path = Path("dev/tests/test_gaussian_inference_no_host_transfer.py")
text = path.read_text()
append = r'''


def test_cupy_device_helper_explicitly_copies_cross_device_native_arrays(monkeypatch):
    import sys
    import types

    from statgpu.backends._utils import _cupy_asarray_on_device

    state = {"current": 0, "copies": 0}

    class FakeArray:
        def __init__(self, device, dtype=np.float64):
            self.device = types.SimpleNamespace(id=int(device))
            self.dtype = np.dtype(dtype)

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    def fake_copy(value):
        state["copies"] += 1
        return FakeArray(state["current"], value.dtype)

    def fake_asarray(value, dtype=None):
        # Deliberately emulate a no-copy asarray for an existing same-dtype
        # native array. The helper must explicitly copy before reaching here.
        if isinstance(value, FakeArray) and (dtype is None or np.dtype(dtype) == value.dtype):
            return value
        return FakeArray(state["current"], dtype or np.float64)

    fake_cupy = types.SimpleNamespace(
        ndarray=FakeArray,
        cuda=types.SimpleNamespace(Device=FakeDevice),
        copy=fake_copy,
        asarray=fake_asarray,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    source = FakeArray(0)
    moved = _cupy_asarray_on_device(source, 1, dtype=np.float64)
    assert moved.device.id == 1
    assert source.device.id == 0
    assert state == {"current": 0, "copies": 1}

    same = _cupy_asarray_on_device(moved, 1, dtype=np.float64)
    assert same is moved
    assert state == {"current": 0, "copies": 1}


def test_torch_l2_fit_defers_parameter_reporting_until_gaussian_inference(monkeypatch):
    torch = pytest.importorskip("torch")

    import statgpu.linear_model._gaussian_inference as gi
    import statgpu.linear_model.penalized._base as pglm_base
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    phase = {"reporting_allowed": False}
    real_fit_to_numpy = fit_mixin._to_numpy

    def guarded_fit_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError("fit parameters crossed to host before Gaussian inference")
        return real_fit_to_numpy(value)

    def fake_reference_inference(
        statistic_abs, *, distribution, alpha, backend, xp, df=None, device=None
    ):
        assert backend == "torch"
        phase["reporting_allowed"] = True
        return torch.full_like(statistic_abs, 0.25), torch.tensor(2.0, dtype=torch.float64)

    monkeypatch.setattr(fit_mixin, "_to_numpy", guarded_fit_to_numpy)
    monkeypatch.setattr(gi, "two_sided_reference_inference", fake_reference_inference)

    X = torch.tensor(
        [[-2.0, 0.5], [-1.0, 1.1], [0.0, 1.4], [1.0, 2.2], [2.0, 2.4], [3.0, 3.1]],
        dtype=torch.float64,
    )
    y = 0.7 * X[:, 0] - 0.2 * X[:, 1] + torch.tensor(
        [0.2, -0.1, 0.15, -0.25, 0.05, -0.05], dtype=torch.float64
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        solver="newton",
    )
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()
    model._nobs = int(X.shape[0])
    model._selected_backend_name = "torch"
    model._selected_backend_device = "cpu"

    model._fit_loss_backend(X, y, None, "newton", "torch")
    assert model.coef_ is None
    assert model._params is None
    assert isinstance(model._native_fit_coef, torch.Tensor)
    assert model._native_fit_coef.device.type == "cpu"

    model._compute_post_fit_gaussian_inference(X, y)
    assert phase["reporting_allowed"] is True
    assert isinstance(model.coef_, np.ndarray)
    assert model._native_fit_coef is None
    assert model._native_fit_intercept is None


def test_gpu_cleanup_is_called_after_post_fit_inference(monkeypatch):
    import types

    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    events = []
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    )

    monkeypatch.setattr(model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="torch"))
    monkeypatch.setattr(model, "_auto_backend_override", lambda backend_name, X: backend_name)
    monkeypatch.setattr(model, "_select_solver", lambda loss, backend_name=None, X=None: "newton")

    def fake_fit_torch(X, y, sample_weight=None):
        events.append("fit")
        model._native_fit_coef = X.new_zeros(X.shape[1])
        model._native_fit_intercept = None
        model.coef_ = None
        model.intercept_ = None
        model._params = None
        model._df_resid = int(X.shape[0] - X.shape[1])

    monkeypatch.setattr(model, "_fit_torch", fake_fit_torch)
    monkeypatch.setattr(
        model,
        "_compute_post_fit_gaussian_inference",
        lambda X, y, sample_weight=None: events.append("inference"),
    )
    monkeypatch.setattr(model, "_cleanup_torch_memory", lambda: events.append("cleanup"))

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(0.0, 1.0, 6)
    model.fit(X, y)

    assert events[-2:] == ["inference", "cleanup"]
'''
if "test_cupy_device_helper_explicitly_copies_cross_device_native_arrays" in text:
    raise RuntimeError("review-fix tests already present")
path.write_text(text.rstrip() + append + "\n")

print("PR129 Codex P2 patch applied")
