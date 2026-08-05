from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''            if sample_weight is not None:
                sw = np.asarray(_to_numpy(sample_weight), dtype=float).ravel()
                if is_gpu:
                    self._sample_weight_inf = self._to_array(
                        sw, backend=inf_backend)
                else:
                    self._sample_weight_inf = sw
            else:
                self._sample_weight_inf = None
''',
    '''            if sample_weight is not None:
                if is_gpu:
                    # sample_weight is already validated on the selected backend;
                    # preserve device residency instead of copying the full vector
                    # to NumPy and immediately transferring it back to the GPU.
                    self._sample_weight_inf = self._to_array(
                        sample_weight, backend=inf_backend
                    )
                else:
                    self._sample_weight_inf = np.asarray(
                        sample_weight, dtype=float
                    ).ravel()
            else:
                self._sample_weight_inf = None
''',
)

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
marker = "# PR87_GLM_WEIGHT_INFERENCE_DEVICE_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_WEIGHT_INFERENCE_DEVICE_TESTS
def test_torch_glm_formula_weight_inference_avoids_cpu_roundtrip(monkeypatch):
    torch = _require_modern_torch_cuda()
    pd = pytest.importorskip("pandas")
    import statgpu.backends as backends
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0., 1., 2., 3., 4., 5.]}
    )
    weights = torch.tensor(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        dtype=torch.float64,
        device="cuda",
    )
    original_to_numpy = backends._to_numpy

    def guarded_to_numpy(value):
        if (
            torch.is_tensor(value)
            and value.is_cuda
            and tuple(value.shape) == tuple(weights.shape)
            and bool(torch.allclose(value, weights).item())
        ):
            raise AssertionError("formula sample_weight copied to CPU")
        return original_to_numpy(value)

    monkeypatch.setattr(backends, "_to_numpy", guarded_to_numpy)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        device="torch",
        compute_inference=True,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    assert torch.is_tensor(model._sample_weight_inf)
    assert model._sample_weight_inf.is_cuda
    assert weights.is_cuda


def test_cupy_glm_formula_weight_inference_avoids_cpu_roundtrip(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    pd = pytest.importorskip("pandas")
    import statgpu.backends as backends
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0., 1., 2., 3., 4., 5.]}
    )
    weights = cp.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=cp.float64)
    original_to_numpy = backends._to_numpy

    def guarded_to_numpy(value):
        if (
            isinstance(value, cp.ndarray)
            and tuple(value.shape) == tuple(weights.shape)
            and bool(cp.allclose(value, weights).item())
        ):
            raise AssertionError("formula sample_weight copied to CPU")
        return original_to_numpy(value)

    monkeypatch.setattr(backends, "_to_numpy", guarded_to_numpy)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        device="cuda",
        compute_inference=True,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    assert isinstance(model._sample_weight_inf, cp.ndarray)
    assert int(model._sample_weight_inf.device.id) == int(weights.device.id)
    assert isinstance(weights, cp.ndarray)
'''
    path.write_text(text, encoding="utf-8")
