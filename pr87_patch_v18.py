from pathlib import Path

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
marker = "# PR87_FORMULA_WEIGHT_GPU_DEVICE_TESTS"
if marker not in text:
    text += '''

# PR87_FORMULA_WEIGHT_GPU_DEVICE_TESTS
def test_torch_formula_sample_weight_alignment_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.core.formula import align_formula_sample_weight

    weights = torch.tensor(
        [1.0, float("nan"), 3.0, 4.0],
        dtype=torch.float64,
        device="cuda",
    )
    aligned = align_formula_sample_weight(
        weights,
        data_length=4,
        retained_rows=np.array([0, 2, 3], dtype=np.int64),
        retained_length=3,
    )
    assert aligned.is_cuda
    assert aligned.device == weights.device
    assert torch.isfinite(aligned).all()
    torch.testing.assert_close(
        aligned,
        torch.tensor([1.0, 3.0, 4.0], dtype=torch.float64, device="cuda"),
    )
    assert weights.is_cuda

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        align_formula_sample_weight(
            weights,
            data_length=4,
            retained_rows=np.array([0, 1, 3], dtype=np.int64),
            retained_length=3,
        )
    assert weights.is_cuda


def test_cupy_formula_sample_weight_alignment_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.core.formula import align_formula_sample_weight

    weights = cp.asarray([1.0, cp.nan, 3.0, 4.0], dtype=cp.float64)
    aligned = align_formula_sample_weight(
        weights,
        data_length=4,
        retained_rows=np.array([0, 2, 3], dtype=np.int64),
        retained_length=3,
    )
    assert isinstance(aligned, cp.ndarray)
    assert int(aligned.device.id) == int(weights.device.id)
    assert bool(cp.isfinite(aligned).all().item())
    cp.testing.assert_allclose(aligned, cp.asarray([1.0, 3.0, 4.0]))
    assert isinstance(weights, cp.ndarray)

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        align_formula_sample_weight(
            weights,
            data_length=4,
            retained_rows=np.array([0, 1, 3], dtype=np.int64),
            retained_length=3,
        )
    assert isinstance(weights, cp.ndarray)
'''
    path.write_text(text, encoding="utf-8")
