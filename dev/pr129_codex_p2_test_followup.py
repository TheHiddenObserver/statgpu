from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old!r}")
    p.write_text(text.replace(old, new, 1))


path = "dev/tests/test_gaussian_inference_gpu_runner_contract.py"
replace_once(
    path,
    '''    shared_source = (\n        Path(__file__).parents[2]\n        / "statgpu"\n        / "linear_model"\n        / "_gaussian_inference.py"\n    ).read_text()\n\n    assert 'return f"cuda:{int(value.device.id)}"' in shared_source\n''',
    '''    shared_source = (\n        Path(__file__).parents[2]\n        / "statgpu"\n        / "linear_model"\n        / "_gaussian_inference.py"\n    ).read_text()\n    utils_source = (\n        Path(__file__).parents[2] / "statgpu" / "backends" / "_utils.py"\n    ).read_text()\n\n    assert 'return f"cuda:{int(value.device.id)}"' in shared_source\n''',
)
replace_once(
    path,
    '''    assert 'target_device = int(cp.cuda.runtime.getDevice())' in shared_source\n    assert 'with cp.cuda.Device(target_device):' in shared_source\n    assert 'device_id = int(X_arr.device.id)' in shared_source\n''',
    '''    assert 'target_device = int(cp.cuda.runtime.getDevice())' in shared_source\n    assert 'return _cupy_asarray_on_device(' in shared_source\n    assert 'def _cupy_asarray_on_device' in utils_source\n    assert 'with cp.cuda.Device(target_device):' in utils_source\n    assert 'value = cp.copy(value)' in utils_source\n    assert 'device_id = int(X_arr.device.id)' in shared_source\n''',
)
replace_once(
    path,
    '''    assert "cupy_device_id = int(X_arr.device.id)" in fit_source\n    assert "y_arr = cp.asarray(y_arr)" in fit_source\n    assert "_sw_arr = cp.asarray(_sw_arr)" in fit_source\n    assert "with cp.cuda.Device(cupy_device_id)" in fit_source\n''',
    '''    assert "cupy_device_id = int(X_arr.device.id)" in fit_source\n    assert "_cupy_asarray_on_device(y_arr, cupy_device_id)" in fit_source\n    assert "_cupy_asarray_on_device(_sw_arr, cupy_device_id)" in fit_source\n    assert "with cp.cuda.Device(cupy_device_id)" in fit_source\n''',
)

path = "dev/tests/test_gaussian_inference_no_host_transfer.py"
replace_once(
    path,
    '''def test_gpu_cleanup_is_called_after_post_fit_inference(monkeypatch):\n    import types\n\n    from statgpu.linear_model import PenalizedGeneralizedLinearModel\n''',
    '''def test_gpu_cleanup_is_called_after_post_fit_inference(monkeypatch):\n    import types\n\n    pytest.importorskip("torch")\n    from statgpu.linear_model import PenalizedGeneralizedLinearModel\n''',
)

for name in (
    "dev/tests/test_gaussian_inference_gpu_runner_contract.py",
    "dev/tests/test_gaussian_inference_no_host_transfer.py",
):
    p = Path(name)
    p.write_text(p.read_text().rstrip() + "\n")

print("PR129 hosted test follow-up applied")
