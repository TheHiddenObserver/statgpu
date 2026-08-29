from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        try:\n            self._compute_post_fit_gaussian_inference(\n                X, y, sample_weight=_sw_arr\n            )\n        finally:\n            if backend_name == "cupy":\n                self._cleanup_cuda_memory()\n            elif backend_name == "torch":\n                self._cleanup_torch_memory()\n        self._fitted = True\n''',
    '''        try:\n            self._compute_post_fit_gaussian_inference(\n                X, y, sample_weight=_sw_arr\n            )\n        except Exception:\n            # A failed GPU inference must not leave live fitted-parameter\n            # tensors attached to a half-fitted estimator.  Allocator cleanup\n            # cannot release memory that is still referenced here.\n            self._native_fit_coef = None\n            self._native_fit_intercept = None\n            raise\n        finally:\n            if backend_name == "cupy":\n                self._cleanup_cuda_memory()\n            elif backend_name == "torch":\n                self._cleanup_torch_memory()\n        self._fitted = True\n''',
)

replace_once(
    "dev/tests/test_gaussian_inference_no_host_transfer.py",
    '''    with pytest.raises(RuntimeError, match="synthetic post-fit inference failure"):\n        model.fit(X, y)\n\n    assert events[-2:] == ["inference", "cleanup"]\n''',
    '''    with pytest.raises(RuntimeError, match="synthetic post-fit inference failure"):\n        model.fit(X, y)\n\n    assert events[-2:] == ["inference", "cleanup"]\n    assert model._native_fit_coef is None\n    assert model._native_fit_intercept is None\n    assert model._fitted is False\n''',
)

print("PR129 native-state rollback fix applied")
