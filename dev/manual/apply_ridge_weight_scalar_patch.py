"""Temporary patch to avoid full GPU sample-weight transfers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


path = ROOT / "statgpu/linear_model/penalized/_fit_mixin.py"
text = path.read_text()
text = replace_once(
    text,
    '_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})\n',
    '''_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})


def _validate_sample_weight_backend(sample_weight, n_samples, backend_name):
    """Validate sample weights in place and synchronize only scalar reductions."""
    if getattr(sample_weight, "ndim", None) != 1:
        raise ValueError("sample_weight must be one-dimensional")
    if int(sample_weight.shape[0]) != int(n_samples):
        raise ValueError("sample_weight must have length n_samples")

    if backend_name == "torch":
        import torch
        if not bool(torch.all(torch.isfinite(sample_weight)).item()):
            raise ValueError("sample_weight must be finite")
        if bool(torch.any(sample_weight < 0).item()):
            raise ValueError("sample_weight must be non-negative")
        total = float(torch.sum(sample_weight).item())
    elif backend_name == "cupy":
        import cupy as cp
        if not bool(cp.all(cp.isfinite(sample_weight)).item()):
            raise ValueError("sample_weight must be finite")
        if bool(cp.any(sample_weight < 0).item()):
            raise ValueError("sample_weight must be non-negative")
        total = float(cp.sum(sample_weight).item())
    else:
        weights = np.asarray(sample_weight)
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must be finite")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        total = float(np.sum(weights))

    if total <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return total
''',
    "sample weight helper",
)
text = replace_once(
    text,
    '''        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name)
            _sw_check = np.asarray(_to_numpy(_sw_arr), dtype=np.float64).reshape(-1)
            if _sw_check.shape[0] != int(X.shape[0]):
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(_sw_check)):
                raise ValueError("sample_weight must be finite")
            if np.any(_sw_check < 0):
                raise ValueError("sample_weight must be non-negative")
            if float(np.sum(_sw_check)) <= 0.0:
                raise ValueError("sample_weight must have a positive sum")
''',
    '''        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name).reshape(-1)
            _validate_sample_weight_backend(_sw_arr, X.shape[0], backend_name)
''',
    "central sample weight validation",
)
text = replace_once(
    text,
    '''            if sample_weight is not None:
                sw = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X).reshape(-1)
                n_eff = float(np.sum(np.asarray(_to_numpy(sw), dtype=np.float64)))
''',
    '''            if sample_weight is not None:
                sw = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X).reshape(-1)
                n_eff = _validate_sample_weight_backend(sw, n_samples, backend_name)
''',
    "GPU exact normalization",
)
text = replace_once(
    text,
    '''        ridge_normalization = (
            float(n_samples)
            if sample_weight is None
            else float(np.sum(np.asarray(_to_numpy(sample_weight), dtype=np.float64)))
        )
''',
    '''        ridge_normalization = (
            float(n_samples)
            if sample_weight is None
            else _validate_sample_weight_backend(
                sample_weight, n_samples, backend_name
            )
        )
''',
    "IRLS normalization",
)
path.write_text(text)


test_path = ROOT / "dev/tests/test_ridge_weighted_consistency.py"
test = test_path.read_text()
test += '''


def test_backend_weight_validation_returns_scalar_sum_without_host_vector_conversion():
    from statgpu.linear_model.penalized._fit_mixin import _validate_sample_weight_backend

    weights = np.array([0.5, 1.5, 2.0])
    assert _validate_sample_weight_backend(weights, 3, "numpy") == 4.0
    with pytest.raises(ValueError, match="non-negative"):
        _validate_sample_weight_backend(np.array([1.0, -0.1]), 2, "numpy")

    torch = pytest.importorskip("torch")
    torch_weights = torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64)
    assert _validate_sample_weight_backend(torch_weights, 3, "torch") == 4.0
    with pytest.raises(ValueError, match="finite"):
        _validate_sample_weight_backend(
            torch.tensor([1.0, float("nan")], dtype=torch.float64), 2, "torch"
        )
'''
test_path.write_text(test)

print("Scalar-only sample-weight validation patch applied")
