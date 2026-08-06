from pathlib import Path

source_path = Path("statgpu/linear_model/wrappers/_logistic.py")
source = source_path.read_text(encoding="utf-8")
old = '''                self._sample_weight = np.asarray(
                    self._to_numpy(sample_weight_arr), dtype=np.float64
                ).reshape(-1)
'''
new = '''                # CPU covariance inference consumes a NumPy weight cache.
                # CuPy/Torch inference already uses the device-native
                # ``sample_weight_arr`` inside the backend fit and must not pay
                # for an otherwise unused full device-to-host copy.
                self._sample_weight = (
                    np.asarray(sample_weight_arr, dtype=np.float64).reshape(-1)
                    if backend_name == "numpy"
                    else None
                )
'''
if source.count(old) != 1:
    raise RuntimeError(f"sample-weight cache anchor count={source.count(old)}")
source = source.replace(old, new, 1)
source_path.write_text(source, encoding="utf-8")

test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
tests = test_path.read_text(encoding="utf-8")
addition = '''

def test_logistic_torch_fit_does_not_copy_weights_to_numpy(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.linear_model.wrappers._logistic as module
    from statgpu.backends import get_backend

    monkeypatch.setattr(module, "_get_torch_device_str", lambda: "cpu")
    X = torch.tensor(
        [[-2.0], [-1.0], [-0.25], [0.25], [1.0], [2.0]],
        dtype=torch.float64,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64)
    weights = torch.tensor(
        [1.0, 2.0, 1.5, 3.0, 0.75, 4.0], dtype=torch.float64
    )
    model = module.LogisticRegression(
        C=1.0,
        max_iter=200,
        tol=1e-10,
        device="cpu",
        compute_inference=False,
    )
    monkeypatch.setattr(
        model,
        "_get_backend",
        lambda backend="auto": get_backend("torch", device="cpu"),
    )
    original_to_numpy = model._to_numpy

    def guarded_to_numpy(value):
        if torch.is_tensor(value) and value.data_ptr() == weights.data_ptr():
            raise AssertionError("sample weights were copied to NumPy")
        return original_to_numpy(value)

    monkeypatch.setattr(model, "_to_numpy", guarded_to_numpy)
    model.fit(X, y, sample_weight=weights)

    assert model._sample_weight is None
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_)


def test_logistic_cpu_fit_retains_weight_cache_for_cpu_inference():
    from statgpu.linear_model import LogisticRegression

    X = np.array([[-1.5], [-0.5], [0.25], [1.0], [1.75]], dtype=float)
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 1.5, 4.0])
    model = LogisticRegression(
        C=2.0,
        max_iter=200,
        tol=1e-10,
        device="cpu",
        compute_inference=True,
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_array_equal(model._sample_weight, weights)
    assert model._bse is not None
'''
if "test_logistic_torch_fit_does_not_copy_weights_to_numpy" not in tests:
    tests += addition
test_path.write_text(tests, encoding="utf-8")

for changelog, entry in [
    (
        Path("docs/en/changelog.md"),
        "- Kept direct LogisticRegression analytic weights device-native on CuPy/Torch fits instead of copying the full vector to NumPy solely for the CPU inference cache.\n\n",
    ),
    (
        Path("docs/cn/changelog.md"),
        "- 直接 LogisticRegression 的解析权重在 CuPy/Torch 拟合中保持设备原生，不再仅为 CPU 推断缓存将整条权重向量复制到 NumPy。\n\n",
    ),
]:
    current = changelog.read_text(encoding="utf-8")
    heading = "# Changelog\n\n"
    if not current.startswith(heading):
        raise RuntimeError(f"unexpected changelog heading: {changelog}")
    if entry not in current:
        current = heading + entry + current[len(heading):]
    changelog.write_text(current, encoding="utf-8")
