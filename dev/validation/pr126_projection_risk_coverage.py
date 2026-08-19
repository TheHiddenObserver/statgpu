from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:200]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text + "\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "from statgpu.panel._utils import _zero_safe_statistic_ratio\n",
    '''from statgpu.panel._reductions import stable_reduction_flags\nfrom statgpu.panel._utils import (\n    _zero_safe_statistic_ratio,\n    demean_variables,\n    within_transform,\n)\n''',
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "def _nonfinite_covariance_guard_audit(backend):\n",
    '''def _projection_created_dynamic_range_audit(backend):\n    """Exercise a stability flag that appears only after the first FE projection."""\n    upper = np.nextafter(1.0, 2.0)\n    y_np = np.asarray([1.0, -1.0, upper, 1.0, -1.0, 1.0], dtype=np.float64)\n    X_np = np.asarray([0.3, -0.5, 0.7, 0.2, -0.8, 0.6], dtype=np.float64)[:, None]\n    entity = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)\n    time = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)\n    X, y, entity_b, time_b = _to_backend(X_np, y_np, entity, time, backend)\n    if backend == "numpy":\n        xp = np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n    elif backend == "torch":\n        import torch\n        xp = torch\n    else:\n        raise ValueError(backend)\n\n    raw_flag = bool(stable_reduction_flags(y, xp)[0])\n    entity_projected = within_transform(y, entity_b, xp=xp)\n    projected_flag = bool(stable_reduction_flags(entity_projected, xp)[0])\n    if raw_flag or not projected_flag:\n        raise AssertionError(\n            f"{backend}: projected-risk fixture did not transition False -> True: "\n            f"raw={raw_flag}, projected={projected_flag}"\n        )\n\n    y_d, X_d = demean_variables(\n        y, X, entity_b, time_b, xp=xp, max_iter=200, tol=1.0e-12\n    )\n    y_ref, X_ref = demean_variables(\n        y_np, X_np, entity, time, xp=np, max_iter=200, tol=1.0e-12\n    )\n    y_actual = _array(y_d)\n    X_actual = _array(X_d)\n    np.testing.assert_allclose(\n        y_actual, y_ref, rtol=0.0, atol=2.0e-15,\n        err_msg=f"{backend}: projection-created y stability path",\n    )\n    np.testing.assert_allclose(\n        X_actual, X_ref, rtol=2.0e-14, atol=2.0e-15,\n        err_msg=f"{backend}: projection-created X stability path",\n    )\n    for codes in (entity, time):\n        for level in np.unique(codes):\n            if abs(float(np.mean(y_actual[codes == level]))) > 2.0e-15:\n                raise AssertionError(f"{backend}: projected-risk y group mean did not converge")\n    return {\n        "status": "success",\n        "backend": backend,\n        "raw_stability_flag": raw_flag,\n        "post_entity_stability_flag": projected_flag,\n        "max_abs_y_vs_numpy": _max_abs(y_actual, y_ref),\n        "max_abs_X_vs_numpy": _max_abs(X_actual, X_ref),\n    }\n\n\ndef _nonfinite_covariance_guard_audit(backend):\n''',
)

replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),\n''',
    '''            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n            "projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),\n''',
)

append_once(
    "dev/tests/test_panel_stage_c_torch_cpu.py",
    "test_stage_c_projection_created_dynamic_range_torch_cpu",
    r'''
def test_stage_c_projection_created_dynamic_range_torch_cpu():
    from statgpu.panel._reductions import stable_reduction_flags
    from statgpu.panel._utils import demean_variables, within_transform

    upper = np.nextafter(1.0, 2.0)
    y = np.asarray([1.0, -1.0, upper, 1.0, -1.0, 1.0], dtype=np.float64)
    X = np.asarray([0.3, -0.5, 0.7, 0.2, -0.8, 0.6], dtype=np.float64)[:, None]
    entity = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)

    y_t = torch.as_tensor(y, dtype=torch.float64)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    entity_t = torch.as_tensor(entity, dtype=torch.int64)
    time_t = torch.as_tensor(time, dtype=torch.int64)
    assert bool(stable_reduction_flags(y_t, torch)[0]) is False
    projected = within_transform(y_t, entity_t, xp=torch)
    assert bool(stable_reduction_flags(projected, torch)[0]) is True

    y_expected, X_expected = demean_variables(
        y, X, entity, time, xp=np, max_iter=200, tol=1.0e-12
    )
    y_actual, X_actual = demean_variables(
        y_t, X_t, entity_t, time_t, xp=torch, max_iter=200, tol=1.0e-12
    )
    assert_allclose(y_actual.detach().cpu().numpy(), y_expected, rtol=0.0, atol=2.0e-15)
    assert_allclose(X_actual.detach().cpu().numpy(), X_expected, rtol=2.0e-14, atol=2.0e-15)
''',
)

append_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "test_stage_c_runner_registers_projection_created_dynamic_range_gpu_audit",
    r'''
def test_stage_c_runner_registers_projection_created_dynamic_range_gpu_audit():
    audit_source = inspect.getsource(_MOD._projection_created_dynamic_range_audit)
    for token in (
        "stable_reduction_flags",
        "within_transform",
        "demean_variables",
        "raw_stability_flag",
        "post_entity_stability_flag",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert '"projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend)' in main_source
''',
)

replace_once(
    "CHANGELOG.md",
    '''Two-way demeaning refreshes its packed stability classification after the first projection so projection-created dynamic range enters the stable path before alternating dimensions.''',
    '''Two-way demeaning refreshes its packed stability classification after the first projection so projection-created dynamic range enters the stable path before alternating dimensions; the maintained Torch CPU and physical CuPy/Torch validators include a fixture whose risk classification is safe before the entity projection and risky immediately afterwards.''',
)
