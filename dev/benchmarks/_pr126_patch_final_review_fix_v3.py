from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Public covariance primitives infer the array backend from X when xp is omitted.
path = Path("statgpu/panel/_covariance.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '''from statgpu.backends import (\n    _LINALG_ERRORS,\n''',
    '''from statgpu.backends import (\n    _LINALG_ERRORS,\n    _get_xp,\n    _resolve_backend,\n''',
    1,
)
old = '''def _ensure_xp(xp=None):\n    \"\"\"Return the array module, defaulting to numpy.\"\"\"\n    return xp if xp is not None else np\n'''
new = '''def _ensure_xp(xp=None, *arrays):\n    \"\"\"Return an explicit array module or infer it from public inputs.\"\"\"\n    if xp is not None:\n        return xp\n    return _get_xp(_resolve_backend("auto", *arrays))\n'''
if old not in text:
    raise SystemExit("expected _ensure_xp definition not found")
text = text.replace(old, new, 1)
count = text.count("xp = _ensure_xp(xp)\n")
if count != 5:
    raise SystemExit(f"expected 5 public covariance backend resolution sites, got {count}")
text = text.replace("xp = _ensure_xp(xp)\n", "xp = _ensure_xp(xp, X)\n")
path.write_text(text, encoding="utf-8")

# Maintained Torch CPU tests must exercise the public default routing (no xp=).
torch_test = Path("dev/tests/test_panel_stage_c_torch_cpu.py")
text = torch_test.read_text(encoding="utf-8")
text = text.replace(
    "actual = ols_covariance(X_t, resid_t, cov_type=cov_type, xp=torch)",
    "actual = ols_covariance(X_t, resid_t, cov_type=cov_type)",
    1,
)
text = text.replace(
    '''        c1,\n        c2,\n        xp=torch,\n        group_debias=True,\n''',
    '''        c1,\n        c2,\n        group_debias=True,\n''',
    1,
)
text = text.replace(
    '''        extra_df=3,\n        xp=torch,\n    )\n''',
    '''        extra_df=3,\n    )\n''',
    1,
)
if "xp=torch" in text:
    raise SystemExit("Torch Stage-C primitive tests still contain explicit xp=torch")
text = text.replace(
    '''    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n\n\ndef test_stage_c_two_way_cluster''',
    '''    assert torch.is_tensor(actual)\n    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n\n\ndef test_stage_c_two_way_cluster''',
    1,
)
text = text.replace(
    '''    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n\n\n@pytest.mark.parametrize("kernel"''',
    '''    assert torch.is_tensor(actual)\n    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n\n\n@pytest.mark.parametrize("kernel"''',
    1,
)
# Add the tensor assertion to the DK primitive test only if not already present.
needle = '''    actual = driscoll_kraay_covariance(\n        torch.as_tensor(X, dtype=torch.float64),\n        torch.as_tensor(resid, dtype=torch.float64),\n        time,\n        bandwidth=2,\n        kernel=kernel,\n        extra_df=3,\n    )\n    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n'''
replacement = needle.replace(
    "    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n",
    "    assert torch.is_tensor(actual)\n    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)\n",
)
if needle not in text:
    raise SystemExit("expected DK Torch primitive block not found")
text = text.replace(needle, replacement, 1)
torch_test.write_text(text, encoding="utf-8")

# Physical acceptance also proves direct public CuPy/Torch primitive routing.
runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = runner.read_text(encoding="utf-8")
text = text.replace(
    '''from statgpu.backends import _to_numpy\nfrom statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects\n''',
    '''from statgpu.backends import _is_cupy_array, _is_torch_array, _to_numpy\nfrom statgpu.panel import (\n    BetweenOLS,\n    FirstDifferenceOLS,\n    PanelOLS,\n    PooledOLS,\n    RandomEffects,\n    clustered_covariance,\n    driscoll_kraay_covariance,\n)\n''',
    1,
)
insert_after = '''def _array(value):\n    return np.asarray(_to_numpy(value), dtype=np.float64)\n\n\n'''
addition = '''def _array_backend_name(value):\n    if _is_cupy_array(value):\n        return "cupy"\n    if _is_torch_array(value):\n        return "torch"\n    return "numpy"\n\n\ndef _public_primitive_cases(X, y, entity, time, clusters, backend):\n    X_design = np.column_stack([np.ones(len(y)), X])\n    params = np.linalg.lstsq(X_design, y, rcond=None)[0]\n    resid = y - X_design @ params\n    Xb, rb, _eb, _tb = _to_backend(X_design, resid, entity, time, backend)\n    return {\n        "cluster_group_debias": clustered_covariance(\n            Xb, rb, clusters[:, 0], group_debias=True\n        ),\n        "driscoll_kraay_qs": driscoll_kraay_covariance(\n            Xb, rb, time, bandwidth=2, kernel="qs"\n        ),\n    }\n\n\n'''
if insert_after not in text:
    raise SystemExit("expected _array insertion point not found")
text = text.replace(insert_after, insert_after + addition, 1)
old = '''    reference_models = _fit_cases(X, y, entity, time, clusters, "numpy")\n    reference = {name: _snapshot(model) for name, model in reference_models.items()}\n\n    results = {}\n'''
new = '''    reference_models = _fit_cases(X, y, entity, time, clusters, "numpy")\n    reference = {name: _snapshot(model) for name, model in reference_models.items()}\n    primitive_reference = {\n        name: _array(value)\n        for name, value in _public_primitive_cases(\n            X, y, entity, time, clusters, "numpy"\n        ).items()\n    }\n    required_public_primitives = {"cluster_group_debias", "driscoll_kraay_qs"}\n    if set(primitive_reference) != required_public_primitives:\n        raise AssertionError("NumPy public primitive acceptance matrix drifted")\n\n    results = {}\n'''
if old not in text:
    raise SystemExit("expected physical reference block not found")
text = text.replace(old, new, 1)
old = '''        payload = {"status": "success", "requested_backend": backend, "cases": {}}\n        if set(models) != set(reference):\n'''
new = '''        payload = {\n            "status": "success",\n            "requested_backend": backend,\n            "cases": {},\n            "public_primitives": {},\n        }\n        if set(models) != set(reference):\n'''
if old not in text:
    raise SystemExit("expected physical backend payload block not found")
text = text.replace(old, new, 1)
old = '''        results[backend] = payload\n\n    required_cases = {\n'''
new = '''        primitive_values = _public_primitive_cases(\n            X, y, entity, time, clusters, backend\n        )\n        if set(primitive_values) != required_public_primitives:\n            raise AssertionError(\n                f"{backend}: public primitive acceptance matrix drifted"\n            )\n        for name, value in primitive_values.items():\n            executed = _array_backend_name(value)\n            if executed != backend:\n                raise AssertionError(\n                    f"public primitive {name}: requested {backend}, executed {executed}"\n                )\n            actual = _array(value)\n            np.testing.assert_allclose(\n                actual,\n                primitive_reference[name],\n                rtol=args.rtol,\n                atol=args.atol,\n                err_msg=f"public primitive {name}",\n            )\n            payload["public_primitives"][name] = {\n                "status": "success",\n                "executed_backend": executed,\n                "max_abs_difference": _max_abs(actual, primitive_reference[name]),\n            }\n        results[backend] = payload\n\n    required_cases = {\n'''
if old not in text:
    raise SystemExit("expected physical result insertion point not found")
text = text.replace(old, new, 1)
text = text.replace(
    '''        "case_count_per_backend": len(reference),\n        "backends": results,\n''',
    '''        "case_count_per_backend": len(reference),\n        "public_primitive_count_per_backend": len(required_public_primitives),\n        "backends": results,\n''',
    1,
)
runner.write_text(text, encoding="utf-8")

contract = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = contract.read_text(encoding="utf-8")
if "test_stage_c_runner_public_primitive_matrix_is_complete" not in text:
    text += '''\n\ndef test_stage_c_runner_public_primitive_matrix_is_complete():\n    X, y, entity, time, clusters = _MOD._dataset()\n    values = _MOD._public_primitive_cases(\n        X, y, entity, time, clusters, "numpy"\n    )\n    assert set(values) == {"cluster_group_debias", "driscoll_kraay_qs"}\n    for value in values.values():\n        arr = np.asarray(value, dtype=np.float64)\n        assert arr.shape == (3, 3)\n        assert np.all(np.isfinite(arr))\n'''
    contract.write_text(text, encoding="utf-8")

plan = Path("dev/plans/panel_p1_stage_c_covariance_plan.md")
text = plan.read_text(encoding="utf-8")
needle = '''Every new public covariance integration reaches both CuPy and Torch CUDA at least once.\n'''
replacement = needle + '''The exported covariance primitives also receive direct-call CuPy/Torch acceptance with `xp` omitted, proving public backend auto-detection rather than only estimator-mediated execution.\n'''
if needle not in text:
    raise SystemExit("expected physical plan sentence not found")
plan.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
