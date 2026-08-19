from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:200]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_in_function(path: str, function: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.index(f"def {function}(")
    end = text.find("\ndef ", start + 1)
    if end < 0:
        end = len(text)
    section = text[start:end]
    if new in section:
        return
    if old not in section:
        raise RuntimeError(f"anchor not found in {path}:{function}: {old[:200]!r}")
    section = section.replace(old, new, 1)
    p.write_text(text[:start] + section + text[end:], encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text + "\n" + addition.strip() + "\n", encoding="utf-8")


# Public covariance helpers must reject non-finite numeric inputs before any
# magnitude-tiered signed reduction can reinterpret NaN/Inf as zero contribution.
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _ensure_xp(xp=None, *arrays):\n    """Return an explicit array module or infer it from public inputs."""\n    if xp is not None:\n        return xp\n    return _get_xp(_resolve_backend("auto", *arrays))\n''',
    '''def _ensure_xp(xp=None, *arrays):\n    """Return an explicit array module or infer it from public inputs."""\n    if xp is not None:\n        return xp\n    return _get_xp(_resolve_backend("auto", *arrays))\n\n\ndef _validate_covariance_finite_inputs(X, resid, xp):\n    """Fail closed before non-finite scores enter signed/group reductions."""\n    finite = xp.all(xp.isfinite(X)) & xp.all(xp.isfinite(resid))\n    if not bool(_to_float_scalar(finite)):\n        raise ValueError("X and resid must contain only finite values")\n''',
)

replace_in_function(
    "statgpu/panel/_covariance.py",
    "clustered_covariance",
    '''    if resid.shape[0] != n:\n        raise ValueError("X and resid must have the same number of observations")\n''',
    '''    if resid.shape[0] != n:\n        raise ValueError("X and resid must have the same number of observations")\n    _validate_covariance_finite_inputs(X, resid, xp)\n''',
)
for function in ("two_way_clustered_covariance", "hac_covariance", "driscoll_kraay_covariance"):
    replace_in_function(
        "statgpu/panel/_covariance.py",
        function,
        '''    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n''',
        '''    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n    _validate_covariance_finite_inputs(X, resid, xp)\n''',
    )

append_once(
    "dev/tests/test_panel_stage_c_edge_contracts.py",
    "test_public_covariance_primitives_reject_nonfinite_residuals_numpy",
    r'''
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", ["cluster", "two_way", "hac", "dk"])
def test_public_covariance_primitives_reject_nonfinite_residuals_numpy(kind, bad):
    from statgpu.panel import (
        clustered_covariance as public_clustered_covariance,
        driscoll_kraay_covariance as public_dk_covariance,
        hac_covariance as public_hac_covariance,
        two_way_clustered_covariance as public_two_way_covariance,
    )

    X = np.column_stack([np.ones(6), np.arange(6.0)])
    resid = np.linspace(-0.3, 0.4, 6)
    resid[2] = bad
    c1 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    time = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        if kind == "cluster":
            public_clustered_covariance(X, resid, c1)
        elif kind == "two_way":
            public_two_way_covariance(X, resid, c1, c2)
        elif kind == "hac":
            public_hac_covariance(X, resid, bandwidth=1)
        else:
            public_dk_covariance(X, resid, time, bandwidth=1)


def test_public_covariance_primitives_reject_nonfinite_design_numpy():
    from statgpu.panel import clustered_covariance as public_clustered_covariance

    X = np.column_stack([np.ones(6), np.arange(6.0)])
    X[1, 1] = np.nan
    resid = np.linspace(-0.3, 0.4, 6)
    groups = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        public_clustered_covariance(X, resid, groups)
''',
)

append_once(
    "dev/tests/test_panel_stage_c_torch_cpu.py",
    "test_stage_c_public_covariance_rejects_nonfinite_residual_torch_cpu",
    r'''
@pytest.mark.parametrize("kind", ["cluster", "two_way", "hac", "dk"])
def test_stage_c_public_covariance_rejects_nonfinite_residual_torch_cpu(kind):
    from statgpu.panel import clustered_covariance, hac_covariance

    X = torch.column_stack(
        [torch.ones(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64)]
    )
    resid = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64)
    resid[2] = float("nan")
    c1 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    time = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        if kind == "cluster":
            clustered_covariance(X, resid, c1)
        elif kind == "two_way":
            two_way_clustered_covariance(X, resid, c1, c2)
        elif kind == "hac":
            hac_covariance(X, resid, bandwidth=1)
        else:
            driscoll_kraay_covariance(X, resid, time, bandwidth=1)
''',
)

# Keep the physical CUDA runner responsible for the same public fail-closed
# contract on both CuPy and Torch rather than relying only on CPU tests.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''def _cancellation_safe_mean_audit(backend):\n''',
    '''def _nonfinite_covariance_guard_audit(backend):\n    X_np = np.column_stack([np.ones(6), np.arange(6.0)])\n    resid_np = np.linspace(-0.3, 0.4, 6)\n    resid_np[2] = np.nan\n    c1 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)\n    c2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)\n    time = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)\n    dummy = np.arange(6, dtype=np.int64)\n    X, resid, _entity, _time = _to_backend(X_np, resid_np, dummy, time, backend)\n    if backend == "numpy":\n        xp = np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n    elif backend == "torch":\n        import torch\n        xp = torch\n    else:\n        raise ValueError(backend)\n\n    calls = {\n        "cluster": lambda: clustered_covariance(X, resid, c1, xp=xp),\n        "two_way": lambda: two_way_clustered_covariance(X, resid, c1, c2, xp=xp),\n        "hac": lambda: hac_covariance(X, resid, bandwidth=1, xp=xp),\n        "dk": lambda: driscoll_kraay_covariance(X, resid, time, bandwidth=1, xp=xp),\n    }\n    guards = {}\n    for name, call in calls.items():\n        try:\n            call()\n        except ValueError as exc:\n            if "X and resid must contain only finite values" not in str(exc):\n                raise AssertionError(f"{backend}: {name} raised wrong nonfinite error: {exc}") from exc\n            guards[name] = True\n        else:\n            raise AssertionError(f"{backend}: {name} accepted a NaN residual")\n    return {"status": "success", "backend": backend, "guards": guards}\n\n\ndef _cancellation_safe_mean_audit(backend):\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n''',
    '''            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),\n''',
)

append_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "test_stage_c_runner_registers_nonfinite_covariance_gpu_guards",
    r'''
def test_stage_c_runner_registers_nonfinite_covariance_gpu_guards():
    audit_source = inspect.getsource(_MOD._nonfinite_covariance_guard_audit)
    for token in (
        "clustered_covariance",
        "two_way_clustered_covariance",
        "hac_covariance",
        "driscoll_kraay_covariance",
        "accepted a NaN residual",
    ):
        assert token in audit_source
    main_source = inspect.getsource(_MOD.main)
    assert '"nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend)' in main_source
''',
)

replace_once(
    "docs/en/panel/covariance.md",
    '''The physical P100 validation additionally records the actual per-field `max_abs_differences` in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`; its summary is in `results/pr126_p100_fresh/validation_summary.txt`.''',
    '''Historical P100 validation records actual per-field `max_abs_differences` in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`, with a summary in `results/pr126_p100_fresh/validation_summary.txt`. Those artifacts predate the later shared reduction and public covariance fail-closed fixes and are reference-only; fresh exact-head CuPy/Torch CUDA validation is required for current acceptance.''',
)
replace_once(
    "docs/cn/panel/covariance.md",
    '''表中的 CI tolerance 是 pass/fail threshold，不是实际观测误差。P100 physical validation 另外保存每个字段实际的 `max_abs_differences`，位于 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`；summary 位于 `results/pr126_p100_fresh/validation_summary.txt`。''',
    '''表中的 CI tolerance 是 pass/fail threshold，不是实际观测误差。历史 P100 validation 保存了每个字段实际的 `max_abs_differences`，位于 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`，summary 位于 `results/pr126_p100_fresh/validation_summary.txt`。这些 artifact 早于后续 shared reduction 与 public covariance fail-closed 修复，只能作为历史参考；当前 acceptance 仍需要在 exact head 上重新完成 CuPy/Torch CUDA 验证。''',
)

replace_once(
    "CHANGELOG.md",
    '''Covariance symmetrization and inclusion-exclusion remain range-aware, while HAC/Driscoll-Kraay protect both pre-Gram products and the complete lag-sequence accumulator from avoidable overflow. The physical Stage-C validator exercises pre-Gram cancellation, tiny-design grouping, mixed-dynamic-range cluster/DK, nested-code permutation, and the earlier extreme-scale cases on CuPy and Torch CUDA.''',
    '''Covariance symmetrization and inclusion-exclusion remain range-aware, while HAC/Driscoll-Kraay protect both pre-Gram products and the complete lag-sequence accumulator from avoidable overflow. Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions. The physical Stage-C validator exercises this fail-closed contract together with pre-Gram cancellation, tiny-design grouping, mixed-dynamic-range cluster/DK, nested-code permutation, and the earlier extreme-scale cases on CuPy and Torch CUDA.''',
)
