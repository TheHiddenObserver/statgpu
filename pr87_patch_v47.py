from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path, marker, addition):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if marker in text:
        return
    file_path.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "statgpu/backends/_array_ops.py",
    '''def _solve_linear_system(A, b, backend="auto"):\n''',
    '''def _linear_solve_runtime_is_rank_failure(exc):\n    """Classify backend solve errors that may safely use least squares."""\n    message = str(exc).lower()\n    return any(\n        marker in message\n        for marker in (\n            "singular",\n            "not invertible",\n            "zero pivot",\n            "rank deficient",\n            "ill-conditioned",\n            "not positive-definite",\n            "not positive definite",\n        )\n    )\n\n\ndef _solve_linear_system(A, b, backend="auto"):\n''',
)
replace_once(
    "statgpu/backends/_array_ops.py",
    '''    except (np.linalg.LinAlgError, RuntimeError):\n        # LinAlgError for numpy/cupy singular matrices\n        # RuntimeError for torch singular matrices\n        if backend == "torch":\n            import torch\n            b_col = b.unsqueeze(1) if b.ndim == 1 else b\n            sol = torch.linalg.lstsq(A, b_col).solution\n            return sol.squeeze(1) if b.ndim == 1 else sol\n        if backend == "cupy":\n            import cupy as cp\n            return cp.linalg.lstsq(A, b)[0]\n        return np.linalg.lstsq(A, b, rcond=None)[0]\n''',
    '''    except np.linalg.LinAlgError:\n        pass\n    except RuntimeError as exc:\n        if not _linear_solve_runtime_is_rank_failure(exc):\n            raise\n\n    if backend == "torch":\n        import torch\n        b_col = b.unsqueeze(1) if b.ndim == 1 else b\n        sol = torch.linalg.lstsq(A, b_col).solution\n        return sol.squeeze(1) if b.ndim == 1 else sol\n    if backend == "cupy":\n        import cupy as cp\n        return cp.linalg.lstsq(A, b)[0]\n    return np.linalg.lstsq(A, b, rcond=None)[0]\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V47
def test_shared_linear_solve_does_not_mask_runtime_failures(monkeypatch):
    from statgpu.backends._array_ops import _solve_linear_system

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def forbidden(*args, **kwargs):
        raise AssertionError("lstsq must not mask infrastructure failures")

    monkeypatch.setattr(np.linalg, "solve", oom)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _solve_linear_system(np.eye(2), np.ones(2), backend="numpy")


def test_shared_linear_solve_retains_rank_failure_fallback(monkeypatch):
    from statgpu.backends._array_ops import _solve_linear_system

    expected = np.array([0.25, -0.5])

    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)
    monkeypatch.setattr(np.linalg, "lstsq", lambda *args, **kwargs: (expected, None, None, None))
    result = _solve_linear_system(np.eye(2), np.ones(2), backend="numpy")
    np.testing.assert_allclose(result, expected)


def test_shared_linear_solve_runtime_classifier_is_narrow():
    from statgpu.backends._array_ops import _linear_solve_runtime_is_rank_failure

    assert _linear_solve_runtime_is_rank_failure(RuntimeError("singular matrix"))
    assert _linear_solve_runtime_is_rank_failure(RuntimeError("rank deficient"))
    assert not _linear_solve_runtime_is_rank_failure(RuntimeError("CUDA out of memory"))
    assert not _linear_solve_runtime_is_rank_failure(RuntimeError("device-side assert"))
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V47", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Narrowed the shared backend linear-system fallback to genuine rank "
    "failures; CUDA OOM, device, and unrelated RuntimeError failures now "
    "propagate instead of being silently retried with least squares.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Shared backend linear solves now use least-squares fallback only for "
    "recognized rank failures and preserve CUDA OOM/device RuntimeErrors.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- shared backend 线性方程求解现在仅对明确的秩失败使用 least-squares "
    "降级，并保留 CUDA OOM/device RuntimeError。\n",
)
