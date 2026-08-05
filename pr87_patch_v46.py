import pr87_patch_v45  # apply staged iterative-solver fixes first
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


# NumPy must honor the same ref-tensor floating dtype contract as CuPy/Torch.
replace_once(
    "statgpu/backends/_array_ops.py",
    '''    if backend == "numpy":\n        return np.zeros(n, dtype=dtype)\n''',
    '''    if backend == "numpy":\n        ref_dtype = getattr(ref_tensor, "dtype", None)\n        out_dtype = dtype\n        if out_dtype is None:\n            out_dtype = (\n                ref_dtype\n                if ref_dtype is not None and np.issubdtype(ref_dtype, np.floating)\n                else np.float64\n            )\n        return np.zeros(n, dtype=out_dtype)\n''',
)
replace_once(
    "statgpu/backends/_array_ops.py",
    '''    return np.asarray(arr, dtype=dtype or float)\n''',
    '''    out_dtype = dtype\n    if out_dtype is None:\n        ref_dtype = getattr(ref_tensor, "dtype", None)\n        out_dtype = (\n            ref_dtype\n            if ref_dtype is not None and np.issubdtype(ref_dtype, np.floating)\n            else float\n        )\n    return np.asarray(arr, dtype=out_dtype)\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V46
def test_numpy_backend_constructors_follow_floating_reference_dtype():
    from statgpu.backends._array_ops import _to_backend, _zeros

    ref32 = np.ones(3, dtype=np.float32)
    assert _zeros(3, "numpy", ref_tensor=ref32).dtype == np.float32
    assert _to_backend([1.0, 2.0], "numpy", ref_tensor=ref32).dtype == np.float32

    ref_int = np.ones(3, dtype=np.int64)
    assert _zeros(3, "numpy", ref_tensor=ref_int).dtype == np.float64
    assert _to_backend([1, 2], "numpy", ref_tensor=ref_int).dtype == np.float64
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V46", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Made shared NumPy zero/conversion helpers honor a floating reference "
    "array dtype, matching the existing CuPy/Torch backend contract while "
    "retaining float64 defaults for integer references.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Shared NumPy constructors now follow floating reference dtypes like the "
    "CuPy/Torch implementations, while integer references retain float64 "
    "numerical defaults.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- shared NumPy constructor 现在与 CuPy/Torch 一样跟随浮点 reference "
    "dtype；整数 reference 仍采用 float64 数值默认值。\n",
)
