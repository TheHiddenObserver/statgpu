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


# "out of range" is also the canonical wording for index/device programming
# errors, so it is not safe as a generic Armijo numerical-domain marker.
replace_once(
    "statgpu/solvers/_utils.py",
    '''            "domain error",\n            "out of range",\n''',
    '''            "domain error",\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V49
def test_trial_error_classifier_does_not_mask_index_out_of_range():
    from statgpu.solvers._utils import _trial_error_is_numerical

    assert not _trial_error_is_numerical(
        RuntimeError("index out of range in self")
    )
    assert not _trial_error_is_numerical(
        ValueError("coefficient index out of range")
    )


def test_proximal_newton_propagates_index_out_of_range_trial_error():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class IndexFailingTrialLoss:
        name = "index_failing_trial"
        has_hessian = True

        def __init__(self):
            self.value_calls = 0

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
            return np.ones_like(coef), np.eye(coef.shape[0], dtype=coef.dtype)

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            self.value_calls += 1
            if self.value_calls == 1:
                return np.asarray(1.0), np.ones_like(coef)
            raise RuntimeError("index out of range in self")

    with pytest.raises(RuntimeError, match="index out of range"):
        proximal_newton_solver(
            IndexFailingTrialLoss(),
            get_penalty("l2", alpha=0.0),
            np.ones((4, 1), dtype=np.float64),
            np.ones(4, dtype=np.float64),
            max_iter=1,
        )
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V49", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Removed the over-broad Armijo `out of range` numerical marker so index "
    "and device programming errors propagate instead of being mistaken for "
    "recoverable trial-point domain failures.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Armijo backtracking no longer treats generic `out of range` errors as "
    "recoverable numerical trials, preserving index/device programming errors.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- Armijo 回溯不再把通用 `out of range` 错误当作可恢复数值 trial，"
    "因此 index/device 编程错误会原样抛出。\n",
)
