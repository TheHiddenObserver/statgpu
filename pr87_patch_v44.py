import pr87_patch_v43  # apply the staged validation fix first
from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Repair the runtime-failure test double: Python resolves xp.all before
# evaluating xp.isfinite(values), so both attributes must exist.
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    class RuntimeFailingXP:\n        @staticmethod\n        def isfinite(values):\n            raise RuntimeError("CUDA out of memory")\n''',
    '''    class RuntimeFailingXP:\n        @staticmethod\n        def all(value):\n            return value\n\n        @staticmethod\n        def isfinite(values):\n            raise RuntimeError("CUDA out of memory")\n''',
)

# Share a narrow classifier for expected trial-point numerical failures.
replace_once(
    "statgpu/solvers/_utils.py",
    '''def _native_sample_weight(sample_weight):\n''',
    '''def _trial_error_is_numerical(exc):\n    """Return whether a trial-point exception is an expected numeric-domain failure."""\n    message = str(exc).lower()\n    return any(\n        marker in message\n        for marker in (\n            "overflow",\n            "invalid value",\n            "nan",\n            "non-finite",\n            "nonfinite",\n            "domain error",\n            "out of range",\n        )\n    )\n\n\ndef _native_sample_weight(sample_weight):\n''',
)

replace_once(
    "statgpu/solvers/_newton.py",
    '''    _validate_smooth_penalty,\n)\n''',
    '''    _validate_smooth_penalty,\n    _trial_error_is_numerical,\n)\n''',
)
replace_once(
    "statgpu/solvers/_newton.py",
    '''            except (ValueError, RuntimeError, FloatingPointError):\n                pass\n''',
    '''            except FloatingPointError:\n                pass\n            except (ValueError, RuntimeError) as exc:\n                if not _trial_error_is_numerical(exc):\n                    raise\n''',
)

# Reuse the same classifier in proximal Newton rather than maintaining a
# second, subtly different marker list.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    _as_backend_vector,\n)\n''',
    '''    _as_backend_vector,\n    _trial_error_is_numerical,\n)\n''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''            except RuntimeError as exc:\n                # Only swallow trial-point numerical failures; infrastructure\n                # and device errors remain visible to the caller.\n                err_msg = str(exc).lower()\n                if not any(\n                    marker in err_msg\n                    for marker in ("overflow", "invalid value", "nan")\n                ):\n                    raise\n''',
    '''            except RuntimeError as exc:\n                # Only swallow trial-point numerical failures; infrastructure\n                # and device errors remain visible to the caller.\n                if not _trial_error_is_numerical(exc):\n                    raise\n''',
)

additional_tests = r'''

# PR87_REVIEW_FIX_V44
def test_newton_line_search_does_not_mask_runtime_failures():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    class RuntimeFailingTrialLoss:
        name = "runtime_failing_trial"
        _has_constant_hessian = False

        def __init__(self):
            self.value_calls = 0

        def preprocess(self, X, y):
            return np.asarray(X), np.asarray(y)

        def gradient(self, X, y, coef):
            return np.ones(X.shape[1], dtype=np.float64)

        def hessian(self, X, y, coef):
            return np.eye(X.shape[1], dtype=np.float64)

        def fused_value_and_gradient(self, X, y, coef):
            self.value_calls += 1
            if self.value_calls == 1:
                return np.array(1.0), np.ones(X.shape[1], dtype=np.float64)
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        newton_solver(
            RuntimeFailingTrialLoss(),
            get_penalty("l2", alpha=0.1),
            np.ones((4, 1), dtype=np.float64),
            np.ones(4, dtype=np.float64),
            max_iter=2,
        )


def test_trial_error_classifier_is_narrow():
    from statgpu.solvers._utils import _trial_error_is_numerical

    assert _trial_error_is_numerical(RuntimeError("invalid value in log"))
    assert _trial_error_is_numerical(ValueError("domain error"))
    assert not _trial_error_is_numerical(RuntimeError("CUDA out of memory"))
    assert not _trial_error_is_numerical(RuntimeError("device-side assert"))
'''
path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
if "# PR87_REVIEW_FIX_V44" not in text:
    path.write_text(text.rstrip() + "\n" + additional_tests + "\n", encoding="utf-8")

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Narrowed Newton-family Armijo trial exception handling to expected "
    "numeric-domain failures so CUDA OOM, device, and infrastructure errors "
    "remain visible to callers.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Newton-family Armijo backtracking now suppresses only recognized "
    "numeric-domain trial failures and propagates CUDA OOM/device/runtime "
    "infrastructure errors.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- Newton 系列 Armijo 回溯现在仅忽略明确的数值域 trial failure，并保留 "
    "CUDA OOM/device/runtime 基础设施错误。\n",
)
