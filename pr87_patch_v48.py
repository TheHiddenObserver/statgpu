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
    "statgpu/solvers/_proximal_newton.py",
    '''            except RuntimeError as exc:\n                # Only swallow trial-point numerical failures; infrastructure\n                # and device errors remain visible to the caller.\n                if not _trial_error_is_numerical(exc):\n                    raise\n''',
    '''            except (ValueError, RuntimeError) as exc:\n                # Only swallow recognized trial-point numerical-domain\n                # failures; input-contract, infrastructure, and device errors\n                # remain visible to the caller.\n                if not _trial_error_is_numerical(exc):\n                    raise\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V48
def test_proximal_newton_backtracks_on_numeric_domain_value_error():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class DomainTrialLoss:
        name = "domain_trial"
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
            if self.value_calls == 2:
                raise ValueError("domain error at trial point")
            return np.asarray(0.25), np.ones_like(coef)

    coef, n_iter = proximal_newton_solver(
        DomainTrialLoss(),
        get_penalty("l2", alpha=0.0),
        np.ones((4, 1), dtype=np.float64),
        np.ones(4, dtype=np.float64),
        max_iter=1,
    )
    assert n_iter == 1
    assert np.all(np.isfinite(coef))
    assert not np.allclose(coef, 0.0)
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V48", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Made proximal-Newton Armijo backtracking treat recognized numeric-domain "
    "ValueError trials consistently with Newton while still propagating "
    "input-contract and infrastructure failures.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Proximal-Newton now backtracks on recognized numeric-domain ValueError "
    "trials while preserving unrelated contract and runtime failures.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- proximal-Newton 现在会对明确的数值域 ValueError trial 执行回溯，"
    "同时保留无关的契约与 runtime failure。\n",
)
