import pr87_patch_v40  # applies the staged warm-start patch
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


# Repair the v40 test fixture import before validation.
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    "# PR87_REVIEW_FIX_V40\ndef _run_warm_start_solver_matrix",
    "# PR87_REVIEW_FIX_V40\nimport warnings\n\n\ndef _run_warm_start_solver_matrix",
)

# Smooth solvers must reject any penalty whose non-smooth part they would
# otherwise silently omit.
replace_once(
    "statgpu/solvers/_utils.py",
    '''def _penalty_name(penalty):
    return str(getattr(penalty, "name", "none")).lower()


def _smooth_penalty_value(penalty, coef):
''',
    '''def _penalty_name(penalty):
    return str(getattr(penalty, "name", "none")).lower()


def _validate_smooth_penalty(penalty, solver_name):
    """Reject penalties whose non-smooth terms a smooth solver cannot optimize."""
    name = _penalty_name(penalty)
    if name not in ("l2", "none", "null", ""):
        raise ValueError(
            f"{solver_name} supports only l2/none penalties; got penalty='{name}'. "
            "Use fista or another proximal solver for non-smooth penalties."
        )


def _smooth_penalty_value(penalty, coef):
''',
)

for path, solver_name, import_anchor in (
    (
        "statgpu/solvers/_newton.py",
        "newton_solver",
        '''    _as_backend_vector,\n)\n''',
    ),
    (
        "statgpu/solvers/_lbfgs.py",
        "lbfgs_solver",
        '''    _as_backend_vector,\n)\n''',
    ),
    (
        "statgpu/solvers/_lbfgs_b.py",
        "lbfgs_b_solver",
        '''    _as_backend_vector,\n)\n''',
    ),
):
    replace_once(
        path,
        import_anchor,
        import_anchor.replace(")\n", "    _validate_smooth_penalty,\n)\n"),
    )
    replace_once(
        path,
        f'''    backend = _resolve_backend("auto", X)\n''',
        f'''    _validate_smooth_penalty(penalty, "{solver_name}")\n    backend = _resolve_backend("auto", X)\n''',
    )

replace_once(
    "statgpu/solvers/_lbfgs.py",
    "Smooth penalty (l2, elasticnet, none).",
    "Smooth penalty (l2 or none).",
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    "Smooth penalty (l2, elasticnet, none).",
    "Smooth penalty (l2 or none).",
)

# Keep maintained compatibility docs aligned with the explicit fallback.
replace_once(
    "docs/en/guides/solver-penalty-matrix.md",
    '''| `proximal_newton` | scalar scad, mcp, adaptive_l1 (Hessian losses) | group penalties and unsupported penalties | Newton direction + Armijo + proximal operator |''',
    '''| `proximal_newton` | l2 / none use Newton; non-smooth direct calls delegate visibly to FISTA | group penalties and unsupported penalties | no silent Euclidean-prox approximation |''',
)
replace_once(
    "docs/cn/guides/solver-penalty-matrix.md",
    '''| `newton` | 光滑目标 | l1、非凸及全部 group penalty | Newton + 线搜索 |''',
    '''| `newton` | l2 / none | l1、elasticnet、非凸及全部 group penalty | Newton + 线搜索 |''',
)
replace_once(
    "docs/cn/guides/solver-penalty-matrix.md",
    '''| `lbfgs` | 光滑目标 | l1、非凸及全部 group penalty | L-BFGS |''',
    '''| `lbfgs` | l2 / none | l1、elasticnet、非凸及全部 group penalty | L-BFGS |''',
)
replace_once(
    "docs/cn/guides/solver-penalty-matrix.md",
    '''| `proximal_newton` | 支持的标量非凸 Hessian 路径 | 全部 group penalty | Newton + Armijo + proximal |''',
    '''| `proximal_newton` | l2 / none 使用 Newton；非光滑 direct 调用显式转到 FISTA | 全部 group penalty 与不支持组合 | 不再静默使用 Euclidean-prox 近似 |''',
)


tests = r'''
# PR87_REVIEW_FIX_V41
def test_smooth_solvers_reject_elasticnet_before_numerical_work():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import lbfgs_b_solver, lbfgs_solver, newton_solver

    class GuardedLoss:
        def preprocess(self, *args, **kwargs):
            raise AssertionError("penalty validation must precede preprocessing")

    penalty = get_penalty("elasticnet", alpha=0.2, l1_ratio=0.5)
    X = np.ones((3, 1), dtype=np.float64)
    y = np.ones(3, dtype=np.float64)
    for solver in (newton_solver, lbfgs_solver, lbfgs_b_solver):
        with pytest.raises(ValueError, match="supports only l2/none"):
            solver(GuardedLoss(), penalty, X, y)
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V41", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Smooth Newton/L-BFGS solvers now reject Elastic Net and other "
    "non-smooth penalties before preprocessing instead of silently omitting "
    "their non-smooth objective component.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Newton, L-BFGS, and L-BFGS-B now fail explicitly for Elastic Net and "
    "other non-smooth penalties rather than optimizing only their smooth part.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- Newton、L-BFGS 与 L-BFGS-B 现在会对 Elastic Net 和其他非光滑惩罚"
    "显式失败，不再只优化其中的光滑部分。\n",
)
