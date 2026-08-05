import pr87_patch_v41  # applies v40 and v41 staged fixes
from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Align the executable matrix truth with the maintained public compatibility
# matrix: Elastic Net is non-smooth whenever l1_ratio > 0.
replace_once(
    "dev/tests/test_loss_penalty_solver_matrix.py",
    '''# Penalties that are non-smooth (L-BFGS/Newton can't handle)
# ElasticNet has a smooth L2 component, so L-BFGS/Newton handle it via the smooth part
NON_SMOOTH_PENALTIES = {"l1", "scad", "mcp", "adaptive_l1", "group_lasso", "group_mcp", "group_scad"}
''',
    '''# Penalties with a non-smooth component. Smooth solvers must not silently
# optimize only the L2 part of Elastic Net.
NON_SMOOTH_PENALTIES = {
    "l1",
    "elasticnet",
    "scad",
    "mcp",
    "adaptive_l1",
    "group_lasso",
    "group_mcp",
    "group_scad",
}
''',
)
replace_once(
    "dev/tests/test_loss_penalty_solver_matrix.py",
    '''    def test_newton_with_l1_raises_or_skips(self, continuous_data):
        """Newton + L1 should either raise or be handled gracefully."""
        X, y, _ = continuous_data
        loss = HuberLoss(delta=1.0)
        penalty = L1Penalty(0.01)
        try:
            coef, _ = newton_solver(loss, penalty, X, y, max_iter=10)
            # If it doesn't raise, it should still produce finite results
            assert np.all(np.isfinite(coef.cpu().numpy() if hasattr(coef, 'cpu') else coef))
        except (NotImplementedError, ValueError, TypeError):
            pass  # Expected
''',
    '''    def test_newton_with_l1_raises_explicitly(self, continuous_data):
        """Newton must reject L1 before silently changing the objective."""
        X, y, _ = continuous_data
        loss = HuberLoss(delta=1.0)
        penalty = L1Penalty(0.01)
        with pytest.raises(ValueError, match="supports only l2/none"):
            newton_solver(loss, penalty, X, y, max_iter=10)
''',
)
replace_once(
    "dev/tests/test_loss_penalty_solver_matrix.py",
    '''    def test_huber_elasticnet(self, continuous_data):
        """HuberLoss + ElasticNet should work."""
        X, y, _ = continuous_data
        loss = HuberLoss(delta=1.0)
        penalty = ElasticNetPenalty(alpha=0.01, l1_ratio=0.5)
        coef, _ = lbfgs_solver(loss, penalty, X, y, max_iter=200, tol=1e-6)
        coef_np = coef.cpu().numpy() if hasattr(coef, 'cpu') else np.asarray(coef)
        assert np.all(np.isfinite(coef_np))
''',
    '''    def test_huber_elasticnet(self, continuous_data):
        """HuberLoss + ElasticNet is optimized by a proximal solver."""
        X, y, _ = continuous_data
        loss = HuberLoss(delta=1.0)
        penalty = ElasticNetPenalty(alpha=0.01, l1_ratio=0.5)
        coef, _ = fista_solver(loss, penalty, X, y, max_iter=500, tol=1e-6)
        coef_np = coef.cpu().numpy() if hasattr(coef, 'cpu') else np.asarray(coef)
        assert np.all(np.isfinite(coef_np))
''',
)

# Update the public function docstring to match the explicit delegation contract.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    """Proximal Newton solver for smooth loss + non-smooth penalty.

    Parameters
    ----------
    loss : LossBase
        Must have gradient(), hessian(), fused_value_and_gradient().
    penalty : Penalty
        Non-smooth penalty with proximal() method.
''',
    '''    """Newton solver for smooth penalties with explicit FISTA delegation.

    L2/no-penalty objectives use Newton updates. A non-smooth penalty emits a
    ``RuntimeWarning`` and is delegated to ``fista_solver`` because the
    Hessian-metric proximal subproblem is not implemented.

    Parameters
    ----------
    loss : LossBase
        Must expose the operations required by the selected solver path.
    penalty : Penalty or None
        L2/None for Newton; non-smooth penalties are delegated to FISTA.
''',
)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Aligned the executable loss/penalty/solver matrix with the maintained "
    "compatibility contract: Elastic Net precision is tested through FISTA, "
    "while smooth solvers are tested to reject it explicitly.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- The executable solver matrix now treats Elastic Net as non-smooth and "
    "validates its precision through FISTA rather than a smooth-only solver.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- 可执行 solver matrix 现在将 Elastic Net 视为非光滑惩罚，并通过 "
    "FISTA 而不是仅支持光滑目标的 solver 验证其精度。\n",
)
