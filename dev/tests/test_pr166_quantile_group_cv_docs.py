"""Public documentation contract for PR #166 Quantile Group CV convergence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_quantile_model_pages_document_complete_fold_cv_eligibility():
    en = (ROOT / "docs/en/models/quantile.md").read_text(encoding="utf-8")
    cn = (ROOT / "docs/cn/models/quantile.md").read_text(encoding="utf-8")

    assert "complete finite fold evidence" in en
    assert "target-level convergence failure signal" in en
    assert "A generic solver `ConvergenceWarning` alone does not erase" in en
    assert "selected full-data refit follows the direct-estimator convergence-reporting contract" in en

    assert "完整且有限的折级证据" in cn
    assert "target-level 收敛失败信号" in cn
    assert "`ConvergenceWarning` 并不会自动抹去" in cn
    assert "全数据最终重拟合沿用直接估计器的收敛报告语义" in cn
