"""Public documentation contract for PR #166 Quantile Group CV convergence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_quantile_model_pages_document_complete_fold_cv_eligibility():
    en = (ROOT / "docs/en/models/quantile.md").read_text(encoding="utf-8")
    cn = (ROOT / "docs/cn/models/quantile.md").read_text(encoding="utf-8")

    assert "a target-level non-converged fold fit is not scored" in en
    assert "only when every fold has a finite score" in en
    assert "selected full-data refit follows direct-estimator convergence reporting" in en
    assert "ConvergenceWarning" in en

    assert "某一折在目标 α 上未建立收敛，则该折不会计分" in cn
    assert "只接受所有折都有有限得分的 α" in cn
    assert "全数据最终重拟合沿用直接估计器的收敛报告语义" in cn
    assert "ConvergenceWarning" in cn
