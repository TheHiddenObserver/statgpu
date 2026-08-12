from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "dev/tests/test_pr79_final_review_fixes.py"
text = PATH.read_text(encoding="utf-8")
old = '''    assert model.rank_ == expected_rank\n    assert model.df_resid == X.shape[0] - expected_rank\n    assert model.df_resid == int(reference.df_resid)\n    assert_allclose(model.bse_, reference.bse, rtol=1e-8, atol=1e-10)\n'''
new = '''    assert model.rank_ == expected_rank\n    assert model.df_resid == X.shape[0] - expected_rank\n    assert model.df_resid == int(reference.df_resid)\n\n    # Exact collinearity leaves the fitted subspace identified but not the\n    # original coefficient coordinates.  Stage C therefore keeps the\n    # Moore-Penrose fit/covariance substrate while failing closed on ordinary\n    # coordinate-wise BSE/test/p-value/CI publication.\n    assert_allclose(\n        design @ model._panel_cov_params_raw @ design.T,\n        design @ reference.cov_params() @ design.T,\n        rtol=1e-8,\n        atol=1e-10,\n    )\n    assert model._coefficient_inference_available is False\n    assert model.bse_ is None\n    assert model.tvalues_ is None\n    assert model.pvalues_ is None\n    assert model.conf_int_ is None\n    assert model._inference_result.metadata["applicable"] is False\n    assert "rank deficient" in model._inference_result.metadata["reason"]\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one stale PR79 rank-deficient inference assertion, got {text.count(old)}")
PATH.write_text(text.replace(old, new, 1), encoding="utf-8")
print("PR126 full-suite rank-deficient contract synchronized")
