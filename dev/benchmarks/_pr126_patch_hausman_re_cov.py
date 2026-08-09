from pathlib import Path

path = Path("statgpu/panel/_diagnostics.py")
text = path.read_text(encoding="utf-8")
needle = '''    if str(getattr(fe_model, "_cov_type", "nonrobust")).lower() != "nonrobust":
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Hausman requires nonrobust FE covariance; robust auxiliary Hausman is not implemented in Stage B",
        )

    left_id = getattr(fe_model, "_panel_diagnostic_identity", None)
'''
replacement = '''    if str(getattr(fe_model, "_cov_type", "nonrobust")).lower() != "nonrobust":
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Hausman requires nonrobust FE covariance; robust auxiliary Hausman is not implemented in Stage B",
        )
    if str(getattr(re_model, "_cov_type", "nonrobust")).lower() != "nonrobust":
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            reason="classical Hausman requires nonrobust RE covariance; robust auxiliary Hausman is not implemented in Stage C",
        )

    left_id = getattr(fe_model, "_panel_diagnostic_identity", None)
'''
if needle not in text:
    raise SystemExit("expected Hausman FE covariance guard not found")
path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
