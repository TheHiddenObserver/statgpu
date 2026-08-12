from pathlib import Path

p = Path("statgpu/panel/_base.py")
text = p.read_text(encoding="utf-8")
old = '''        diag = xp.diag(cov_params)\n        diag_np = np.diag(self._panel_cov_params_raw).astype(\n            np.float64, copy=False\n        )\n        # A variance is invalid whenever it is strictly negative. There is no\n'''
new = '''        diag = xp.diag(cov_params)\n        cov_np = self._panel_cov_params_raw\n        if not np.all(np.isfinite(cov_np)):\n            raise ValueError(\n                "covariance contains non-finite values; inference is not numerically valid"\n            )\n        diag_np = np.diag(cov_np).astype(np.float64, copy=False)\n        # A variance is invalid whenever it is strictly negative. There is no\n'''
if text.count(old) != 1:
    raise SystemExit(f"_base.py replacement count={text.count(old)}")
p.write_text(text.replace(old, new), encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_inference_guard.py")
text = p.read_text(encoding="utf-8")
addition = r'''


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, -np.inf]]),
    ],
)
def test_inference_rejects_nonfinite_covariance(monkeypatch, covariance):
    with pytest.raises(ValueError, match="covariance contains non-finite values"):
        _store_with_mock_covariance(monkeypatch, covariance)
'''
if "test_inference_rejects_nonfinite_covariance" in text:
    raise SystemExit("nonfinite covariance regression already exists")
p.write_text(text + addition, encoding="utf-8")
