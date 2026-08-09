from pathlib import Path

path = Path("statgpu/panel/_base.py")
text = path.read_text(encoding="utf-8")
needle = '''    def _panel_set_index_info(self, nobs, *, entity_ids=None, time_ids=None):
        info = build_panel_index_info(
            nobs, entity_ids=entity_ids, time_ids=time_ids
        )
        self._panel_index_info = info
        return info

    @property
'''
replacement = '''    def _panel_set_index_info(self, nobs, *, entity_ids=None, time_ids=None):
        info = build_panel_index_info(
            nobs, entity_ids=entity_ids, time_ids=time_ids
        )
        self._panel_index_info = info
        return info

    def set_params(self, **params):
        """Set estimator parameters and refresh panel covariance aliases.

        ``BaseEstimator`` deliberately preserves the exact user-supplied public
        constructor value for sklearn constructor identity.  Panel covariance
        dispatch, however, uses the normalized private ``_cov_type`` runtime
        value.  Refresh only that private value after sklearn-style parameter
        updates so ``hc1``/``dk``/``kernel`` behave exactly like direct
        construction without changing the public raw parameter.
        """
        result = super().set_params(**params)
        if "cov_type" in params and hasattr(self, "cov_type"):
            from statgpu.panel._covariance import normalize_covariance_type

            self._cov_type = normalize_covariance_type(self.cov_type)
        return result

    @property
'''
if needle not in text:
    raise SystemExit("expected BasePanelModel index-info block not found")
path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
