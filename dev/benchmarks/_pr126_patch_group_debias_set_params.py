from pathlib import Path

path = Path('statgpu/panel/_base.py')
text = path.read_text(encoding='utf-8')
old = '''    def set_params(self, **params):
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
'''
new = '''    def set_params(self, **params):
        """Set estimator parameters and refresh panel covariance aliases.

        ``BaseEstimator`` deliberately preserves the exact user-supplied public
        constructor value for sklearn constructor identity.  Panel covariance
        dispatch, however, uses the normalized private ``_cov_type`` runtime
        value.  Refresh only that private value after sklearn-style parameter
        updates so ``hc1``/``dk``/``kernel`` behave exactly like direct
        construction without changing the public raw parameter.
        """
        if "group_debias" in params and not isinstance(
            params["group_debias"], (bool, np.bool_)
        ):
            raise ValueError("group_debias must be boolean")
        result = super().set_params(**params)
        if "cov_type" in params and hasattr(self, "cov_type"):
            from statgpu.panel._covariance import normalize_covariance_type

            self._cov_type = normalize_covariance_type(self.cov_type)
        return result
'''
if old not in text:
    raise SystemExit('expected BasePanelModel.set_params block not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
