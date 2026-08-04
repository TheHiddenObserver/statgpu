"""Prediction mixin for PenalizedGeneralizedLinearModel."""

from __future__ import annotations

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_numpy

# Eta (linear predictor) clipping bound for numerical stability in GLM link functions.
# Prevents overflow in exp(eta) for log-link families and sigmoid(eta) for logistic.
# Value of 500 is safe because exp(500) ≈ 1.4e217 (within float64 range).
_ETA_CLIP = 500.0


class _PenalizedPredictMixin:

    def _prepare_predict_X(self, X):
        """Apply stored formula design metadata to DataFrame inputs."""
        if self._design_info is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None and isinstance(X, pd.DataFrame):
                from statgpu.core.formula import FormulaParser

                n_input_rows = len(X)
                parser = FormulaParser.__new__(FormulaParser)
                parser._design_info = self._design_info
                parser.formula = None
                X = parser.transform(X)
                if X.shape[0] != n_input_rows:
                    raise ValueError(
                        "formula prediction data contains missing values; "
                        "rows cannot be dropped silently"
                    )
                col_names = list(self._design_info.column_names)
                if self._formula_has_intercept and "Intercept" in col_names:
                    X = np.delete(X, col_names.index("Intercept"), axis=1)
            # Formula processing produces numpy arrays
            return np.asarray(X)
        # No formula: return X as-is to avoid unnecessary GPU→CPU→GPU round-trip
        return X

    def _prediction_backend_name(self):
        backend_name = getattr(self, "_selected_backend_name", None)
        if backend_name == "cupy" and self._cupy_available():
            return "cupy"
        if backend_name == "torch" and self._torch_cuda_available():
            return "torch"
        if backend_name == "numpy":
            return "numpy"
        if self.device == Device.AUTO:
            return "numpy"
        device = self._get_compute_device()
        if device == Device.CUDA:
            if self._cupy_available():
                return "cupy"
            raise RuntimeError(
                "device='cuda' was explicitly requested, but CuPy/CUDA is unavailable at prediction time."
            )
        if device == Device.TORCH:
            if self._torch_cuda_available():
                return "torch"
            raise RuntimeError(
                "device='torch' was explicitly requested, but Torch CUDA is unavailable at prediction time."
            )
        return "numpy"

    def predict(self, X, return_cpu=True):
        """
        Predict using fitted model.

        For squared_error: returns linear prediction.
        For logistic: returns binary class labels.
        For poisson: returns exp(linear prediction) (count values).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        return_cpu : bool, default=True
            If True, always return a numpy ndarray (GPU→CPU transfer happens
            automatically when the model was fitted on GPU).  If False, return
            the result in the same backend as the fitted coefficients (cupy/
            torch when fitted on GPU, numpy when fitted on CPU).  Setting to
            False avoids an unnecessary D→H transfer when chaining GPU
            operations (e.g., ``model.predict(X_gpu) - y_gpu``).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        backend_name = self._prediction_backend_name()
        if backend_name == "cupy":
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self._effective_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            if self.loss == "logistic":
                p = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
                result = (p > 0.5).astype(float)
            elif self.loss == "cox_ph":
                result = cp.exp(cp.clip(raw, -_ETA_CLIP, _ETA_CLIP))
            elif self.loss != "squared_error":
                result = self._family_for_loss().link.inverse(raw)
            else:
                result = raw
            return _to_numpy(result) if return_cpu else result
        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            if self.loss == "logistic":
                p = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -_ETA_CLIP, _ETA_CLIP)))
                result = (p > 0.5).to(raw.dtype)
            elif self.loss == "cox_ph":
                result = torch.exp(torch.clamp(raw, -_ETA_CLIP, _ETA_CLIP))
            elif self.loss != "squared_error":
                result = self._family_for_loss().link.inverse(raw)
            else:
                result = raw
            return _to_numpy(result) if return_cpu else result

        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_

        # Apply link inverse for GLM losses
        if self.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
            return (p > 0.5).astype(float)
        elif self.loss == "cox_ph":
            return np.exp(np.clip(raw, -_ETA_CLIP, _ETA_CLIP))
        elif self.loss != "squared_error":
            return self._family_for_loss().link.inverse(raw)
        return raw

    def score(self, X, y, sample_weight=None):
        """
        Return goodness-of-fit score (R² = 1 - SS_res/SS_tot).

        For all loss types, computes the standard R² metric on the response
        scale. For squared_error this is the classical R². For GLM losses
        (logistic, Poisson, Gamma, etc.) this is R² on the original y scale,
        not the deviance-based pseudo-R².

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. When provided, returns weighted R².

        Returns
        -------
        score : float
            R² or pseudo-R² score.
        """
        # Use predict(return_cpu=True) to avoid device mismatch between
        # predict() and score() backend resolution logic.
        y_pred_np = np.asarray(_to_numpy(self.predict(X, return_cpu=True)))
        y = np.asarray(y)
        sw = np.asarray(sample_weight, dtype=np.float64).ravel() if sample_weight is not None else None
        resid_sq = (y - y_pred_np) ** 2
        if sw is not None:
            w_sum = float(np.sum(sw))
            if w_sum <= 0:
                return 0.0
            ss_res = float(np.sum(sw * resid_sq))
            ss_tot = float(np.sum(sw * (y - np.average(y, weights=sw)) ** 2))
        else:
            ss_res = float(np.sum(resid_sq))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
