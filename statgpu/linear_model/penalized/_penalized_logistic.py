"""PenalizedLogisticRegression — thin wrapper over PenalizedGeneralizedLinearModel."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._predict_mixin import _ETA_CLIP


class PenalizedLogisticRegression(PenalizedGeneralizedLinearModel):
    """Binomial/logistic penalized GLM."""

    def __init__(
        self,
        penalty: Union[str, "Penalty"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        penalty_kwargs: Optional[dict] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        cpu_solver: str = "fista",
        solver: str = "auto",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = False,
        inference_method: str = "debiased",
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
        stopping: str = "coef_delta",
        lla: bool = True,
        max_lla_iters: int = 50,
        lla_tol: float = 1e-6,
    ):
        super().__init__(
            loss="logistic",
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            cpu_solver=cpu_solver,
            solver=solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            compute_inference=compute_inference,
            inference_method=inference_method,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
            stopping=stopping,
            lla=lla,
            max_lla_iters=max_lla_iters,
            lla_tol=lla_tol,
        )

    def predict_proba(self, X):
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
            p1 = 1.0 / (1.0 + cp.exp(-cp.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
            return cp.column_stack([1.0 - p1, p1])
        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(
                    self.intercept_, dtype=raw.dtype, device=raw.device
                )
            p1 = 1.0 / (1.0 + torch.exp(-torch.clamp(raw, -_ETA_CLIP, _ETA_CLIP)))
            return torch.column_stack([1.0 - p1, p1])
        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -_ETA_CLIP, _ETA_CLIP)))
        return np.column_stack([1.0 - p1, p1])



