"""Hosted Torch-CPU coverage for the shared Panel Stage-A substrate."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose


torch = pytest.importorskip("torch")


def _problem(dtype):
    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        dtype=dtype,
    )
    y = torch.tensor([0.2, 1.1, 2.4, 2.7, 4.5], dtype=dtype)
    beta = torch.linalg.pinv(X) @ y
    resid = y - X @ beta
    df = int(X.shape[0] - torch.linalg.matrix_rank(X).item())
    scale = float((resid @ resid).item()) / df
    return X, resid, scale, df


def test_panel_index_info_accepts_torch_cpu_metadata():
    from statgpu.panel._results import build_panel_index_info

    entity = torch.tensor([1, 1, 2, 2], dtype=torch.int64)
    time = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    info = build_panel_index_info(4, entity_ids=entity, time_ids=time)
    assert info.is_balanced is True
    assert info.n_entities == 2
    assert info.n_times == 2
    assert info.original_order.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("cov_type", ["nonrobust", "robust"])
def test_panel_ols_covariance_registry_torch_cpu_matches_numpy(cov_type):
    from statgpu.panel._covariance import ols_covariance

    X, resid, scale, df = _problem(torch.float64)
    actual = ols_covariance(
        X,
        resid,
        cov_type=cov_type,
        scale=scale,
        df_resid=df,
        xp=torch,
        allowed=("nonrobust", "robust"),
    )

    X_np = X.numpy()
    resid_np = resid.numpy()
    expected = ols_covariance(
        X_np,
        resid_np,
        cov_type=cov_type,
        scale=scale,
        df_resid=df,
        xp=np,
        allowed=("nonrobust", "robust"),
    )
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-13)


def test_panel_shared_inference_finalizer_torch_cpu_matches_numpy():
    from statgpu.backends import TorchBackend
    from statgpu.panel._base import BasePanelModel

    class Dummy(BasePanelModel):
        def fit(self, X, y=None, **fit_params):  # pragma: no cover - unused
            return self

        def predict(self, X):  # pragma: no cover - unused
            return X

    X, resid, scale, df = _problem(torch.float64)
    params = torch.linalg.pinv(X) @ (
        X @ torch.tensor([0.7, -0.3], dtype=torch.float64) + resid
    )
    model = Dummy(device="cpu")
    model.alpha = 0.05
    model._panel_store_ols_inference(
        X,
        resid,
        params,
        scale=scale,
        df_resid=df,
        backend=TorchBackend(device="cpu"),
        cov_type="robust",
        allowed=("nonrobust", "robust"),
        hc1_correction=int(X.shape[0]) / df,
        distribution_df=df,
        diag_floor=1e-30,
    )

    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.bse_).all()
    assert np.isfinite(model.pvalues_).all()
    assert model.conf_int_.shape == (2, 2)
