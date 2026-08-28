"""Behavioral guard for the issue #127 numerical/reporting boundary."""

from __future__ import annotations

import numpy as np
import pytest


def test_pglm_torch_numerics_finish_before_any_reporting_snapshot(monkeypatch):
    torch = pytest.importorskip("torch")

    import statgpu.linear_model._gaussian_inference as gi
    import statgpu.linear_model.penalized._base as pglm_base
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    phase = {"reporting_allowed": False, "gi_snapshots": 0, "pglm_snapshots": 0}
    real_gi_to_numpy = gi._to_numpy
    real_pglm_to_numpy = pglm_base._to_numpy

    def guarded_gi_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "Gaussian numerical inference attempted a host snapshot before "
                "reference-distribution work completed"
            )
        phase["gi_snapshots"] += 1
        return real_gi_to_numpy(value)

    def guarded_pglm_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "PGLM reporting state attempted a host snapshot before numerical "
                "Gaussian inference completed"
            )
        phase["pglm_snapshots"] += 1
        return real_pglm_to_numpy(value)

    def fake_reference_inference(
        statistic_abs,
        *,
        distribution,
        alpha,
        backend,
        xp,
        df=None,
        device=None,
    ):
        assert distribution == "t"
        assert backend == "torch"
        assert str(device) == "cpu"
        assert isinstance(statistic_abs, torch.Tensor)
        assert statistic_abs.device.type == "cpu"
        assert alpha == pytest.approx(0.05)
        assert df is not None and int(df) > 0
        phase["reporting_allowed"] = True
        return (
            torch.full_like(statistic_abs, 0.25),
            torch.tensor(2.0, dtype=torch.float64),
        )

    monkeypatch.setattr(gi, "_to_numpy", guarded_gi_to_numpy)
    monkeypatch.setattr(pglm_base, "_to_numpy", guarded_pglm_to_numpy)
    monkeypatch.setattr(
        gi, "two_sided_reference_inference", fake_reference_inference
    )

    X = np.asarray(
        [[-2.0, 0.5], [-1.0, 1.0], [0.0, 1.5], [1.0, 2.0], [2.0, 2.5], [3.0, 3.0]],
        dtype=np.float64,
    )
    coef = np.asarray([0.75, -0.4])
    intercept = 1.2
    y = intercept + X @ coef + np.asarray([0.2, -0.1, 0.15, -0.25, 0.05, -0.05])

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        device="cpu",
        compute_inference=True,
    )
    model._penalty = model._resolve_penalty()
    model._selected_backend_name = "torch"
    model.coef_ = coef.copy()
    model.intercept_ = float(intercept)

    model._compute_post_fit_gaussian_inference(X, y)

    assert phase["reporting_allowed"] is True
    assert phase["gi_snapshots"] == 5  # params, bse, statistic, pvalue, CI
    assert phase["pglm_snapshots"] == 5  # design, y, residual, params, scale
    assert model._inference_result.metadata["numerical_backend"] == "torch"
    assert model._inference_result.metadata["reporting_boundary"] == "post_numerical_inference"
