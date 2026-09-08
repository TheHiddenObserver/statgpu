import numpy as np
import pytest

from statgpu.linear_model import Lasso


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan, np.inf])
def test_simultaneous_alpha_rejected_before_backend_dispatch(device, alpha):
    with pytest.raises(ValueError, match=r"simultaneous_alpha must be in \(0, 1\)"):
        Lasso(
            alpha=0.1,
            inference_method="debiased",
            compute_inference=True,
            enable_simultaneous_inference=True,
            simultaneous_alpha=alpha,
            device=device,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
@pytest.mark.parametrize("n_bootstrap", [0, -1, -20])
def test_simultaneous_bootstrap_count_rejected_before_backend_dispatch(
    device,
    n_bootstrap,
):
    with pytest.raises(
        ValueError,
        match="simultaneous_n_bootstrap must be a positive integer",
    ):
        Lasso(
            alpha=0.1,
            inference_method="debiased",
            compute_inference=True,
            enable_simultaneous_inference=True,
            simultaneous_n_bootstrap=n_bootstrap,
            device=device,
        )


def test_disabled_simultaneous_preserves_constructor_compatibility():
    model = Lasso(
        alpha=0.1,
        inference_method="debiased",
        compute_inference=True,
        enable_simultaneous_inference=False,
        simultaneous_alpha=0.0,
        simultaneous_n_bootstrap=0,
        device="cpu",
    )
    assert model.enable_simultaneous_inference is False
    assert model.simultaneous_alpha == 0.0
    assert model.simultaneous_n_bootstrap == 0


def test_set_params_reuses_simultaneous_validation_contract():
    model = Lasso(
        alpha=0.1,
        inference_method="debiased",
        compute_inference=True,
        enable_simultaneous_inference=True,
        simultaneous_alpha=0.05,
        simultaneous_n_bootstrap=32,
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"simultaneous_alpha must be in \(0, 1\)"):
        model.set_params(simultaneous_alpha=0.0)
