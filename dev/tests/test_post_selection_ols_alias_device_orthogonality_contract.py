import pytest

from statgpu._config import Device
from statgpu.linear_model import Lasso
from statgpu.linear_model import _penalized_inference_api_contract as api_contract


@pytest.mark.parametrize(
    ("alias", "device", "expected_device"),
    [
        ("gpu_ols", "cpu", Device.CPU),
        ("cpu_ols", "cuda", Device.CUDA),
        ("cpu_ols", "torch", Device.TORCH),
    ],
)
def test_legacy_inference_alias_never_selects_execution_device(
    monkeypatch,
    alias,
    device,
    expected_device,
):
    with pytest.warns(FutureWarning, match="post_selection_ols"):
        model = Lasso(
            inference_method=alias,
            compute_inference=False,
            device=device,
        )

    assert model._inference_method == "post_selection_ols"
    assert model._device is expected_device

    # Even when the input looks GPU-native and the global policy is AUTO, an
    # explicit estimator device remains authoritative. The deprecated method
    # spelling may normalize the statistical method but must never repin device.
    fake_native = object()
    monkeypatch.setattr(api_contract, "_get_configured_device", lambda: Device.AUTO)
    monkeypatch.setattr(api_contract, "_is_cupy_array", lambda value: value is fake_native)
    monkeypatch.setattr(api_contract, "_is_torch_array", lambda value: False)
    assert api_contract._input_native_device(model, fake_native) is None
