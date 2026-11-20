"""Canonical PGLM cleanup ownership and legacy entry point regressions."""

import pickle
import subprocess
import sys
import textwrap
import types

import pytest

from statgpu.linear_model.penalized import (
    _final_finite_validation_cleanup_contract as direct,
    _latest_review_fix_contract as latest,
    _no_inference_public_validation_reset_contract as reset,
)


def test_historical_installers_do_not_replace_active_cleanup():
    # Use a fresh interpreter: reloading private installers must not leak
    # state into the other fit/validation tests in this pytest process.
    script = """
        import importlib
        from statgpu.linear_model import PenalizedGeneralizedLinearModel as PGLM
        from statgpu.linear_model.penalized import (
            _no_inference_public_validation_reset_contract as reset,
            _latest_review_fix_contract as latest,
            _final_finite_validation_cleanup_contract as direct,
        )

        fit = PGLM.fit
        reset_fit = PGLM._reset_fit_state
        for modules in [(latest, direct, reset), (reset, direct, latest)]:
            for module in modules:
                importlib.reload(module)
                cleanup = reset._cleanup_failed_public_finite_validation
                assert cleanup.__module__ == reset.__name__
                latest._install_public_validation_cleanup_contract()
                direct._install_direct_public_validation_cleanup_contract()
                reset._install_no_inference_public_validation_reset_contract()
                assert reset._cleanup_failed_public_finite_validation is cleanup
                assert PGLM.fit is fit
                assert PGLM._reset_fit_state is reset_fit
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True)


@pytest.mark.parametrize(
    ("entry", "owner"),
    [
        (reset._cleanup_failed_public_finite_validation, "captured"),
        (direct._cleanup_failed_public_finite_validation_direct, "captured"),
        (latest._cleanup_failed_public_finite_validation_safe, "live"),
    ],
)
@pytest.mark.parametrize("backend", [" CuPy ", " Torch ", "numpy"])
@pytest.mark.parametrize("context_failure", [None, "constructor", "enter", "exit"])
def test_finite_cleanup_entry_routes_once_and_preserves_error(
    monkeypatch, entry, owner, backend, context_failure
):
    events = []

    class Device:
        def __init__(self, device_id):
            events.append(("device", device_id))
            if context_failure == "constructor":
                raise RuntimeError("constructor failed")

        def __enter__(self):
            events.append("enter")
            if context_failure == "enter":
                raise RuntimeError("entry failed")

        def __exit__(self, *args):
            events.append(("exit", args))
            if context_failure == "exit":
                raise RuntimeError("exit failed")

    def cleanup(name):
        events.append(name)
        raise RuntimeError("advisory cleanup failed")

    monkeypatch.setitem(
        sys.modules, "cupy", types.SimpleNamespace(cuda=types.SimpleNamespace(Device=Device))
    )
    estimator = types.SimpleNamespace(
        _cleanup_cuda_memory=lambda: cleanup(("live", "cupy")),
        _cleanup_torch_memory=lambda: cleanup(("live", "torch")),
        _statgpu_finite_direct_cleanup_cuda=lambda: cleanup(("captured", "cupy")),
        _statgpu_finite_direct_cleanup_torch=lambda: cleanup(("captured", "torch")),
    )
    error = ValueError("invalid input")
    error._statgpu_finite_backend = backend
    error._statgpu_finite_device = " CUDA:2 "
    with pytest.raises(ValueError) as caught:
        try:
            raise error
        except ValueError:
            assert entry(estimator) is None
            raise

    assert caught.value is error
    if backend == " CuPy ":
        expected = [("device", 2)]
        if context_failure != "constructor":
            expected.append("enter")
        expected.append((owner, "cupy"))
        if context_failure not in ("constructor", "enter"):
            expected.append(("exit", (None, None, None)))
        assert events == expected
    elif backend == " Torch ":
        assert events == [(owner, "torch")]
    else:
        assert events == []


def test_canonical_cleanup_can_run_before_direct_capture():
    calls = []
    estimator = types.SimpleNamespace(_cleanup_cuda_memory=lambda: calls.append("live"))
    error = ValueError("invalid input")
    error._statgpu_finite_backend = "cupy"
    error._statgpu_finite_device = "cuda:unknown"
    try:
        raise error
    except ValueError:
        reset._cleanup_failed_public_finite_validation(estimator)
    assert calls == ["live"]


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (direct, "_cleanup_failed_public_finite_validation_direct"),
        (direct, "_install_direct_public_validation_cleanup_contract"),
        (latest, "_cleanup_failed_public_finite_validation_safe"),
        (latest, "_install_public_validation_cleanup_contract"),
        (reset, "_cleanup_failed_public_finite_validation"),
    ],
)
def test_historical_callable_pickle_paths(module, name):
    # Protocol 0 names the old global directly, independent of the current
    # function object's module and name metadata.
    data = f"c{module.__name__}\n{name}\n.".encode()
    assert pickle.loads(data) is getattr(module, name)
