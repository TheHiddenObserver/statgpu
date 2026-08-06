"""Regression coverage for the opt-in torch.compile policy."""

from __future__ import annotations


def test_auto_mode_returns_observable_eager_wrapper(monkeypatch):
    from statgpu.backends._torch_compile import (
        compile_torch,
        get_torch_compile_diagnostics,
    )

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    get_torch_compile_diagnostics(clear=True)

    wrapped = compile_torch(lambda value: value + 1, workload="iterative")

    assert wrapped(2) == 3
    assert wrapped.__statgpu_compile_mode__ is None
    assert wrapped.__statgpu_compile_status__ == "disabled"
    events = get_torch_compile_diagnostics(clear=True)
    assert events[-1]["status"] == "disabled"
    assert events[-1]["mode"] is None


def test_requested_mode_cannot_silently_override_auto(monkeypatch):
    from statgpu.backends._torch_compile import resolve_torch_compile_mode

    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "auto")
    assert resolve_torch_compile_mode(
        workload="general", requested_mode="default"
    ) is None
    assert resolve_torch_compile_mode(
        workload="general", requested_mode="reduce-overhead"
    ) is None
