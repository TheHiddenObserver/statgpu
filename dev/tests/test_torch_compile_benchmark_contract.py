"""Regression tests for the physical torch.compile benchmark evidence policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "dev"
    / "benchmarks"
    / "benchmark_torch_compile_maintenance.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_torch_compile_maintenance", _BENCHMARK_PATH
)
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BENCHMARK)


def test_compile_evidence_accepts_new_and_cached_compiled_callables():
    assert (
        _BENCHMARK._validate_compile_evidence(
            "default", "scad", ({"status": "compiled"},), 2
        )
        == "compiled-diagnostic-and-dynamo-graph"
    )
    assert (
        _BENCHMARK._validate_compile_evidence("default", "mcp", (), 2)
        == "cached-callable-and-dynamo-graph"
    )


def test_compile_evidence_is_not_required_when_compilation_is_disabled():
    assert (
        _BENCHMARK._validate_compile_evidence("disable", "mcp", (), 0)
        == "not-applicable"
    )


@pytest.mark.parametrize("graph_delta", [0, -1])
def test_compile_evidence_requires_case_local_dynamo_graph(graph_delta):
    with pytest.raises(RuntimeError, match="did not create a Dynamo graph"):
        _BENCHMARK._validate_compile_evidence(
            "default", "mcp", (), graph_delta
        )


@pytest.mark.parametrize(
    "events, message",
    [
        (({"status": "runtime-fallback"},), "entered fallback"),
        (({"status": "construction-fallback"},), "entered fallback"),
        (({"status": "disabled"},), "no compiled event"),
        (({"status": "unavailable"},), "no compiled event"),
    ],
)
def test_compile_evidence_rejects_noncompiled_diagnostics(events, message):
    with pytest.raises(RuntimeError, match=message):
        _BENCHMARK._validate_compile_evidence(
            "default", "mcp", events, 1
        )
