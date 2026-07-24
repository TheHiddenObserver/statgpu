"""Git-integrity tests for PR79 canonical accuracy evidence."""

from __future__ import annotations

import copy

import pytest

from dev.benchmarks.pr79 import aggregate_results as aggregate_module
from dev.benchmarks.pr79 import run_accuracy


SHA = "a" * 40


def _snapshot(*, clean: bool, sha: str = SHA) -> dict:
    return {
        "git_sha": sha,
        "worktree_clean": clean,
        "dirty_entries": [] if clean else [" M statgpu/example.py"],
        "inspection_error": None,
    }


def _provenance(*, clean: bool, allow_dirty: bool = False) -> dict:
    return run_accuracy._repository_provenance(
        _snapshot(clean=clean),
        _snapshot(clean=clean),
        allow_dirty=allow_dirty,
    )


def _empty_manifest() -> dict:
    return {
        "configurations": {
            "empty": {
                "cases": [],
                "backends": [],
                "iterations": 1,
                "warmup": 0,
            }
        }
    }


def test_collection_rejects_dirty_snapshot_by_default(monkeypatch):
    monkeypatch.setattr(run_accuracy, "_git_snapshot", lambda: _snapshot(clean=False))
    with pytest.raises(run_accuracy.RepositoryIntegrityError, match="non-clean"):
        run_accuracy.collect_accuracy(
            config_name="unused",
            manifest={"configurations": {}},
        )


def test_collection_allow_dirty_marks_raw_noncanonical(monkeypatch):
    snapshots = iter((_snapshot(clean=False), _snapshot(clean=False)))
    monkeypatch.setattr(run_accuracy, "_git_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(run_accuracy, "_case_specs", lambda: {})
    raw = run_accuracy.collect_accuracy(
        config_name="empty",
        manifest=_empty_manifest(),
        allow_dirty=True,
    )
    assert raw["source_schema_version"] == "pr79-benchmark-source-2.1"
    assert raw["repository_provenance"]["allow_dirty_requested"] is True
    assert raw["repository_provenance"]["canonical_eligible"] is False


def test_collection_clean_snapshot_is_canonical(monkeypatch):
    snapshots = iter((_snapshot(clean=True), _snapshot(clean=True)))
    monkeypatch.setattr(run_accuracy, "_git_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(run_accuracy, "_case_specs", lambda: {})
    raw = run_accuracy.collect_accuracy(
        config_name="empty", manifest=_empty_manifest()
    )
    assert raw["git_sha"] == SHA
    assert raw["repository_provenance"]["canonical_eligible"] is True


def test_aggregate_dirty_raw_cannot_be_canonical_pass(monkeypatch):
    monkeypatch.setattr(
        aggregate_module, "_git_snapshot", lambda: _snapshot(clean=True)
    )
    raw = {
        "source_schema_version": "pr79-benchmark-source-2.1",
        "git_sha": SHA,
        "repository_provenance": _provenance(clean=False, allow_dirty=True),
        "cases": {},
    }
    monkeypatch.setattr(
        aggregate_module, "_validate_raw_schema", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(aggregate_module, "expected_checks", lambda *args: [])
    with pytest.raises(
        aggregate_module.AggregationError, match="non-canonical|refused"
    ):
        aggregate_module.aggregate_results(
            raw,
            {},
            config_name="smoke",
            expected_sha=SHA,
        )


def test_aggregate_clean_provenance_can_pass(monkeypatch):
    monkeypatch.setattr(
        aggregate_module, "_git_snapshot", lambda: _snapshot(clean=True)
    )
    raw = {
        "repository_provenance": _provenance(clean=True),
        "cases": {},
    }
    monkeypatch.setattr(
        aggregate_module, "_validate_raw_schema", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(aggregate_module, "expected_checks", lambda *args: [])
    report = aggregate_module.aggregate_results(
        raw,
        {},
        config_name="smoke",
        expected_sha=SHA,
    )
    assert report["status"] == "pass"
    assert report["canonical_eligible"] is True
    assert report["summary"]["gate_verdict"] == "PASS"
    assert report["repository_provenance"]["noncanonical_reasons"] == []


def test_aggregate_clean_but_different_head_cannot_pass(monkeypatch):
    monkeypatch.setattr(
        aggregate_module,
        "_git_snapshot",
        lambda: _snapshot(clean=True, sha="b" * 40),
    )
    raw = {
        "repository_provenance": _provenance(clean=True),
        "cases": {},
    }
    monkeypatch.setattr(
        aggregate_module, "_validate_raw_schema", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(aggregate_module, "expected_checks", lambda *args: [])
    with pytest.raises(aggregate_module.AggregationError) as caught:
        aggregate_module.aggregate_results(
            raw,
            {},
            config_name="smoke",
            expected_sha=SHA,
        )
    assert caught.value.report is not None
    assert caught.value.report["status"] == "fail"
    assert any(
        "HEAD does not match" in reason
        for reason in caught.value.report["repository_provenance"][
            "noncanonical_reasons"
        ]
    )


def test_aggregate_allow_dirty_emits_failed_noncanonical_report(monkeypatch):
    monkeypatch.setattr(
        aggregate_module, "_git_snapshot", lambda: _snapshot(clean=False)
    )
    raw = {
        "repository_provenance": _provenance(clean=False, allow_dirty=True),
        "cases": {},
    }
    monkeypatch.setattr(
        aggregate_module, "_validate_raw_schema", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(aggregate_module, "expected_checks", lambda *args: [])
    with pytest.raises(aggregate_module.AggregationError) as caught:
        aggregate_module.aggregate_results(
            raw,
            {},
            config_name="smoke",
            expected_sha=SHA,
            allow_dirty=True,
        )
    report = caught.value.report
    assert report is not None
    assert report["status"] == "fail"
    assert report["canonical_eligible"] is False
    assert report["summary"]["gate_verdict"] == "NONCANONICAL_FAIL"
    assert report["repository_provenance"]["noncanonical_reasons"]


def test_cli_allow_dirty_is_explicit_and_default_is_false():
    assert run_accuracy._parse_args([]).allow_dirty is False
    assert run_accuracy._parse_args(["--allow-dirty"]).allow_dirty is True
    assert aggregate_module._parse_args([]).allow_dirty is False
    assert aggregate_module._parse_args(["--allow-dirty"]).allow_dirty is True


def test_dirty_provenance_cannot_be_hidden_by_claiming_canonical():
    forged = copy.deepcopy(_provenance(clean=False, allow_dirty=False))
    forged["canonical_eligible"] = True
    reasons = aggregate_module._provenance_noncanonical_reasons(forged)
    assert any("not clean" in reason for reason in reasons)
