from pathlib import Path

import dev.benchmarks.pr79.emit_final_report as renderer
import pytest


def test_full_renderer_defaults_to_canonical_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        renderer, "load_json_strict", lambda _path: {"configuration": "full"}
    )
    captured = {}

    def fake_emit(validated, output_json, output_markdown):
        captured.update(
            validated=validated,
            output_json=output_json,
            output_markdown=output_markdown,
        )

    monkeypatch.setattr(renderer, "emit_report", fake_emit)

    assert renderer.main(["--config", "full"]) == 0
    assert captured["output_json"] == Path(
        "results/pr79/final/final_accuracy_report.json"
    )
    assert captured["output_markdown"] == Path(
        "results/pr79/final/final_accuracy_report.md"
    )


def test_nonfull_renderer_keeps_artifacts_out_of_canonical_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        renderer, "load_json_strict", lambda _path: {"configuration": "smoke"}
    )
    captured = {}
    monkeypatch.setattr(
        renderer,
        "emit_report",
        lambda _validated, output_json, output_markdown: captured.update(
            output_json=output_json, output_markdown=output_markdown
        ),
    )

    assert renderer.main(["--config", "smoke"]) == 0
    assert captured["output_json"] == Path(
        "results/pr79/accuracy/smoke_final_report.json"
    )
    assert captured["output_markdown"] == Path(
        "results/pr79/accuracy/smoke_final_report.md"
    )


def test_renderer_rejects_configuration_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(
        renderer, "load_json_strict", lambda _path: {"configuration": "smoke"}
    )
    monkeypatch.setattr(
        renderer,
        "emit_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("emit_report must not run")
        ),
    )

    assert renderer.main(["--config", "full"]) == 1
    assert "configuration does not match" in capsys.readouterr().err


def test_renderer_rejects_noncanonical_pass_claim():
    with pytest.raises(renderer.ReportValidationError, match="exact-head"):
        renderer.validate_aggregated_report(
            {
                "validated_schema_version": "pr79-validated-accuracy-1.0",
                "status": "pass",
                "canonical_eligible": False,
            }
        )
