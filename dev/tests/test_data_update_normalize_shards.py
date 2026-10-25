"""Stdlib synthetic normalizer orchestration tests; subprocesses are mocked."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'benchmarks/data_update'
sys.path.insert(0, str(TOOLS))

import normalize_shards as ns
from preview_export import canonical, digest


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"; directory = self.repo / "dev/benchmarks"; directory.mkdir(parents=True)
        for name in ("export_qualification_preview.py", "generate_benchmark_data.py"):
            (directory / name).write_text("# synthetic only\n")
        self.source = self.base / "source"; self.source.mkdir()
        self.raw = self.source / "preview.json"; self.raw.write_text("{}")
        self.index = self.source / "preview_index.json"
        self.doc = {"schema_version": "statgpu-preview-shard-index-v1", "status": "complete", "shards": [
            {"status": "complete", "path": self.raw.name, "sha256": digest(self.raw)}]}
        self.save()

    def tearDown(self): self.tmp.cleanup()

    def save(self):
        self.doc.pop("index_sha256", None)
        self.doc["index_sha256"] = hashlib.sha256(canonical(self.doc).encode()).hexdigest()
        self.index.write_text(json.dumps(self.doc))

    def execute(self): return ns.normalize(self.repo, self.index, self.base / "out", "2026-10-25")

    def test_hash_mismatch_refuses_before_any_subprocess(self):
        self.raw.write_text('{"changed":true}')
        with patch.object(ns.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"): self.execute()
            run.assert_not_called()

    def test_partial_index_is_not_silently_successful(self):
        self.doc["status"] = "partial"; self.save()
        with self.assertRaisesRegex(ValueError, "partial"): self.execute()

    def test_traversal_is_rejected(self):
        self.doc["shards"][0]["path"] = "../outside.json"; self.save()
        with self.assertRaises(ValueError): self.execute()

    def test_fifo_is_rejected_before_hashing(self):
        self.raw.unlink(); os.mkfifo(self.raw)
        with self.assertRaisesRegex(ValueError, "regular JSON"): self.execute()

    def test_nonobject_index_is_controlled_error(self):
        self.index.write_text("[]")
        with self.assertRaisesRegex(ValueError, "must be an object"): self.execute()

    def test_duplicate_shard_rejected(self):
        self.doc["shards"].append(dict(self.doc["shards"][0])); self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"): self.execute()

    def test_exact_converter_and_generator_commands_only(self):
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            if "--manifest" in command:
                for flag, value in (("--out", {"meta": {"generation_id": "x"}, "runs": []}),
                                    ("--report", {"generation_id": "x", "issues": []}),
                                    ("--inventory-out", {"generation_id": "x"})):
                    target = Path(command[command.index(flag) + 1]); target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(value))
            return SimpleNamespace(returncode=0, stdout="synthetic conversion")
        with patch.object(ns.subprocess, "run", side_effect=fake): result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 2)
        self.assertIn("--manifest-output", calls[0])
        self.assertIn("--strict-sources", calls[1])
        self.assertFalse(any("--execute" in command for command in calls))

    def test_conversion_failure_is_recorded(self):
        with patch.object(ns.subprocess, "run", return_value=SimpleNamespace(returncode=2, stdout="failed")):
            result = self.execute()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["shards"][0]["step"], 1)
        self.assertTrue((self.base / "out/normalization_report.json").is_file())


if __name__ == "__main__": unittest.main()
