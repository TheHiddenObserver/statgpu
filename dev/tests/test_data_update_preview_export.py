"""Synthetic stdlib contract tests. These never import numerical runners."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'benchmarks/data_update'
sys.path.insert(0, str(TOOLS))

import preview_export as ex


class FakeValidators:
    source = {"git_commit": "a" * 40, "helper_sha256": {}, "test_only": True}

    def assess(self, kind, job, folder, process, expected_sha):
        raw = ex.read_json(folder / "raw.json")
        if kind == "registered":
            ok = raw.get("source", {}).get("git_commit") == expected_sha and raw.get("rows", [{}])[0].get("status") == "pass"
            return {"correctness": {"status": "complete" if ok else "failed", "errors": [], "uncovered": []},
                    "expanded_inventory": {"exact_match": ok}}
        return {"status": "complete", "correctness_status": "pass"}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, allow_nan=False))


def seal(root):
    write(root / "SHA256SUMS.json", {p.relative_to(root).as_posix(): ex.digest(p)
                                     for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"})


def fixture(root):
    root.mkdir()
    source = {"git_commit": "a" * 40, "working_tree": "clean", "content_sha256": "b" * 64}
    job = {"id": "case-t4", "suite": "model-families", "cpu_threads": 4,
           "configuration": {"seed": 17}, "expected_rows": [{"case": "ols", "backend": "numpy", "n": 12, "p": 2}]}
    write(root / "plan.json", {"jobs": [job, dict(job, id="never-started")]})
    write(root / "summary.json", {"schema_version": "statgpu-qualification-campaign-v1", "status": "partial",
                                  "results": [{"id": job["id"], "status": "complete"}]})
    write(root / "source-before.json", source)
    write(root / "source-after.json", source)
    folder = root / job["id"]
    write(folder / "source-after.json", source)
    write(folder / "qualification.json", {"correctness": {"status": "complete"}})
    write(folder / "process.json", {"exit_code": 0, "reason": None})
    snapshot = {"cap_verified": True, "pools": [{"num_threads": 1}], "torch_threads": None}
    write(folder / "runtime.before.json", {"before": snapshot})
    write(folder / "runtime.after.json", {"before": snapshot, "after": snapshot, "exit_code": 0})
    write(folder / "raw.json", {"source": source, "rows": [{"case": "ols", "status": "pass", "backend": "numpy",
          "problem": {"n": 12, "p": 2, "seed": 17}, "timing": {"scope": "predict-only-public-api",
          "raw_seconds": [0.1, 0.3], "median_seconds": 0.2, "warmup": 1, "repeats": 2}}]})
    seal(root)
    return folder


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "campaign"
        self.folder = fixture(self.root)

    def tearDown(self): self.tmp.cleanup()

    def run_export(self, **kwargs):
        return ex.export_roots([self.root], self.base / "out", repo=self.base,
                               _validators=FakeValidators(), **kwargs)

    def test_diagnostic_only_and_missing_jobs_preserved(self):
        got = self.run_export()
        self.assertFalse(got["performance_eligible"])
        self.assertFalse(got["canonical_dashboard_ready"])
        self.assertEqual(got["comparisons"], [])
        self.assertEqual(got["jobs"][0]["validation_status"], "validated_diagnostic")
        self.assertEqual(got["jobs"][1]["validation_status"], "not_run")
        self.assertEqual(got["records"][0]["timing_scope"], "predict-only-public-api")
        self.assertEqual(got["records"][0]["summary_seconds"], 0.2)
        self.assertEqual(got["records"][0]["raw_pointer"], "/rows/0/timing/raw_seconds")

    def test_cap_is_not_exact_thread_count(self):
        got = self.run_export()["jobs"][0]["cpu_thread_profile"]
        self.assertTrue(got["cap_verified"])
        self.assertEqual(got["requested_cap"], 4)
        self.assertFalse(got["all_observed_equal_requested"])

    def test_tampered_raw_is_not_promoted(self):
        raw = ex.read_json(self.folder / "raw.json")
        raw["rows"][0]["timing"]["raw_seconds"] = [0.2, 0.4]
        write(self.folder / "raw.json", raw)
        got = self.run_export()
        self.assertEqual(got["jobs"][0]["validation_status"], "withheld")
        self.assertTrue(any("checksum" in error for error in got["jobs"][0]["evidence_errors"]))
        self.assertTrue(got["records"])

    def test_operator_attestation_does_not_verify_machine(self):
        path = self.base / "attestation.json"
        write(path, {"operator": "owner", "recorded_at": "2026-10-25T00:00:00Z", "environment_id": "p100",
                     "statement": "No competing tasks are expected", "scope": "this new batch only"})
        got = self.run_export(operator_attestation=path)
        self.assertEqual(got["exclusivity"]["operator_attestation"]["kind"], "operator_attested")
        self.assertFalse(got["exclusivity"]["machine_verified"])
        self.assertFalse(got["performance_eligible"])

    def test_failed_process_dominates_pass_sidecar(self):
        write(self.folder / "process.json", {"exit_code": 2, "reason": "timeout"})
        seal(self.root)
        self.assertEqual(self.run_export()["jobs"][0]["validation_status"], "withheld")

    def test_source_drift_withholds_whole_campaign(self):
        source = ex.read_json(self.root / "source-after.json")
        source["content_sha256"] = "c" * 64
        write(self.root / "source-after.json", source)
        seal(self.root)
        self.assertEqual(self.run_export()["jobs"][0]["validation_status"], "withheld")

    def test_unknown_scope_and_zero_stay_raw(self):
        values = list(ex.timing_records({"timing": {"raw_seconds": [0.0, 0.0]}}))
        self.assertIsNone(values[0]["timing_scope"])
        self.assertNotIn("summary_seconds", values[0])

    def test_units_and_dispersion_not_confused(self):
        values = list(ex.timing_records({"timing": {"scope": "operation", "times_ms": [2, 4], "time_ms_std": 2}}))
        self.assertEqual(values[0]["summary_seconds"], .003)
        self.assertNotIn("summary_seconds", values[1])

    def test_compile_mode_arrays_use_recorded_fit_contract(self):
        doc = {"timing_scope": {"fit": "construction plus fit; excludes data and prediction"},
               "results": [{"case": "lasso", "raw_fit_seconds": {"disable": [1, 2], "default": [3, 4]}}]}
        rows = list(ex.timing_records(doc))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["raw_pointer"], "/results/0/raw_fit_seconds/disable")
        self.assertEqual(rows[0]["timing_scope"], doc["timing_scope"]["fit"])
        self.assertEqual(rows[1]["context"]["mode"], "default")

    def test_kde_global_timing_scope_is_not_lost(self):
        doc = {"timing": {"scope": "public-bootstrap-end-to-end", "warmup": 1},
               "rows": [{"backend": "cupy", "timing": {"samples_seconds": [1, 2]}}]}
        row = list(ex.timing_records(doc))[0]
        self.assertEqual(row["timing_scope"], "public-bootstrap-end-to-end")
        self.assertEqual(row["context"]["warmup"], 1)
        self.assertEqual(row["row_pointer"], "/rows/0")

    def test_unknown_seconds_like_field_is_not_manufactured_timing(self):
        self.assertEqual(list(ex.timing_records({"not_a_timer_seconds_scale": [1, 2, 3]})), [])

    def test_missing_and_empty_samples_stay_visible_without_central_value(self):
        rows = list(ex.timing_records({"timing": {"scope": "x", "raw_seconds": [], "median_seconds": None, "iqr_seconds": .2}}))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["raw_value"], [])
        self.assertIsNone(rows[1]["raw_value"])
        for row in rows: self.assertNotIn("summary_seconds", row)

    def test_dispersion_and_deltas_never_become_duration_arrays(self):
        for field in ("iqr_seconds", "std_seconds", "min_seconds", "digest_overhead_seconds", "representative_seconds"):
            row = list(ex.timing_records({"timing": {"scope": "x", field: [.1, .2]}}))[0]
            self.assertNotIn("summary_seconds", row)

    def test_comparison_candidates_require_exact_metadata_and_never_compute_ratio(self):
        base = {"record_id": "cpu", "campaign_id": "session", "suite_or_lane": "gaussian-refresh",
                "source_commit": "a" * 40, "source_fingerprint": "c" * 64, "environment": {"environment_id": "same"},
                "validation_status": "validated_diagnostic", "timing_scope": "fit-only", "sample_count": 2,
                "summary_seconds": .2, "configuration": {}, "cpu_thread_profile": {"requested_cap": 1},
                "context": {"backend": "numpy", "problem": {"data_id": "b" * 64, "seed": 17},
                            "transfer_policy": "excluded", "warmup": 1, "repeats": 2}}
        gpu = copy.deepcopy(base); gpu.update(record_id="gpu", summary_seconds=.1)
        gpu["context"].update(backend="cupy", synchronized_device="cuda:0")
        candidates, _ = ex.comparison_candidates([base, gpu])
        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0]["speedup"])
        self.assertFalse(candidates[0]["performance_eligible"])
        gpu["context"]["problem"]["seed"] = 18
        self.assertEqual(ex.comparison_candidates([base, gpu])[0], [])
        gpu["context"]["problem"]["seed"] = 17; gpu["source_fingerprint"] = "d" * 64
        self.assertEqual(ex.comparison_candidates([base, gpu])[0], [])

    def test_cv_stages_and_derived_differences_are_retained_separately(self):
        doc = {"repeat_samples": [{"cv_evaluation_ms": 2, "final_refit_ms": 3, "total_fit_ms": 5}],
               "digest_overhead_seconds": -0.01, "first_fit_extra_seconds_vs_eager": 3,
               "representative_seconds": .2}
        rows = list(ex.timing_records(doc))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["unit"], "milliseconds")
        for row in rows[3:]: self.assertNotIn("summary_seconds", row)

    def test_runtime_failure_dominates_valid_pool_snapshots(self):
        state = ex.read_json(self.folder / "runtime.after.json"); state["exit_code"] = 86
        write(self.folder / "runtime.after.json", state); seal(self.root)
        self.assertEqual(self.run_export()["jobs"][0]["validation_status"], "withheld")

    def test_torch_profile_requires_observed_torch_pools(self):
        state = {"before": {"cap_verified": True, "pools": [{"num_threads": 1}], "torch_threads": None},
                 "after": {"cap_verified": True, "pools": [{"num_threads": 1}], "torch_threads": {}}}
        self.assertFalse(ex.thread_profile(state, {"cpu_threads": 1, "backend": "torch"}, "registered")["cap_verified"])

    def test_duplicate_keys_nonfinite_and_traversal_rejected(self):
        for payload in ('{"a": 1, "a": 2}', '{"x":NaN}', '{"x":1e999}'):
            path = self.base / "bad.json"; path.write_text(payload)
            with self.assertRaises(ValueError): ex.read_json(path)
        with self.assertRaises(ValueError): ex.safe_path(self.root, "../secret")

    def test_symlink_evidence_rejected(self):
        link = self.root / "link.json"
        link.symlink_to(self.folder / "raw.json")
        with self.assertRaises(ValueError): ex.safe_path(self.root, "link.json")

    def test_no_overwrite(self):
        self.run_export()
        with self.assertRaises(ValueError): self.run_export()

    def test_output_budget_checked_before_any_files_written(self):
        original = ex.read_json
        with patch.object(ex, "MAX_JSON_BYTES", 10), patch.object(ex, "read_json", side_effect=lambda path: self.read_with_normal_budget(path, original)):
            with self.assertRaisesRegex(ValueError, "budget"): self.run_export()
        self.assertFalse((self.base / "out").exists())

    @staticmethod
    def read_with_normal_budget(path, original):
        with patch.object(ex, "MAX_JSON_BYTES", 64 * 1024**2): return original(path)

    def test_invalid_thread_type_is_withheld_not_crashed(self):
        state = {"before": {"cap_verified": True, "pools": [{"num_threads": 1}]},
                 "after": {"cap_verified": True, "pools": [{"num_threads": 1}]}}
        self.assertFalse(ex.thread_profile(state, {"cpu_threads": "4"}, "registered")["cap_verified"])

    def test_malformed_runtime_object_withholds_instead_of_crashing(self):
        write(self.folder / "runtime.after.json", []); seal(self.root)
        self.assertEqual(self.run_export()["jobs"][0]["validation_status"], "withheld")

    def test_malformed_source_object_withholds_instead_of_crashing(self):
        write(self.root / "source-before.json", []); seal(self.root)
        self.assertEqual(self.run_export()["jobs"][0]["validation_status"], "withheld")

    def test_results_discovery_does_not_scan_arbitrary_nested_json(self):
        nested = self.base / "untrusted" / "extra"
        fixture(nested.parent)
        nested.mkdir()
        self.assertEqual(ex.discover_roots([self.base / "untrusted"]), [(self.base / "untrusted", "registered")])

    def test_campaign_shards_preserve_every_job_record_and_ledger(self):
        other = self.base / "second"; fixture(other)
        ledger = self.base / "ledger.json"
        write(ledger, {"schema_version": "statgpu-batch-coverage-ledger-v1", "counts": {"not_run": 9}, "slots": [{"status": "not_run"}]})
        result = ex.export_shards([self.root, other], self.base / "shards", repo=self.base,
                                 coverage_ledger=ledger, _validators=FakeValidators())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"]["jobs"], 4)
        self.assertEqual(result["counts"]["timing_fields"], 4)
        self.assertEqual(len(result["shards"]), 2)
        self.assertEqual(ex.digest(self.base / "shards" / "coverage_ledger.json"), ex.digest(ledger))
        for row in result["shards"]:
            self.assertEqual(ex.digest(self.base / "shards" / row["path"]), row["sha256"])




class CVScopeTests(unittest.TestCase):
    @staticmethod
    def document():
        return {"schema_version": "statgpu-scoped-cv-candidate-v2", "results": [{
            "model_id": "RidgeCV", "backend": "torch", "n_samples": 20, "n_features": 2,
            "case": {"case_id": "cv-ridge", "model_id": "RidgeCV", "runs": [
                {"framework": "statgpu", "backend": "torch", "timing": {"cv_evaluation_ms": 100},
                 "repeat_samples": [{"cv_evaluation_ms": 100}]},
                {"framework": "sklearn", "backend": None, "timing": {"total_fit_ms": 300}}]},
            "protocol": {"timing_scope": "profiling overhead included", "repeats": 1, "warmup": 1},
            "diagnostic_timing": {"instrumented": True},
            "clean_public_fit": {"scope": "clean public fit", "instrumented": False,
                                 "raw_seconds": [.2], "timing": {"median_seconds": .2},
                                 "repeat_samples": [{"total_fit_ms": 200}]},
            "other": {"total_fit_ms": 700}}]}

    def test_protocol_scope_is_bound_to_exact_cv_phase_paths(self):
        rows = list(ex.timing_records(self.document()))
        stages = [row for row in rows if "/case/runs/" in row["raw_pointer"]]
        self.assertEqual(len(stages), 3)
        for row in stages:
            self.assertEqual(row["timing_scope"], "profiling overhead included")
            self.assertEqual(row["context"]["scope_source_pointer"], "/results/0/protocol/timing_scope")
            self.assertEqual(row["context"]["timing_kind"], "cv_diagnostic_stage")
        self.assertTrue(stages[0]["context"]["instrumented"])
        self.assertEqual(stages[0]["context"]["case_id"], "cv-ridge")
        self.assertEqual(stages[-1]["context"]["framework"], "sklearn")
        self.assertIsNone(stages[-1]["context"]["backend"])
        clean = [row for row in rows if "/clean_public_fit/" in row["raw_pointer"]]
        self.assertEqual(len(clean), 3)
        for row in clean:
            self.assertEqual(row["timing_scope"], "clean public fit")
            self.assertFalse(row["context"]["instrumented"])
            self.assertEqual(row["context"]["timing_kind"], "clean_public_fit")
        self.assertIsNone(rows[-1]["timing_scope"])

    def test_unknown_schema_does_not_borrow_cv_protocol_scope(self):
        doc = self.document(); doc["schema_version"] = "unknown"
        row = next(ex.timing_records(doc))
        self.assertIsNone(row["timing_scope"])
        self.assertNotIn("source_schema", row)

    def test_nested_identity_conflict_survives_context_traversal(self):
        doc = self.document(); doc["results"][0]["case"]["model_id"] = "OtherCV"
        row = next(ex.timing_records(doc))
        self.assertEqual(row["context"]["identity_conflicts"], ["model_id"])

    def test_sibling_result_protocols_do_not_leak(self):
        doc = self.document(); second = copy.deepcopy(doc["results"][0])
        second["protocol"]["timing_scope"] = "other result diagnostic scope"
        doc["results"].append(second)
        rows = list(ex.timing_records(doc))
        for row in rows:
            if "/case/runs/" not in row["raw_pointer"]: continue
            expected = "other result diagnostic scope" if row["raw_pointer"].startswith("/results/1/") else "profiling overhead included"
            self.assertEqual(row["timing_scope"], expected)


if __name__ == "__main__": unittest.main()
