#!/usr/bin/env python3
"""Exact-source wrapper for the PR #164 physical CUDA acceptance matrix.

The numerical matrix runner writes to a temporary directory first. This wrapper
requires the repository to be clean before and after that run, verifies the
recorded source SHA/status, then publishes the final JSON artifact. The final
artifact write is deliberately last so it does not invalidate the clean-source
check it records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


RUNNER = Path(__file__).with_name("validate_quantile_solver_provenance_gpu.py")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "PR164 physical acceptance requires a clean exact-source worktree"
        )
    return sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr164_quantile_solver_provenance_gpu.json",
    )
    args = parser.parse_args()

    source_before = _require_clean_source()
    with tempfile.TemporaryDirectory(prefix="statgpu-pr164-") as temp_dir:
        temp_output = Path(temp_dir) / "pr164-gpu.json"
        subprocess.run(
            [sys.executable, str(RUNNER), "--output", str(temp_output)],
            check=True,
        )
        payload = json.loads(temp_output.read_text(encoding="utf-8"))

    source_after = _require_clean_source()
    if source_after != source_before:
        raise RuntimeError(
            "PR164 source HEAD changed during physical validation: "
            f"{source_before} -> {source_after}"
        )
    if payload.get("source_sha") != source_before:
        raise RuntimeError(
            "PR164 inner-runner source SHA mismatch: "
            f"{payload.get('source_sha')!r} != {source_before!r}"
        )
    if payload.get("source_clean") is not True:
        raise RuntimeError("PR164 inner runner did not record a clean source")
    if payload.get("status") != "success":
        raise RuntimeError(
            f"PR164 inner runner did not succeed: {payload.get('status')!r}"
        )

    payload["source_sha_before"] = source_before
    payload["source_sha_after_execution"] = source_after
    payload["source_clean_before"] = True
    payload["source_clean_after_execution"] = True
    payload["evidence_wrapper"] = str(Path(__file__).as_posix())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
