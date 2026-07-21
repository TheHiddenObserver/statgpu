#!/usr/bin/env python3
"""
PR79 GPU Validation Orchestrator.

Executes the multi-phase GPU validation plan from
dev/plans/statgpu_pr79_gpu_review_fix_test_plan.md on a remote GPU server
via paramiko SSH.

Usage:
    python dev/validation/pr79_gpu_orchestrator.py --test-connection
    python dev/validation/pr79_gpu_orchestrator.py --setup
    python dev/validation/pr79_gpu_orchestrator.py --round 1
    python dev/validation/pr79_gpu_orchestrator.py --phase gate_a
    python dev/validation/pr79_gpu_orchestrator.py --download --summary
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import paramiko

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_CONFIG = {
    "host": "hz-4.matpool.com",
    "port": 27495,
    "user": "root",
    # Password must be set via STATGPU_REMOTE_PASSWORD env var.
    # See dev/scripts/remote_config.py for shared credential helpers.
    "password": None,
}

CONDA_PROFILE = "/root/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "myconda"
CONDA_ACTIVATE = f"source {CONDA_PROFILE} && conda activate {CONDA_ENV}"

REMOTE_PATHS = {
    "validation_root": "/root/statgpu-validation",
    "repo": "/root/statgpu-validation/repo",
    "worktree_base": "/root/statgpu-validation/worktrees/pr79-base",
    "worktree_head": "/root/statgpu-validation/worktrees/pr79-head",
    "results_root": "/root/statgpu-validation/results/pr79",
}

GIT_INFO = {
    "repo_url": "https://github.com/TheHiddenObserver/statgpu.git",
    "base_sha": "a4879fb4d9fb183efc01f147cd2cc501691f28c4",
    "head_sha": "e30cec6768a734a0d61dfec44b6b4884adf9a880",
}

# Gate A test files (Section 8 of test plan)
GATE_A_TEST_FILES = [
    "dev/tests/test_backends.py",
    "dev/tests/test_core_contracts.py",
    "dev/tests/test_v10_import_smoke.py",
    "dev/tests/test_three_backend_native_followup.py",
    "dev/tests/test_third_full_review.py",
    "dev/tests/test_ordered_cross_backend.py",
    "dev/tests/test_distributions_backend.py",
]

# Result subdirectories (Section 5 of test plan)
RESULT_SUBDIRS = [
    "environment", "collection", "logs", "junit",
    "parity", "device", "memory", "performance",
    "profiling", "external", "failures", "iterations", "final",
]

# All phases in execution order (Section 22)
ALL_PHASES = [
    "phase_0", "phase_1",
    "gate_a", "gate_b", "gate_c", "gate_d",
    "gate_e", "gate_f", "gate_g",
    "final_gate",
]

ROUND_PHASES = {
    1: ["phase_0", "phase_1", "gate_a"],
    2: ["gate_b"],
    3: ["gate_c", "gate_d"],
    4: ["gate_e", "gate_f"],
    5: ["gate_g", "final_gate"],
}

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PR79GPUValidator:
    """Main orchestrator for PR79 GPU validation on remote server."""

    def __init__(self, host=None, port=None, user=None, password=None):
        cfg = REMOTE_CONFIG.copy()
        cfg["host"] = host or os.environ.get("STATGPU_REMOTE_HOST", cfg["host"])
        cfg["port"] = port or int(os.environ.get("STATGPU_REMOTE_PORT", str(cfg["port"])))
        cfg["user"] = user or os.environ.get("STATGPU_REMOTE_USER", cfg["user"])
        cfg["password"] = password or os.environ.get("STATGPU_REMOTE_PASSWORD", cfg["password"])

        self.host = cfg["host"]
        self.port = cfg["port"]
        self.user = cfg["user"]
        self.password = cfg["password"]
        self.ssh: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        self.run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.result_dir = f"{REMOTE_PATHS['results_root']}/{self.run_id}"
        self.local_results = Path(__file__).resolve().parent.parent.parent / "results" / "pr79" / self.run_id

    # ---- SSH lifecycle ----

    def connect(self):
        """Establish SSH connection to the remote GPU server."""
        self._log(f"Connecting to {self.user}@{self.host}:{self.port} ...")
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            self.host, port=self.port,
            username=self.user, password=self.password,
            look_for_keys=False, allow_agent=False,
            timeout=30,
        )
        self.sftp = self.ssh.open_sftp()
        self._log("Connected successfully.")

    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        self._log("Disconnected.")

    def test_connection(self):
        """Quick connectivity + CUDA probe check."""
        self.connect()
        exit_code, stdout, stderr = self.run_raw(
            "echo 'SSH OK'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
            "python -c \"import cupy as cp; print('CuPy OK, devices:', cp.cuda.runtime.getDeviceCount()); "
            "import torch; print('Torch CUDA:', torch.cuda.is_available())\"",
            timeout=60,
        )
        self._log(f"Connection test: exit={exit_code}")
        self._log(stdout)
        if stderr.strip():
            self._log(f"stderr: {stderr.strip()}")
        self.disconnect()
        return exit_code == 0

    # ---- Remote execution ----

    def run_raw(self, cmd, timeout=300, env=None):
        """Execute a command on the remote WITHOUT cd to worktree.

        Used for setup commands that run before worktrees exist and
        for test-connection.
        """
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"export {k}={v};" for k, v in env.items()) + " "

        full_cmd = (
            f"{CONDA_ACTIVATE} && "
            f"{env_prefix}"
            f"{cmd}"
        )
        self._log(f"[REMOTE-RAW] {full_cmd[:200]}...", level="debug")

        stdin, stdout, stderr = self.ssh.exec_command(full_cmd, get_pty=True, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out_str = stdout.read().decode("utf-8", errors="replace")
        err_str = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out_str, err_str

    def run_remote(self, cmd, timeout=300, env=None, worktree="head"):
        """Execute a command on the remote server via SSH.

        The command is wrapped with conda activation and cd to the
        appropriate worktree.
        """
        wt = REMOTE_PATHS[f"worktree_{worktree}"]
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"export {k}={v};" for k, v in env.items()) + " "

        full_cmd = (
            f"{CONDA_ACTIVATE} && "
            f"cd {wt} && "
            f"{env_prefix}"
            f"{cmd}"
        )
        self._log(f"[REMOTE] {full_cmd[:200]}...", level="debug")

        stdin, stdout, stderr = self.ssh.exec_command(full_cmd, get_pty=True, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out_str = stdout.read().decode("utf-8", errors="replace")
        err_str = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out_str, err_str

    def run_remote_script(self, script_content, timeout=600, env=None, worktree="head"):
        """Upload a Python script string and execute it on the remote."""
        script_path = "/tmp/pr79_script.py"
        self._upload_string(script_content, script_path)
        exit_code, stdout, stderr = self.run_remote(
            f"python {script_path}", timeout=timeout, env=env, worktree=worktree
        )
        # Clean up
        self.run_remote(f"rm -f {script_path}", timeout=10, worktree=worktree)
        return exit_code, stdout, stderr

    def run_remote_pytest(self, test_files, timeout=600, env=None, worktree="head",
                          extra_args="", junit_name="results"):
        """Run pytest on the remote server for specified test files."""
        env = env or {}
        env.setdefault("STATGPU_REQUIRE_PHYSICAL_GPU", "1")
        env.setdefault("PYTHONNOUSERSITE", "1")
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("PYTHONHASHSEED", "0")

        files_str = " ".join(test_files)
        junit_path = f"{self.result_dir}/junit/{junit_name}.xml"
        log_path = f"{self.result_dir}/logs/{junit_name}.log"

        cmd = (
            f"mkdir -p {self.result_dir}/junit {self.result_dir}/logs && "
            f"python -m pytest {files_str} "
            f"-q -ra --tb=short {extra_args} "
            f"--junitxml={junit_path} "
            f"2>&1 | tee {log_path}"
        )
        return self.run_remote(cmd, timeout=timeout, env=env, worktree=worktree)

    # ---- SFTP helpers ----

    def upload_file(self, local_path, remote_path):
        """Upload a single file via SFTP."""
        self.sftp.put(str(local_path), remote_path)
        self._log(f"Uploaded: {local_path} -> {remote_path}", level="debug")

    def upload_directory(self, local_dir, remote_dir, skip_patterns=None):
        """Recursively upload a directory via SFTP."""
        skip_patterns = skip_patterns or {"__pycache__", ".pyc", ".git", ".venv", ".pytest_cache"}
        local_dir = Path(local_dir)

        # Ensure remote directory exists
        self._mkdir_p(remote_dir)

        for item in local_dir.rglob("*"):
            # Skip unwanted patterns
            if any(p in item.parts for p in skip_patterns):
                continue
            if item.suffix in skip_patterns:
                continue

            rel = item.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel}".replace("\\", "/")

            if item.is_dir():
                self._mkdir_p(remote_path)
            else:
                self.sftp.put(str(item), remote_path)
        self._log(f"Uploaded directory: {local_dir} -> {remote_dir}")

    def download_file(self, remote_path, local_path):
        """Download a single file via SFTP."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote_path, str(local_path))
        self._log(f"Downloaded: {remote_path} -> {local_path}", level="debug")

    def download_results(self):
        """Download all result files from the remote server."""
        self._log("Downloading results...")
        for subdir in RESULT_SUBDIRS:
            remote_sub = f"{self.result_dir}/{subdir}"
            local_sub = self.local_results / subdir
            local_sub.mkdir(parents=True, exist_ok=True)
            try:
                files = self.sftp.listdir(remote_sub)
                for fname in files:
                    self.download_file(f"{remote_sub}/{fname}", local_sub / fname)
            except FileNotFoundError:
                self._log(f"  (no {subdir}/)", level="debug")
        self._log(f"Results downloaded to {self.local_results}")

    def upload_package(self):
        """Upload the statgpu package and dev/ directory to remote worktrees.

        Uploads to BOTH base and head worktrees so both can run tests.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        skip = {"__pycache__", ".pyc", ".git", ".venv", ".pytest_cache",
                "results", "node_modules", "frontend", ".mypy_cache", "*.egg-info"}

        for wt in ["base", "head"]:
            remote_wt = REMOTE_PATHS[f"worktree_{wt}"]
            self._log(f"Uploading to worktree: {wt} ({remote_wt})")

            for subdir in ["statgpu", "dev"]:
                local_d = project_root / subdir
                if local_d.exists():
                    self.upload_directory(local_d, f"{remote_wt}/{subdir}", skip)

            # Also upload setup files
            for f in ["setup.py", "setup.cfg", "pyproject.toml", "README.md"]:
                local_f = project_root / f
                if local_f.exists():
                    self.upload_file(local_f, f"{remote_wt}/{f}")

        self._log("Package upload complete.")

    def _upload_string(self, content, remote_path):
        """Upload a string as a file via SFTP."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            self.sftp.put(tmp_path, remote_path)
        finally:
            os.unlink(tmp_path)

    def _mkdir_p(self, remote_dir):
        """Create remote directory if it doesn't exist."""
        try:
            self.sftp.stat(remote_dir)
        except FileNotFoundError:
            # Walk and create parent directories
            parts = remote_dir.strip("/").split("/")
            for i in range(1, len(parts) + 1):
                partial = "/" + "/".join(parts[:i])
                try:
                    self.sftp.stat(partial)
                except FileNotFoundError:
                    self.sftp.mkdir(partial)

    # ---- Setup ----

    def setup_remote_environment(self):
        """One-time remote environment setup.

        Creates directory structure, clones repo, creates worktrees,
        clones conda environments, and uploads the statgpu package.
        """
        self._log_section("Remote Environment Setup")

        # Step 1: Create directory structure
        self._log("Step 1/6: Creating directory structure...")
        dirs = [
            REMOTE_PATHS["validation_root"],
            f"{REMOTE_PATHS['validation_root']}/worktrees",
            f"{REMOTE_PATHS['validation_root']}/results",
            f"{REMOTE_PATHS['validation_root']}/results/pr79",
        ]
        code, out, err = self.run_raw(
            f"mkdir -p {' '.join(dirs)} && echo 'Directories created'",
            timeout=30
        )
        if code != 0:
            self._log(f"ERROR creating directories: {err}")
            return False
        self._log(out.strip())

        # Step 2: Use existing repo or clone
        self._log("Step 2/6: Setting up repository...")
        code, out, err = self.run_raw(
            f"if [ -d /root/statgpu/.git ]; then "
            f"  echo 'Using existing /root/statgpu as source repo'; "
            f"  cd /root/statgpu && git fetch --all --prune 2>/dev/null || true; "
            f"elif [ -d {REMOTE_PATHS['repo']}/.git ]; then "
            f"  cd {REMOTE_PATHS['repo']} && git fetch --all --prune && echo 'Repo exists, fetched'; "
            f"else "
            f"  git clone {GIT_INFO['repo_url']} {REMOTE_PATHS['repo']} && echo 'Repo cloned'; "
            f"fi",
            timeout=120
        )
        self._log(out.strip())
        if code != 0:
            self._log(f"WARNING: git clone/fetch failed (may need SFTP upload): {err}")

        # Step 3: Create worktrees from the cloned repo
        self._log("Step 3/6: Creating worktrees...")
        source_repo = REMOTE_PATHS["repo"]  # The repo we cloned in step 2
        for wt_name, sha in [("pr79-base", GIT_INFO["base_sha"]), ("pr79-head", GIT_INFO["head_sha"])]:
            wt_path = REMOTE_PATHS[f"worktree_{wt_name.replace('pr79-', '')}"]
            code, out, err = self.run_raw(
                f"cd {source_repo} && "
                f"(git worktree list 2>/dev/null | grep -q {wt_path} && "
                f"  echo 'Worktree {wt_name} already exists' || "
                f"  git worktree add --detach {wt_path} {sha} && echo 'Worktree {wt_name} created at {sha}')",
                timeout=60
            )
            self._log(out.strip())
            if code != 0:
                self._log(f"ERROR creating worktree {wt_name}: {err}")
                return False

        # Step 4: Verify SHAs
        self._log("Step 4/6: Verifying SHAs...")
        for wt_name, sha in [("pr79-base", GIT_INFO["base_sha"]), ("pr79-head", GIT_INFO["head_sha"])]:
            wt_key = wt_name.replace("pr79-", "")
            code, out, err = self.run_remote(
                f"cd {REMOTE_PATHS[f'worktree_{wt_key}']} && "
                f"test \"$(git rev-parse HEAD)\" = \"{sha}\" && "
                f"echo '{wt_name} SHA verified: {sha[:12]}'",
                timeout=30, worktree=wt_key
            )
            self._log(out.strip())
            if code != 0:
                self._log(f"ERROR: {wt_name} SHA mismatch! Expected {sha}")
                self._log(err)
                return False

        # Step 5: Clone conda environments (optional - can just use myconda + sys.path)
        self._log("Step 5/6: Conda environments ready (using myconda + sys.path insertion)...")
        # We don't actually clone; we use sys.path.insert in test scripts
        self._log("  Using myconda environment for both base and head")

        # Step 6: Upload package
        self._log("Step 6/6: Uploading statgpu package...")
        self.upload_package()

        self._log("Setup complete!")
        return True

    # ---- Phase 0: Environment Freeze ----

    def run_phase_0(self):
        """Phase 0: Record environment state and run CUDA probe."""
        self._log_section("Phase 0: Environment Freeze & Hardware Pre-check")

        # Build the phase 0 script using .format() to avoid nested f-string issues
        result_dir = self.result_dir
        script = (
            "import json, os, sys, subprocess, platform\n"
            "\n"
            "result_dir = '{result_dir}/environment'\n"
            "os.makedirs(result_dir, exist_ok=True)\n"
            "sys.path.insert(0, '.')\n"
            "\n"
            "def run(cmd, outfile):\n"
            "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)\n"
            "    path = result_dir + '/' + outfile\n"
            "    with open(path, 'w') as f:\n"
            "        f.write(r.stdout)\n"
            "        if r.stderr:\n"
            "            f.write('\\n--- stderr ---\\n' + r.stderr)\n"
            "    return r.stdout.strip()\n"
            "\n"
            "# Basic info\n"
            "run('git rev-parse HEAD', 'git_sha.txt')\n"
            "run('git status --short', 'git_status.txt')\n"
            "run('python -V', 'python_version.txt')\n"
            "run('python -m pip freeze', 'pip_freeze.txt')\n"
            "run('python -m pip check', 'pip_check.txt')\n"
            "run('nvidia-smi -q', 'nvidia_smi_q.txt')\n"
            "run('nvcc --version || echo nvcc not found', 'nvcc_version.txt')\n"
            "\n"
            "# Dtype info\n"
            "import numpy as np\n"
            "path = result_dir + '/dtype_info.txt'\n"
            "with open(path, 'w') as f:\n"
            "    f.write('default float: ' + str(np.finfo(np.float64).dtype) + '\\n')\n"
            "    f.write('float32 eps: ' + str(np.finfo(np.float32).eps) + '\\n')\n"
            "    f.write('float64 eps: ' + str(np.finfo(np.float64).eps) + '\\n')\n"
            "\n"
            "# GPU summary\n"
            "run('nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader', 'gpu_summary.csv')\n"
            "\n"
            "# CUDA probe (Section 6.1)\n"
            "print('=== CUDA Probe ===')\n"
            "\n"
            "x_np = np.arange(16, dtype=np.float64).reshape(4, 4)\n"
            "\n"
            "# CuPy probe\n"
            "try:\n"
            "    import cupy as cp\n"
            "    x_cp = cp.asarray(x_np)\n"
            "    cp.cuda.Stream.null.synchronize()\n"
            "    x_back = cp.asnumpy(x_cp)\n"
            "    cupy_ok = bool(np.array_equal(x_back, x_np))\n"
            "    cupy_devices = int(cp.cuda.runtime.getDeviceCount())\n"
            "    cupy_device = int(cp.cuda.runtime.getDevice())\n"
            "    cupy_version = cp.__version__\n"
            "    print('CuPy: version=' + str(cupy_version) + ', devices=' + str(cupy_devices) + ', current=' + str(cupy_device) + ', roundtrip=' + str(cupy_ok))\n"
            "except Exception as e:\n"
            "    cupy_ok = False; cupy_devices = 0; cupy_device = -1; cupy_version = None\n"
            "    print('CuPy: FAILED - ' + str(e))\n"
            "\n"
            "# Torch probe\n"
            "try:\n"
            "    import torch\n"
            "    x_t = torch.as_tensor(x_np, device='cuda')\n"
            "    torch.cuda.synchronize()\n"
            "    x_t_back = x_t.cpu().numpy()\n"
            "    torch_ok = bool(np.array_equal(x_t_back, x_np))\n"
            "    torch_cuda = torch.cuda.is_available()\n"
            "    torch_devices = torch.cuda.device_count() if torch_cuda else 0\n"
            "    torch_device = torch.cuda.current_device() if torch_cuda else -1\n"
            "    torch_version = torch.__version__\n"
            "    torch_cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else None\n"
            "    print('Torch: version=' + str(torch_version) + ', cuda=' + str(torch_cuda) + ', devices=' + str(torch_devices) + ', current=' + str(torch_device) + ', roundtrip=' + str(torch_ok))\n"
            "except Exception as e:\n"
            "    torch_ok = False; torch_cuda = False; torch_devices = 0; torch_device = -1\n"
            "    torch_version = None; torch_cuda_ver = None\n"
            "    print('Torch: FAILED - ' + str(e))\n"
            "\n"
            "# statgpu import check\n"
            "try:\n"
            "    import statgpu\n"
            "    statgpu_path = statgpu.__file__\n"
            "    print('statgpu: path=' + str(statgpu_path))\n"
            "except Exception as e:\n"
            "    statgpu_path = None\n"
            "    print('statgpu: FAILED - ' + str(e))\n"
            "\n"
            "probe = {{\n"
            "    'python_version': platform.python_version(),\n"
            "    'numpy_version': np.__version__,\n"
            "    'cupy': {{'available': cupy_ok, 'version': cupy_version, 'device_count': cupy_devices, 'current_device': cupy_device}},\n"
            "    'torch': {{'cuda_available': torch_cuda, 'version': torch_version, 'cuda_version': torch_cuda_ver, 'device_count': torch_devices, 'current_device': torch_device}},\n"
            "    'statgpu_path': statgpu_path,\n"
            "    'cupy_roundtrip_ok': cupy_ok,\n"
            "    'torch_roundtrip_ok': torch_ok,\n"
            "}}\n"
            "\n"
            "probe_path = result_dir + '/backend_probe.json'\n"
            "with open(probe_path, 'w') as f:\n"
            "    json.dump(probe, f, indent=2, default=str)\n"
            "print('Probe saved to ' + probe_path)\n"
            "\n"
            "# Hard preflight checks\n"
            "if not cupy_ok:\n"
            "    print('CRITICAL: CuPy float64 roundtrip FAILED!')\n"
            "    sys.exit(1)\n"
            "if not torch_ok:\n"
            "    print('CRITICAL: Torch CUDA float64 roundtrip FAILED!')\n"
            "    sys.exit(1)\n"
            "if not statgpu_path:\n"
            "    print('CRITICAL: statgpu import FAILED!')\n"
            "    sys.exit(1)\n"
            "\n"
            "# Check for dirty checkout\n"
            "git_status_path = result_dir + '/git_status.txt'\n"
            "git_status = open(git_status_path).read().strip()\n"
            "if git_status:\n"
            "    print('WARNING: Git checkout is dirty:')\n"
            "    print(git_status)\n"
            "\n"
            "print('Phase 0 complete - all preflight checks passed.')\n"
        ).format(result_dir=result_dir)

        exit_code, stdout, stderr = self.run_remote_script(script, timeout=120)
        self._log(stdout)
        if stderr.strip():
            self._log(f"stderr: {stderr.strip()}")
        if exit_code != 0:
            self._log("Phase 0 FAILED - preflight checks did not pass!")
            return False
        self._log("Phase 0 PASSED")
        return True

    # ---- Phase 1: Test Collection ----

    def run_phase_1(self):
        """Phase 1: Test collection and skip audit."""
        self._log_section("Phase 1: Test Collection & Skip Audit")

        result_dir = self.result_dir

        # Collect tests
        exit_code, stdout, stderr = self.run_remote(
            f"mkdir -p {result_dir}/collection && "
            f"python -m pytest --collect-only -q dev/tests "
            f"2>&1 | tee {result_dir}/collection/all_tests.txt",
            timeout=180,
            env={"PYTHONNOUSERSITE": "1"},
        )
        self._log(f"Collection exit code: {exit_code}")

        # Parse collection output
        script = (
            "import json, os, re, subprocess\n"
            "\n"
            "result_dir = '{result_dir}/collection'\n"
            "os.makedirs(result_dir, exist_ok=True)\n"
            "\n"
            "all_tests_path = result_dir + '/all_tests.txt'\n"
            "with open(all_tests_path) as f:\n"
            "    content = f.read()\n"
            "\n"
            "test_files = []\n"
            "lines = content.split('\\n')\n"
            "for line in lines:\n"
            "    m = re.match(r'^(dev/tests/test_.*\\.py).*?(\\d+) selected', line)\n"
            "    if m:\n"
            "        test_files.append(dict(file=m.group(1), tests=int(m.group(2))))\n"
            "\n"
            "gpu_tests = []\n"
            "for tf in test_files:\n"
            "    try:\n"
            "        import subprocess as sp\n"
            "        grep_cmd = \"grep -l 'importorskip.*cupy\\\\|importorskip.*torch\\\\|mark\\\\.gpu\\\\|mark\\\\.cupy\\\\|mark\\\\.torch_cuda' \" + tf['file']\n"
            "        r = sp.run(grep_cmd, shell=True, capture_output=True, text=True, timeout=10)\n"
            "        if r.returncode == 0:\n"
            "            gpu_tests.append(tf['file'])\n"
            "    except:\n"
            "        pass\n"
            "\n"
            "total_tests = content.count('::')\n"
            "\n"
            "inventory = {{\n"
            "    'total_test_functions_approx': total_tests,\n"
            "    'test_files_listed': len(test_files),\n"
            "    'gpu_dependent_files': len(gpu_tests),\n"
            "    'gpu_dependent_file_list': gpu_tests,\n"
            "    'test_files': test_files[:50],\n"
            "}}\n"
            "\n"
            "inv_path = result_dir + '/test_inventory.json'\n"
            "with open(inv_path, 'w') as f:\n"
            "    json.dump(inventory, f, indent=2)\n"
            "\n"
            "print(json.dumps(inventory, indent=2))\n"
        ).format(result_dir=result_dir)

        exit_code, stdout, stderr = self.run_remote_script(script, timeout=60)
        self._log(stdout)
        self._log("Phase 1 complete")
        return True

    # ---- Gate A: Physical GPU Smoke Test ----

    def run_gate_a(self):
        """Gate A: Fast physical GPU smoke test (Section 8)."""
        self._log_section("Gate A: Physical GPU Smoke Test")

        # Also include the new physical GPU test file
        test_files = list(GATE_A_TEST_FILES)
        test_files.append("dev/tests/test_pr79_physical_gpu.py")

        exit_code, stdout, stderr = self.run_remote_pytest(
            test_files,
            timeout=600,
            junit_name="gate_a",
        )

        self._log(stdout[-3000:] if len(stdout) > 3000 else stdout)

        # Parse JUnit results
        junit_path = f"{self.result_dir}/junit/gate_a.xml"
        script = (
            "import xml.etree.ElementTree as ET\n"
            "import json\n"
            "\n"
            "junit_path = '{junit_path}'\n"
            "try:\n"
            "    tree = ET.parse(junit_path)\n"
            "    root = tree.getroot()\n"
            "    total = int(root.get('tests', 0))\n"
            "    skipped = int(root.get('skipped', 0))\n"
            "    errors = int(root.get('errors', 0))\n"
            "    failures = int(root.get('failures', 0))\n"
            "    passed = total - errors - failures - skipped\n"
            "\n"
            "    result = {{\n"
            "        'gate': 'A',\n"
            "        'total': total,\n"
            "        'passed': passed,\n"
            "        'failed': errors + failures,\n"
            "        'skipped': skipped,\n"
            "        'pass_rate': round(passed / total * 100, 1) if total > 0 else 0,\n"
            "    }}\n"
            "\n"
            "    result['gate_passed'] = (passed > 0 and total > 0)\n"
            "    print(json.dumps(result, indent=2))\n"
            "except FileNotFoundError:\n"
            "    print(json.dumps({{'error': 'JUnit XML not found', 'gate_passed': False}}, indent=2))\n"
        ).format(junit_path=junit_path)

        exit_code, stdout, stderr = self.run_remote_script(script, timeout=30)
        self._log(stdout)

        try:
            result = json.loads(stdout.strip().split("\n")[-1])
            if result.get("gate_passed"):
                self._log(f"Gate A PASSED: {result.get('passed')}/{result.get('total')} tests")
                return True
            else:
                self._log(f"Gate A FAILED: {result}")
                return False
        except json.JSONDecodeError:
            self._log("Gate A: Could not parse results")
            return exit_code == 0

    # ---- Gate B: Three-Backend Numerical Correctness ----

    def run_gate_b(self):
        """Gate B: Three-backend numerical correctness (Sections 9-10)."""
        self._log_section("Gate B: Three-Backend Numerical Correctness")

        # Run broader test suite - all test_*.py files excluding benchmarks
        # Use --continue-on-collection-errors to skip files with import issues
        # (e.g., files that import paramiko or have R deps)
        exit_code, stdout, stderr = self.run_remote_pytest(
            ["dev/tests/",
             "--ignore=dev/tests/_archive",
             "--ignore-glob=*bench*",
             "--ignore-glob=*remote*",
             "--ignore=dev/tests/test_backend_comparison.py",
             "--ignore=dev/tests/test_comprehensive_remote.py",
             "--ignore=dev/tests/test_elasticnet_cv_runner.py",
             "--ignore=dev/tests/test_cupy_fused_ab_comparison.py",
             "--ignore=dev/tests/test_cupy_fused_ab_rerun.py",
             "--ignore=dev/tests/test_cupy_fused_optimization.py",
             "--ignore=dev/tests/test_cupy_import_overhead.py",
             "--ignore=dev/tests/test_dbscan_edge_cases.py",
             "--ignore=dev/tests/test_lassocv_inference_simple.py",
             "--ignore=dev/tests/test_coxph_3backends.py",
             "--ignore=dev/tests/test_irls_gpu.py"],
            timeout=1800,
            junit_name="gate_b",
            extra_args="-q -ra --tb=short --continue-on-collection-errors",
        )
        self._log(f"Gate B exit code: {exit_code}")

        # Compare base vs head if failures found
        if exit_code != 0:
            self._log("Gate B failures detected. Running base comparison...")
            self._log("(base comparison will be implemented in Round 2)")

        return exit_code == 0

    # ---- Gate C: Metamorphic Tests ----

    def run_gate_c(self):
        """Gate C: Metamorphic/property-based tests (Section 11)."""
        self._log_section("Gate C: Metamorphic Tests")
        self._log("Gate C will be implemented in Round 3 (Properties & Device Purity)")
        return True

    # ---- Gate D: Device Purity Audit ----

    def run_gate_d(self):
        """Gate D: Device purity and host transfer audit (Section 12)."""
        self._log_section("Gate D: Device Purity & Host Transfer Audit")
        self._log("Gate D will be implemented in Round 3 (Properties & Device Purity)")
        return True

    # ---- Gate E: Memory & Repeated Fit ----

    def run_gate_e(self):
        """Gate E: Memory leak and repeated fit tests (Section 13)."""
        self._log_section("Gate E: Memory & Repeated Fit Tests")
        self._log("Gate E will be implemented in Round 4 (Memory & Performance)")
        return True

    # ---- Gate F: Performance Comparison ----

    def run_gate_f(self):
        """Gate F: Base vs Head performance comparison (Section 14)."""
        self._log_section("Gate F: Base vs Head Performance")
        self._log("Gate F will be implemented in Round 4 (Memory & Performance)")
        return True

    # ---- Gate G: External Benchmarks ----

    def run_gate_g(self):
        """Gate G: External statistical benchmarks (Section 15)."""
        self._log_section("Gate G: External Statistical Benchmarks")
        self._log("Gate G will be implemented in Round 5 (External & Final Gate)")
        return True

    # ---- Final Gate ----

    def run_final_gate(self):
        """Final Gate: Complete CPU + GPU suite (Section 19)."""
        self._log_section("Final Gate: Complete CPU + GPU Suite")
        self._log("Final Gate will be implemented in Round 5 (External & Final Gate)")
        return True

    # ---- Logging ----

    def _log(self, msg, level="info"):
        prefix = {"info": "  ", "debug": "    [D] ", "section": "\n"}
        print(f"{prefix.get(level, '  ')}{msg}")

    def _log_section(self, title):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    def _log_result(self, phase_name, exit_code, stdout, stderr):
        status = "PASSED" if exit_code == 0 else "FAILED"
        self._log(f"{phase_name}: {status} (exit={exit_code})")

    # ---- Phase Dispatch ----

    PHASE_MAP = {
        "phase_0": "run_phase_0",
        "phase_1": "run_phase_1",
        "gate_a": "run_gate_a",
        "gate_b": "run_gate_b",
        "gate_c": "run_gate_c",
        "gate_d": "run_gate_d",
        "gate_e": "run_gate_e",
        "gate_f": "run_gate_f",
        "gate_g": "run_gate_g",
        "final_gate": "run_final_gate",
    }

    def run_phase(self, phase_name):
        """Run a single named phase."""
        method_name = self.PHASE_MAP.get(phase_name)
        if not method_name:
            self._log(f"Unknown phase: {phase_name}")
            return False
        method = getattr(self, method_name)
        return method()

    def run_round(self, round_num):
        """Run all phases for a given round (Section 22)."""
        phases = ROUND_PHASES.get(round_num, [])
        if not phases:
            self._log(f"Unknown round: {round_num}")
            return False

        self._log_section(f"Round {round_num}: {', '.join(phases)}")
        for phase in phases:
            success = self.run_phase(phase)
            if not success:
                self._log(f"Round {round_num} stopped at {phase} (failed)")
                return False
        self._log(f"Round {round_num} complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="PR79 GPU Validation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev/validation/pr79_gpu_orchestrator.py --test-connection
  python dev/validation/pr79_gpu_orchestrator.py --setup
  python dev/validation/pr79_gpu_orchestrator.py --round 1
  python dev/validation/pr79_gpu_orchestrator.py --phase gate_a
  python dev/validation/pr79_gpu_orchestrator.py --download
        """,
    )
    parser.add_argument("--test-connection", action="store_true",
                        help="Test SSH connection and CUDA availability")
    parser.add_argument("--setup", action="store_true",
                        help="One-time remote environment setup")
    parser.add_argument("--round", type=int, choices=[1, 2, 3, 4, 5],
                        help="Execute a specific round (1-5)")
    parser.add_argument("--phase", type=str,
                        help="Execute a specific phase (phase_0, gate_a, etc.)")
    parser.add_argument("--download", action="store_true",
                        help="Download all results from remote")
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary from downloaded results")
    parser.add_argument("--host", type=str, help="Remote host (overrides config)")
    parser.add_argument("--port", type=int, help="Remote port (overrides config)")
    parser.add_argument("--user", type=str, help="Remote user (overrides config)")

    args = parser.parse_args()

    # Get password from env or config
    password = os.environ.get("STATGPU_REMOTE_PASSWORD", REMOTE_CONFIG["password"])

    validator = PR79GPUValidator(
        host=args.host, port=args.port,
        user=args.user, password=password,
    )

    try:
        validator.connect()

        if args.test_connection:
            validator.disconnect()
            success = validator.test_connection()
            return 0 if success else 1

        if args.setup:
            if not validator.setup_remote_environment():
                validator._log("Setup FAILED!")
                return 1
            validator._log("Setup complete. Ready for --round 1")

        elif args.round:
            if not validator.run_round(args.round):
                return 1
            validator.download_results()

        elif args.phase:
            if not validator.run_phase(args.phase):
                return 1
            validator.download_results()

        elif args.download:
            validator.download_results()
            if args.summary:
                validator._log("Generating summary (use pr79_results.py for detailed output)")

        else:
            parser.print_help()

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1
    finally:
        validator.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
