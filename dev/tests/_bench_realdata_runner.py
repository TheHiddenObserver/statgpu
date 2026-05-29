"""6-stage progressive real-data benchmark runner.

Uploads stage scripts to RTX 4090 server, runs them sequentially.
On PASS: automatically proceeds to next stage.
On FAIL: diagnoses failure, prints fix instructions, restarts from Stage 1.

Usage:
  export STATGPU_BENCH_SERVER=your-server
  export STATGPU_BENCH_PORT=22
  export STATGPU_BENCH_USER=root
  export STATGPU_BENCH_PASS=your-password
  python _bench_realdata_runner.py [--start-stage N]
"""
import paramiko
import time
import re
import sys
import os

SERVER = os.environ.get("STATGPU_BENCH_SERVER", "")
PORT = int(os.environ.get("STATGPU_BENCH_PORT", "22"))
USER = os.environ.get("STATGPU_BENCH_USER", "root")
PASS = os.environ.get("STATGPU_BENCH_PASS", "")

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/root/statgpu/dev/tests"

STAGES = [
    {"id": 1, "name": "Smoke Test",        "script": "_bench_realdata_s1.py", "timeout": 300},
    {"id": 2, "name": "Precision Validation", "script": "_bench_realdata_s2.py", "timeout": 600},
    {"id": 3, "name": "Full Single-Module",  "script": "_bench_realdata_s3.py", "timeout": 900},
    {"id": 4, "name": "High-Dim CoxPH",     "script": "_bench_realdata_s4.py", "timeout": 900},
    {"id": 5, "name": "Penalized Models",    "script": "_bench_realdata_s5.py", "timeout": 1200},
]

PASS_CRITERIA = {
    1: {"check": "no_errors"},
    2: {
        "poisson_cupy_sklearn": {"coef_corr": 0.98},
        "gamma_cupy_sklearn":   {"coef_corr": 0.98},
        "adj_pval_100k":        {"reject_agreement": 0.99},
        "coxph_cupy_lifelines": {"coef_corr": 0.99},
    },
    3: {
        "poisson_full":    {"coef_corr": 0.93, "speedup": 1.0},
        "gamma_full":      {"coef_corr": 0.98},
        "adj_pval_1m":     {"reject_agreement": 0.99},
    },
    4: {
        "metabric_coxph_cupy": {"coef_corr": 0.99},
    },
    5: {
        "adj_pval_5m": {"reject_agreement": 0.99},
    },
}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    # ESC[ ... letter (CSI sequences)
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # ESC[? ... letter (private mode sequences like bracket paste)
    text = re.sub(r'\x1b\[\?[0-9]*[a-zA-Z]', '', text)
    # Bare [?2004h/l (already stripped of ESC)
    text = re.sub(r'\[\?[0-9]+[a-zA-Z]', '', text)
    # Any remaining ESC sequences
    text = re.sub(r'\x1b[^a-zA-Z]*[a-zA-Z]', '', text)
    return text


def parse_results(output: str) -> dict:
    """Parse === BENCH_RESULT === blocks."""
    clean = strip_ansi(output)
    results = {}
    # Use flexible regex that allows whitespace around markers
    for match in re.finditer(
        r'=+\s*BENCH_RESULT\s*=+\s*\n(.*?)=+\s*END\s*=+', clean, re.DOTALL
    ):
        entry = {}
        for line in match.group(1).strip().split('\n'):
            line = line.strip()
            if ': ' in line:
                k, v = line.split(': ', 1)
                entry[k.strip()] = v.strip()
        if 'name' in entry:
            results[entry['name']] = entry
    return results


def validate(stage_id: int, results: dict, output: str) -> tuple:
    """Check pass criteria. Returns (passed, failures)."""
    criteria = PASS_CRITERIA.get(stage_id, {})
    if criteria.get("check") == "no_errors":
        # Check for Python tracebacks before the PASSED marker
        before_passed = output.split('PASSED')[0] if 'PASSED' in output else output
        if 'Traceback' in before_passed:
            return False, ["Traceback detected in output"]
        if 'Stage 1 PASSED' not in output:
            return False, ["Stage 1 did not report PASSED"]
        return True, []
    failures = []
    for name, reqs in criteria.items():
        if name not in results:
            failures.append(f"Missing result: {name}")
            continue
        r = results[name]
        for metric, threshold in reqs.items():
            actual_str = r.get(metric, '0')
            try:
                actual = float(actual_str)
            except ValueError:
                failures.append(f"{name}.{metric}: cannot parse '{actual_str}'")
                continue
            if metric in ("speedup",):
                # speedup is a minimum threshold
                if actual < threshold:
                    failures.append(f"{name}.{metric}: {actual:.2f} < {threshold}")
            elif metric in ("coef_corr", "reject_agreement"):
                if actual < threshold:
                    failures.append(f"{name}.{metric}: {actual:.6f} < {threshold}")
    return len(failures) == 0, failures


def connect():
    """Create a fresh SSH connection."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER, PORT, USER, PASS, timeout=30)
    return client


def ssh_execute(client, script_name: str, timeout: int) -> str:
    """Upload script and execute on remote server using exec_command (no PTY)."""
    local_path = os.path.join(LOCAL_DIR, script_name)
    remote_path = f"{REMOTE_DIR}/{script_name}"

    # Upload
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()

    # Execute via exec_command (non-interactive, no ANSI escape codes)
    cmd = (
        f"cd /root/statgpu && "
        f"source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate myconda && "
        f"PYTHONUNBUFFERED=1 python -u {remote_path}"
    )

    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    channel = stdout.channel
    # Use a generous per-read timeout — we check total elapsed time ourselves
    channel.settimeout(60)

    output = ""
    start = time.time()
    while True:
        if channel.recv_ready():
            data = channel.recv(8192).decode('utf-8', errors='replace')
            print(data, end="", flush=True)
            output += data
            if '__STAGE_DONE__' in output:
                # Drain remaining output
                import time as _t
                _t.sleep(0.5)
                while channel.recv_ready():
                    data = channel.recv(8192).decode('utf-8', errors='replace')
                    print(data, end="", flush=True)
                    output += data
                break
        elif channel.exit_status_ready():
            while channel.recv_ready():
                data = channel.recv(8192).decode('utf-8', errors='replace')
                print(data, end="", flush=True)
                output += data
            break
        else:
            if time.time() - start > timeout:
                print(f"\n[TIMEOUT after {timeout}s]")
                output += "\n[TIMEOUT]"
                break
            time.sleep(0.5)

    # Drain stderr (non-blocking)
    err_parts = []
    try:
        while channel.recv_stderr_ready():
            err_parts.append(channel.recv_stderr(8192).decode('utf-8', errors='replace'))
    except Exception:
        pass
    err_output = ''.join(err_parts)
    if err_output.strip():
        print(f"\n[STDERR]\n{err_output}", flush=True)

    return output


def clear_cache(client):
    """Clear remote __pycache__."""
    client.exec_command(
        "find /root/statgpu -name '__pycache__' -exec rm -rf {} + 2>/dev/null",
        timeout=10
    )
    time.sleep(1)


def main():
    if not SERVER or not PASS:
        print("Error: Set STATGPU_BENCH_SERVER and STATGPU_BENCH_PASS environment variables.")
        print("  export STATGPU_BENCH_SERVER=your-server")
        print("  export STATGPU_BENCH_PASS=your-password")
        sys.exit(1)

    start_stage = 1
    if len(sys.argv) > 2 and sys.argv[1] == '--start-stage':
        start_stage = int(sys.argv[2])

    print("=" * 60)
    print("Real-Data Benchmark Runner (RTX 4090)")
    print("=" * 60)

    # Initial connect and upload
    client = connect()
    print(f"Connected to {SERVER}:{PORT}")

    # Upload statgpu package
    print("Uploading statgpu package...")
    sftp = client.open_sftp()
    for root, dirs, files in os.walk(os.path.join(LOCAL_DIR, '..', 'statgpu')):
        for f in files:
            if f.endswith('.py'):
                local = os.path.join(root, f)
                rel = os.path.relpath(local, os.path.join(LOCAL_DIR, '..'))
                remote = f"/root/statgpu/{rel.replace(os.sep, '/')}"
                try:
                    sftp.put(local, remote)
                except Exception:
                    pass
    sftp.close()
    clear_cache(client)
    client.close()
    print("Upload complete.\n")

    current_stage = start_stage - 1
    max_restarts = 3
    restart_count = 0

    while current_stage < len(STAGES):
        stage = STAGES[current_stage]
        print(f"\n{'='*60}")
        print(f"Stage {stage['id']}: {stage['name']}")
        print(f"{'='*60}")

        # Fresh connection per stage (server may drop idle connections)
        client = connect()
        output = ssh_execute(client, stage['script'], stage['timeout'])
        client.close()

        # Save output
        out_path = os.path.join(LOCAL_DIR, f"_bench_realdata_s{stage['id']}_output.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(output)

        # Parse and validate
        results = parse_results(output)
        passed, failures = validate(stage['id'], results, output)

        if not passed:
            print(f"\n*** FAIL at Stage {stage['id']}: ***")
            for fail in failures:
                print(f"  - {fail}")
            print(f"\nDiagnosis needed. Fix instructions:")
            print(f"  1. Analyze failure above")
            print(f"  2. Fix code in {stage['script']} or statgpu source")
            print(f"  3. Re-run: python _bench_realdata_runner.py --start-stage {stage['id']}")
            if restart_count < max_restarts:
                restart_count += 1
                c = connect()
                clear_cache(c)
                c.close()
                current_stage = start_stage - 1
                print(f"\nAuto-restarting ({restart_count}/{max_restarts})...")
                continue
            else:
                print(f"\nMax restarts ({max_restarts}) reached. Stopping.")
                sys.exit(1)

        print(f"\n*** PASS: Stage {stage['id']} ***")
        current_stage += 1

    print("\n" + "=" * 60)
    print("ALL STAGES PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
