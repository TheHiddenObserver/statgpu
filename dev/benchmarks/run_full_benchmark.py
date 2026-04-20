"""
Upload and run complete benchmark (R glmnet + statgpu) on remote server.
"""
import paramiko
import json
import os
import sys

# Import remote configuration from environment or local config file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from remote_config import get_remote_config

config = get_remote_config()
HOST = config['host']
PORT = config['port']
USERNAME = config['username']
PASSWORD = config['password']
SSH_KEY_PATH = config.get('ssh_key_path')


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600, print_output: bool = True):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()

    if print_output:
        if out:
            for line in out.split('\n'):
                try:
                    print(line)
                except UnicodeEncodeError:
                    pass
        if err:
            print("[STDERR]")
            for line in err.split('\n'):
                try:
                    print(line)
                except UnicodeEncodeError:
                    pass

    return code, out, err


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {HOST}:{PORT}...")
    connect_kwargs = dict(
        hostname=HOST,
        port=PORT,
        username=USERNAME,
        timeout=30,
    )
    if SSH_KEY_PATH:
        connect_kwargs["key_filename"] = SSH_KEY_PATH
        connect_kwargs["allow_agent"] = True
        connect_kwargs["look_for_keys"] = True
    else:
        connect_kwargs["allow_agent"] = False
        connect_kwargs["look_for_keys"] = False
    if PASSWORD:
        connect_kwargs["password"] = PASSWORD
    client.connect(**connect_kwargs)
    print("Connected successfully!\n")

    try:
        # Step 1: Upload statgpu package
        print("[Step 1] Uploading statgpu package...")
        sftp = client.open_sftp()

        for root, dirs, files in os.walk("statgpu"):
            for file in files:
                if file.endswith('.py'):
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, "statgpu")
                    remote_path = f"/root/statgpu/statgpu/{relative_path}".replace('\\', '/')

                    remote_subdir = os.path.dirname(remote_path)
                    try:
                        sftp.stat(remote_subdir)
                    except FileNotFoundError:
                        sftp.makedirs(remote_subdir)

                    sftp.put(local_path, remote_path)

        sftp.close()
        print("  Upload complete!\n")

        # Step 2: Upload R benchmark script
        print("[Step 2] Uploading R benchmark script...")
        sftp = client.open_sftp()
        sftp.put("dev/benchmarks/benchmark_glmnet_full.R", "/root/benchmark_glmnet.R")
        sftp.close()
        print("  Uploaded: /root/benchmark_glmnet.R\n")

        # Step 3: Upload Python benchmark script
        print("[Step 3] Uploading Python benchmark script...")
        sftp = client.open_sftp()
        sftp.put("dev/benchmarks/benchmark_statgpu_full.py", "/root/benchmark_statgpu.py")
        sftp.close()
        print("  Uploaded: /root/benchmark_statgpu.py\n")

        # Step 4: Run R glmnet benchmark
        print("[Step 4] Running R glmnet benchmark...")
        cmd = f"/usr/bin/Rscript /root/benchmark_glmnet.R"
        code, out, err = run(client, cmd, timeout=600, print_output=True)

        if code != 0:
            print("\n[WARNING] R benchmark failed, continuing with Python...\n")

        # Step 5: Run Python statgpu benchmark
        print("\n[Step 5] Running Python statgpu benchmark...")
        cmd = f"/root/miniconda3/envs/myconda/bin/python /root/benchmark_statgpu.py"
        code, out, err = run(client, cmd, timeout=600, print_output=True)

        if code != 0:
            print("\n[WARNING] Python benchmark failed\n")

        # Step 6: Download results
        print("\n[Step 6] Downloading results...")
        sftp = client.open_sftp()

        results_dir = Path("results/benchmark_full")
        results_dir.mkdir(exist_ok=True)

        # Download glmnet results
        for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
            try:
                sftp.get(f"/root/glmnet_result_{name}.json", str(results_dir / f"glmnet_result_{name}.json"))
                print(f"  Downloaded: glmnet_result_{name}.json")
            except FileNotFoundError:
                print(f"  Not found: glmnet_result_{name}.json")

        try:
            sftp.get("/root/benchmark_glmnet_all.json", str(results_dir / "benchmark_glmnet_all.json"))
            print("  Downloaded: benchmark_glmnet_all.json")
        except FileNotFoundError:
            print("  Not found: benchmark_glmnet_all.json")

        try:
            sftp.get("/root/benchmark_statgpu_all.json", str(results_dir / "benchmark_statgpu_all.json"))
            print("  Downloaded: benchmark_statgpu_all.json")
        except FileNotFoundError:
            print("  Not found: benchmark_statgpu_all.json")

        sftp.close()

        print("\n" + "=" * 60)
        print("Benchmark complete!")
        print(f"Results saved to: {results_dir}/")
        print("=" * 60)

    finally:
        client.close()
        print("\nConnection closed.")


# Need to import Path
from pathlib import Path

if __name__ == "__main__":
    main()
