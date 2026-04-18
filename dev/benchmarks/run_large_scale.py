"""
Upload and run large-scale benchmark on remote server.
"""
import paramiko
import os
from pathlib import Path
import sys

# Import remote configuration from environment or local config file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from remote_config import get_remote_config, REMOTE_WORK_DIR

config = get_remote_config()
HOST = config['host']
PORT = config['port']
USERNAME = config['username']
PASSWORD = config['password']
SSH_KEY_PATH = config.get('ssh_key_path')


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 1800, print_output: bool = True):
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
    client.connect(
        hostname=HOST,
        port=PORT,
        username=USERNAME,
        password=PASSWORD,
        timeout=30,
        allow_agent=False,
        look_for_keys=False
    )
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

        # Step 2: Upload benchmark script
        print("[Step 2] Uploading benchmark script...")
        sftp = client.open_sftp()
        sftp.put("dev/benchmarks/benchmark_large_scale.py", "/root/benchmark_large_scale.py")
        sftp.close()
        print("  Uploaded: /root/benchmark_large_scale.py\n")

        # Step 3: Create results directory
        print("[Step 3] Creating results directory...")
        run(client, "mkdir -p /root/results/large_scale", timeout=30)

        # Step 4: Run benchmark
        print("[Step 4] Running large-scale benchmark (this may take a while)...")
        cmd = f"/root/miniconda3/envs/myconda/bin/python /root/benchmark_large_scale.py"
        code, out, err = run(client, cmd, timeout=1800, print_output=True)

        if code == 0:
            print("\n" + "=" * 60)
            print("Benchmark completed successfully!")
            print("=" * 60)

            # Step 5: Download results
            print("\n[Step 5] Downloading results...")
            sftp = client.open_sftp()

            results_dir = Path("results/large_scale")
            results_dir.mkdir(exist_ok=True)

            # Get the latest result files
            remote_files = sftp.listdir('/root/results/large_scale/')
            json_files = [f for f in remote_files if f.endswith('.json')]
            md_files = [f for f in remote_files if f.endswith('.md')]

            for f in json_files + md_files:
                remote_path = f'/root/results/large_scale/{f}'
                local_path = results_dir / f
                print(f"  Downloading: {f}")
                sftp.get(remote_path, str(local_path))

            sftp.close()
            print(f"\nResults downloaded to: {results_dir}/")
        else:
            print("\n" + "=" * 60)
            print("Benchmark FAILED!")
            print("=" * 60)

    finally:
        client.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
