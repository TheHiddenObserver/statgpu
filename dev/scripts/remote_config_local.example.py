"""
Remote GPU Server Configuration - LOCAL EXAMPLE

This file is an EXAMPLE. Create a copy named remote_config_local.py
with your actual credentials. DO NOT commit remote_config_local.py to git.

The actual credentials should be kept secret and never committed.
"""

# Remote server configuration
HOST = "<your-gpu-server-hostname>"
PORT = <ssh-port>
USERNAME = "<username>"
PASSWORD = "<your-password>"  # Or use STATGPU_REMOTE_PASSWORD env var

# Or use SSH key authentication (recommended)
# SSH_KEY_PATH = "/path/to/your/ssh_private_key"

# Remote work directory (where statgpu source is uploaded)
REMOTE_WORK_DIR = "/path/to/statgpu"
