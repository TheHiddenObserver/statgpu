"""
Remote GPU Server Configuration - LOCAL EXAMPLE

This file is an EXAMPLE. Create a copy named remote_config_local.py
with your actual credentials. DO NOT commit remote_config_local.py to git.

The actual credentials should be kept secret and never committed.
"""

# Remote server configuration
HOST = "hz-4.matpool.com"
PORT = 27609
USERNAME = "root"
PASSWORD = "YOUR_PASSWORD_HERE"  # Replace with your actual password

# Or use SSH key authentication (recommended)
# SSH_KEY_PATH = "/path/to/your/ssh_private_key"

# Remote work directory
REMOTE_WORK_DIR = "/root/statgpu"
