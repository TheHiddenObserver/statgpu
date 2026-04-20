"""
Remote GPU Server Configuration

This module provides configuration for connecting to remote GPU servers.
Sensitive credentials should be set via environment variables or a local
config file that is NOT committed to git.

Usage:
------
1. Set environment variables:
   export STATGPU_REMOTE_HOST="hz-4.matpool.com"
   export STATGPU_REMOTE_PORT="27609"
   export STATGPU_REMOTE_USER="root"
   export STATGPU_REMOTE_PASSWORD="your_password_here"

2. Or create a local config file (dev/scripts/remote_config_local.py):
   HOST = "hz-4.matpool.com"
   PORT = 27609
   USERNAME = "root"
   PASSWORD = "your_password_here"

3. Or use SSH key authentication (recommended):
   export STATGPU_SSH_KEY_PATH="/path/to/your/ssh_key"
"""

import os
import importlib.util
from pathlib import Path


def get_remote_config():
    """
    Get remote server configuration from environment variables or local config file.

    Priority:
    1. Environment variables (STATGPU_REMOTE_*)
    2. Local config file (dev/scripts/remote_config_local.py)
    3. Raise error if neither is available

    Returns
    -------
    dict
        Dictionary with keys: host, port, username, password, ssh_key_path
    """
    # Try environment variables first
    host = os.environ.get('STATGPU_REMOTE_HOST')
    port = os.environ.get('STATGPU_REMOTE_PORT')
    username = os.environ.get('STATGPU_REMOTE_USER')
    password = os.environ.get('STATGPU_REMOTE_PASSWORD')
    ssh_key_path = os.environ.get('STATGPU_SSH_KEY_PATH')

    if host and (password or ssh_key_path):
        return {
            'host': host,
            'port': int(port) if port else 27609,
            'username': username or 'root',
            'password': password,
            'ssh_key_path': ssh_key_path,
        }

    # Try local config file
    local_config_path = Path(__file__).parent / 'remote_config_local.py'
    if local_config_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("remote_config_local", str(local_config_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load spec from {local_config_path}")
            remote_config_local = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(remote_config_local)
            return {
                'host': getattr(remote_config_local, 'HOST', 'hz-4.matpool.com'),
                'port': getattr(remote_config_local, 'PORT', 27609),
                'username': getattr(remote_config_local, 'USERNAME', 'root'),
                'password': getattr(remote_config_local, 'PASSWORD', None),
                'ssh_key_path': getattr(remote_config_local, 'SSH_KEY_PATH', None),
            }
        except ImportError as e:
            raise ImportError(f"Failed to import local config: {e}")

    raise ValueError(
        "Remote server configuration not found. "
        "Please set environment variables (STATGPU_REMOTE_HOST, STATGPU_REMOTE_PASSWORD) "
        "or create dev/scripts/remote_config_local.py"
    )


# Default remote work directory
REMOTE_WORK_DIR = "/root/statgpu"
