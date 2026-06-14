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
    # Load local config file as base (if exists)
    config = {}
    local_config_path = Path(__file__).parent / 'remote_config_local.py'
    if local_config_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("remote_config_local", str(local_config_path))
            if spec is not None and spec.loader is not None:
                remote_config_local = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(remote_config_local)
                config = {
                    'host': getattr(remote_config_local, 'HOST', 'hz-4.matpool.com'),
                    'port': getattr(remote_config_local, 'PORT', 27609),
                    'username': getattr(remote_config_local, 'USERNAME', 'root'),
                    'password': getattr(remote_config_local, 'PASSWORD', None),
                    'ssh_key_path': getattr(remote_config_local, 'SSH_KEY_PATH', None),
                }
        except ImportError:
            pass

    # Override with environment variables (any env var takes precedence)
    env_host = os.environ.get('STATGPU_REMOTE_HOST')
    env_port = os.environ.get('STATGPU_REMOTE_PORT')
    env_user = os.environ.get('STATGPU_REMOTE_USER')
    env_password = os.environ.get('STATGPU_REMOTE_PASSWORD')
    env_key = os.environ.get('STATGPU_SSH_KEY_PATH')

    if env_host:
        config['host'] = env_host
    if env_port:
        config['port'] = int(env_port)
    if env_user:
        config['username'] = env_user
    if env_password:
        config['password'] = env_password
    if env_key:
        config['ssh_key_path'] = env_key

    if config:
        return config

    raise ValueError(
        "Remote server configuration not found. "
        "Please set environment variables (STATGPU_REMOTE_HOST, STATGPU_REMOTE_PASSWORD) "
        "or create dev/scripts/remote_config_local.py"
    )


# Default remote work directory
REMOTE_WORK_DIR = "/root/statgpu"
