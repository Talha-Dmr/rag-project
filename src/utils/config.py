"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    # Try multiple possible locations
    config_paths = [
        Path(f"config/{config_name}.yaml"),
        Path(f"../config/{config_name}.yaml"),
        Path(f"/kaggle/working/config/{config_name}.yaml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Config file '{config_name}.yaml' not found in: {[str(p) for p in config_paths]}"
    )
