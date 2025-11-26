"""
Configuration loading and management utilities.

Handles loading YAML configurations with environment variable substitution.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """
    Loads and manages configuration files with environment variable substitution
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        load_dotenv()  # Load .env file if it exists

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        config = self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax

        Args:
            config: Configuration object (dict, list, str, etc.)

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string(config)
        else:
            return config

    def _substitute_string(self, value: str) -> Any:
        """
        Substitute environment variables in a string

        Supports:
        - ${VAR_NAME} - Replace with env var, error if not found
        - ${VAR_NAME:default} - Replace with env var or default if not found

        Args:
            value: String potentially containing ${VAR} patterns

        Returns:
            String with substituted values, or original type if conversion possible
        """
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # If no default and env var not found, keep original
                return match.group(0)

        result = re.sub(pattern, replacer, value)

        # Try to convert to appropriate type
        return self._convert_type(result)

    def _convert_type(self, value: str) -> Any:
        """
        Convert string to appropriate Python type

        Args:
            value: String value

        Returns:
            Converted value (int, float, bool, or str)
        """
        # Don't convert if it still has ${...} patterns (unresolved vars)
        if '${' in value:
            return value

        # Try boolean
        if value.lower() in ('true', 'yes', 'on'):
            return True
        if value.lower() in ('false', 'no', 'off'):
            return False

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def save(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary
            config_name: Name of config file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files

        Later configs override earlier ones

        Args:
            *config_names: Names of config files to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}

        for name in config_names:
            config = self.load(name)
            merged = self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base dictionary
            override: Dictionary to merge in (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# Convenience function
def load_config(config_name: str = "base_config", config_dir: str = "config") -> Dict[str, Any]:
    """
    Quick function to load a config file

    Args:
        config_name: Name of config file (without .yaml extension)
        config_dir: Directory containing config files

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)
