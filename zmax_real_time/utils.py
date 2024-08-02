from pathlib import Path

import yaml


def load_yaml_config(config_path: Path) -> dict:
    """Load and return the configuration from a YAML file."""
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
