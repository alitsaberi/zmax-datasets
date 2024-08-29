from pathlib import Path

import yaml


def load_yaml_config(config_path: Path) -> dict:
    """Load and return the configuration from a YAML file."""
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def remove_tree(dir_path: Path) -> None:
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()  # Remove file
        elif item.is_dir():
            remove_tree(item)  # Recursively remove subdirectory

    # Finally, remove the directory itself
    dir_path.rmdir()
