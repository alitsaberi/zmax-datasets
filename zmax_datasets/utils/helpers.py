import inspect
import pkgutil
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TypeVar

import yaml
from loguru import logger

T = TypeVar("T")


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


def generate_timestamped_file_name(file_name: str, extension: str | None = None) -> str:
    timestamped_file_name = (
        f"{file_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    if extension:
        timestamped_file_name += f".{extension}"

    return timestamped_file_name


def get_class_by_name(
    class_name: str,
    module: ModuleType,
    base_class: type[T] | None = None,
    search_submodules: bool = True,
) -> type[T]:
    """
    Retrieve a class by its name in a module and its submodules,
    ensuring it is a subclass of a specified base class.

    Args:
        module (ModuleType): The Python module to search within.
        class_name (str): The name of the class to retrieve.
        base_class (type, optional): The base class that the target class must
            inherit from.
        search_submodules (bool): Whether to search in submodules recursively.

    Returns:
        type: the class object.
    """

    def find_class_in_module(mod: ModuleType) -> type[T] | None:
        """Helper function to find the class within a module."""
        try:
            cls = getattr(mod, class_name)
            if inspect.isclass(cls) and (
                base_class is None or issubclass(cls, base_class)
            ):
                return cls
        except AttributeError:
            pass  # Class not found in this module
        return None

    # Check in the main module
    if cls := find_class_in_module(module):
        return cls

    if search_submodules and hasattr(module, "__path__"):
        # Recursively search submodules
        for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            try:
                submodule = import_module(name)
                if cls := find_class_in_module(submodule):
                    return cls
            except ImportError as e:
                logger.debug(f"Failed to import submodule {name}: {e}")

    raise ValueError(
        f"No class named '{class_name}' found in module"
        f"'{module.__name__}' or its submodules."
    )
