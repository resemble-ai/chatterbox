"""
Extensions data loader module.

This module provides functions to load extensions data from various sources.
It abstracts the data loading process to enable more complex data sources in the future,
such as multiple JSON file merges or fetching from external sources.
"""

import json
import os
from typing import Dict, List, Any, Optional


# Default paths for extensions files
DEFAULT_EXTENSIONS_FILE = "extensions.json"
EXTERNAL_EXTENSIONS_FILE = "extensions.external.json"


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: The contents of the JSON file as a dictionary.
        Returns an empty dict if the file cannot be loaded.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"\n! Failed to load {file_path}: {e}")
        return {}


def load_extensions_json() -> Dict[str, Any]:
    """
    Load the extensions.json file.

    Returns:
        Dict[str, Any]: The contents of extensions.json as a dictionary.
        Returns an empty dict if the file cannot be loaded.
    """
    return load_json_file(DEFAULT_EXTENSIONS_FILE)


def merge_extensions_data(base_data: Dict[str, Any], additional_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two extension data dictionaries.

    Args:
        base_data (Dict[str, Any]): Base extensions data.
        additional_data (Dict[str, Any]): Additional extensions data to merge.

    Returns:
        Dict[str, Any]: Merged extensions data.
    """
    result = base_data.copy()

    for key, value in additional_data.items():
        if key in result and isinstance(value, list) and isinstance(result[key], list):
            # For lists (like tabs and decorators), extend the existing list
            # but avoid duplicates based on package_name
            existing_package_names = {item.get("package_name") for item in result[key] if isinstance(item, dict) and "package_name" in item}

            for item in value:
                if isinstance(item, dict) and "package_name" in item:
                    if item["package_name"] not in existing_package_names:
                        result[key].append(item)
                        existing_package_names.add(item["package_name"])
                else:
                    # If it doesn't have a package_name, just append it
                    result[key].append(item)
        elif key not in result:
            # If the key doesn't exist in the base data, add it
            result[key] = value

    return result


def load_merged_extensions_data() -> Dict[str, Any]:
    """
    Load and merge extensions data from multiple sources.

    Returns:
        Dict[str, Any]: Merged extensions data from all sources.
    """
    # Load base extensions data
    extensions_data = load_extensions_json()

    # Load external extensions data if it exists
    if os.path.exists(EXTERNAL_EXTENSIONS_FILE):
        external_data = load_json_file(EXTERNAL_EXTENSIONS_FILE)
        if external_data:
            extensions_data = merge_extensions_data(extensions_data, external_data)

    return extensions_data


def get_decorator_extensions() -> List[Dict[str, Any]]:
    """
    Get the list of decorator extensions.

    Returns:
        List[Dict[str, Any]]: List of decorator extensions.
    """
    extensions_data = load_merged_extensions_data()
    return extensions_data.get("decorators", [])


def get_interface_extensions() -> List[Dict[str, Any]]:
    """
    Get the list of interface extensions (tabs).

    Returns:
        List[Dict[str, Any]]: List of interface extensions.
    """
    extensions_data = load_merged_extensions_data()
    return extensions_data.get("tabs", [])


def filter_extensions_by_type_and_class(
    extensions: List[Dict[str, Any]],
    extension_type: str,
    extension_class: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter extensions by type and class.

    Args:
        extensions (List[Dict[str, Any]]): List of extensions to filter.
        extension_type (str): Extension type to filter by.
        extension_class (Optional[str], optional): Extension class to filter by.
            If None, only filters by type. Defaults to None.

    Returns:
        List[Dict[str, Any]]: Filtered list of extensions.
    """
    if extension_class is None:
        return [x for x in extensions if x.get("extension_type") == extension_type]

    return [
        x for x in extensions
        if x.get("extension_type") == extension_type and x.get("extension_class") == extension_class
    ]


def get_decorator_extensions_by_class(class_name: str) -> List[Dict[str, Any]]:
    """
    Get decorator extensions filtered by class.

    Args:
        class_name (str): Class name to filter by (e.g., "outer", "inner").

    Returns:
        List[Dict[str, Any]]: Filtered list of decorator extensions.
    """
    decorators = get_decorator_extensions()
    return filter_extensions_by_type_and_class(decorators, "decorator", class_name)


def get_interface_extensions_by_class(class_name: str) -> List[Dict[str, Any]]:
    """
    Get interface extensions filtered by class.

    Args:
        class_name (str): Class name to filter by.

    Returns:
        List[Dict[str, Any]]: Filtered list of interface extensions.
    """
    interfaces = get_interface_extensions()
    return filter_extensions_by_type_and_class(interfaces, "interface", class_name)


def get_extension_example() -> Dict[str, Any]:
    """
    Get the example extension template.

    Returns:
        Dict[str, Any]: Example extension template.
    """
    extensions_data = load_merged_extensions_data()
    return extensions_data.get("example_extension", {})


def create_empty_external_extensions_file() -> bool:
    """
    Create an empty external extensions file if it doesn't exist.

    Returns:
        bool: True if the file was created, False otherwise.
    """
    if not os.path.exists(EXTERNAL_EXTENSIONS_FILE):
        try:
            with open(EXTERNAL_EXTENSIONS_FILE, "w") as f:
                json.dump({"tabs": [], "decorators": []}, f, indent=4)
            print(f"Created empty {EXTERNAL_EXTENSIONS_FILE}")
            return True
        except Exception as e:
            print(f"Failed to create {EXTERNAL_EXTENSIONS_FILE}: {e}")
            return False
    return False
