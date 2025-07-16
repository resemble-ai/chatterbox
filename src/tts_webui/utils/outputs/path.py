import os

_dir_cache = {}


def _get_relative_output_path(result_dict, *args):
    """
    Get the relative path to the output directory.

    Args:
        result_dict (dict): The result dictionary.
        *args (str): The path arguments, e.g., "filename.wav".

    Returns:
        str: The relative path to the output directory.
    """
    dir = result_dict["folder_root"]
    if dir not in _dir_cache:
        os.makedirs(dir, exist_ok=True)
        _dir_cache[dir] = True
    return os.path.join(dir, *args)


def get_relative_output_path_ext(result_dict, ext: str):
    """
    Get the relative path to the output directory with an extension.

    Args:
        result_dict (dict): The result dictionary.
        ext (str): The extension, e.g., ".wav".

    Returns:
        str: The relative path to the output directory with an extension.
    """
    return _get_relative_output_path(result_dict, result_dict["filename"] + ext)
