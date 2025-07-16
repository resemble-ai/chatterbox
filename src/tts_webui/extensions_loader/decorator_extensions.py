import importlib
import importlib.util
from importlib.metadata import version
import time
from types import ModuleType
from typing import Literal
import functools

import gradio as gr

from tts_webui.utils.pip_install import pip_install_wrapper, pip_uninstall_wrapper
from tts_webui.utils.generic_error_tab_advanced import generic_error_tab_advanced
from tts_webui.extensions_loader.extensions_data_loader import (
    get_decorator_extensions,
    get_decorator_extensions_by_class,
)


def check_if_package_installed(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None


# A list of disabled extensions and decorators
disabled_extensions = ["decorator_disabled"]


# Get the decorator extensions list from the data loader
extension_list_json = get_decorator_extensions()


def extension_decorator_list_tab():
    with gr.Tab("Decorator Extensions List"):
        gr.Markdown("List of all extensions")
        table_string = """| Title | Description |\n| --- | --- |\n"""
        for x in extension_list_json:
            table_string += (
                # f"| {x['name']} (v{x['version']}) "
                f"| {x['name']} "
                + f"| {x['description']} (website: {x['website']}) (extension_website: {x['extension_website']}) |\n"
            )
        gr.Markdown(table_string)

        external_extension_list = [
            x for x in extension_list_json if "builtin" not in x["package_name"]
        ]

        with gr.Row():
            with gr.Column():
                gr.Markdown("Install/Uninstall Extensions")

                install_dropdown = gr.Dropdown(
                    label="Select Extension to Install",
                    choices=[x["package_name"] for x in external_extension_list],
                )

                install_button = gr.Button("Install extension")

                def install_extension(package_name):
                    requirements = [
                        x["requirements"]
                        for x in external_extension_list
                        if x["package_name"] == package_name
                    ][0]
                    yield from pip_install_wrapper(requirements, package_name)()

                install_button.click(
                    fn=install_extension,
                    inputs=[install_dropdown],
                    outputs=[gr.HTML()],
                    api_name="install_extension",
                )

            with gr.Column():
                gr.Markdown("Uninstall Extensions")
                uninstall_dropdown = gr.Dropdown(
                    label="Select Extension to Uninstall",
                    choices=[x["package_name"] for x in external_extension_list],
                )
                uninstall_button = gr.Button("Uninstall extension")

                def uninstall_extension(package_name):
                    yield from pip_uninstall_wrapper(package_name, package_name)()

                uninstall_button.click(
                    fn=uninstall_extension,
                    inputs=[uninstall_dropdown],
                    outputs=[gr.HTML()],
                    api_name="uninstall_extension",
                )


def _load_decorators(class_name: Literal["outer", "inner"]):
    """
    Loads all decorators from extensions.

    The decorators are loaded from the "main" module of the extension.
    Decorators must be functions prefixed with "decorator_".
    Generators are detected by the suffix "_generator".

    For example:
    def decorator_save_ogg(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper

    def decorator_save_ogg_generator(fn):
        def wrapper(*args, **kwargs):
            yield from fn(*args, **kwargs)
        return wrapper

    Args:
        class_name (str): "outer" or "inner"

    Returns:
        wrappers (list): List of decorators.
        gen_wrappers (list): List of decorators for generators.
    """
    wrappers = []
    gen_wrappers = []

    def _parse_module(module: ModuleType, name: str):
        if name.startswith("decorator_"):
            if name in disabled_extensions:
                print(f"  Skipping disabled decorator extension {name}")
                return
            if name.endswith("_generator"):
                gen_wrappers.append(getattr(module, name))
                print(f"  Decorator {name} loaded")
                return
            wrappers.append(getattr(module, name))
            print(f"  Decorator {name} loaded")

    def _load(x: dict):
        if x["package_name"] in disabled_extensions:
            print(f"Skipping disabled decorator extension {x['name']}")
            return
        module = importlib.import_module(f"{x['package_name']}.main")
        for name in dir(module):
            _parse_module(module, name)

    # Get decorator extensions filtered by class from the data loader
    filtered_extensions = get_decorator_extensions_by_class(class_name)

    for x in filtered_extensions:
        print(f"Loading decorator extension {x['name']}")
        start_time = time.time()
        try:
            _load(x)
        except Exception as e:
            print(f"Failed to load decorator extension {x['name']}: {e}")
        finally:
            elapsed_time = time.time() - start_time
            print(f"  Done in {elapsed_time:.2f} seconds. ({x['name']})\n")

    wrappers.reverse()
    gen_wrappers.reverse()
    return wrappers, gen_wrappers


OUTER_WRAPPERS, OUTER_WRAPPERS_GEN = _load_decorators("outer")
INNER_WRAPPERS, INNER_WRAPPERS_GEN = _load_decorators("inner")


def _create_decorator(wrappers_list):
    def decorator(fn0):
        for wrapper in wrappers_list:
            fn0 = wrapper(fn0)

        @functools.wraps(fn0)
        def wrapped(*args, **kwargs):
            return fn0(*args, **kwargs)

        return wrapped

    return decorator


def _create_decorator_generator(wrappers_list):
    def decorator(fn0):
        for wrapper in wrappers_list:
            fn0 = wrapper(fn0)

        @functools.wraps(fn0)
        def wrapped(*args, **kwargs):
            yield from fn0(*args, **kwargs)

        return wrapped

    return decorator


# Define the four decorators using the helper function
decorator_extension_outer = _create_decorator(OUTER_WRAPPERS)
decorator_extension_inner = _create_decorator(INNER_WRAPPERS)
decorator_extension_outer_generator = _create_decorator_generator(OUTER_WRAPPERS_GEN)
decorator_extension_inner_generator = _create_decorator_generator(INNER_WRAPPERS_GEN)

if __name__ == "__main__":
    pass
