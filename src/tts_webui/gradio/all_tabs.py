import gradio as gr

from tts_webui.extensions_loader import (
    extension_decorator_list_tab,
    extension_list_tab,
    handle_extension_class,
)
from tts_webui.extensions_loader.LoadingIndicator import LoadingIndicator
from tts_webui.history_tab.collections_directories_atom import (
    collections_directories_atom,
)
from tts_webui.utils.generic_error_tab_advanced import generic_error_tab_advanced


def reload_config_and_restart_ui():
    import os

    os._exit(0)
    # print("Reloading config and restarting UI...")
    # config = load_config()
    # gradio_interface_options = config["gradio_interface_options"] if "gradio_interface_options" in config else {}
    # demo.close()
    # time.sleep(1)
    # demo.launch(**gradio_interface_options)


def run_tab(module_name, function_name, name, requirements=None):
    import importlib

    with LoadingIndicator(name):
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            func()
        except Exception as e:
            generic_error_tab_advanced(e, name=name, requirements=requirements)


def load_tabs(list_of_tabs):
    for tab in list_of_tabs:
        module_name, function_name, name = tab[:3]
        requirements = tab[3] if len(tab) > 3 else None
        run_tab(module_name, function_name, name, requirements)


def all_tabs(config):
    with gr.Tab("💬 Text-to-Speech"), gr.Tabs():
        handle_extension_class("text-to-speech", config)
    with gr.Tab("🎼 Audio/Music Generation"), gr.Tabs():
        handle_extension_class("audio-music-generation", config)
    with gr.Tab("🎙️ Audio Conversion"), gr.Tabs():
        handle_extension_class("audio-conversion", config)
    with gr.Tab("📁 Outputs"), gr.Tabs():
        from tts_webui.history_tab.main import history_tab

        collections_directories_atom.render()
        try:
            history_tab()
            history_tab(directory="favorites")
            history_tab(
                directory="outputs",
                show_collections=True,
            )
        except Exception as e:
            generic_error_tab_advanced(e, name="History", requirements=None)

        outputs_tabs = []
        load_tabs(outputs_tabs)

        handle_extension_class("outputs", config)

    with gr.Tab("🔧 Tools"), gr.Tabs():
        handle_extension_class("tools", config)
    with gr.Tab("⚙️ Settings"), gr.Tabs():
        from tts_webui.settings_tab_gradio import settings_tab_gradio

        settings_tab_gradio(
            reload_config_and_restart_ui, config["gradio_interface_options"]
        )

        settings_tabs = [
            (
                "tts_webui.utils.model_location_settings_tab",
                "model_location_settings_tab",
                "Model Location Settings",
            ),
            ("tts_webui.utils.gpu_info_tab", "gpu_info_tab", "GPU Info"),
            ("tts_webui.utils.pip_list_tab", "pip_list_tab", "Installed Packages"),
        ]
        load_tabs(settings_tabs)

        extension_list_tab()
        extension_decorator_list_tab()

        handle_extension_class("settings", config)
    with gr.Tab("📚 Tutorials"), gr.Tabs():
        from tts_webui.tutorials.tab import tutorial_tab

        tutorial_tab()
