import os
import gradio as gr
from gradio_iconbutton import IconButton

from tts_webui.utils.OpenFolderButton import OpenFolderButton
from tts_webui.utils.get_path_from_root import get_path_from_root
from tts_webui.utils.manage_model_state import unload_model


def _list_dir_models(abs_dir: str):
    try:
        # return [x for x in os.listdir(abs_dir) if x not in [".gitkeep", "cache"]]
        os.makedirs(abs_dir, exist_ok=True)
        walk_result = next(os.walk(abs_dir), (None, [], []))
        return [x for x in walk_result[1] if x not in ["cache"]]
    except FileNotFoundError as e:
        print(e)
        return []


def _get_models(repos, abs_dir):
    return repos + [(x, os.path.join(abs_dir, x)) for x in _list_dir_models(abs_dir)]


def model_select_ui(
    repos,
    prefix: str,
    Component: type[gr.Radio | gr.Dropdown] = gr.Radio,
):
    with gr.Column(variant="panel"):
        gr.Markdown("Model")
        with gr.Row():
            abs_dir = get_path_from_root("data", "models", prefix)
            models = _get_models(repos, abs_dir)
            model = Component(
                choices=models,
                # label="Model",
                show_label=False,
                value=models[0][1],
            )
            IconButton("refresh").click(
                fn=lambda: Component(choices=_get_models(repos, abs_dir)),
                outputs=[model],
                api_name=f"{prefix}_get_models",
            )
            OpenFolderButton(abs_dir, api_name=f"{prefix}_open_model_dir")
    return model


def unload_model_button(prefix: str):
    button = gr.Button(value="Unload Model", variant="stop")

    button.click(
        fn=lambda: gr.Button(
            value="Unloading...", variant="primary", interactive=False
        ),
        outputs=[button],
    ).then(
        fn=lambda: unload_model(model_namespace=prefix),
        api_name=f"{prefix}_unload_model",
    ).then(
        fn=lambda: gr.Button(value="Unload Model", variant="stop", interactive=True),
        outputs=[button],
    )
    return button
