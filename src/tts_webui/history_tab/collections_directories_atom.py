import os
import gradio as gr


def get_collections():
    from tts_webui.utils.get_path_from_root import get_path_from_root

    path = get_path_from_root("collections")
    dirs = os.listdir(path)
    dirs.sort()

    def get_collection_path(d):
        return os.path.join("collections", d)

    return ["outputs", "favorites"] + [
        get_collection_path(d) for d in dirs if os.path.isdir(get_collection_path(d))
    ]


collections_directories_atom = gr.JSON(visible=False, value=get_collections())
