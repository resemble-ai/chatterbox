import os
import gradio as gr

import json
import shutil
from tts_webui.history_tab.delete_generation import delete_generation
from tts_webui.history_tab.collections_directories_atom import (
    collections_directories_atom,
    get_collections,
)
from tts_webui.history_tab.get_wav_files import get_wav_files
from tts_webui.history_tab.delete_generation_cb import delete_generation_cb
from tts_webui.history_tab.save_to_favorites import (
    save_to_collection,
    save_to_favorites,
)
from tts_webui.utils.open_folder import open_folder


def _get_row_index(evt: gr.SelectData):
    index: int | tuple[int, int] = evt.index
    return index[0] if isinstance(index, (list, tuple)) else index


def _get_filename(table, index):
    return table["data"][index][-1]


def get_json_text(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except:
        return None


def _select_audio(table, evt: gr.SelectData):
    index = _get_row_index(evt)
    filename = _get_filename(table, index)
    return filename, get_json_text(filename.replace(".wav", ".json"))


def clear_audio():
    return [
        gr.Audio(value=None),
        gr.Image(value=None),
        gr.JSON(value=None),
        gr.Button(visible=False),
    ]


def save_to_voices_cb(npz_filename: str):
    shutil.copy(npz_filename, "voices/")
    return gr.Button(value="Saved")


def history_tab(directory="outputs", show_collections=False):
    with gr.Tab(
        show_collections and "Collections" or directory.capitalize()
    ) as history_tab:
        return history_content(directory, history_tab, show_collections)


def history_content(directory, history_tab, show_collections):
    directories = get_collections()
    directory_dropdown = gr.Dropdown(
        value=directory,
        choices=directories,
        label="Select directory of the collection",
        visible=show_collections,
    )
    collections_directories_atom.change(
        fn=lambda x: gr.Dropdown(choices=x),
        inputs=[collections_directories_atom],
        outputs=[directory_dropdown],
    )

    if show_collections:
        create_collection_ui(collections_directories_atom)

    # with gr.Accordion("Gallery Selector (Click to Open)", open=False):
    #     history_list_as_gallery = gr.Gallery(
    #         value=[], columns=8, object_fit="contain", height="auto"
    #     )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                button_output = gr.Button(
                    value=f"Open {show_collections and 'collection' or directory} folder"
                )
                reload_button = gr.Button(value="Refresh", variant="secondary")
            button_output.click(
                lambda x: open_folder(x),
                inputs=[directory_dropdown],
                api_name=directory == "favorites" and "open_folder" or None,
            )

            datatypes = ["date", "str", "str", "str"]
            # headers = ["Date and Time", directory.capitalize(), "When", "Filename"]
            headers = ["Date and Time", "Name", "When", "Filename"]

            history_list = gr.Dataframe(
                value=[],
                type="array",
                interactive=False,
                col_count=len(datatypes),
                datatype=datatypes,
                headers=headers,
                max_height=800,
            )

        with gr.Column():
            history_bundle_name = gr.Markdown(visible=True)
            folder_root = gr.Textbox(visible=False)
            history_audio = gr.Audio(visible=True, type="filepath", show_label=False)
            history_image = gr.Image(show_label=False)
            history_json = gr.JSON()
            history_npz = gr.Textbox(visible=False)

            with gr.Row():
                delete_from_history = gr.Button(
                    value="Delete", variant="stop", visible=False
                )
                save_to_favorites_history = gr.Button(
                    value="Save to favorites", variant="primary", visible=False
                )
                gr.Markdown("""Use as voice button is now only available in React UI""")
                save_to_voices = gr.Button(
                    value="Save to voices", variant="secondary", visible=False
                )
                open_folder_button = gr.Button(
                    value="Open folder", variant="secondary", visible=False
                )
                open_folder_button.click(open_folder, inputs=folder_root)

                save_to_favorites_history.click(
                    fn=save_to_favorites,
                    inputs=folder_root,
                    outputs=save_to_favorites_history,
                )

                save_to_voices.click(
                    fn=save_to_voices_cb,
                    inputs=history_npz,
                    outputs=save_to_voices,
                    api_name=directory == "favorites" and "save_to_voices" or None,
                )

            save_to_collection_ui(
                directory,
                directories,
                folder_root,
                collections_directories_atom,
            )

    def _select_audio_history(filename: str, json_text):
        return {
            history_bundle_name: gr.Textbox(value=os.path.dirname(filename)),
            folder_root: os.path.dirname(filename),
            history_audio: gr.Audio(value=filename, label=filename),
            history_image: (
                gr.Image(value=filename.replace(".wav", ".png"))
                if os.path.exists(filename.replace(".wav", ".png"))
                else gr.Image(value=None)
            ),
            history_json: gr.JSON(value=json_text),
            history_npz: gr.Textbox(value=filename.replace(".wav", ".npz")),
            delete_from_history: gr.Button(visible=True),
            save_to_favorites_history: gr.Button(
                visible=directory != "favorites", value="Save to favorites"
            ),
            save_to_voices: gr.Button(visible=True, value="Save to voices"),
            open_folder_button: gr.Button(visible=True),
        }

    def select_audio_history(table, evt: gr.SelectData):
        return _select_audio_history(*_select_audio(table, evt))

    def select_audio_history2(_list, evt: gr.SelectData, table):
        return _select_audio_history(*_select_audio(table, evt))

    outputs = [
        history_bundle_name,
        folder_root,
        history_audio,
        history_image,
        history_json,
        history_npz,
        delete_from_history,
        save_to_favorites_history,
        save_to_voices,
        open_folder_button,
    ]

    history_list.select(
        fn=select_audio_history,
        inputs=[history_list],
        outputs=outputs,
        preprocess=False,
    )

    def update_history_tab(directory: str):
        return gr.Dataframe(value=get_wav_files(directory))

    delete_from_history.click(
        fn=clear_audio,
        outputs=[history_audio, history_image, history_json, delete_from_history],
    )
    delete_from_history.click(
        fn=delete_generation_cb(update_history_tab),
        inputs=[folder_root, directory_dropdown],
        # outputs=[history_list, history_list_as_gallery],
        outputs=[history_list],
    )
    # API ONLY
    gr.Button(
        value="Delete (API ONLY)",
        visible=False,
    ).click(
        fn=delete_generation,
        inputs=[folder_root],
        api_name=directory == "favorites" and "delete_generation" or None,
    )
    history_tab.select(
        fn=update_history_tab,
        inputs=[directory_dropdown],
        # outputs=[history_list, history_list_as_gallery],
        outputs=[history_list],
    )

    directory_dropdown.change(
        fn=update_history_tab,
        inputs=[directory_dropdown],
        # outputs=[history_list, history_list_as_gallery],
        outputs=[history_list],
    )

    reload_button.click(
        fn=lambda x: gr.Dataframe(value=get_wav_files(x)),
        inputs=[directory_dropdown],
        outputs=[history_list],
        api_name=f"{'collections' if show_collections else directory}_refresh_history",
    )


def save_to_collection_ui(
    directory: str,
    directories: list[str],
    folder_root: gr.Textbox,
    directories_state: gr.JSON,
):
    with gr.Row():
        move_to_collection = gr.Dropdown(
            label="Save to collection",
            choices=directories,
            value=directory,
        )

    move_to_collection.select(
        fn=save_to_collection,
        inputs=[folder_root, move_to_collection],
        outputs=[move_to_collection],
    )

    directories_state.change(
        fn=lambda x: gr.Dropdown(choices=x),
        inputs=[directories_state],
        outputs=[move_to_collection],
    )


def create_collection_ui(directories_state):
    new_collection_name = gr.Textbox(label="New collection name", value="")

    def create_collection(new_collection_name):
        os.makedirs(os.path.join("collections", new_collection_name))
        return [
            get_collections(),
            gr.Button(value="Created"),
        ]

    create_collection_button = gr.Button(value="Create collection")

    new_collection_name.change(
        fn=lambda: gr.Button(value="Create collection"),
        outputs=[create_collection_button],
    )
    create_collection_button.click(
        fn=create_collection,
        inputs=[new_collection_name],
        outputs=[directories_state, create_collection_button],
    )
