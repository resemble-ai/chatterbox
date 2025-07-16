from tts_webui.utils.open_folder import open_folder
from gradio_iconbutton import IconButton


def OpenFolderButton(dirname="", api_name=None, **kwargs):
    button = IconButton(
        value="folder_open",
        **kwargs,
    )
    button.click(
        fn=lambda: open_folder(dirname),
        api_name=api_name,
    )
    return button
