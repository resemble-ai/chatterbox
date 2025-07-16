from deprecated import deprecated
from tts_webui.utils.open_folder import open_folder
from gradio_iconbutton import IconButton


@deprecated(
    version="0.0.1",
    reason="Use IconButton instead",
)
def gr_reload_button(**kwargs):
    return IconButton(
        value="refresh",
        size="sm",
        **kwargs,
    )


@deprecated(
    version="0.0.1",
    reason="Use OpenFolderButton instead",
)
def gr_open_button_simple(dirname="", api_name=None, **kwargs):
    return IconButton(
        value="folder_open",
        size="sm",
        **kwargs,
    ).click(
        fn=lambda: open_folder(dirname),
        api_name=api_name,
    )
