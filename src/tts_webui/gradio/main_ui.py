import gradio as gr
from tts_webui.gradio.fix_gradio_tabs import fix_gradio_tabs
from tts_webui.gradio.all_tabs import all_tabs
from tts_webui.gradio.css import full_css
from tts_webui.gradio.get_theme import get_theme


def main_ui(config):
    fix_gradio_tabs()

    with gr.Blocks(
        css=full_css,
        title="TTS WebUI",
        analytics_enabled=False,  # it broke too many times
        theme=get_theme(config),
    ) as blocks:
        gr.HTML(
            """
        <div style="display: flex; align-items: baseline; gap: 1rem; font-family: sans-serif;">
            <h1>TTS WebUI</h1>
            | <h4>(The Gradio UI has more tools, but the UI is basic)</h4>
            | <h3><a href="http://localhost:3000">React UI</a></h3>
            | <h3><a href="https://forms.gle/2L62owhBsGFzdFBC8">Feedback / Bug reports</a></h3>
            | <h3><a href="https://discord.gg/V8BKTVRtJ9">Discord Server</a></h3>
        </div>
        """
        )

        with gr.Tabs():
            all_tabs(config)

    return blocks


if __name__ == "__main__":
    from tts_webui.config.config import config

    main_ui(config).launch(
        server_port=7770,
    )
