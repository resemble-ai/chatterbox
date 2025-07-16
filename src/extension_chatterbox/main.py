import gradio as gr


def extension__tts_generation_webui():
    ui_wrapper()
    return {
        "package_name": "extension_chatterbox",
        "name": "Chatterbox",
        "requirements": "git+https://github.com/rsxdalv/extension_chatterbox@main",
        "description": "Chatterbox, Resemble AI's first production-grade open source TTS model",
        "extension_type": "interface",
        "extension_class": "tools",
        "author": "Resemble AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/resemble-ai/chatterbox",
        "extension_website": "https://github.com/rsxdalv/extension_chatterbox",
        "extension_platform_version": "0.0.1",
    }


def ui_wrapper():
    from .gradio_app import ui

    ui()


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch()
