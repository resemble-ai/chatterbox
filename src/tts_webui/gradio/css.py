import os


def load_css(filename: str):
    with open(filename, "r") as f:
        return f.read()


full_css = load_css(os.path.join(os.path.dirname(__file__), "tts_webui.css"))
