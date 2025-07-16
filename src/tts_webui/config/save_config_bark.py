from tts_webui.config.config import config
from tts_webui.config._save_config import _save_config


def save_config_bark(
    text_use_gpu: bool,
    text_use_small: bool,
    coarse_use_gpu: bool,
    coarse_use_small: bool,
    fine_use_gpu: bool,
    fine_use_small: bool,
    codec_use_gpu: bool,
):
    global config
    config["model"] = {
        "text_use_gpu": text_use_gpu,
        "text_use_small": text_use_small,
        "coarse_use_gpu": coarse_use_gpu,
        "coarse_use_small": coarse_use_small,
        "fine_use_gpu": fine_use_gpu,
        "fine_use_small": fine_use_small,
        "codec_use_gpu": codec_use_gpu,
    }
    _save_config(config)

    return f"Saved: {str(config['model'])}"
