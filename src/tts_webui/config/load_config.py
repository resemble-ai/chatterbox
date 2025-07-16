import json
import os

default_config = {
    "model": {
        "text_use_gpu": True,
        "text_use_small": True,
        "coarse_use_gpu": True,
        "coarse_use_small": True,
        "fine_use_gpu": True,
        "fine_use_small": True,
        "codec_use_gpu": True,
    },
    "gradio_interface_options": {
        "inline": False,
        "inbrowser": True,
        "share": False,
        "debug": False,
        "max_threads": 40,
        "auth": None,
        "auth_message": None,
        "prevent_thread_lock": False,
        "show_error": False,
        # "server_name": "0.0.0.0",
        "server_name": "127.0.0.1",
        "server_port": 7770,
        # "show_tips": False, # DEPRECATED
        "height": 500,
        "width": "100%",
        "favicon_path": None,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_keyfile_password": None,
        "ssl_verify": True,
        "quiet": True,
        "show_api": True,
        "_frontend": True,
    },
    "extensions": {
        "disabled": [],
    },
}


def load_config():
    if os.path.exists("config.json"):
        with open("config.json", "r") as infile:
            return json.load(infile)
    else:
        print("Config file not found. Creating default config.")
        with open("config.json", "w") as outfile:
            json.dump(default_config, outfile, indent=2)
        return default_config
