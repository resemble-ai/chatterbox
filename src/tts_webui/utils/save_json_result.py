import json
from tts_webui.utils.outputs.path import get_relative_output_path_ext


def save_json_result(result_dict, metadata):
    with open(get_relative_output_path_ext(result_dict, ".json"), "w") as outfile:
        json.dump(metadata, outfile, indent=2)
