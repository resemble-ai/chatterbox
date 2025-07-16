import os
import datetime

from tts_webui.history_tab.generate_pretty_name import generate_pretty_name
from tts_webui.history_tab.generate_relative_date import generate_relative_date
from tts_webui.history_tab.parse_time import extract_and_parse_time


def get_wav_files(directory: str):
    def get_wav_in_dir(dir_path: str):
        return os.path.join(dir_path, f"{os.path.basename(dir_path)}.wav")

    file_date_list = []

    root, list_of_directories, _ = os.walk(directory).__next__()

    def get_directory_info(directory: str):
        full_path = os.path.join(root, directory)
        timestamp = extract_and_parse_time(full_path)
        return [
            timestamp or datetime.datetime(1970, 1, 1),
            generate_pretty_name(full_path),
            generate_relative_date(timestamp),  # type: ignore
            get_wav_in_dir(full_path),
        ]

    file_date_list = [
        get_directory_info(directory) for directory in list_of_directories
    ]

    # order by date
    file_date_list.sort(key=lambda x: x[0], reverse=True)
    return file_date_list


def generate_pretty_name_npz(name: str):
    return name.replace(".npz", "").strip().capitalize()
