import subprocess
import sys

if sys.platform == "darwin":

    def open_folder(folder_path: str):
        subprocess.check_call(["open", "--", folder_path])

elif sys.platform.startswith("linux"):

    def open_folder(folder_path: str):
        subprocess.check_call(["xdg-open", folder_path])

elif sys.platform == "win32":

    def open_folder(folder_path: str):
        subprocess.Popen(["explorer", folder_path])


if __name__ == "__main__":
    # open_folder("./data/models/")
    import os

    open_folder(os.path.join(os.path.dirname(__file__), "..", "data", "models"))
