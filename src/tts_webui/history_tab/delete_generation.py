import shutil


def delete_generation(directory: str):
    shutil.rmtree(directory)
