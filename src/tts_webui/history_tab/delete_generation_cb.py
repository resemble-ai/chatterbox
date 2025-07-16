import shutil


def delete_generation_cb(refresh):
    def delete_generation(directory: str, *args):
        shutil.rmtree(directory)
        return refresh(*args)

    return delete_generation
