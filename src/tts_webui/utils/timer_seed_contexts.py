import time
from tts_webui.utils.set_seed import set_seed
from contextlib import contextmanager


@contextmanager
def Timer():
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Generated in", "{:.3f}".format(elapsed_time), "seconds")


@contextmanager
def Seed(seed):
    original_seed = set_seed(seed)
    yield
    set_seed(original_seed)
