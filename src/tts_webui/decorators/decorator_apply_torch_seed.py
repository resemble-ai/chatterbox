from tts_webui.utils.timer_seed_contexts import Seed


def decorator_apply_torch_seed(fn):
    def wrapper(*args, **kwargs):
        with Seed(int(kwargs.get("seed", -1))):
            return fn(*args, **kwargs)

    return wrapper


def decorator_apply_torch_seed_generator(fn):
    def wrapper(*args, **kwargs):
        with Seed(int(kwargs.get("seed", -1))):
            yield from fn(*args, **kwargs)

    return wrapper
