from tts_webui.utils.timer_seed_contexts import Timer


def log_function_time(fn):
    def wrapper(*args, **kwargs):
        with Timer():
            return fn(*args, **kwargs)

    return wrapper


def log_generator_time(generator):
    def wrapper(*args, **kwargs):
        with Timer():
            yield from generator(*args, **kwargs)

    return wrapper
