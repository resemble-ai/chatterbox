from tts_webui.utils.log_generation import middleware_log_generation


def decorator_log_generation(fn):
    def wrapper(*args, **kwargs):
        middleware_log_generation(kwargs)
        return fn(*args, **kwargs)

    return wrapper


def decorator_log_generation_generator(fn):
    def wrapper(*args, **kwargs):
        middleware_log_generation(kwargs)
        yield from fn(*args, **kwargs)

    return wrapper
