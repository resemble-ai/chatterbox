from datetime import datetime


def decorator_add_date(fn):
    def wrapper(*args, **kwargs):
        result_dict = fn(*args, **kwargs)
        result_dict["date"] = datetime.now()
        return result_dict

    return wrapper


def decorator_add_date_generator(fn):
    def wrapper(*args, **kwargs):
        for result_dict in fn(*args, **kwargs):
            if result_dict is None:
                continue
            result_dict["date"] = datetime.now()
            yield result_dict

    return wrapper
