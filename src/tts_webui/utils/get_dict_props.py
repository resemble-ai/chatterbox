def get_dict_props(kwargs, props):
    return {k: kwargs[k] for k in props if k in kwargs}
