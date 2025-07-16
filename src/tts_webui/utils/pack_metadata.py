import numpy as np
import json
from typing import Any


def pack_metadata(metadata: dict[str, Any]):
    # return list(json.dumps(metadata))
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o.__dict__

    return np.array(json.dumps(metadata, default=default))


def unpack_metadata(metadata: np.ndarray):
    def join_list(x: list | np.ndarray):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return "".join(x)

    return json.loads(join_list(metadata))
