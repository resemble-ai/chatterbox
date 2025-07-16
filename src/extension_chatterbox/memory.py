import sys
from collections import defaultdict
from tts_webui.utils.manage_model_state import get_current_model


def estimate_model_memory(model):
    memory_by_component = defaultdict(float)
    total_memory = 0

    for name, param in model.named_parameters():
        # Calculate memory usage
        memory_mb = sys.getsizeof(param.storage()) / 1024 / 1024
        total_memory += memory_mb

        # Group by component (everything before first dot)
        component = name.split(".")[0] if "." in name else name
        memory_by_component[component] += memory_mb

    return memory_by_component, total_memory


def estimate_combined_memory(model):
    memory_by_component, total_memory = estimate_model_memory(model.t3)
    memory_by_component_s3gen, total_memory_s3gen = estimate_model_memory(model.s3gen)
    for component, memory in memory_by_component_s3gen.items():
        memory_by_component[component] += memory
    total_memory += total_memory_s3gen
    return memory_by_component, total_memory


def memory_to_string(memory_by_component, total_memory):
    markdown = "| Component | Memory |\n| --- | --- |\n"
    for component, memory in memory_by_component.items():
        if memory < 20:
            continue
        markdown += f"| {component:<16} | {memory:>10.0f} MB |\n"
    markdown += f"| {'Total':<16} | {total_memory:>10.0f} MB |\n"
    return markdown


class Pipeline:
    def __init__(self, func=None):
        self.func = func or (lambda x: x)

    def __or__(self, next_func):
        def composed(*args):
            result = self.func(*args)
            if isinstance(result, tuple):
                return next_func(*result)
            return next_func(result)

        return Pipeline(composed)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


get_chatterbox_memory_usage = (
    Pipeline(lambda: "chatterbox")
    | get_current_model
    | estimate_combined_memory
    | memory_to_string
)
