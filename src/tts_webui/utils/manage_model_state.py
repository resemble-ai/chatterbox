import gradio as gr
from tts_webui.utils.torch_clear_memory import torch_clear_memory


def show(message):
    print(message)
    gr.Info(message)


class ModelState:
    def __init__(self):
        self._model = None
        self._model_name = None

    def set_model(self, model, model_name):
        self._model = model
        self._model_name = model_name

    def get_model(self):
        return self._model

    def is_model_loaded(self, model_name):
        return self._model is not None and self._model_name == model_name

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, model_name):
        self._model_name = model_name


model_states = {}


def manage_model_state(model_namespace):
    """Decorator to manage the model state."""

    def decorator(func):
        def wrapper(model_name, *args, **kwargs):
            global model_states
            if model_namespace not in model_states:
                model_states[model_namespace] = ModelState()

            model_state = model_states[model_namespace]

            if not model_state.is_model_loaded(model_name):
                show(
                    f"Model '{model_name}' in namespace '{model_namespace}' is not loaded or is different. Loading model..."
                )
                unload_model(model_namespace)
                model = func(model_name, *args, **kwargs)
                model_state.set_model(model, model_name)
            else:
                print(
                    f"Using cached model '{model_name}' in namespace '{model_namespace}'."
                )

            return model_state.get_model()

        return wrapper

    return decorator


def unload_model(model_namespace):
    if (
        model_namespace in model_states
        and model_states[model_namespace].get_model() is not None
    ):
        model_states[model_namespace].set_model(None, None)
        # del model_states[model_namespace]
        torch_clear_memory()
        show(f"Model in namespace '{model_namespace}' has been unloaded.")
    else:
        show(f"No model loaded in namespace '{model_namespace}'.")


def unload_all_models():
    for model_namespace in list(model_states.keys()):
        unload_model(model_namespace)


def list_loaded_models_as_markdown():
    lines = ["| Model Namespace | Model Name |", "|-----------------|------------|"]

    for namespace, state in model_states.items():
        model_name = state.get_model_name()
        if model_name:
            lines.append(f"| {namespace} | {model_name} |")
        else:
            lines.append(f"| {namespace} | Not Loaded |")

    return "\n".join(lines)


def is_model_loaded(model_namespace):
    return (
        model_namespace in model_states
        and model_states[model_namespace].get_model() is not None
    )


def rename_model(model_namespace, new_name):
    if model_namespace in model_states:
        model_states[model_namespace].set_model_name(new_name)


def get_current_model(model_namespace):
    if model_namespace in model_states:
        return model_states[model_namespace].get_model()
    else:
        return None
